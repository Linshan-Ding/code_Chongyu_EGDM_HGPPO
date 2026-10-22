"""Regression checks for the self-contained current experimental workflow."""
import ast
import copy
import importlib
import pkgutil
import tempfile
import unittest
from pathlib import Path
import torch
from agent.paper.config import ROOT, FORMAT, settings, digest, project_config
from agent.paper.data import rows, prepare, restore_registered_file
from agent.paper.storage import save_checkpoint, load_checkpoint
from agent.paper.matrix import matrix
from data.generator import InstanceGenerator


class CleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'result').mkdir(exist_ok=True)
        cls.options = settings()

    def test_fixed_files_and_registration(self):
        records = prepare(self.options)
        self.assertEqual(len(records), 92)
        from agent.paper.data import registered_manifest
        manifest = registered_manifest(self.options)
        self.assertEqual(manifest['inactive_instances'], [])
        self.assertEqual(len(manifest['removed_instances']), 36)
        # prepare must not recreate deleted XL or retired L training/development data.
        prepare(self.options)
        self.assertTrue(all(not (ROOT / r['path']).exists() for r in manifest['removed_instances']))
        root = ROOT / 'data/instances/paper_batch_epoch'
        self.assertEqual({p.resolve() for p in root.rglob('*.csv')}, {(ROOT / r['path']).resolve() for r in records})
        for row in records:
            self.assertEqual(digest(ROOT / row['path']), row['sha256'])
            self.assertEqual(Path(row['path']).parent.name, row.get('storage_split', row['split']))

    def test_missing_file_exact_recovery_and_corruption_rejection(self):
        original = rows(self.options, 'tiny')[0]
        generator = InstanceGenerator(project_config(self.options))
        with tempfile.TemporaryDirectory(dir=ROOT / 'result') as tmp:
            row = dict(original, path=str(Path(tmp) / 'restored.csv'))
            restore_registered_file(row, generator)
            self.assertEqual(Path(row['path']).read_bytes(), (ROOT / original['path']).read_bytes())
            Path(row['path']).write_text('corrupted', 'utf-8')
            with self.assertRaisesRegex(ValueError, 'Frozen data changed'):
                restore_registered_file(row, generator)
            bad = dict(original, path=str(Path(tmp) / 'unpublished.csv'), sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'registered hash'):
                restore_registered_file(bad, generator)
            self.assertFalse(Path(bad['path']).exists())

    def test_current_import_closure(self):
        forbidden = ('agent.v2', 'agent.atc_residual', 'agent.training', 'agent.evaluation', 'agent.experiments', 'agent.encoder',
                     'environment.replay_state', 'environment.graph_normalization')
        for folder in ('agent', 'environment', 'data', 'configs', 'scripts', 'tests'):
            for file in (ROOT / folder).rglob('*.py'):
                tree = ast.parse(file.read_text('utf-8-sig'))
                for node in ast.walk(tree):
                    names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                             [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                    self.assertFalse(any(n.startswith(forbidden) for n in names), str(file))
        for package_name in ('agent', 'environment', 'data', 'configs'):
            package = importlib.import_module(package_name)
            for module in pkgutil.walk_packages(package.__path__, package_name+'.'):
                importlib.import_module(module.name)
        from agent.paper.dan_official import official_classes
        self.assertIn('DualAttentionNetwork', official_classes())

    def test_matrix_counts_and_missing_reports(self):
        from agent.paper.report import report
        o = self.options
        per_learner = 1 + 2 * o['replays']
        from agent.paper.config import LEARNERS
        learners = 3 * len(LEARNERS)
        expected = {'pilot-single': (6 * per_learner, 6 * o['pilot_updates']),
                    'pilot-joint': (learners * per_learner + 8, learners * o['pilot_updates']),
                    'formal': (learners * per_learner + 9, learners * o['formal_updates'])}
        for suite, (jobs, epochs) in expected.items():
            plan = matrix(self.options, suite)
            training = [j for j in plan['jobs'] if j['kind']=='train']
            self.assertEqual(len(plan['jobs']), jobs)
            self.assertEqual(sum(j['budget_epochs'] for j in training), epochs)
            self.assertEqual(sum(j['budget_episodes'] for j in training), epochs * o['environments'])
            with tempfile.TemporaryDirectory(dir=ROOT / 'result') as tmp:
                result = report(dict(self.options, result_root=tmp), suite)
                self.assertFalse(result['complete'])
                self.assertTrue(all(r['status']=='missing' for r in result['all_results']))

    def test_reject_pre_cleanup_checkpoints(self):
        self.assertEqual(FORMAT, 'paper-sm100-2')
        with tempfile.TemporaryDirectory(dir=ROOT / 'result') as tmp:
            path = Path(tmp) / 'legacy.pt'
            torch.save(dict(format='paper-sm100-1', signature='same'), path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, 'same')


if __name__ == '__main__': unittest.main()
