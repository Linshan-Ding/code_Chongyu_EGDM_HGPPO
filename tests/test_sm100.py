import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from agent.paper.config import ROOT, FORMAT, settings, scientific_options, source_manifest
from agent.paper.data import rows, load
from agent.paper.experiment import protocol_hash, suite_root, host_id, local_root
from agent.paper.matrix import matrix, select_jobs
from agent.paper.storage import write_json, read_json, save_checkpoint, publish_commit
from agent.paper.rollout import batch_summary
from agent.paper.report import training_tail, report
from agent.paper.train import signature
from agent.paper.transfer import seal, collect, job_identity


def episode_fixture(epoch, iid, env_id, twt, baseline=1000., group=None):
    return dict(epoch=epoch, instance_id=iid, instance_index=0, replicate=env_id, group=group or iid, perturbation_seed=None,
                instance_sha256='hash', initial_state_sha256='initial', policy_version=epoch - 1, env_id=env_id, episode=epoch,
                action_seed=epoch * 1000 + env_id, baseline_twt=baseline, twt=float(twt), rel_twt=float(twt) / baseline,
                feasible=True, failure='', episode_events=10, moves=0, travel=0., anchor_fraction=1., gate_fraction=0.,
                wall_seconds=0., decision_seconds=0., step_seconds=0., trace_sha256='t')


class SM100Tests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'result').mkdir(exist_ok=True)
        self.options = settings()

    def test_portable_identity_and_filters(self):
        o = self.options
        other = dict(o, result_root='Z:/different/result', workers=2, update_threads=2, device='cuda', data_manifest=o['data_manifest'])
        self.assertEqual(protocol_hash(o), protocol_hash(other))
        self.assertNotEqual(protocol_hash(o), protocol_hash(dict(o, actor_lr=.001)))
        self.assertNotEqual(protocol_hash(o), protocol_hash(dict(o, candidate_version='v1')))
        plan = matrix(o, 'formal')
        chosen = select_jobs(plan, phase='train', methods=['no_graph'], seeds=[2])
        self.assertEqual([(j['method'], j['seed']) for j in chosen], [('mlp', 2)])
        self.assertEqual(chosen[0]['budget_epochs'], o['formal_updates'])
        self.assertEqual(chosen[0]['budget_episodes'], o['formal_updates'] * o['environments'])
        self.assertAlmostEqual(set(chosen[0]['expected_visits'].values()).pop(), o['formal_updates'] * o['instances_per_update'] / 36)
        self.assertTrue(all(r['parameters']['scale'] in ('S', 'M') for r in chosen[0]['training']))
        self.assertEqual(len(chosen[0]['validation']), 10)
        self.assertTrue(all(r['split'] == 'validation' for r in chosen[0]['validation']))
        with self.assertRaises(ValueError): select_jobs(plan, phase='train', methods=['unknown'])
        with self.assertRaises(ValueError): settings(evaluation_epochs=2)
        with self.assertRaises(ValueError): settings(pilot_epochs=300)
        with self.assertRaises(ValueError): settings(formal_epochs_per_instance=100)
        from agent.paper.runtime import alive
        self.assertFalse(alive(dict(host_id='foreign-host', pid=1, created_at=1, command=[])))

    def test_exclusive_execution_lease(self):
        from agent.paper.runtime import exclusive_lease
        with tempfile.TemporaryDirectory(dir=ROOT/'result') as tmp:
            path=Path(tmp)/'execution.lease'
            with exclusive_lease(path):
                with self.assertRaisesRegex(RuntimeError,'execution lease'):
                    with exclusive_lease(path): pass
            with exclusive_lease(path): pass

    def test_tail_is_per_case_and_never_partial_mean(self):
        o = dict(self.options, instances_per_update=2, episodes_per_instance=4, environments=8, tail_epochs=3)
        with tempfile.TemporaryDirectory(dir=ROOT / 'result') as tmp:
            folder = Path(tmp)
            cases = rows(o, 'pilot')
            ids = [r['instance_id'] for r in cases]
            job = dict(method='full', seed=0, output=str(folder), training=cases, budget_epochs=5,
                       expected_visits={iid: 5. for iid in ids})
            for epoch in range(6):
                eps, sums = [], []
                if epoch:
                    # Both cases appear in every update: four replicates each.
                    eps = [episode_fixture(epoch, ids[i // 4], i, 1000 + epoch + i) for i in range(8)]
                    sums = [dict(batch_summary(eps, 8), epoch=epoch, environments=8)]
                publish_commit(folder, dict(epoch=epoch, events=epoch*80, policy_version=epoch, progress=dict(committed_episodes=epoch*8),
                    last_bundle=dict(epoch=epoch, episodes=eps, summaries=sums, evaluations=[], validation_summaries=[])))
            tail = training_tail(o, job)
            self.assertEqual([r['status'] for r in tail], ['complete', 'complete'])
            self.assertEqual([r['tail_episodes'] for r in tail], [12, 12])
            # Last three updates (3, 4, 5): case 0 slots 0-3, case 1 slots 4-7.
            self.assertAlmostEqual(tail[0]['mean_twt'], np.mean([1000 + e + i for e in (3, 4, 5) for i in range(4)]))
            self.assertAlmostEqual(tail[1]['mean_twt'], np.mean([1000 + e + i for e in (3, 4, 5) for i in range(4, 8)]))
            self.assertEqual([r['committed_visits'] for r in tail], [5, 5])
            damaged = folder / 'epochs/000004.json'
            bundle = read_json(damaged); bundle['episodes'][-1]['feasible'] = False; write_json(damaged, bundle)
            self.assertTrue(all(r['mean_twt'] is None and r['status'] == 'invalid' for r in training_tail(o, job)))

    def test_transfer_idempotence_conflicts_and_missing(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'result') as tmp:
            o = dict(self.options, result_root=str(Path(tmp)/'nodeA'))
            dest_options = dict(o, result_root=str(Path(tmp)/'hub'),workers=2)
            job = next(j for j in matrix(o,'pilot-joint')['jobs'] if j['kind']=='train')
            folder=Path(job['output']); baseline={r['instance_id']:1000. for r in job['training'] + job['validation']}
            sig=signature(o,job['method'],job['seed'],job['training'],job['validation'],baseline)
            write_json(folder/'run.json',dict(format=FORMAT,protocol_hash=protocol_hash(o),job=job_identity(job)))
            write_json(folder/'baselines.json',baseline)
            write_json(folder/'config_snapshot.json',dict(signature=sig,source=source_manifest(),options=o,
                target_epochs=job['budget_epochs'],instance_order=[r['instance_id'] for r in job['training']]))
            progress=dict(epoch=0,committed_episodes=0,committed_events=0)
            write_json(folder/'progress.json',progress)
            save_checkpoint(folder/'latest.pt',dict(signature=sig,scientific_options=scientific_options(o),source_digest=source_manifest()['digest'],
                epoch=0,policy_version=0,progress=progress))
            write_json(folder/'job-status.json',dict(status='failed',failure='fixture interrupted before sampling'))
            seal(folder)
            a=collect(dest_options,folder);self.assertEqual(a['imported'],[job['id']]);self.assertFalse(a['rejected'])
            b=collect(dest_options,folder);self.assertEqual(b['unchanged'],[job['id']])
            target=Path(next(j for j in matrix(dest_options,'pilot-joint')['jobs'] if j['id']==job['id'])['output'])
            write_json(folder/'job-status.json',dict(status='failed',failure='second conflicting attempt'));seal(folder)
            c=collect(dest_options,folder);self.assertEqual(len(c['conflicts']),1)
            self.assertEqual(read_json(target/'job-status.json')['failure'],'fixture interrupted before sampling')
            self.assertFalse(report(dest_options,'pilot-joint')['complete'])
            (folder/'progress.json').write_text('{}','utf-8')
            self.assertEqual(len(collect(dest_options,folder)['rejected']),1)

    def test_training_validates_only_on_development_cases_and_formal_is_guarded(self):
        from agent.paper.train import train
        from agent.paper.runtime import check_formal_approval
        with tempfile.TemporaryDirectory(dir=ROOT/'result') as tmp:
            o=dict(self.options,result_root=tmp,workers=1,update_threads=1,ppo_update_epochs=1,minibatch=32,microbatch=16,
                   instances_per_update=1,episodes_per_instance=8,environments=8)
            tiny=rows(o,'tiny');case=load(tiny[0]);bs={case.instance_id:1000.}
            with patch('agent.paper.evaluate.evaluate',side_effect=AssertionError('Training invoked evaluation')):
                result=train(o,'mlp',0,[case],[],bs,Path(tmp)/'train',1,'fixture')
            self.assertEqual(result['completed_episodes'],8)
            self.assertFalse((Path(tmp)/'train/best.pt').exists())
            self.assertTrue(all(not read_json(p)['evaluations'] for p in (Path(tmp)/'train/epochs').glob('*.json')))
            # Development selection may only read the registered validation split.
            with self.assertRaisesRegex(ValueError,'development split'):
                signature(o,'mlp',0,tiny[:1],tiny[1:2],bs)
            main=[r for r in rows(o,'main')][:1]
            with self.assertRaisesRegex(ValueError,'development split'):
                signature(o,'mlp',0,tiny[:1],main,bs)
            signature(o,'mlp',0,tiny[:1],rows(o,'validation')[:1],bs)
            with self.assertRaisesRegex(ValueError,'baselines missing'):
                train(o,'mlp',0,[case],[load(tiny[1])],bs,Path(tmp)/'bad',1,'fixture')
            with self.assertRaisesRegex(RuntimeError,'reviewed pilot'):
                check_formal_approval(o)

    def test_registered_node_to_hub_final_replay(self):
        """Exercise real checkpoint/optimizer transport through orchestration on a tiny fixture."""
        from agent.paper import runtime
        with tempfile.TemporaryDirectory(dir=ROOT/'result') as tmp:
            a=dict(self.options,result_root=str(Path(tmp)/'node'),workers=1,update_threads=1,
                   ppo_update_epochs=1,minibatch=32,microbatch=16,instances_per_update=1,episodes_per_instance=8,environments=8)
            b=dict(a,result_root=str(Path(tmp)/'hub'),workers=2)
            tiny=rows(a,'tiny')[0]
            def fixture_plan(options,suite):
                plan=matrix(options,suite)
                train_job=next(j for j in plan['jobs'] if j['kind']=='train' and j['method']=='mlp' and j['seed']==0)
                train_job.update(training=[tiny],validation=[],budget_epochs=1,budget_episodes=8,expected_visits={tiny['instance_id']:1.})
                replay=next(j for j in plan['jobs'] if j['kind']=='replay' and j['training_job']==train_job['id']
                            and j['repeat']==0 and j['checkpoint_kind']=='latest')
                replay.update(evaluation=[tiny],budget_epochs=1,budget_episodes=8)
                plan['jobs']=[train_job,replay]
                return plan
            with patch('agent.paper.runtime.matrix',side_effect=fixture_plan):
                runtime.run_suite(a,'pilot-joint',phase='train',methods=['mlp'],seeds=[0])
            ajob=fixture_plan(a,'pilot-joint')['jobs'][0]
            status=read_json(Path(ajob['output'])/'job-status.json')
            self.assertEqual(status['status'],'completed',status)
            with patch('agent.paper.transfer.matrix',side_effect=fixture_plan):
                result=collect(b,Path(ajob['output']))
                self.assertFalse(result['rejected']);self.assertEqual(len(result['imported']),1)
            with patch('agent.paper.runtime.matrix',side_effect=fixture_plan):
                runtime.run_suite(b,'pilot-joint',phase='evaluate',methods=['mlp'],seeds=[0])
            replay=fixture_plan(b,'pilot-joint')['jobs'][1]
            self.assertEqual(read_json(Path(replay['output'])/'job-status.json')['status'],'completed')
            self.assertTrue(read_json(Path(replay['output'])/'results.json')[0]['feasible'])


if __name__ == '__main__': unittest.main()
