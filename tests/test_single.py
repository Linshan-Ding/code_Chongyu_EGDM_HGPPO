"""Single-case convergence diagnostic and parameter-derived case names."""
import re
import tempfile
import unittest
from pathlib import Path

import torch

from agent.paper.config import ROOT, RULES, settings
from agent.paper.data import case_id, rows
from agent.paper.storage import read_csv, read_json

CASE_NAME = re.compile(r"^(S|M|L)_D[134]_rho\d+(p\d)?_tau\d+(p\d)?$|^tiny_S\d+_J\d+_rho\d+(p\d)?$")


class SingleCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "result").mkdir(exist_ok=True)
        torch.set_num_threads(1)

    def test_case_ids_are_derived_from_design_parameters(self):
        records = rows(settings())
        for r in records:
            self.assertRegex(r["instance_id"], CASE_NAME)
            self.assertEqual(r["instance_id"], case_id(r["parameters"], r["split"]))
            self.assertEqual(Path(r["path"]).stem, r["instance_id"])
        self.assertEqual(len({r["instance_id"] for r in records}), len(records))
        self.assertEqual({r["instance_id"] for r in records if r["split"] == "pilot"},
                         {"S_D3_rho72p5_tau100", "M_D3_rho72p5_tau100"})
        s = next(r for r in records if r["instance_id"] == "S_D3_rho72p5_tau100")
        self.assertEqual((s["parameters"]["scale"], s["parameters"]["scenario"], s["parameters"]["load_ratio"],
                          s["parameters"]["relocation_multiplier"]), ("S", "D3", 0.725, 1.0))
        tiny = next(r for r in records if r["instance_id"] == "tiny_S2_J6_rho95")
        self.assertEqual((tiny["parameters"]["instance_parameters"]["S"], tiny["parameters"]["instance_parameters"]["J"],
                          tiny["parameters"]["load_ratio"]), (2, 6, 0.95))
        self.assertEqual(case_id(dict(scale="M", scenario="D1", load_ratio=0.9, relocation_multiplier=1.25), "train"),
                         "M_D1_rho90_tau125")
        self.assertEqual(case_id(dict(scale="L", scenario="D4", load_ratio=0.95, relocation_multiplier=2.0), "generalization"),
                         "L_D4_rho95_tau200")

    def test_single_case_diagnostic_reports_rules_learners_and_convergence(self):
        from agent.paper.single import run_single, convergence
        self.assertFalse(convergence([])["converged"])
        flat = [dict(rel_twt_geomean=1.0), dict(rel_twt_geomean=0.9), dict(rel_twt_geomean=0.9), dict(rel_twt_geomean=0.9)]
        self.assertTrue(convergence(flat)["converged"])
        self.assertEqual(convergence(flat)["best"], 0.9)
        self.assertFalse(convergence(flat[:2] + [dict(rel_twt_geomean=0.8)])["converged"])
        with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
            options = settings(result_root=tmp, run_id="single-test", workers=1, update_threads=1, instances_per_update=1,
                               episodes_per_instance=4, environments=4, minibatch=16, microbatch=8, ppo_update_epochs=1,
                               pilot_updates=1, validation_every_updates=1)
            with self.assertRaisesRegex(ValueError, "Unknown registered case"):
                run_single(options, instance_id="nope", methods=["full"], seeds=[0])
            with self.assertRaisesRegex(ValueError, "Unknown learners"):
                run_single(options, instance_id="tiny_S2_J6_rho95", methods=["pilot"], seeds=[0])
            with self.assertRaisesRegex(ValueError, "one case per update"):
                run_single(dict(options, instances_per_update=2, environments=8), instance_id="tiny_S2_J6_rho95",
                           methods=["full"], seeds=[0])
            summary = run_single(options, instance_id="tiny_S2_J6_rho95", methods=["full"], seeds=[0])
            reports = Path(summary["reports"])
            self.assertEqual(summary["case"]["instance_id"], "tiny_S2_J6_rho95")
            baselines = read_csv(reports / "baselines.csv")
            self.assertEqual(len(baselines), len(RULES) - 1 + len(options["seeds"]))
            self.assertEqual({r["method"] for r in baselines}, set(RULES))
            anchor = next(r for r in baselines if r["method"] == "Periodic-ATC")
            self.assertEqual(float(anchor["twt"]), summary["reference_twt"])
            self.assertGreater(summary["reference_twt"], 0.0)
            self.assertEqual(float(anchor["rel_twt"]), 1.0)
            learners = read_csv(reports / "learners.csv")
            self.assertEqual({(r["method"], r["kind"]) for r in learners}, {("full", "latest"), ("full", "best")})
            self.assertEqual(int(next(r for r in learners if r["kind"] == "latest")["epoch"]), 1)
            self.assertTrue(all(r["feasible"] == "True" for r in learners))
            curve = read_csv(reports / "validation-curves.csv")
            self.assertEqual([int(r["epoch"]) for r in curve], [0, 1])
            # The initial greedy policy reproduces the Periodic-ATC anchor exactly.
            self.assertAlmostEqual(float(curve[0]["rel_twt_geomean"]), 1.0, places=9)
            convergences = read_csv(reports / "convergence.csv")
            self.assertEqual(len(convergences), 1)
            self.assertEqual(int(convergences[0]["points"]), 2)
            ranking = read_csv(reports / "ranking.csv")
            self.assertEqual(len(ranking), len(baselines) + len(learners))
            self.assertEqual([float(r["twt"]) for r in ranking], sorted(float(r["twt"]) for r in ranking))
            text = (reports / "summary.md").read_text("utf-8")
            self.assertIn("tiny_S2_J6_rho95", text)
            self.assertIn("not generalization evidence", text)
            self.assertEqual(read_json(reports / "summary.json")["suite"], "single")
            case_dir = reports.parents[1]
            self.assertEqual(case_dir.name, "tiny_S2_J6_rho95")
            self.assertEqual(case_dir.parent.name, "single")
            folder = case_dir / "full" / "seed_0" / "single-test"
            self.assertTrue((folder / "latest.pt").exists() and (folder / "best.pt").exists())
            self.assertEqual(read_json(folder / "single-case.json")["note"], "diagnostic: training and evaluation case coincide")
            self.assertEqual(sorted(p.name for p in case_dir.iterdir() if p.is_dir()), sorted(set(RULES) | {"full", "_reports"}))
            # The diagnostic suite never writes into the pilot/formal result trees.
            self.assertEqual(sorted(p.name for p in case_dir.parent.parent.iterdir() if p.is_dir()), ["single"])


if __name__ == "__main__":
    unittest.main()
