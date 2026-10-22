import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from agent.paper.config import settings, project_config, LEARNERS, ROOT
from agent.paper.data import rows, load
from agent.paper.candidates import Builder
from agent.paper.network import create_network
from agent.paper.policies import Policy, learned_action
from agent.paper.rollout import Collector, Pool, batch_summary, update_plan
from agent.paper.ppo import PPO
from agent.paper.storage import save_checkpoint, load_checkpoint, write_json, append_csv
from agent.paper.ppo import returns_and_advantages, self_critical_advantages
from agent.paper.popart import PopArt
from environment.env import AssemblyEnv
from environment.masks import validate_composite_action
from environment.reward import interval_twt_reward, tardiness_potential
from environment.entities import OrderRuntime


def single_plan(index=0, slots=8):
    return [dict(env_id=i, instance_index=index, replicate=i, perturbation_seed=None) for i in range(slots)]


def episode_fixture(epoch, iid, env_id, twt, baseline=100., seed_offset=0, moves=0):
    return dict(epoch=epoch, instance_id=iid, instance_index=0, replicate=env_id, group=iid, perturbation_seed=None,
                instance_sha256="fixed", initial_state_sha256="initial", policy_version=epoch - 1, env_id=env_id,
                episode=epoch, action_seed=1000 * epoch + env_id + seed_offset, baseline_twt=baseline, twt=twt,
                rel_twt=twt / baseline, feasible=True, failure="", episode_events=10, moves=moves, travel=float(moves),
                anchor_fraction=1., gate_fraction=0., wall_seconds=0., decision_seconds=0., step_seconds=0., trace_sha256="t")


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tests must also work in a fresh checkout with no experiment outputs.
        (ROOT / "result").mkdir(exist_ok=True)
        torch.set_num_threads(1)
        cls.options = settings(workers=1, instances_per_update=1, episodes_per_instance=8,
                               minibatch=16, microbatch=8, ppo_update_epochs=1)
        cls.instance = load(rows(cls.options, "tiny")[0])

    def env(self, options=None):
        env = AssemblyEnv(project_config(options or self.options))
        env.reset(copy.deepcopy(self.instance))
        return env

    def choice_state(self, options=None):
        options = options or self.options
        env = self.env(options)
        rule = Policy("Periodic-ATC", options)
        for _ in range(200):
            obs = Builder(env, options).build()
            if len(obs.actions) > 1:
                return env, obs
            _, _, done, _ = env.step(rule.act(env))
            if done:
                break
        self.fail("test fixture has no reconfiguration choice")

    def test_data_design(self):
        from collections import Counter
        self.assertEqual(Counter(r["split"] for r in rows(self.options)),
            dict(train=36, pilot=2, validation=10, main=24, generalization=12, tiny=8))
        from agent.paper.matrix import matrix
        for suite, n in (("pilot-single", 6), ("pilot-joint", 3 * len(LEARNERS)), ("formal", 3 * len(LEARNERS))):
            plan = matrix(self.options, suite)
            trains = [j for j in plan["jobs"] if j["kind"] == "train"]
            self.assertEqual(len(trains), n)
            self.assertFalse(any(j.get("method") == "no_graph" for j in plan["jobs"]))
            self.assertTrue(all(j["budget_epochs"] > 0 for j in trains))
            # Development cases only; the test splits never steer selection.
            self.assertTrue(all(r["split"] == "validation" for j in trains for r in j["validation"]))
            self.assertTrue(all(j["expected_visits"] for j in trains))
            kinds = {j["checkpoint_kind"] for j in plan["jobs"] if j["kind"] == "replay"}
            self.assertEqual(kinds, {"latest", "best"})
        joint = next(j for j in matrix(self.options, "pilot-joint")["jobs"] if j["kind"] == "train")
        self.assertEqual(len(joint["training"]), 12)
        self.assertEqual(Counter(r["parameters"]["scale"] for r in joint["training"]), {"S": 6, "M": 6})
        self.assertTrue(all(r["parameters"]["relocation_multiplier"] == 1.25 for r in joint["training"]))

    def test_process_guard_checks_command_not_suite_option(self):
        from agent.paper.runtime import ensure_idle, ENTRY
        with tempfile.TemporaryDirectory(dir=ROOT / 'result') as tmp:
            options = dict(self.options, result_root=tmp)
            for command in ('matrix', 'report', 'status', 'prepare', 'monitor'):
                process = SimpleNamespace(pid=999999999, info=dict(pid=999999999, name='python.exe',
                    cmdline=['python.exe', str(ENTRY), command, '--suite', 'formal']))
                with patch('agent.paper.runtime.psutil.process_iter', return_value=[process]):
                    ensure_idle(options)
            for command in ('pilot-joint', 'formal', 'resume', 'evaluate-final', '_execute', '_replay-file', 'smoke', 'single'):
                process = SimpleNamespace(pid=999999999, info=dict(pid=999999999, name='python.exe',
                    cmdline=['python.exe', str(ENTRY), command, '--suite', 'pilot-joint']))
                with patch('agent.paper.runtime.psutil.process_iter', return_value=[process]):
                    with self.assertRaisesRegex(RuntimeError, 'Existing experiment process'):
                        ensure_idle(options)

    def test_orphan_worker_blocks_launch(self):
        from agent.paper.runtime import ensure_idle
        with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
            o = dict(self.options, result_root=tmp)
            from agent.paper.experiment import local_root
            write_json(local_root(o) / "workers/train_full.json",
                       dict(workers=[dict(identity=dict(pid=123, created_at=456, command=["worker"]))]))
            with patch("agent.paper.runtime.alive", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "Existing sampling worker"):
                    ensure_idle(o)
            with patch("agent.paper.runtime.alive", return_value=False), patch("agent.paper.runtime.psutil.process_iter", return_value=[]):
                ensure_idle(o)

    def test_reward_bootstrap_and_self_critical(self):
        instance = SimpleNamespace(orders=[SimpleNamespace(weight=50., due_date=1.)])
        order = [OrderRuntime(0, arrived=True)]
        self.assertEqual(interval_twt_reward(instance, order, 10000., 10000.000001), -50 * (10000.000001 - 10000.))
        self.assertEqual(interval_twt_reward(instance, order, .5, 1.5), -25.)
        a = SimpleNamespace(reward=-3., done=False, value=0., env_id=0)
        b = SimpleNamespace(reward=-7., done=True, value=0., env_id=0)
        returns_and_advantages([a, b], {0: 99.}, 1.)
        self.assertEqual((a.target, b.target), (-10., -7.))
        b.done = False
        returns_and_advantages([a, b], {0: 5.}, 1.)
        self.assertEqual((a.target, b.target), (-5., -2.))
        # Episode-level self-critical credit: every decision of a slot shares one
        # advantage, groups of one case are centred, and Monte Carlo targets remain.
        records = [SimpleNamespace(reward=-1., done=False, env_id=s) for s in (0, 1, 2)] + \
                  [SimpleNamespace(reward=-2., done=True, env_id=s) for s in (0, 1, 2)]
        episodes = [dict(env_id=0, group="a", baseline_twt=100., twt=90.), dict(env_id=1, group="a", baseline_twt=100., twt=110.),
                    dict(env_id=2, group="b", baseline_twt=50., twt=40.)]
        raw = self_critical_advantages(records, episodes, group_center=True)
        self.assertAlmostEqual(raw[0], .1)
        self.assertAlmostEqual(raw[1], -.1)
        self.assertAlmostEqual(raw[2], 0.)
        self.assertTrue(all(r.advantage == raw[r.env_id] for r in records))
        self.assertEqual([r.target for r in records], [-3., -3., -3., -2., -2., -2.])
        raw = self_critical_advantages(records, episodes, group_center=False)
        self.assertAlmostEqual(raw[2], .2)
        # The future-only potential never refunds already charged lateness and is zero at the end.
        env = self.env()
        legacy = tardiness_potential(env.sim.instance, env.sim.orders, env.sim.operations, env.sim.time, 1., "legacy")
        future = tardiness_potential(env.sim.instance, env.sim.orders, env.sim.operations, env.sim.time, 1., "future_only")
        self.assertLessEqual(legacy, future)
        self.assertLessEqual(future, 0.)
        pop = PopArt(4)
        x = torch.randn(9, 4)
        old = pop.denormalize(pop(x)).detach()
        pop.update(torch.randn(100) * 100 - 500)
        torch.testing.assert_close(old, pop.denormalize(pop(x)), atol=1e-4, rtol=1e-4)

    def test_prior_gate_candidates_and_graph_gradients(self):
        env, obs = self.choice_state()
        net = create_network(self.options, env.graph(), "full")
        ds, parts = net.distributions([obs])
        torch.testing.assert_close(ds[0].probs, torch.from_numpy(obs.prior), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ds[0].logits, parts[0][0] + parts[0][1], atol=1e-6, rtol=1e-6)
        self.assertLessEqual(len(obs.actions), self.options["max_candidates"])
        self.assertEqual(len(obs.actions), len(set(obs.actions)))
        self.assertEqual(obs.identifiers[0].split("|")[0], "periodic_anchor")
        for action, group in zip(obs.actions, obs.groups):
            plan = validate_composite_action(env.sim, action)
            self.assertEqual(bool(group), bool(plan.worker_changes + plan.robot_changes))
            self.assertLessEqual(len(plan.worker_changes) + len(plan.robot_changes), self.options["candidate_max_moves"])
        # Zero residual heads: zero divergence from the prior and an ATC-identical dispatch.
        lp, ent, kl, _ = net.evaluate_actions([obs], [0], need_values=False)
        self.assertAlmostEqual(float(kl[0]), 0., places=6)
        builder = Builder(env, self.options)
        steps, schedule, total, atc_steps = net.dispatch(obs, 0, builder, deterministic=True)
        self.assertEqual(schedule, obs.actions[0].schedule_assignments)
        self.assertEqual(atc_steps, len(steps))
        if steps:
            lp2, _, kl2, _ = net.evaluate_actions([obs], [(0, tuple(s.choice for s in steps))], need_values=False)
            self.assertAlmostEqual(float(kl2[0]), 0., places=6)
            self.assertAlmostEqual(float(lp2[0]), float(lp[0]) + total, places=5)
        # Heads start at zero intentionally. Perturb them to test that graph and
        # moved-resource embeddings actually carry policy gradients after learning.
        torch.nn.init.normal_(net.plan[-1].weight, std=.01)
        torch.nn.init.normal_(net.gate[-1].weight, std=.01)
        ds, _ = net.distributions([obs])
        (-ds[0].log_prob(torch.tensor(0))).backward()
        gg = [p.grad for p in net.actor_encoder.parameters() if p.grad is not None]
        self.assertTrue(any(g.abs().sum() > 0 for g in gg))
        self.assertTrue(all(torch.isfinite(g).all() for g in gg))
        self.assertTrue(all(p.grad is None for p in net.critic_parameters()))
        # v1 candidates remain registered for the ablation configuration.
        v1 = Builder(env, dict(self.options, candidate_version="v1")).build()
        self.assertLessEqual(len(v1.actions), 15)
        restricted = Builder(env, dict(self.options, reconfiguration_budget={"S": 0, "M": 0})).build()
        self.assertTrue(restricted.restricted)
        self.assertTrue(all(n.split("|")[0] in ("periodic_anchor", "keep") for n in restricted.identifiers))

    def test_future_isolation_and_graph_relations(self):
        env = self.env()
        with patch.object(env, "step", side_effect=AssertionError("No lookahead")):
            before = Builder(env, self.options).build()
        modified = copy.deepcopy(env)
        hidden = [r for r in modified.sim.orders if not r.arrived]
        self.assertTrue(hidden)
        for r in hidden:
            modified.sim.instance.orders[r.order_id].weight = 99999
            modified.sim.instance.orders[r.order_id].due_date += 100000
        after = Builder(modified, self.options).build()
        self.assertEqual(before.actions, after.actions)
        for name in ("stages", "state", "features"):
            np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
        for kind in before.graph.nodes:
            torch.testing.assert_close(before.graph.nodes[kind].continuous, after.graph.nodes[kind].continuous)
        self.assertEqual(set(before.graph.nodes), {"operation", "stage", "cell", "worker", "robot"})
        self.assertIn("in_transit_resources", before.graph.gate_continuous_names)
        for key, edge in before.graph.edges.items():
            if "compet" in key[1]:
                self.assertEqual(edge.num_edges, 0)
        env, _ = self.choice_state()
        graph = env.graph()
        ids = set(graph.nodes["operation"].ids.tolist())
        self.assertTrue(all(op.operation_id not in ids for op in env._last_state.operations if op.status.value == "DONE"))

    def test_transit_reservations_and_singleton_gradient(self):
        env, obs = self.choice_state()
        action = next(a for a, g in zip(obs.actions, obs.groups) if g)
        env.sim.apply_validated_action(validate_composite_action(env.sim, action))
        env._last_state = env.sim.snapshot()
        graph = Builder(env, self.options).build().graph
        total = 0
        for kind in ("worker", "robot"):
            expected = [(getattr(c, "reserved_" + kind), c.cell_id) for c in env._last_state.cells
                        if getattr(c, "reserved_" + kind) is not None]
            edge = graph.edges[(kind, "reserved_at", "cell")]
            self.assertEqual(edge.edge_index.T.tolist(), [list(p) for p in expected])
            self.assertTrue(torch.all(edge.continuous > 0))
            torch.testing.assert_close(graph.edges[("cell", "reserved_for_" + kind, kind)].edge_index, edge.edge_index.flip(0))
            total += edge.num_edges
        self.assertGreater(total, 0)
        # A single legal action has exactly zero log probability and gradient,
        # including a nonzero residual head (not just its initialization).
        env = self.env()
        obs = Builder(env, self.options).build()
        self.assertEqual(len(obs.actions), 1)
        net = create_network(self.options, env.graph(), "full")
        torch.nn.init.normal_(net.plan[-1].weight)
        ds, _ = net.distributions([obs])
        loss = -ds[0].log_prob(torch.tensor(0))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.all(p.grad == 0) for p in net.actor_parameters()))

    def test_training_resume_budget_and_optimizer(self):
        from agent.paper.train import train
        from agent.paper.storage import read_csv, committed_bundles, read_json
        options = dict(self.options, update_threads=1, ppo_update_epochs=3)
        baseline = {self.instance.instance_id: 1000.}
        with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
            whole, resumed = Path(tmp) / "whole", Path(tmp) / "resumed"
            final = train(options, "mlp", 3, [self.instance], [], baseline, whole, 3, "fixture")
            self.assertEqual((final["epoch"], final["completed_episodes"]), (3, 24))
            real_update = PPO.update
            calls = []
            def interrupted_update(ppo, records, bootstrap, **kwargs):
                result = real_update(ppo, records, bootstrap, **kwargs)
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("injected interruption AFTER optimizer changed")
                return result
            with patch.object(PPO, "update", interrupted_update):
                with self.assertRaisesRegex(RuntimeError, "AFTER optimizer"):
                    train(options, "mlp", 3, [self.instance], [], baseline, resumed, 3, "fixture")
            self.assertEqual(read_json(resumed / "committed.json")["epoch"], 1)
            failed = list((resumed / "failures").glob("*.json"))
            self.assertEqual(len(read_json(failed[0])["episodes"]), 8)
            train(options, "mlp", 3, [self.instance], [], baseline, resumed, 3, "fixture", resume=True)
            a, b = (load_checkpoint(p / "latest.pt", "fixture") for p in (whole, resumed))
            self.assertEqual(a["events"], b["events"])
            self.assertEqual(a["policy_version"], b["policy_version"])
            for k in a["network"]:
                torch.testing.assert_close(a["network"][k], b["network"][k], rtol=0, atol=0)
            for key in ("actor_optimizer", "critic_optimizer"):
                for i in a[key]["state"]:
                    for k, value in a[key]["state"][i].items():
                        torch.testing.assert_close(value, b[key]["state"][i][k], rtol=0, atol=0)
            self.assertEqual(a["actor_scheduler"]["last_epoch"], 3)
            self.assertEqual(a["actor_scheduler"], b["actor_scheduler"])
            bundles = committed_bundles(resumed)
            self.assertEqual(len(bundles), 4)
            self.assertEqual([int(r["epoch"]) for r in read_csv(resumed / "episodes.csv")], [1]*8+[2]*8+[3]*8)
            logs = read_csv(resumed / "log.csv")
            self.assertEqual(len(logs), 3)
            self.assertTrue(all(r["ppo_update_epochs"] == "3" and r["ppo_update_cycles"] == "1" for r in logs))
            self.assertTrue(all(r["advantage_mode"] == "self_critical" for r in logs))
            self.assertEqual([int(r["epoch"]) for r in read_csv(resumed / "evaluations.csv")], [])
            self.assertFalse((resumed / "best.pt").exists())
            for bundle in bundles[1:]:
                rows = bundle["episodes"]
                summary = bundle["summaries"][0]
                self.assertEqual(summary["mean_twt"], np.mean([r["twt"] for r in rows]))
                self.assertEqual(summary["std_twt"], np.std([r["twt"] for r in rows]))
                self.assertEqual(summary["sampling_policy_version"], bundle["epoch"]-1)
                self.assertEqual(summary["policy_version"], bundle["epoch"])
                self.assertEqual(summary["instances"], self.instance.instance_id)
            before = (resumed / "progress.json").read_bytes()
            train(options, "mlp", 3, [self.instance], [], baseline, resumed, 3, "fixture", resume=True)
            self.assertEqual(before, (resumed / "progress.json").read_bytes())

    def test_validation_curve_and_best_checkpoint(self):
        from agent.paper.train import train
        from agent.paper.storage import read_csv, committed_bundles, read_json
        other = load(rows(self.options, "tiny")[1])
        options = dict(self.options, update_threads=1, validation_every_updates=1)
        baseline = {self.instance.instance_id: 1000., other.instance_id: 1000.}
        with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
            folder = Path(tmp) / "run"
            result = train(options, "full", 1, [self.instance], [other], baseline, folder, 2, "fixture")
            self.assertIsNotNone(result["best"])
            self.assertTrue((folder / "best.pt").exists())
            best = load_checkpoint(folder / "best.pt", "fixture")
            self.assertEqual(best["kind"], "best_validation_policy")
            summaries = read_csv(folder / "validation_summary.csv")
            self.assertEqual([int(r["epoch"]) for r in summaries], [0, 1, 2])
            evaluations = read_csv(folder / "evaluations.csv")
            self.assertEqual([r["instance_id"] for r in evaluations], [other.instance_id] * 3)
            self.assertTrue(all(r["feasible"] == "True" for r in evaluations))
            scores = [float(r["rel_twt_geomean"]) for r in summaries]
            self.assertEqual(best["epoch"], int(summaries[int(np.argmin(scores))]["epoch"]))
            self.assertEqual(committed_bundles(folder)[0]["evaluations"][0]["epoch"], 0)
            self.assertTrue(all((folder / "validation" / f"epoch_{e:06d}" / f"{other.instance_id}.json").exists() for e in (0, 1, 2)))
            self.assertFalse(any((folder / "validation").rglob("*.trace.jsonl")))
            self.assertEqual(read_json(folder / "progress.json")["best_validation"]["epoch"], best["epoch"])

    def test_resume_repairs_publication_after_atomic_checkpoint(self):
        from agent.paper.train import train
        from agent.paper.storage import publish_commit, committed_bundles, read_json
        o = dict(self.options, episodes_per_instance=2, environments=2, update_threads=1)
        bs = {self.instance.instance_id: 1000.}
        with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
            folder = Path(tmp)
            def fail_publish(folder, state, **kwargs):
                if state["epoch"] == 1:
                    raise RuntimeError("injected publication interruption")
                return publish_commit(folder, state, **kwargs)
            with patch("agent.paper.train.publish_commit", fail_publish):
                with self.assertRaisesRegex(RuntimeError, "publication interruption"):
                    train(o, "mlp", 1, [self.instance], [], bs, folder, 1, "fixture")
            self.assertEqual(load_checkpoint(folder / "latest.pt", "fixture")["epoch"], 1)
            failed_tensor = next((folder / "attempts").glob("epoch_*/failed-rollout.pt"))
            failed = load_checkpoint(failed_tensor, "fixture")
            self.assertEqual({r.version for r in failed["records"]}, {failed["policy_version"]})
            self.assertEqual(read_json(folder / "committed.json")["epoch"], 0)
            train(o, "mlp", 1, [self.instance], [], bs, folder, 1, "fixture", resume=True)
            self.assertEqual(len(committed_bundles(folder)), 2)
            self.assertEqual(read_json(folder / "progress.json")["committed_episodes"], 2)

    def test_serial_parallel_probability_and_resume(self):
        env = self.env()
        torch.manual_seed(8)
        network = create_network(self.options, env.graph(), "full")
        baseline = {self.instance.instance_id: 1000.}
        plan = single_plan()
        collector = Collector(self.options, [self.instance], "full", 4, list(range(8)))
        serial = collector.collect(network.state_dict(), 0, 1, plan, baseline)
        for workers in (1, 2):
            with Pool(dict(self.options, workers=workers), [self.instance], "full", 4) as pool:
                parallel, boot, eps = pool.collect(network, 0, 1, plan, baseline)
            self.assertEqual([r.action for r in serial["records"]], [r.action for r in parallel])
            self.assertEqual([r.reward for r in serial["records"]], [r.reward for r in parallel])
            self.assertEqual([r["trace_sha256"] for r in serial["episodes"]], [r["trace_sha256"] for r in eps])
            self.assertEqual(len(eps), 8)
            self.assertEqual(sum(r.done for r in parallel), 8)
            batch_summary(eps, 8)
        sample = parallel[:16]
        lp = network.evaluate_actions([r.observation for r in sample], [r.action for r in sample])[0]
        torch.testing.assert_close(lp, torch.tensor([r.logp for r in sample]), atol=2e-6, rtol=2e-6)
        self.assertTrue(any(isinstance(r.action, tuple) and len(r.action[1]) > 0 for r in parallel))
        aa = collector.collect(network.state_dict(), 1, 2, plan, baseline)
        bb = Collector(self.options, [self.instance], "full", 4, list(range(8))).collect(network.state_dict(), 1, 2, plan, baseline)
        self.assertEqual([r.action for r in aa["records"]], [r.action for r in bb["records"]])
        det = collector.collect(network.state_dict(), 0, 1, plan, baseline, deterministic=True)
        self.assertEqual(len({r["trace_sha256"] for r in det["episodes"]}), 1)
        self.assertEqual(len({r["action_seed"] for r in aa["episodes"] + serial["episodes"]}), 16)
        old = copy.deepcopy(network.state_dict())
        ppo = PPO(network, self.options, total_updates=5)
        metrics = ppo.update(parallel, boot, episodes=eps, step=0)
        self.assertTrue(all(np.isfinite(v) for k, v in metrics.items() if isinstance(v, float)))
        self.assertEqual(metrics["advantage_mode"], "self_critical")
        self.assertTrue(any(not torch.equal(old[k], network.state_dict()[k]) for k in old))
        self.assertAlmostEqual(ppo.kl_coefficient(0), self.options["kl_prior_coef_start"])
        self.assertAlmostEqual(ppo.kl_coefficient(10**6), self.options["kl_prior_coef_end"])
        self.assertLess(ppo.lr_factor(0), 1.)
        with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
            p = Path(tmp) / "checkpoint.pt"
            save_checkpoint(p, dict(signature="new", network=network.state_dict()))
            load_checkpoint(p, "new")
            with self.assertRaises(ValueError):
                load_checkpoint(p, "legacy")
            torch.save(dict(format="paper-sm100-1", signature="new"), p)
            with self.assertRaises(ValueError):
                load_checkpoint(p, "new")
        parallel[0].version += 1
        with self.assertRaises(ValueError):
            PPO(network, self.options).update(parallel, boot, episodes=eps)

    def test_worker_exception_and_update_schedule(self):
        from agent.paper.worker_pool import WorkerFailure
        from agent.paper.matrix import matrix
        env = self.env()
        net = create_network(self.options, env.graph(), "full")
        with Pool(dict(self.options, event_limit=1, workers=2), [self.instance], "full", 0) as pool:
            with self.assertRaises(WorkerFailure) as raised:
                pool.collect(net, 0, 1, single_plan(), {self.instance.instance_id: 1.})
            evidence = raised.exception.evidence
            self.assertEqual(len(evidence["episodes"]), 8)
            self.assertTrue(all(not r["feasible"] for r in evidence["episodes"]))
            with self.assertRaises(ValueError):
                batch_summary(evidence["episodes"], 8)
        formal = next(j for j in matrix(self.options, "formal")["jobs"] if j["kind"] == "train")
        self.assertEqual(formal["budget_episodes"], self.options["formal_updates"] * self.options["environments"])
        self.assertEqual(len(formal["training"]), 36)
        order = [(r["parameters"]["scenario"], r["parameters"]["scale"]) for r in formal["training"][:6]]
        self.assertEqual(order, [(d, s) for d in ("D1", "D3", "D4") for s in ("S", "M")])
        # Seeded case cycles visit every registered case equally often.
        o = dict(self.options, instances_per_update=3, episodes_per_instance=2, environments=6)
        visits = {}
        for epoch in range(1, 9):
            for slot in update_plan(o, 5, epoch, 4):
                visits[slot["instance_index"]] = visits.get(slot["instance_index"], 0) + 1
        self.assertEqual(visits, {0: 12, 1: 12, 2: 12, 3: 12})
        self.assertEqual(update_plan(o, 5, 3, 4), update_plan(o, 5, 3, 4))
        self.assertNotEqual([s["instance_index"] for s in update_plan(o, 5, 1, 4)], [s["instance_index"] for s in update_plan(o, 6, 1, 4)])
        with Pool(self.options, [self.instance]*54, "mlp", 0) as pool:
            self.assertEqual(pool.nslots, 8)
        with self.assertRaises(ValueError):
            settings(complete_epochs_per_update=8)
        with self.assertRaises(ValueError):
            settings(pilot_epochs_per_instance=100)

    def test_update_mixes_cases_and_rejects_partial_batches(self):
        other = load(rows(self.options, "tiny")[1])
        cases = [self.instance, other]
        o = dict(self.options, instances_per_update=2, episodes_per_instance=4, environments=8)
        net = create_network(o, self.env(o).graph(), "mlp")
        collector = Collector(o, cases, "mlp", 0, list(range(8)))
        baselines = {case.instance_id: 1000. for case in cases}
        for epoch in (1, 2):
            plan = [dict(env_id=i, instance_index=i // 4, replicate=i % 4, perturbation_seed=None) for i in range(8)]
            part = collector.collect(net.state_dict(), 0, epoch, plan, baselines)
            self.assertEqual({r["instance_id"] for r in part["episodes"]}, {c.instance_id for c in cases})
            self.assertEqual(sum(r.done for r in part["records"]), 8)
            summary = batch_summary(part["episodes"], 8)
            self.assertEqual(summary["groups"], 2)
            self.assertEqual(summary["instances"], "|".join(sorted(c.instance_id for c in cases)))
            # Neither a missing eighth row nor a failed eighth episode may
            # produce the mean of seven successful trajectories.
            with self.assertRaises(ValueError):
                batch_summary(part["episodes"][:-1], 8)
            partial = copy.deepcopy(part["episodes"])
            partial[-1]["feasible"] = False
            with self.assertRaises(ValueError):
                batch_summary(partial, 8)
        # Arrival perturbation keeps the case file, reseeds the stream per update and remeasures the anchor.
        from data.generator import perturb_arrivals
        perturbed = perturb_arrivals(self.instance, 7)
        self.assertEqual(len(perturbed.orders), len(self.instance.orders))
        self.assertNotEqual([o.release_time for o in perturbed.orders], [o.release_time for o in self.instance.orders])
        self.assertEqual(sorted(o.weight for o in perturbed.orders), sorted(o.weight for o in self.instance.orders))
        po = dict(o, train_instance_perturbation="arrivals")
        plan = update_plan(po, 0, 1, 2)
        self.assertTrue(all(s["perturbation_seed"] is not None for s in plan))
        part = Collector(po, cases, "mlp", 0, list(range(8))).collect(net.state_dict(), 0, 1, plan, baselines)
        rows_ = part["episodes"]
        self.assertTrue(all("#" in r["group"] for r in rows_))
        self.assertTrue(all(r["baseline_twt"] != 1000. for r in rows_))
        batch_summary(rows_, 8)

    def test_monitor_update_points_and_missing_report(self):
        from agent.paper.monitor import figures
        from agent.paper.report import report
        from agent.paper.storage import publish_commit
        with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
            p = Path(tmp)
            write_json(p / "baselines.json", {"case": 100.})
            ev = dict(instance_id="case", epoch=0, index=0, twt=100., baseline_twt=100., rel_twt=1., feasible=True)
            vs = dict(epoch=0, rel_twt_geomean=1., best_rel_twt_geomean=1., improved=True)
            def publish(epoch, episodes, summaries, evaluations, validation_summaries=()):
                publish_commit(p, dict(epoch=epoch, events=epoch*80, policy_version=epoch,
                    progress=dict(committed_episodes=epoch*8),
                    last_bundle=dict(epoch=epoch, episodes=episodes, summaries=summaries, evaluations=evaluations,
                                     validation_summaries=list(validation_summaries))))
            publish(0, [], [], [ev], [vs])
            f = figures(p)
            self.assertNotIn("case_TWT", f)
            self.assertEqual(next(t for t in f["validation_rel_TWT"]["data"] if t["name"].startswith("Greedy validation"))["y"], [1.])
            for epoch in (1, 2):
                eps = [episode_fixture(epoch, "case", i, 80. + i - epoch) for i in range(8)]
                summary = dict(batch_summary(eps, 8), epoch=epoch, environments=8)
                publish(epoch, eps, [summary], [])
                if epoch == 1:
                    single = next(t for t in figures(p)["case_TWT"]["data"] if t["name"].startswith("Batch mean"))
                    self.assertEqual(single["mode"], "lines+markers")
                    self.assertEqual(single["error_y"]["array"], [summary["std_twt"]])
                    self.assertEqual(single["y"], [summary["mean_twt"]])
            f = figures(p)["case_TWT"]
            mean = next(t for t in f["data"] if t["name"].startswith("Batch mean"))
            self.assertEqual(mean["x"], [1., 2.])
            self.assertEqual(mean["y"], [82.5, 81.5])
            self.assertNotIn("error_y", mean)
            upper = next(t for t in f["data"] if t.get("fill") == "tonexty")
            np.testing.assert_allclose(upper["y"], np.array(mean["y"])+np.std(np.arange(8)))
            base = next(t for t in f["data"] if t["name"] == "Periodic-ATC beta=1")
            self.assertEqual(base["y"], [100., 100.])
            write_json(p / "failures/failure.json", dict(instances=["case"], epoch=3, failure="fixture", episodes=[]))
            write_json(p / "failures/supervisor.json", dict(status="failed", failure="job-level fixture"))
            self.assertEqual(figures(p)["job_failures"]["data"][0]["y"], [1])
            f = figures(p)["case_TWT"]
            self.assertEqual(f["data"][-1]["y"], [100.])
            self.assertIn("not objective", f["data"][-1]["name"])
            self.assertEqual(next(t for t in f["data"] if t["name"].startswith("Batch mean"))["x"], [1., 2.])
            result = report(dict(self.options, result_root=tmp), "pilot-joint")
            self.assertFalse(result["complete"])
            self.assertFalse(result["admission"]["passed"])
            self.assertEqual(result["admission"]["judged_on"], "latest")
            self.assertTrue(all(r["status"] == "missing" for r in result["all_results"]))
            from agent.paper.matrix import matrix
            job = next(j for j in matrix(dict(self.options, result_root=tmp), "pilot-joint")["jobs"] if j["kind"] == "train")
            write_json(Path(job["output"]) / "failures/retained.json", dict(epoch=1, committed_epoch=0,
                attempt="fixture", failure="historical failed attempt", episodes=[]))
            audited = report(dict(self.options, result_root=tmp), "pilot-joint")
            self.assertEqual(len(audited["attempt_failures"]), 1)

    def test_compact_trajectory_replays_physical_actions_and_rewards(self):
        import hashlib
        import json
        from dataclasses import asdict
        from agent.paper.trajectory import write_trajectory, read_trajectory, physical_action
        for method in ("full", "dan"):
            network = create_network(self.options, self.env().graph(), method)
            collector = Collector(self.options, [self.instance], method, 2, [0, 1])
            part = collector.collect(network.state_dict(), 0, 1, single_plan(), {self.instance.instance_id: 1000.})
            with tempfile.TemporaryDirectory(dir=ROOT / "result") as tmp:
                path = Path(tmp) / "events.jsonl.gz"
                meta = write_trajectory(path, part["records"], "fixture")
                restored = list(read_trajectory(path))
                self.assertEqual(meta["events"], len(part["records"]))
                for episode in part["episodes"]:
                    env, digest = self.env(), hashlib.sha256()
                    rows = [r for r in restored if r["env_id"] == episode["env_id"]]
                    for row in rows:
                        self.assertEqual(row["details"]["physical_time"], env.sim.time)
                        action = physical_action(row["details"]["physical_action"])
                        canonical = dict(time=env.sim.time, action=asdict(action))
                        digest.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())
                        _, reward, done, info = env.step(action)
                        self.assertEqual(reward / 1000., row["reward"])
                        self.assertEqual(done, row["done"])
                    self.assertTrue(done)
                    self.assertEqual(info["twt"], episode["twt"])
                    self.assertEqual(digest.hexdigest(), episode["trace_sha256"])


if __name__ == "__main__":
    unittest.main()
