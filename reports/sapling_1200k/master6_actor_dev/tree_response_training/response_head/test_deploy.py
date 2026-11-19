"""Replay all registered isolates through the real decoder; dummy coefficients remain in process memory."""

import gzip
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import deploy

sys.path.insert(0, str(deploy.ROOT / "code"))
from fastsim import FastCarbon


def read(path):
    b = path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix == ".gz" else b)


class TestDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = sorted(deploy.TRAINING.glob("case_*/root.json.gz"))
        if len(cls.cases) != 8:
            raise ValueError("the eight original geometry witnesses changed")
        # Never write a runnable agent or test weights to disk.
        cls.original = compile(
            deploy.PARENT.read_bytes(), "<source-locked frozen v1>", "exec"
        )
        cls.overlay = compile(deploy.runtime_source(), "<prospective runtime>", "exec")

    def test_eight_ordered_pairs_and_rejections(self):
        audited = []
        directions = set()
        for path in self.cases:
            case = read(path)
            item = case["candidate"]
            with self.subTest(root=path.parent.name):
                raw = read(
                    deploy.TRAINING
                    / "source"
                    / f'raw_{item["seed"]}_{item["seat"]}.json.gz'
                )
                cfg = raw["game_configuration"]
                sim = FastCarbon(cfg).reset(seed=item["seed"])
                module = types.ModuleType("training_tree_response_test")
                exec(self.original, module.__dict__)
                module._tree_weights = {}
                exec(self.overlay, module.__dict__)
                head = module._tree_head
                uid = item["uid"]
                t = item["t"]
                seat = item["seat"]
                for step in range(t):
                    obs = sim.obs(seat)
                    if step == 0:
                        self.assertEqual(
                            module.agent(obs, cfg), raw["actions"][step][seat]
                        )
                    else:
                        module._FEAT.observe(obs)
                        head.observe(obs, cfg)
                    sim.step(*raw["actions"][step])
                obs = sim.obs(seat)
                self.assertEqual(obs, case["public_at_root"])
                obs["full_carbon"] = ["ignored"] * 225
                base = module._tree_original(obs, cfg)
                module._tree_parent._FEAT = module._FEAT
                self.assertEqual(list(base.items()), list(case["v1"].items()))
                p = module._tree_proposals
                st = p.state(obs, cfg)
                commit = p.commitments(st, base)
                original_choose = head.learned.choose
                original_features = head.learned.features
                normal_pair = head.pair
                tested = []
                pay_v1 = case["paid_arm"] == "parent"
                direction = "defer" if pay_v1 else "pay"
                dirs = ("pay", "defer")
                names = tuple(module._tree_policy.FEATURE_NAMES)
                mock_weights = dict(
                    features=list(names),
                    beta=[-100000.0 if pay_v1 else 100000.0] + [0] * len(names),
                    low=[-10000.0] * len(names),
                    high=[10000.0] * len(names),
                    support_pad=0.0,
                    minimum_predicted_departure=1.0,
                )
                focus = []

                def features(*args):
                    focus[:] = [args[2]]
                    return original_features(*args)

                def choose(weights, x, pay_is_v1):
                    # Only test the registered focal worker; other workers retain the v1 action.
                    if focus[0] != uid:
                        return dirs[0 if pay_is_v1 else 1]
                    choice = original_choose(mock_weights, x, pay_is_v1)
                    self.assertEqual(choice, direction)
                    return choice

                def capture(obs, st, base, worker, commit):
                    pair = normal_pair(obs, st, base, worker, commit)
                    if worker == uid:
                        tested.append(pair)
                    return pair

                with patch.object(head.learned, "features", features), patch.object(
                    head.learned, "choose", choose
                ), patch.object(head, "pair", capture):
                    actual = head.consider(obs, cfg, base)
                self.assertEqual(len(tested), 1)
                pair = tested[0]
                self.assertIsNotNone(pair)
                self.assertEqual(
                    pair["ranked_alternatives_in_order"],
                    case["ranked_alternatives_in_order"],
                )
                self.assertEqual(pair["true_prefix_slot"], case["true_prefix_slot"])
                self.assertEqual(
                    list(actual.items()), list(case["alternative"].items())
                )
                self.assertEqual(
                    list(pair["alternative"].items()), list(case["alternative"].items())
                )
                self.assertEqual(list(pair["v1"].items()), list(case["v1"].items()))
                self.assertEqual(
                    (pair["old"], pair["new"], pair["paid_arm"]),
                    (case["old"], case["new"], case["paid_arm"]),
                )
                self.assertEqual(
                    [
                        k
                        for k in set(actual) | set(base)
                        if actual.get(k) != base.get(k)
                    ],
                    [uid],
                )
                self.assertTrue(head.used)
                self.assertTrue(head.last["changed"])
                # Suppress an out-of-support prediction rather than executing a stale edit.
                shield = dict(
                    mock_weights,
                    low=[100000.0] * len(names),
                    high=[100001.0] * len(names),
                )
                self.assertEqual(
                    original_choose(shield, [0] * len(names), pay_v1),
                    "pay" if pay_v1 else "defer",
                )
                self.assertIsNone(
                    normal_pair(obs, dict(st, cash=0.0), base, uid, commit)
                )
                # Other truly reserved landing cells block a pay or defer, including the base hire.
                hires, dests, requests = commit
                y = st["own"][2][uid][0]
                self.assertIsNone(
                    normal_pair(
                        obs,
                        st,
                        base,
                        uid,
                        (dict(hires, other_hire=(y, 0.0)), dests, requests),
                    )
                )
                x, aux, enc, labels, mode = module._tree_parent._root
                slot = int(pair["true_prefix_slot"])
                nonfocal = next(
                    i for i in range(len(labels)) if i != slot and labels[i] >= 0
                )

                def corrupt(*args):
                    altered = labels.copy()
                    altered[slot] = (int(labels[slot]) + 1) % 5
                    altered[nonfocal] = (int(labels[nonfocal]) + 1) % 3
                    return altered

                with patch.object(module._tree_parent, "_forced_decode", corrupt):
                    self.assertIsNone(normal_pair(obs, st, base, uid, commit))
                sim.step(*raw["actions"][t])
                obs2 = sim.obs(seat)
                self.assertIs(head.consider(obs2, cfg, base), base)
                self.assertFalse(head.last["changed"])
                directions.add(case["paid_arm"])
                audited.append((item["seed"], seat, t, uid))
        self.assertEqual(len(set(audited)), 8)
        self.assertEqual(directions, {"parent", "alternative"})

    def test_unapproved_fit_cannot_write(self):
        import tempfile
        import train

        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "must_not_exist.py"
            with patch.object(
                train, "rebuild_release", side_effect=ValueError("held-map gate closed")
            ):
                with self.assertRaises(ValueError):
                    deploy.publish(target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
