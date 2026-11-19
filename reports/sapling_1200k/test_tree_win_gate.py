"""Regressions for same-map seats, genuine zero-depth returns, and terminal scoring."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import tree_win_gate as gate


def game(seed, seat, win):
    score = [3., 1.]
    if (seat == 1) == win:
        score.reverse()
    margin = score[seat] - score[1-seat]
    return dict(seed=seed, candidate_seat=seat, result="win" if win else "loss",
                margin=margin, official_score_by_seat=score, actual_transitions=299,
                agent_errors_by_seat=[0,0], agent_calls_by_seat=[299,299], termination="official_episode_limit",
                audit_messages=[], valid=True, search_invocations=299,
                search=dict(searched=297, single=1, aborted=4),
                search_return_types=dict(searched=297,single=1,depth_zero_return_base=1))


class GateTests(unittest.TestCase):
    def test_observed_threshold_and_both_seat_threshold(self):
        plan={"maps":[{"seed":s} for s in range(100)]}
        rows=[game(s,seat,seat==0 or s<80) for s in range(100) for seat in (0,1)]
        result=gate.verdict(plan,rows)
        self.assertEqual(result["wins"],180)
        self.assertTrue(result["observed_90_percent_met"])
        self.assertFalse(result["each_seat_observed_90_percent_met"])
        self.assertLess(result["win_rate_95pct_map_interval"][0],.90)
        rows[1]=game(0,1,False)
        self.assertFalse(gate.verdict(plan,rows)["observed_90_percent_met"])
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):
            gate.verdict(plan,rows)

    def test_real_fallback_and_fabricated_search(self):
        row=game(1,0,True)
        gate.validate_game(row)
        for patch in (dict(search_invocations=298),
                      dict(search_return_types=dict(searched=297,single=1)),
                      dict(search=dict(searched=0,single=298,aborted=1),
                           search_return_types=dict(searched=0,single=298,depth_zero_return_base=1)),
                      dict(margin=-12.)):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                gate.validate_game(dict(row,**patch))

    def test_extinct_survivor_uses_cash_before_last_transition(self):
        from tree_audited_game import exam, prior
        seed = 231
        sim = exam.FastCarbon(exam.CFG).reset(seed=seed)
        base = [sim.bases[seat][0][0] for seat in (0,1)]
        actions = [[{base[0]: "RECCOLLECTOR"}, {}],
                   [{base[0]: "RECCOLLECTOR", "player-0-worker-0": "UP"}, {}],
                   [{"player-0-worker-0": "DOWN"}, {base[1]: "RECCOLLECTOR"}]]
        for ours, theirs in actions:
            pre = list(sim.cash)
            sim.step(ours, theirs)
        self.assertEqual(pre, [20,80])
        self.assertEqual(sim.cash, [20,50])
        self.assertEqual([exam._extinct(sim.get_state(), p) for p in (0,1)], [True,False])
        row = game(seed,1,True)
        row.update(termination="extinction", actual_transitions=3, agent_calls_by_seat=[3,3],
                   official_score_by_seat=[-298,80],margin=378.,search_invocations=3,
                   search=dict(searched=3,single=0,aborted=0),
                   search_return_types=dict(searched=3),extinct_by_seat=[True, False],
                   final_state_sha256=prior.hashed(sim.get_state()),actions_sha256=prior.hashed(actions))
        raw = dict(seed=seed,candidate_seat=1,pack_sha256="example",weight_identity="weights",
                   margin=378.,result="win",examiner_md5=gate.EXAMINER,game_configuration=exam.CFG,
                   errors=[],actions=actions,steps=3,extinct_by_seat=[True,False],last_pre_cash=pre,
                   last_post_cash=sim.cash,official_score_by_seat=[-298,80],search_calls=[[i,"searched"] for i in range(3)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/f"raw_{seed}_1.json.gz"
            def write(payload):
                path.write_bytes(gzip.compress(json.dumps(payload).encode(),mtime=0))
                row.update(raw_file=path.name,raw_sha256=gate.digest(path))
            write(raw)
            gate.validate_game(row)
            gate.verify_raw(directory,row,"example","weights")
            wrong=copy.deepcopy(raw)
            wrong["official_score_by_seat"][1]=50.
            write(wrong)
            with self.assertRaisesRegex(ValueError, "terminal cash"):
                gate.verify_raw(directory,row,"example","weights")


    def test_real_actions_reject_early_nonterminal_and_invented_final_cash(self):
        folder = Path(__file__).with_name("tree_exam_basecal")
        report = json.loads((folder / "results.json").read_text())
        row = report["games"][0]
        header = report["header"]
        gate.verify_raw(folder, row, header["pack_sha256"], header["weight_sha256"])
        raw = json.loads(gzip.decompress((folder / row["raw_file"]).read_bytes()))
        from tree_audited_game import prior
        short = dict(raw, actions=raw["actions"][:100])
        with self.assertRaisesRegex(ValueError, "first official terminal"):
            gate.verify_terminal_replay(dict(row, actions_sha256=prior.hashed(short["actions"])), short)
        wrong = dict(raw, last_post_cash=[900.,100.])
        with self.assertRaisesRegex(ValueError, "end state"):
            gate.verify_terminal_replay(row, wrong)
        with self.assertRaisesRegex(ValueError, "end state"):
            gate.verify_terminal_replay(dict(row, final_state_sha256="wrong"), raw)
        with self.assertRaisesRegex(ValueError, "identity"):
            gate.verify_raw(folder, row, header["pack_sha256"], "different weights")

    def test_freeze_binds_rules_and_candidate_weights(self):
        weights = gate.ROOT / "data/weights/sapling_1200k"
        v3 = weights / "cash_v3.npz"
        pack = weights / "cash_v3_agent.py"
        self.assertEqual(gate.protocol_hashes()[gate.RULES], gate.digest(gate.ROOT / gate.RULES))
        self.assertEqual(gate.prove_single_pack(v3, pack), (gate.digest(v3),gate.digest(pack)))
        with self.assertRaisesRegex(ValueError, "differs from the exact rebuild"):
            gate.prove_single_pack(weights / "tree_v4.npz", pack)
        with self.assertRaisesRegex(ValueError, "compound"):
            gate.prove_single_pack(Path("two_networks.json"), pack)

    def test_wider_candidate_schema_and_guarded_v5(self):
        import math
        cfg = dict(C=93, NB=12, TA=0, D=96, NH=4, L=1, E=32, G=3)
        meta = dict(cfg=cfg, parent_preset="sapling-1.2m",
                    architecture_id="sapling-experimental-c93-nb12-deepening-v1", params=2_143_916)
        shapes = gate.sapling_shapes(meta)
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), 2_143_916)
        self.assertEqual(len(shapes), 153)
        with self.assertRaisesRegex(ValueError, "unknown deep Sapling"):
            gate.sapling_shapes(dict(meta, cfg=dict(cfg, C=95)))
        weights = gate.ROOT / "data/weights/sapling_1200k"
        v5 = weights / "soft_it50_v5.npz"
        _, packed = gate.prove_single_pack(v5, weights / "soft_it50_v5_guard_agent.py")
        self.assertEqual(packed, gate.GUARDED_V5)

    def test_guarded_formal_verdict_requires_an_executed_network_choice(self):
        plan = dict(pack_sha256=gate.GUARDED_V5, maps=[dict(seed=s) for s in range(100)])
        games = [dict(game(seed, seat, True), candidate_guard_stats=dict(unchanged_slots=1), unchanged_network_action_steps=[])
                 for seed in range(100) for seat in (0,1)]
        with self.assertRaisesRegex(ValueError, "no unmodified network decisions"):
            gate.verdict(plan, games)
        games[0]["unchanged_network_action_steps"] = [2]
        self.assertTrue(gate.verdict(plan,games)["observed_90_percent_met"])

    def test_reject_replay_of_previously_exposed_formal_map(self):
        nonce = "0" * 64
        seeds = gate.seed_list(nonce)
        plan = dict(schema="sapling-one-shot-90/2",weight_sha256="w",pack_sha256="p",
                    examiner_md5=gate.EXAMINER, protocol_sha256=gate.protocol_hashes(),
                    run_directory="trial",nonce=nonce,maps=[{"seed":v} for v in seeds],
                    excluded_reserved_seeds=[seeds[0]])
        with self.assertRaisesRegex(ValueError, "exposed reserved maps"):
            gate.validate(plan,"w","p",gate.EXAMINER,"trial")


if __name__=="__main__":
    unittest.main()
