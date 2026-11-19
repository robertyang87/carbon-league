"""Compare V5 and the production guard on both seats of the observed failure map."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import cnet2_feat as F
import cnet2_search as S
from build_sapling_cash_dagger import module
from eval_v53 import _extinct
from fastsim import FastCarbon


def sha(data):
    return hashlib.sha256(data).hexdigest()


def guard_source(src):
    tree = ast.parse(src)
    return next(ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "BankruptcyGuard")


def summary(sim, seat):
    state = sim.get_state()
    return dict(workers=sum(w["player"] == seat for w in state["workers"].values()),
                cash=float(state["cash"][seat]), extinct=bool(_extinct(state, seat)))


def main():
    models = ROOT / "data/weights/sapling_1200k"
    active = module(models / "soft_it50_v5_guard_agent.py", "guarded_v5")
    source = module(models / "soft_it50_v5_agent.py", "plain_v5")
    production = module(ROOT / "code/packs/ppou2_m_it50.py", "reference_it50_guard")
    assert guard_source(active._SRC["cnet2_search"]) == guard_source(production._SRC["cnet2_search"])
    assert (active._B1, active._Q1, active._M1) == (source._W64, source._QMETA, source._WMETA)
    games = []
    pack_sha = sha((models / "soft_it50_v5_guard_agent.py").read_bytes())
    for seat in (0, 1):
        path = HERE / f"tree_exam_soft_it50_v5/raw_620106811_{seat}.json.gz"
        raw = json.loads(gzip.decompress(path.read_bytes()))
        assert sha((models / "soft_it50_v5_agent.py").read_bytes()) == raw["pack_sha256"]
        sim = FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
        feat, guard = F.Feat(seat), S.BankruptcyGuard(S.DEFAULTS)
        events, openings = [], []
        for t, joint in enumerate(raw["actions"]):
            obs = sim.obs(seat)
            assert source.agent(obs, sim.config) == joint[seat]
            guarded = active.agent(obs, sim.config)
            x = feat.observe(obs)
            if t == 0:
                pol = S.CachePolicy(source._WTS)
            labels, logits, _ = pol.act(x, feat.aux, teacher=2)
            assert feat.to_actions(labels) == joint[seat]
            guard.reset_step()
            checked = guard.apply(x, feat.aux, labels, logits, pol, feat.own_uid, 2)
            assert feat.to_actions(checked) == guarded
            if t == 0:
                assert guarded == joint[seat]
                openings = [kind for kind in guarded.values() if kind.startswith("REC")]
                assert "RECCOLLECTOR" in openings
            if guarded != joint[seat]:
                alternate = FastCarbon(raw["game_configuration"])
                alternate.set_state(sim.get_state())
                alternate_actions = list(joint)
                alternate_actions[seat] = guarded
                alternate.step(*alternate_actions)
                event = dict(t=t, before=summary(sim, seat), original=joint[seat],
                             guarded=guarded, alternative_next=summary(alternate, seat))
                events.append(event)
            sim.step(*joint)
            if events and events[-1]["t"] == t:
                events[-1]["recorded_next"] = summary(sim, seat)
        counts = dict(active.agent.last_stats)
        assert counts["changed_roots"] == len(events)
        assert counts["calls"] == len(raw["actions"]) and counts["unchanged_slots"] > 0
        if seat == 1:
            assert [v["t"] for v in events] == [31, 36]
            assert events[-1]["recorded_next"]["extinct"] and not events[-1]["alternative_next"]["extinct"]
        games.append(dict(seed=raw["seed"], seat=seat, raw_sha256=sha(path.read_bytes()),
                          original_actions_reproduced=len(raw["actions"]), legitimate_opening_hires=openings,
                          guard_stats=counts, changes=events))
    report = dict(candidate_sha256=pack_sha, fp32_sha256=sha((models/"soft_it50_v5.npz").read_bytes()),
                  int8_payload_sha256=sha(base64.b64decode(active._B1)),
                  guard_ast_equals_production_it50=True, guard=(active.CONF), games=games,
                  scope="Original replay states, separately changing one candidate action against the logged opponent move. No replanning after the branch and no claim of eventual victory.")
    (HERE / "v5_guard_replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
