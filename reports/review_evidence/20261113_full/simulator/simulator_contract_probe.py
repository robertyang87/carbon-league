"""Read-only original-engine parity, visibility, clone and snapshot probes.

CARBON_REPO selects the checkout; CARBON_REVIEW_OUT selects a parent for a new
temporary output directory. A snapshot probe records an existing API defect;
exit zero means that the probe completed, not that every recorded check passed.
"""
import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile


def find_repo():
    if os.environ.get("CARBON_REPO"):
        root = pathlib.Path(os.environ["CARBON_REPO"]).expanduser().resolve()
        if not (root / "code/fastsim.py").is_file():
            raise RuntimeError("CARBON_REPO must contain code/fastsim.py")
        return root
    for start in (pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()):
        for root in (start, *start.parents):
            if (root / "code/fastsim.py").is_file():
                return root
    raise RuntimeError("Set CARBON_REPO")


ROOT = find_repo()
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
import test_fastsim_parity as parity
import cnet2_feat as F
import cnet2_search as S
from fastsim import FastCarbon

out = {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
parity.check_map_generation()
parity.check_full_game(7)
parity.check_full_game(123)
parity.check_scenarios()
parity.check_agent_compat()
out["original_board_parity"] = [{"section": s, "name": n, "pass": bool(ok), "detail": d}
                                  for s, n, ok, d in parity.RESULTS]
assert all(r["pass"] for r in out["original_board_parity"]), parity.MISMATCHES

# Legal prefix from the default initial layout and cash. No arbitrary IDs or
# owner changes: an actual planter seizes the other side's first tree.
sim = FastCarbon().reset(seed=7)
ref = parity.RefChain(sim.full_observation())
p0, p1 = "player-0-worker-0", "player-1-worker-0"
b0, b1 = sim.bases[0][0][0], sim.bases[1][0][0]
prefix = [({b0: "RECPLANTER"}, {b1: "RECPLANTER"}),
          ({p0: "RIGHT"}, {p1: "LEFT"}), ({}, {p1: "LEFT"}),
          ({p0: "DOWN"}, {p1: "LEFT"}), ({p0: "DOWN"}, {p1: "LEFT"}),
          ({p0: "DOWN"}, {p1: "LEFT"}), ({p0: "DOWN"}, {})]
for a, b in prefix:
    sim.step(a, b)
    ref.step(a, b)
    assert not parity.diff_states(parity.state_from_obs(ref.obs), parity.sim_state(sim))
snapshot = sim.full_observation()
restored = FastCarbon().load_observation(copy.deepcopy(snapshot))
before = {"step": sim.step_count, "cash": list(sim.cash), "players": snapshot["players"],
          "original_counters": sim.tree_counters, "restored_counters": restored.tree_counters}
sim.step({}, {})
ref.step({}, {})
restored.step({}, {})
ref_state = parity.state_from_obs(ref.obs)
continuous_diff = parity.diff_states(ref_state, parity.sim_state(sim))
snapshot_diff = parity.diff_states(ref_state, parity.sim_state(restored))
assert not continuous_diff
assert snapshot_diff  # Record the defect; not a success assertion on the API.
out["seized_tree_snapshot"] = dict(pass_=False, before=before,
    actions=[{}, {}], expected=ref_state["trees"], restored=parity.sim_state(restored)["trees"],
    restored_players=restored.full_observation()["players"], diff=snapshot_diff,
    continuous_matches_original=True)

# Public information is identical under a hidden-map change. Feat must ignore
# unseen values even if handed a full-map-shaped carbon vector by a caller.
obs = snapshot
visibility_results = []
for seat in (0, 1):
    raw = copy.deepcopy(obs)
    raw["player"] = seat
    own = raw["players"][seat][2]
    visible = set()
    for pos, cargo, kind in own.values():
        visible.update([pos] if kind == "COLLECTOR" else F.DEST[pos, 1:].tolist())
    changed = copy.deepcopy(raw)
    for cell in set(range(225)) - visible:
        changed["carbon"][cell] = 99.0
    a, b = F.Feat(seat), F.Feat(seat)
    xa, xb = a.observe(raw), b.observe(changed)
    diffs = [k for k in xa if isinstance(xa[k], np.ndarray) and not np.array_equal(xa[k], xb[k])]
    assert not diffs, (seat, diffs)
    # Actual simulator observations expose exactly the game's allowed carbon.
    known = FastCarbon().load_observation(changed)
    got = known.obs(seat)["carbon"]
    assert all(got[c] == (known.carbon[c] if c in visible else 0) for c in range(225))
    visibility_results.append(dict(seat=seat, visible_count=len(visible), changed_hidden_cells=225-len(visible),
                                   differing_feature_fields=diffs, pass_=True))
out["hidden_map_noninterference"] = visibility_results

# A branch observing a future state must not mutate its root's estimate,
# previous-state dictionaries, ID/token mapping, or histories.
root = F.Feat(0)
root.observe(dict(snapshot, player=0))
before_root = copy.deepcopy(root.__dict__)
branch = S.feat_copy(root)
branch.observe(sim.obs(0))
def same(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b
changed_fields = [k for k in before_root if not same(before_root[k], root.__dict__[k])]
assert not changed_fields, changed_fields
out["feature_clone_isolation"] = {"pass": True, "changed_root_fields": changed_fields}

base = os.environ.get("CARBON_REVIEW_OUT")
if base:
    pathlib.Path(base).expanduser().mkdir(parents=True, exist_ok=True)
    base = str(pathlib.Path(base).expanduser().resolve())
dest = pathlib.Path(tempfile.mkdtemp(prefix="carbon-contract-r5-", dir=base)) / "simulator-contracts.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"parity_checks": len(out["original_board_parity"]),
                  "snapshot_diff": snapshot_diff, "evidence": str(dest)}, indent=2))
