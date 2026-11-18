"""File-level self-proof for stage0_teacher_homefix2.py.

Run: python3 tests_homefix2.py   (from this directory)

Covers, in order:
  T0  identity: base file untouched (md5), variant stamps itself.
  U1  homefix2_plan unit tests (window edges, lethality grounding,
      policy order, masking, canonical frame, claimed destinations).
  U2  regression: homefix2=False == base evaluate_root, bit for bit;
      homefix2=True on an untriggered root == base, bit for bit.
  C-D candidate presence: A0 stays arm 0; the protected arm is in the
      priced set and carries the planned step verbatim (seat 0 + seat 1).
  C-A exit-override contrast (channel 2.2, pricing/execution
      decoupling): on the crafted root the old form's rewrite overrules
      its own search's priced selection; the new form's executed
      selection is always the priced argmax arm, verbatim.
  C-V H=2 veto: a priced protected arm is rejected by pricing
      (veto counter increments) -- the shallow pricing CAN say no.
  C-B joint-action desync (channel 2.3): old form collides two own
      collectors on one cell and the heavier dies; the protected arm
      keeps destinations distinct and nobody dies.
  C-C unchecked tail death (channel 2.4): old form's direction-first
      step is lethal and the worker dies in fastsim; the new plan
      takes the safe step; all-lethal -> skipped, search untouched.
  C-I instrumentation: recorder arithmetic for the three registered
      readings (trigger rate / adoption rate / delta_q distribution),
      JSONL scenario sink, env switch.
The arithmetic-ceiling channel (explanation section 2.1) is a payoff
property, not an implementation defect -- no counter-example is owed
for it (per the tasking); it is priced into the ruling's line.
"""
import hashlib
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import stage0_teacher as base  # noqa: E402


def _load_variant():
    spec = importlib.util.spec_from_file_location(
        "stage0_teacher_homefix2", os.path.join(HERE, "stage0_teacher_homefix2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load_variant()
WEIGHTS = os.path.expanduser(
    "~/workspace/goals/carbon-league-large-file-leaderboard-challenge/"
    "hidden_files/k0pkg/weights/cnet2_m3s2r1_heads_v2.npz")
POL = V.CachePolicy(dict(np.load(WEIGHTS)))

BASE_POS = 112  # (7,7)
PASS = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name}: {detail}"
    PASS.append(name)
    print(f"  PASS {name} {detail}")


def dest(pos, move):
    return V._dest(pos, move, 15)


def place(moves, start=BASE_POS):
    p = start
    for m in moves:
        p = dest(p, m)
    return p


def make_sim(step, own_workers, enemy_workers, carbon=None,
             own_base=BASE_POS, enemy_base=224, cash=400.0):
    """own/enemy_workers: list of (suffix, pos, cargo, occ)."""
    ow = {f"player-0-worker-{w[0]}": [w[1], w[2], w[3]] for w in own_workers}
    ew = {f"player-1-worker-{w[0]}": [w[1], w[2], w[3]] for w in enemy_workers}
    carb = [0.0] * 225
    for pos, val in (carbon or {}).items():
        carb[pos] = val
    sim = FastCarbon()
    sim.load_observation({
        "step": step, "carbon": carb,
        "players": [
            [cash, {"player-0-base-0": own_base}, ow, {}],
            [cash, {"player-1-base-0": enemy_base}, ew, {}],
        ]})
    return sim


def feats_for(sim, me):
    f = F.Feat(me)
    x = f.observe(sim.obs(me))
    shim = SimpleNamespace(seat=me, own_uid=list(f.own_uid))
    return f, x, f.aux, shim


def eval_root(sim, me, homefix2, recorder=None, mod=None):
    _, x, aux, shim = feats_for(sim, me)
    _, xo, auxo, shimo = feats_for(sim, 1 - me)
    mod = mod or V
    kw = dict(teacher=1, K=16, M=2, H=2, seed_base=7000, guard="on",
              belief="oracle")
    if mod is V:
        return V.evaluate_root_homefix2(sim, me, POL, x, aux, xo, auxo,
                                        shim, shimo, homefix2=homefix2,
                                        hf2_recorder=recorder, **kw)
    return base.evaluate_root(sim, me, POL, x, aux, xo, auxo, shim, shimo, **kw)


def same_eval(a, b):
    if len(a["arms"]) != len(b["arms"]):
        return False
    if any(not np.array_equal(x, y) for x, y in zip(a["arms"], b["arms"])):
        return False
    return (np.array_equal(a["priors"], b["priors"])
            and np.array_equal(a["q"], b["q"])
            and np.array_equal(a["Q_mean"], b["Q_mean"])
            and a["entropy"] == b["entropy"])


# ---- old rule, verbatim from EXPLANATION_HOMEFIX_NEGATIVE_20261118.md s.1
def old_rule_override(workers, bases, step, actions):
    """workers: {uid: (pos, cargo)} own collectors. Returns the joint
    action dict after the old exit override (direction order
    UP, RIGHT, DOWN, LEFT; first distance-reducing step; no checks)."""
    left = 299 - step
    out = dict(actions)
    for uid, (pos, cargo) in workers.items():
        if cargo <= 0:
            continue
        d = min(V._torus_dist(pos, b, 15) for b in bases)
        if d == 0 or left > d + 2:
            continue
        for m in (1, 2, 3, 4):
            z = dest(pos, m)
            if min(V._torus_dist(z, b, 15) for b in bases) == d - 1:
                out[uid] = F.MOVES[m]
                break
    return out


ENEMY_IDLE = None  # filled per-test


def step_with(sim, own_actions, enemy_workers_spec):
    ew = {f"player-1-worker-{w[0]}": "NONE" for w in enemy_workers_spec}
    ew["player-1-base-0"] = "HOLD"
    sim.step(own_actions, ew)
    return sim


# =====================================================================
print("T0 identity")
md5 = hashlib.md5(open(os.path.join(SRC, "stage0_teacher.py"), "rb").read()).hexdigest()
check("T0 base untouched", md5.startswith("c6ade5ce"), md5[:8])
check("T0 rule point id", V.RULE_POINT_ID == "homefix2-candidate-internal-v1")

# =====================================================================
print("U1 plan unit tests")


class FakeSim:
    def __init__(self, players, step):
        self.config = {"size": 15}
        self.step_count = step
        self._players = players

    def obs(self, me):
        return {"players": self._players}


def plan_for(own, enemy, step, logits_map=None, seat=0):
    mine = [400.0, {"b0": BASE_POS}, {u: [p, c, o] for u, (p, c, o) in own.items()}, {}]
    theirs = [400.0, {"b1": 224}, {u: [p, c, o] for u, (p, c, o) in enemy.items()}, {}]
    players = [mine, theirs] if seat == 0 else [theirs, mine]
    sim = FakeSim(players, step)
    shim = SimpleNamespace(seat=seat, own_uid=list(own.keys()) + [None] * (12 - len(own)))
    logits = [np.full(5, -1.0) for _ in range(12)]
    for (s, lab), v in (logits_map or {}).items():
        logits[s][lab] = v
    return V.homefix2_plan(sim, seat, shim, logits)


W = "player-0-worker-0"
E = "player-1-worker-0"
p = plan_for({W: (place([2, 2, 2]), 5.0, "COLLECTOR")}, {}, 295)
check("U1a window trigger", p["triggered"] and p["workers"][0]["move"] == 4)
p = plan_for({W: (place([2] * 5), 5.0, "COLLECTOR")}, {}, 295)
check("U1b d>left no trigger", not p["triggered"] and not p["skipped"])
p = plan_for({W: (place([2]), 5.0, "COLLECTOR")}, {}, 290)
check("U1c d below window no trigger", not p["triggered"])
# worker at (6,8)=98: reducing DOWN(3)->113, LEFT(4)->97
p = plan_for({W: (98, 5.0, "COLLECTOR")}, {E: (113, 5.0, "COLLECTOR")}, 295,
             {(0, 3): 2.0, (0, 4): 1.0})
check("U1d equal-cargo enemy lethal, safe step taken",
      p["workers"][0]["move"] == 4)
p = plan_for({W: (98, 5.0, "COLLECTOR")},
             {E: (113, 3.0, "COLLECTOR"), "player-1-worker-1": (97, 3.0, "COLLECTOR")}, 295)
check("U1e all reducing lethal -> skipped",
      not p["triggered"] and p["skipped"][0]["reason"] == "no_safe_step")
p = plan_for({W: (98, 5.0, "COLLECTOR")}, {E: (113, 9.0, "COLLECTOR")}, 295,
             {(0, 3): 2.0, (0, 4): 1.0})
check("U1f heavier enemy safe", p["workers"][0]["move"] == 3)
p = plan_for({W: (98, 5.0, "COLLECTOR")}, {E: (113, 3.0, "PLANTER")}, 295,
             {(0, 3): 2.0, (0, 4): 1.0})
check("U1g enemy planter safe", p["workers"][0]["move"] == 3)
p = plan_for({W: (98, 5.0, "COLLECTOR")}, {}, 295, {(0, 3): 1.0, (0, 4): 3.0})
check("U1h policy order beats direction order", p["workers"][0]["move"] == 4)
p = plan_for({W: (98, 5.0, "COLLECTOR")}, {}, 295, {(0, 3): 1.0, (0, 4): -20.0})
check("U1i masked step skipped", p["workers"][0]["move"] == 3)
p = plan_for({W: (place([2, 2, 2]), 8.0, "PLANTER")}, {}, 295)
check("U1j planters not protected", not p["triggered"])
# seat 1 canonical frame: canonical label 2 == real LEFT(4), label 3 == real DOWN(3)
p = plan_for({W: (98, 5.0, "COLLECTOR")}, {}, 295, {(0, 2): 5.0, (0, 3): 1.0}, seat=1)
check("U1k seat-1 canonical mapping", p["workers"][0]["move"] == 4)
# claimed destination: A slot0 -> 113 (DOWN preferred), B slot1 wants UP->113 too
own2 = {"player-0-worker-0": (98, 30.0, "COLLECTOR"),
        "player-0-worker-1": (128, 10.0, "COLLECTOR")}
players = [
    [400.0, {"b0": BASE_POS}, {u: [q, c, o] for u, (q, c, o) in own2.items()}, {}],
    [400.0, {"b1": 224}, {}, {}],
]
sim = FakeSim(players, 295)
shim = SimpleNamespace(seat=0, own_uid=list(own2.keys()) + [None] * 10)
logits = [np.full(5, -1.0) for _ in range(12)]
logits[0][3] = 5.0
logits[1][1] = 5.0
logits[1][4] = 1.0
p = V.homefix2_plan(sim, 0, shim, logits)
check("U1l claimed destination avoided",
      p["workers"][0]["dest"] == 113 and p["workers"][1]["dest"] == 127)

# =====================================================================
print("U2 regression (feature off / untriggered == base, exact)")
spec1 = dict(step=150,
             own_workers=[(0, 60, 12.0, "COLLECTOR"), (1, 200, 0.0, "COLLECTOR"),
                          (2, 33, 0.0, "PLANTER")],
             enemy_workers=[(0, 90, 7.0, "COLLECTOR")])
r_base = eval_root(make_sim(**spec1), 0, False, mod=base)
r_off = eval_root(make_sim(**spec1), 0, False)
check("U2a homefix2=False == base", same_eval(r_base, r_off))
check("U2a2 version stamped", r_off["teacher_version"].endswith("+homefix2"))
spec2 = dict(step=285, own_workers=[(0, place([2] * 6), 9.0, "COLLECTOR")],
             enemy_workers=[(0, 14, 0.0, "COLLECTOR")])
r_base2 = eval_root(make_sim(**spec2), 0, False, mod=base)
r_on2 = eval_root(make_sim(**spec2), 0, True)
check("U2b untriggered homefix2=True == base", same_eval(r_base2, r_on2)
      and r_on2["homefix2"]["triggered"] is False)

# =====================================================================
print("C-D candidate presence (triggered root)")
specD = dict(step=295,
             own_workers=[(0, place([2, 2, 2]), 20.0, "COLLECTOR"),
                          (1, 0, 0.0, "COLLECTOR")],
             enemy_workers=[(0, 14, 0.0, "COLLECTOR")])
rec = V.Homefix2Recorder()
rD = eval_root(make_sim(**specD), 0, True, recorder=rec)
rD_base = eval_root(make_sim(**specD), 0, False, mod=base)
blk = rD["homefix2"]
check("C-D1 triggered", blk["triggered"] is True, str(blk["prot_workers"]))
check("C-D2 A0 is base arm 0", np.array_equal(rD["arms"][0], rD_base["arms"][0]))
idx = blk["prot_arm_index"]
check("C-D3 protected arm priced", idx is not None and blk["q_prot"] is not None)
fD, *_ = feats_for(make_sim(**specD), 0)
acts = F.Feat.to_actions(fD, rD["arms"][idx])
w0 = "player-0-worker-0"
planned = F.MOVES[blk["prot_workers"][0]["move"]]
check("C-D4 protected step verbatim in arm", acts[w0] == planned,
      f"{acts[w0]} vs planned {planned}")
check("C-D5 recorder counted", rec.roots == 1 and rec.triggered == 1)
# seat-1 subject: worker near ITS OWN base (224): (14,11)=221, d=3
specD1 = dict(step=295,
              own_workers=[(0, 14, 0.0, "COLLECTOR")],
              enemy_workers=[(0, place([4, 4, 4], start=224), 20.0, "COLLECTOR"),
                             (1, 0, 0.0, "COLLECTOR")])
rD1 = eval_root(make_sim(**specD1), 1, True)
blk1 = rD1["homefix2"]
check("C-D6a seat-1 triggered", blk1["triggered"] is True, str(blk1["prot_workers"]))
fD1, *_ = feats_for(make_sim(**specD1), 1)
acts1 = F.Feat.to_actions(fD1, rD1["arms"][blk1["prot_arm_index"]])
planned1 = F.MOVES[blk1["prot_workers"][0]["move"]]
check("C-D6 seat-1 protected step verbatim",
      blk1["triggered"] and acts1["player-1-worker-0"] == planned1,
      f"{acts1['player-1-worker-0']} vs {planned1}")

# =====================================================================
print("C-A exit-override contrast (pricing/execution decoupling)")
# Craft (probed): single collector cargo 5 at 116 (d=4, step 293, left
# 6, window [4,6]), rich cell 200 at 117. The base search's priced
# selection is A0 (argmax q == 0) sending the worker RIGHT (collect);
# the old override would rewrite the SELECTED action to LEFT (home)
# after pricing -- executing a joint action its own pricing rejected.
specA = dict(step=293,
             own_workers=[(0, place([2, 2, 2, 2]), 5.0, "COLLECTOR")],
             enemy_workers=[(0, 14, 0.0, "COLLECTOR")],
             carbon={place([2, 2, 2, 2, 2]): 200.0})
rA_base = eval_root(make_sim(**specA), 0, False, mod=base)
rA = eval_root(make_sim(**specA), 0, True)
fA, *_ = feats_for(make_sim(**specA), 0)
sel_base = int(np.argmax(rA_base["q"]))
a0_acts = F.Feat.to_actions(fA, rA_base["arms"][sel_base])
own_state = {w0: (place([2, 2, 2, 2]), 5.0)}
old_joint = old_rule_override(own_state, [BASE_POS], 293, a0_acts)
check("C-A1 old override rewrites the priced selection post-pricing",
      sel_base == 0 and a0_acts[w0] == "RIGHT" and old_joint[w0] == "LEFT",
      f"selection arm {sel_base} {w0}={a0_acts[w0]} -> old executes {old_joint[w0]}")
old_labels = F.Feat.labels(fA, old_joint)
match_idx = [i for i, a in enumerate(rA_base["arms"])
             if np.array_equal(old_labels, a)]
check("C-A2 executed product != priced selection (decoupling)",
      all(i != sel_base for i in match_idx),
      f"old product coincides with priced arm(s) {match_idx}, none of "
      f"which is the selection {sel_base}: presence in the set does not "
      f"save the old form -- its pricing picked another arm and the "
      f"override overruled the verdict")
blkA = rA["homefix2"]
sel_var = int(np.argmax(rA["q"]))
prot_acts = F.Feat.to_actions(fA, rA["arms"][blkA["prot_arm_index"]])
check("C-A3 new form: executed selection is the priced arm verbatim",
      blkA["triggered"] and blkA["q_prot"] is not None
      and prot_acts[w0] == "LEFT"
      and blkA["adopted"] == (sel_var == blkA["prot_arm_index"]),
      f"prot idx={blkA['prot_arm_index']} sel={sel_var} "
      f"adopted={blkA['adopted']} vetoed={blkA['vetoed']} dq={blkA['delta_q']}")

# =====================================================================
print("C-V H=2 veto scan")
rec2 = V.Homefix2Recorder()
veto_found = []
scan = []
for step, d, rich_moves in ((290, 8, [2] * 7 + [3, 3]), (293, 5, [2] * 5 + [3])):
    for cargo in (1.0, 5.0):
        wpos = place(([2] * 7 + [3]) if d == 8 else [2] * 5)
        rich = place(rich_moves)
        spec = dict(step=step,
                    own_workers=[(0, wpos, cargo, "COLLECTOR")],
                    enemy_workers=[(0, 14, 0.0, "COLLECTOR")],
                    carbon={rich: 200.0})
        r = eval_root(make_sim(**spec), 0, True, recorder=rec2)
        b = r["homefix2"]
        scan.append((step, cargo, b["triggered"], b["prot_arm_index"],
                     None if b["delta_q"] is None else round(b["delta_q"], 4),
                     b["adopted"], b["vetoed"]))
        if b["vetoed"] and b["delta_q"] is not None and b["delta_q"] < 0:
            veto_found.append((step, cargo))
for row in scan:
    print("   scan step=%s cargo=%s trig=%s idx=%s dq=%s adopted=%s vetoed=%s" % row)
check("C-V1 veto observed (priced protection rejected)", len(veto_found) >= 1,
      str(veto_found))
check("C-V2 recorder veto counter", rec2.vetoed >= 1 and rec2.offered >= 1,
      f"offered={rec2.offered} vetoed={rec2.vetoed} adopted={rec2.adopted}")

# =====================================================================
print("C-B joint-action desync")
specB = dict(step=295,
             own_workers=[(0, 98, 30.0, "COLLECTOR"), (1, 128, 10.0, "COLLECTOR")],
             enemy_workers=[(0, 14, 0.0, "COLLECTOR")])
rB = eval_root(make_sim(**specB), 0, True)
fB, *_ = feats_for(make_sim(**specB), 0)
a0B = F.Feat.to_actions(fB, rB["arms"][0])
ownB = {"player-0-worker-0": (98, 30.0), "player-0-worker-1": (128, 10.0)}
oldB = old_rule_override(ownB, [BASE_POS], 295, a0B)
sim_old = step_with(make_sim(**specB), oldB, specB["enemy_workers"])
dead_old = {e[0] for e in getattr(sim_old, "last_collisions", [])}
check("C-B1 old form: both sent to 113, heavier collector dies",
      oldB["player-0-worker-0"] == "DOWN" and oldB["player-0-worker-1"] == "UP"
      and "player-0-worker-0" in dead_old, f"collisions={dead_old}")
blkB = rB["homefix2"]
prot_acts = F.Feat.to_actions(fB, rB["arms"][blkB["prot_arm_index"]])
dA = dest(98, F.MOVES.index(prot_acts["player-0-worker-0"]))
dB = dest(128, F.MOVES.index(prot_acts["player-0-worker-1"]))
check("C-B2 protected arm destinations distinct", dA != dB, f"{dA} vs {dB}")
sim_new = step_with(make_sim(**specB), prot_acts, specB["enemy_workers"])
dead_new = {e[0] for e in getattr(sim_new, "last_collisions", [])}
check("C-B3 protected arm stepped: nobody dies", not dead_new,
      f"collisions={dead_new}")

# =====================================================================
print("C-C unchecked tail death")
specC = dict(step=295, own_workers=[(0, 98, 10.0, "COLLECTOR")],
             enemy_workers=[(0, 113, 5.0, "COLLECTOR")])
rC = eval_root(make_sim(**specC), 0, True)
fC, *_ = feats_for(make_sim(**specC), 0)
a0C = F.Feat.to_actions(fC, rC["arms"][0])
oldC = old_rule_override({w0: (98, 10.0)}, [BASE_POS], 295, a0C)
sim_oldC = step_with(make_sim(**specC), oldC, specC["enemy_workers"])
deadC = {e[0] for e in getattr(sim_oldC, "last_collisions", [])}
check("C-C1 old form steps onto lighter enemy and dies",
      oldC[w0] == "DOWN" and w0 in deadC, f"collisions={deadC}")
blkC = rC["homefix2"]
protC = F.Feat.to_actions(fC, rC["arms"][blkC["prot_arm_index"]])
check("C-C2 new plan takes the safe step", protC[w0] == "LEFT",
      f"arm action {protC[w0]}")
sim_newC = step_with(make_sim(**specC), protC, specC["enemy_workers"])
deadCn = {e[0] for e in getattr(sim_newC, "last_collisions", [])}
check("C-C3 protected step survives", w0 not in deadCn)
specC2 = dict(step=295, own_workers=[(0, 98, 10.0, "COLLECTOR")],
              enemy_workers=[(0, 113, 5.0, "COLLECTOR"), (1, 97, 5.0, "COLLECTOR")])
rC2 = eval_root(make_sim(**specC2), 0, True)
rC2b = eval_root(make_sim(**specC2), 0, False, mod=base)
check("C-C4 all-lethal -> no protection, search untouched",
      rC2["homefix2"]["triggered"] is False
      and rC2["homefix2"]["skipped"] and same_eval(rC2, rC2b))

# =====================================================================
print("C-I instrumentation arithmetic")
rec3 = V.Homefix2Recorder()
rec3.record({"triggered": True, "prot_arm_index": 3, "prot_is_a0": False,
             "prot_dropped": None, "adopted": True, "vetoed": False, "delta_q": 0.10})
rec3.record({"triggered": True, "prot_arm_index": 2, "prot_is_a0": False,
             "prot_dropped": None, "adopted": False, "vetoed": True, "delta_q": -0.20})
rec3.record({"triggered": False, "prot_arm_index": None, "prot_is_a0": False,
             "prot_dropped": None, "adopted": None, "vetoed": None, "delta_q": None})
s = rec3.summary()
check("C-I1 three readings arithmetic",
      s["roots"] == 3 and s["triggered"] == 2 and abs(s["trigger_rate"] - 2 / 3) < 1e-9
      and s["offered"] == 2 and s["adopted"] == 1 and s["vetoed"] == 1
      and abs(s["adoption_rate_offered"] - 0.5) < 1e-9
      and s["delta_q"]["n"] == 2 and abs(s["delta_q"]["mean"] + 0.05) < 1e-9
      and abs(s["delta_q"]["share_nonneg"] - 0.5) < 1e-9, json.dumps(s))
jsonl_path = os.path.join(HERE, "test_recorder_out.jsonl")
if os.path.exists(jsonl_path):
    os.remove(jsonl_path)
rec4 = V.Homefix2Recorder(jsonl_path)
rec4.record({"triggered": True, "prot_arm_index": 1, "prot_is_a0": False,
             "prot_dropped": None, "adopted": True, "vetoed": False,
             "delta_q": 0.03}, meta={"step": 295, "me": 0})
rows = [json.loads(l) for l in open(jsonl_path)]
check("C-I2 JSONL scenario sink", len(rows) == 1 and rows[0]["meta"]["step"] == 295)
os.environ["CARBON_HOMEFIX2_RECORD"] = jsonl_path
check("C-I3 env switch on", V.recorder_from_env() is not None)
del os.environ["CARBON_HOMEFIX2_RECORD"]
check("C-I4 env switch off", V.recorder_from_env() is None)

print(f"\nALL {len(PASS)} CHECKS PASSED")
