"""Build the A/B target game + sampler spec sets.

Target game: REAL ladder game 899d25b15b488b25ae538a5afc5fbc21-1
(harvest file: 189 frames, true terminal at frame 188 by seat status --
seat1 eliminated). The harvest truncates at the terminal frame, so the
file carries no padding tail. The padded variant extends frames
189..299 with deep copies of the terminal frame -- exactly the padding
construction of the owner's K0 counterexample. Real frames 0..188 are
byte-identical game content; only the padding tail is constructed.
Real roots (step < 188) never read frames past their own step
(extract_scene replays the prefix only) and continuations never read
the replay at all, so the tail's content cannot enter a real root's
computation through any data channel; the channels under test are
root-set composition effects (cross-root process/cache state).
"""
import copy
import gzip
import hashlib
import json
import os
import sys

sys.path.insert(0, "/home/hatch/workspace/padding-ab/src")
import k0_aux_labels as K0  # noqa: E402

ORIG = ("/home/hatch/workspace/carbon-league/replays/replays/"
        "899d25b15b488b25ae538a5afc5fbc21-1.json.gz")
OUTDIR = "/home/hatch/workspace/padding-ab/game"
os.makedirs(OUTDIR, exist_ok=True)

raw = json.load(gzip.open(ORIG))
steps = raw["steps"]
cfg = raw.get("configuration") or {}
cheapest = min(float(cfg.get("recCollectorCost", K0.DEFAULT_RECRUIT_COST)),
               float(cfg.get("recPlanterCost", K0.DEFAULT_RECRUIT_COST)))
ft, cause = K0.first_terminal(steps, cheapest)
print("orig frames:", len(steps), "first_terminal:", ft, "cause:", cause)
assert ft == len(steps) - 1, "expected harvest truncation at terminal"

padded = dict(raw)
padded["steps"] = steps + [copy.deepcopy(steps[-1]) for _ in range(300 - len(steps))]
ft2, cause2 = K0.first_terminal(padded["steps"], cheapest)
print("padded frames:", len(padded["steps"]), "FT still:", ft2, cause2)
assert ft2 == ft

out_path = os.path.join(OUTDIR, os.path.basename(ORIG))
with gzip.open(out_path, "wt") as fh:
    json.dump(padded, fh)


def md5_of(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("orig  md5:", md5_of(ORIG))
print("padded md5:", md5_of(out_path), "->", out_path)

# ---- sampler spec sets (pre-fix gate sampler, used as-is) ----
import importlib.util  # noqa: E402

specmod = importlib.util.spec_from_file_location(
    "gate_prefix", "/home/hatch/workspace/padding-ab/src/stage1_teacher_gate_PREFIX.py")
G = importlib.util.module_from_spec(specmod)
specmod.loader.exec_module(G)

SEEDS = [995300, 995301, 995302, 995303, 995304, 995305]
all_specs, per_seed = [], {}
for sd in SEEDS:
    ss = G.sample_scene_specs([out_path], G.LAYERS_DU, sd,
                              opening_share=G.DU_OPENING_SHARE)
    per_seed[sd] = [(s["step"], s["me"]) for s in ss]
    for s in ss:
        key = (s["step"], s["me"])
        if key not in [(a["step"], a["me"]) for a in all_specs]:
            all_specs.append(s)
all_specs.sort(key=lambda s: (s["step"], s["me"]))
real = [s for s in all_specs if s["step"] < ft]
pad = [s for s in all_specs if s["step"] >= ft]
print("per-seed specs:", per_seed)
print("unique specs:", [(s["step"], s["me"]) for s in all_specs])
print("real roots:", [(s["step"], s["me"]) for s in real])
print("padding roots:", [(s["step"], s["me"]) for s in pad])
json.dump({"game_path": out_path, "ft": ft, "cause": cause,
           "seeds": SEEDS, "per_seed": {str(k): v for k, v in per_seed.items()},
           "specs_A": [{"step": s["step"], "me": s["me"]} for s in all_specs],
           "specs_B": [{"step": s["step"], "me": s["me"]} for s in real]},
          open("/home/hatch/workspace/padding-ab/out/specs.json", "w"), indent=1)
print("specs saved")
