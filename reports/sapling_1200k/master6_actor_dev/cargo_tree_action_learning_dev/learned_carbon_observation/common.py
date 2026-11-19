"""Frozen archives and public-observation replay for the carbon estimator."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

def save(path, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)

PROTOCOL = read(HERE / "protocol.json")
assert sha((HERE.parent / "candidate.py").read_bytes()) == PROTOCOL["incumbent"]
source = (HERE.parent / "candidate.py").read_text()
src = ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign) and n.targets[0].id == "_SRC"))
F = types.ModuleType("frozen_sapling_features")
exec(compile(src["cnet2_feat"], "frozen_sapling_features", "exec"), F.__dict__)

def public_frames(card):
    path = ROOT / card["path"]
    assert sha(path.read_bytes()) == card["sha256"]
    tape = read(path)
    sim = FastCarbon(tape["game_configuration"]).reset(seed=card["seed"], full_carbon=tape["initial_full_carbon"])
    assert sha(bytes(map(int, sim.carbon))) == card["terrain"]
    for step, joint in enumerate(tape["actions"]):
        observations = [sim.obs(seat) for seat in (0, 1)]
        assert all(set(o) == {"players","trees","player","carbon","step","remainingOverageTime"} and o["step"] == step for o in observations)
        yield observations, np.array(sim.carbon, dtype=np.float32), joint, tape["game_configuration"]
        sim.step(*joint)
    assert np.allclose(sim.cash, tape["last_post_cash"], rtol=0, atol=.011)

def selected(card):
    features = [F.Feat(s) for s in (0, 1)]
    for observations, truth, joint, cfg in public_frames(card):
        for s, o in enumerate(observations):
            feat = features[s]
            x = feat.observe(o)
            assert np.all(x["maps"][0].reshape(-1)[x["maps"][3].reshape(-1)>0] == F.q8(np.array(o["carbon"], np.float32)[F.CANON if s else F.GRID]/100)[x["maps"][3].reshape(-1)>0])
            if o["step"] % 6:
                continue
            cpos = F.CANON if s else F.GRID
            visible = x["maps"][3].reshape(-1) > 0
            raw = np.array(o["carbon"])[cpos]
            assert np.allclose(raw[visible], truth[cpos][visible], atol=.001, rtol=0)
            assert np.all(raw[~visible] == 0)
            if not o["step"]:
                base = x["maps"][14:16].reshape(2,225).any(axis=0)
                assert not np.any(visible) and not np.any(x["maps"][2]) and np.all(feat.est[cpos][~base] == 11)
            yield feat, x, truth[cpos], o, joint[s], cfg
