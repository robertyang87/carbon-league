"""Reconstruct the measured bc69 early intervention from its actual shared public prefix."""
import numpy as np
from support import HERE, ROOT, F, read, save, sha, begin, public, option, _carbon_view

native = HERE.parent / "native_response/621758327_0"
shared = read(native / "shared.json.gz")
c, root = shared["case"], shared["root"]
tape = read(ROOT / c["archived_map_fixture"])
assert sha((ROOT / c["archived_map_fixture"]).read_bytes()) == c["fixture_sha256"]
env, seat = begin(tape), c["seat"]
feat = F.Feat(seat)
for both in shared["prefix"]:
    feat.observe(public(env,seat))
    env.step(both)
obs = public(env,seat)
assert obs["step"] == root["t"] == len(shared["prefix"])
x = feat.observe(obs)
params=np.load(HERE.parent/"weights.npz")
maps = [z["maps"][0].astype(float)*(100/255) for z in (x,_carbon_view(x,feat,F,[params[k] for k in "wbva"]))]
us=obs["players"][seat]
units = []
for uid in root["changed"]:
    actions = [root[arm].get(uid,"NONE") for arm in ("parent","student")]
    vectors=[]
    for m in maps:
        arm=[option(m,us[1],us[2],uid,a,F.CANON if seat else F.GRID)[0] for a in actions]
        vectors.append(arm[1]-arm[0])
    units.append(dict(uid=uid,role=us[2][uid][2],actions=actions,delta_contrast=(vectors[1]-vectors[0]).tolist()))
save(HERE/"intervention_support.json",dict(seed=c["seed"],seat=seat,step=root["t"],shared_sha256=sha((native/"shared.json.gz").read_bytes()),changed_cells=int(np.sum(maps[0]!=maps[1])),units=units))
