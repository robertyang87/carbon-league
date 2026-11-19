"""Predict the next 25 turns of worker deliveries from the candidate's public observation."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "midgame_role_value_dev"))
from data import sr
from role import _D
from turning_point_ledger import CashEvents

HORIZON = 25
FEATURES = ["cargo", "cargo squared", "home distance", "nearest enemy collector",
            "friendly collectors nearby", "enemy collectors nearby", "friendly collector count",
            "enemy collector count", "cash below hire price",
            "public step", "remaining horizon", "is Sapling", "cargo times home distance"]


def features(obs, side, uid):
    t = obs["step"]
    own, opp = obs["players"][side], obs["players"][1-side]
    pos, cargo, _ = own[2][uid]
    friends = [w[0] for k, w in own[2].items() if k != uid and w[2] == "COLLECTOR"]
    foes = [w[0] for w in opp[2].values() if w[2] == "COLLECTOR"]
    fd = _D[pos, friends]
    ed = _D[pos, foes]
    home = min(_D[pos, b] for b in own[1].values())
    cargo /= 100.
    return [cargo, cargo*cargo, home, min(ed, default=15), sum(fd<=2), sum(ed<=2),
            len(friends)+1, len(foes), own[0]<30,
            t/299, min(299-t,HORIZON)/HORIZON, side==obs["player"], cargo*home]


def extract(raw, seat):
    sim = sr.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    sim.cash = CashEvents(sim.cash)
    paid = defaultdict(list)
    rows = []
    for t, joint in enumerate(raw["actions"]):
        obs = sim.obs(seat)
        for side in (0,1):
            for uid,w in obs["players"][side][2].items():
                if w[2] == "COLLECTOR":
                    rows.append((t, uid, features(obs, side, uid), w[1]))
        sim.cash.events.clear()
        sim.step(*joint)
        for kind,side,val,*extra in sim.cash.events:
            if kind=="own_delivery":
                paid[extra[0]].append((t+1,val))
    assert list(sim.cash) == raw["official_score_by_seat"]
    return [(t,uid,x,c,sum(v for step,v in paid[uid] if t<step<=t+HORIZON)) for t,uid,x,c in rows]


def predict(x, model):
    z = (np.asarray(x)-model["center"])/model["scale"]
    return np.maximum(0, model["intercept"]+z@model["beta"])


def run():
    cover = sr.load(HERE.parent/"harvest_contact_dev/coverage.json.gz")
    seeds = sorted({g["seed"] for g in cover})
    train, holdout = seeds[:3], seeds[3:]
    grouped = {}
    sources = []
    for g in cover:
        path = sr.ROOT/g["source"]
        grouped[g["seed"],g["seat"]] = extract(sr.load(path),g["seat"])
        sources.append(dict(seed=g["seed"],seat=g["seat"],source=g["source"],sha256=sr.sha(path)))
    rows = [r for (seed,_),batch in grouped.items() if seed in train for r in batch]
    x = np.array([r[2] for r in rows]); y = np.array([r[4] for r in rows])
    center,scale = x.mean(0), x.std(0)
    scale[scale<1e-8] = 1
    z = (x-center)/scale
    beta = np.linalg.solve(z.T@z+100*np.eye(z.shape[1]), z.T@(y-y.mean()))
    model = dict(center=center.tolist(),scale=scale.tolist(),intercept=float(y.mean()),beta=beta.tolist(),features=FEATURES)
    sr.retain(HERE/"weights.json",model)
    result = []
    for (seed,seat), batch in grouped.items():
        if seed not in holdout:continue
        x=np.array([r[2] for r in batch]); actual=np.array([r[4] for r in batch])
        pr=predict(x,model); cargo=np.array([r[3] for r in batch])
        result.append(dict(seed=seed,seat=seat,n=len(pr),mae=float(np.abs(pr-actual).mean()),
                           cargo_mae=float(np.abs(cargo-actual).mean())))
    sr.retain(HERE/"evaluation.json",dict(train=train,holdout=holdout,sources=sources,tests=result))
    print(result)

if __name__=="__main__":
    run()
