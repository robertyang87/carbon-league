"""Score both frozen deployed enemy models against the same observed opposing commands."""
import gzip
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
FEEDBACK = LADDER / "prediction_feedback"
sys.path.insert(0, str(FEEDBACK))
import calibrate


def read(path):
    return json.loads(path.read_bytes())


def packed(path, identity):
    data = path.read_bytes()
    calibrate.check(calibrate.sha(data) == identity["candidate_sha256"], "different live package")
    module = types.ModuleType("signed_feedback_actor")
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def stats(rows):
    return dict(
        workers=len(rows),
        fixed=sum(r["predicted"][0] != r["actual"] == r["predicted"][1] for r in rows),
        broken=sum(r["predicted"][0] == r["actual"] != r["predicted"][1] for r in rows),
        changed=sum(r["predicted"][0] != r["predicted"][1] for r in rows),
        correct=[sum(r["predicted"][i] == r["actual"] for r in rows) for i in (0, 1)],
        nll=[sum(r["nll"][i] for r in rows)/len(rows) if rows else None for i in (0, 1)],
    )


def evaluate(card, v1, net, v2):
    ticket = dict(id=card["match_id"], game=card["game"], seat=card["candidate_seat"], opponent=card["opponent"])
    path = LADDER / "replays" / f"{ticket['id']}-{ticket['game']}.json.gz"
    game = calibrate.import_game(ticket, path, "new_public_ladder")
    rows = []
    joint = [0, 0]
    theta = np.asarray(v2._FEEDBACK_WEIGHTS, np.float32)
    for t, enemy, players, units, cells, phi, x, lb, truth, seed, action in calibrate.predictors(game, v1, net):
        y = v2._feedback_adjust(x.copy(), players, enemy, v2._FEEDBACK_WEIGHTS, v2._OPP_STEPS, v2.FEAT_DIST)
        calibrate.check(np.array_equal(y[0,cells], x[0,cells] + phi @ theta), "packaged readout drift")
        chosen = [m._sample(players,enemy,z[0],lb[0],np.random.RandomState(seed),t) for m,z in ((v1,x),(v2,y))]
        for i,c in enumerate(chosen):
            joint[i] += c == action
        loss = [logsumexp(z[0,cells].astype(np.float64),axis=1)-z[0,cells,truth] for z in (x,y)]
        for j,(uid,cargo) in enumerate(units):
            dest = action.get(uid) or "NONE"
            home_change = (int(phi[j,truth[j],1]>0)-int(phi[j,truth[j],1]<0)) if cargo else None
            rows.append(dict(t=t, unit=uid, cargo=cargo, observed_home_progress=home_change,
                             actual=dest,predicted=[o.get(uid) or "NONE" for o in chosen],
                             nll=[float(l[j]) for l in loss]))
    groups = {
        "all": rows,
        "cargo_ge200": [r for r in rows if r["cargo"]>=200],
        "cargo_ge200_moving_towards_home": [r for r in rows if r["cargo"]>=200 and r["observed_home_progress"]==1],
        "t250_onward": [r for r in rows if r["t"]>=250],
    }
    return dict(ticket=ticket,opening_sha256=game["terrain"],replay_sha256=game["source_gzip_sha256"],
                active_steps=len(game["frames"]),joint_order_correct=joint,groups={k:stats(v) for k,v in groups.items()},
                per_worker=rows)


def main():
    receipts = [read(LADDER / f"ds-pro-v{i}.json") for i in (1,2)]
    v1 = calibrate.model()
    calibrate.check(calibrate.sha(calibrate.PARENT.read_bytes())==receipts[0]["candidate_sha256"], "deployment differs")
    v2 = packed(FEEDBACK / "package/candidate.py", receipts[1])
    calibrate.check(list(v2._FEEDBACK_WEIGHTS)==read(FEEDBACK/"weights.json")["coefficients"], "feedback weights")
    net = v1._net()
    # Lock the observational cohort to this public pagination snapshot; later matches cannot enter silently.
    coverage = read(HERE / "scorecard_coverage.json")
    ids = set(coverage["versions"]["ds-pro-v2"]["exposed_game_keys"])
    cards = read(LADDER / "ds-pro-v2-placement.json")["games"]
    new = [c for c in cards if f"{c['match_id']}-{c['game']}" in ids]
    calibrate.check(len(new)==len(ids)==3, "fixed three-game cohort")
    source = read(FEEDBACK / "weights.json")
    old = set(source["training_terrains"])
    for p in (FEEDBACK/"held").glob("*.json.gz"):
        old.add(json.loads(gzip.decompress(p.read_bytes()))["terrain"])
    games = [evaluate(c,v1,net,v2) for c in new]
    terrains = {g["opening_sha256"] for g in games}
    calibrate.check(len(terrains)==3 and not terrains.intersection(old), "training/holdout overlap")
    out = dict(cohort=calibrate.sha((HERE/"scorecard_coverage.json").read_bytes()),
               deployed_model_sha256=[r["candidate_sha256"] for r in receipts], games=games,
               unit_of_evaluation="paired predictions on one opponent collector per public pre-action observation",
               random_draws="same deterministic diagnostic draw for each policy on each turn; live private RNG unobserved")
    raw = json.dumps(out,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    (HERE/"paired_predictions.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    print([(g["ticket"]["opponent"],g["groups"]) for g in games])


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
