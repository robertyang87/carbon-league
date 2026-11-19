"""Score the already-frozen 30-step head on every eligible full-corpus appearance."""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"state_value"))
from data import view, future_label
from value import CashMargin30, History, HORIZON

# Fixed steps sampled from opening, middle game, and closing. All games still have a row.
ANCHORS = (30, 60, 100, 150, 200, 250)
METHODS = ("head", "trend30", "zero")


def evaluate(path, game, model):
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if game["sha256"] != digest:
        raise ValueError("replay digest mismatch")
    try:
        from pydantic_core import from_json
    except ImportError:
        from_json = json.loads
    replay = from_json(gzip.decompress(blob))
    frames = replay["steps"]
    if replay["rewards"] != game["score_by_seat"]:
        raise ValueError("final scores differ")
    last = len(frames) - 1
    times = [t for t in ANCHORS if t + HORIZON <= last]
    history = [History(), History()]
    rows = []
    for t in range(1 + max(times, default=-1)):
        for seat in (0, 1):
            x = view(frames[t],seat)
            history[seat].update(x)
            if t in times:
                rows.append([t,seat,round(future_label(frames,t,seat),6),
                             round(model.predict(history[seat],x,replay["configuration"]),6),
                             round(history[seat].trend(30),6)])
    return {"id":game["id"], "seed":replay["configuration"]["randomSeed"],
            "sha256":digest, "rows":rows, "steps":last}


def summaries(cases, forecasts, membership):
    groups = defaultdict(lambda: defaultdict(list))
    for g in cases:
        f = forecasts[g["id"]]
        split = membership.get(f["seed"], "other_maps")
        for t,seat,y,head,trend in f["rows"]:
            labels = [split+"/all",split+"/"+g["cohort"]]
            if split=="other_maps":
                labels += ["other_maps/t"+str(t),"other_maps/seat"+str(seat)]
                if g["seat_strategy_ids"][seat] is not None:
                    labels.append("ranked/"+g["seat_names"][seat])
                    ms=g["score_by_seat"][seat]-g["score_by_seat"][1-seat]
                    labels.append("ranked/"+("win" if ms>0 else "loss" if ms<0 else "draw"))
            for label in labels:
                groups[label][f["seed"]].append((y,head,trend,0.0))
    result={}
    for label,by_seed in sorted(groups.items()):
        mses=[]; maes=[]; sign=[]; n=0
        for samples in by_seed.values():
            a=np.array(samples); y=a[:,:1]; pred=a[:,1:]
            mses.append(((pred-y)**2).mean(axis=0)); maes.append(abs(pred-y).mean(axis=0))
            mask=y[:,0]!=0
            if mask.any():sign.append((np.sign(pred[mask])==np.sign(y[mask])).mean(axis=0))
            n+=len(samples)
        result[label]={"maps":len(by_seed),"seat_roots":n,"metrics":{
            name: {"rmse":round(float(np.sqrt(np.mean(mses,axis=0))[i]),3),
                   "mae":round(float(np.mean(maes,axis=0)[i]),3),
                   "direction_accuracy_on_nonzero":round(float(np.mean(sign,axis=0)[i]),4) if sign else None}
            for i,name in enumerate(METHODS)}}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",type=Path,default=HERE/"manifest.json")
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--work",type=Path,required=True)
    args=ap.parse_args()
    manifest=json.loads(args.manifest.read_bytes())
    cases=manifest["games"]
    old=json.loads((HERE.parent/"top5_h2h_data.json").read_bytes())["games"]
    recent=json.loads((HERE.parent/"latest5_inventory.json").read_bytes())["games"]
    membership={g["seed"]:"previous_"+g["split"] for g in old}
    for game in recent:
        if game["seed"] in membership:
            raise ValueError("historical split overlap")
        membership[game["seed"]]="recent_diagnostic"
    checkpoint=HERE.parent/"state_value/cash_margin_30.npz"
    model_digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if model_digest!=json.loads((HERE.parent/"state_value/frozen.json").read_bytes())["checkpoint_sha256"]:
        raise ValueError("unfrozen model")
    model=CashMargin30(checkpoint)
    args.work.mkdir(parents=True,exist_ok=True)
    forecasts={}
    for game in cases:
        if not game.get("frame_verified"):
            continue
        saved=args.work/(game["id"]+".json")
        if saved.exists():
            r=json.loads(saved.read_bytes())
            if r["sha256"]==game["sha256"] and r["model"]==model_digest:
                forecasts[r["id"]]=r
                continue
        r=evaluate(args.cache/game["path"],game,model)
        if r["seed"]!=game["seed"]:
            raise ValueError("map seed differs")
        r["model"]=model_digest
        tmp=saved.with_suffix(".tmp")
        tmp.write_text(json.dumps(r,allow_nan=False,separators=(",",":")))
        tmp.replace(saved)
        forecasts[r["id"]]=r
    done=[g for g in cases if g["id"] in forecasts]
    out={"revision":manifest["revision"],"model_sha256":model_digest,"anchors":ANCHORS,"card_count":len(cases),
         "complete_replays":len(done),"games_without_valid_anchor":sum(not r["rows"] for r in forecasts.values()),
         "aggregates":summaries(done,forecasts,membership),"per_game_columns":["step","seat","actual","head","trend30"],
         "games":[forecasts[g["id"]] for g in done]}
    tmp=HERE/"forecast_full.tmp"
    tmp.write_bytes(gzip.compress(json.dumps(out,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode(),mtime=0))
    tmp.replace(HERE/"forecast_full.json.gz")
    print("scored games",len(done),"of",len(cases),flush=True)


if __name__=="__main__":
    main()
