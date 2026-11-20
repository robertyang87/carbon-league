"""Evaluate one frozen projection on official archived matches against the unmodified opponent."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import fmean
import zipfile

from data import TIMES, digest
from runtime import features, predict_residual

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
WEIGHTS = "ceed9c721d2ff70bcd2a47bb47dacb5ca88b51073e8a7fb9beefe3598e679cd6"
DOMAINS = {
    "historical": ["dev1", "cold_start/development", "ladder/prediction_feedback/development"],
    "prospective": ["ladder/cargo_defense/development"],
}
METHODS = ("cash", "aggregate", "enhanced")


def evaluate(domain):
    raw=(HERE/"weights.json").read_bytes()
    if digest(raw)!=WEIGHTS:
        raise ValueError("Different model from the pre-result freeze")
    model=json.loads(raw)
    studied={t["terrain"] for t in json.loads((HERE/"maps.json").read_bytes())["tickets"]}
    rows, excluded = [], []
    for lane in DOMAINS[domain]:
        artifacts=sorted((FAMILY/lane/"hosted").glob("*.zip"))
        if not artifacts:
            raise ValueError("Missing receipts for "+lane)
        for artifact in artifacts:
            with zipfile.ZipFile(artifact) as z:
                for name in sorted(n for n in z.namelist() if n.endswith("/receipt.json")):
                    arm=name.rsplit("/",1)[0]
                    rec=json.loads(z.read(name))
                    reg=json.loads(z.read(arm+"/registration.json"))
                    terrain=reg["physical_map_sha256"]
                    if reg["files"]["opponent"]["sha256"]!=ORIGINAL or terrain in studied:
                        raise ValueError("Wrong original or duplicate study map")
                    identity=dict(artifact=artifact.name,arm=arm,seed=reg["seed"],seat=reg["candidate_seat"],terrain=terrain)
                    if rec["status"]!="complete" or rec.get("official_scores_by_seat") is None:
                        excluded.append({**identity,"reason":rec["status"]})
                        continue
                    values=rec["official_scores_by_seat"]
                    end=values[0]-values[1]
                    games=[]
                    with gzip.GzipFile(fileobj=z.open(arm+"/observations.jsonl.gz")) as stream:
                        for line in stream:
                            frame=json.loads(line)
                            step=frame["step"]
                            if step not in TIMES: continue
                            p=frame["observations"][0]["players"]
                            if p!=frame["observations"][1]["players"]:
                                raise ValueError("Rosters not public in both seats")
                            cash=p[0][0]-p[1][0]
                            row=features(p,0,299-step)
                            enhanced=cash+predict_residual(row,model["enhanced"])
                            aggregate=cash+predict_residual(row,model["aggregate"])
                            games.append(dict(step=step,actual=end,cash=cash,aggregate=aggregate,enhanced=enhanced))
                    if games:
                        rows.append({**identity,"roots":games})
                    else:
                        excluded.append({**identity,"reason":"no live pre-action study roots"})
    by_map=defaultdict(list)
    for r in rows:
        by_map[r["terrain"]].append({k:fmean(abs(v[k]-v["actual"]) for v in r["roots"]) for k in METHODS})
    means={key:{k:fmean(v[k] for v in games) for k in METHODS} for key,games in by_map.items()}
    if domain=="prospective" and len(rows)+len(excluded)!=8:
        raise ValueError("Incomplete eight-game development archive")
    return dict(domain=domain, original_sha256=ORIGINAL, weights_sha256=digest(raw),
                completed_with_roots=len(rows), map_count=len(means), root_count=sum(len(r["roots"]) for r in rows),
                mae_map_equal={k:fmean(v[k] for v in means.values()) for k in METHODS},
                by_map=means, games=rows, excluded=excluded)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain",choices=DOMAINS)
    args=parser.parse_args()
    out=evaluate(args.domain)
    (HERE/(args.domain+".json")).write_text(json.dumps(out,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    print(out["completed_with_roots"],out["map_count"],out["root_count"],out["mae_map_equal"])
