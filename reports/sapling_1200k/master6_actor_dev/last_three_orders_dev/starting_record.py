"""Report the entire fixed-parent tape panel registered for the terminal analysis."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import fmean

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def load(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes())


def record():
    panel=load(HERE/"protocol.json")
    rows=[]
    seeds=defaultdict(set)
    for game in panel["games"]:
        path=ROOT/game["source"]
        data=load(path)
        receipt=load(path.with_name(path.name.replace("raw_","receipt_").replace(".json.gz",".json")))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==game["sha256"]==receipt["game"]["raw_sha256"]
        assert data["pack_sha256"]==panel["parent_sha256"] and not data["errors"]
        s=game["seat"]
        assert s==data["candidate_seat"] and game["seed"]==data["seed"]
        points=data["official_score_by_seat"]
        margin=round(points[s]-points[1-s],2)
        assert abs(margin-data["margin"])<1e-6
        seeds[game["seed"]].add((s,data["initial_terrain_u8_sha256"]))
        rows.append(dict(**game,points=points,margin=margin,steps=data["steps"],result=data["result"]))
    assert len(rows)==2*len(seeds) and all({s for s,h in v}=={0,1} and len({h for s,h in v})==1 for v in seeds.values())
    by={}
    for key,items in (("seat0",[r for r in rows if not r["seat"]]),("seat1",[r for r in rows if r["seat"]]),("all",rows)):
        by[key]=dict(games=len(items),wins=sum(r["margin"]>0 for r in items),draws=sum(r["margin"]==0 for r in items),mean_margin=fmean(r["margin"] for r in items))
    out=dict(parent=panel["parent_sha256"],registered_maps=len(seeds),scores=by,rows=rows,
             caveat="Fixed historical development tapes only. Includes exposed maps, not the independent formal exam.")
    (HERE/"starting_record.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(by)


if __name__=="__main__":
    record()
