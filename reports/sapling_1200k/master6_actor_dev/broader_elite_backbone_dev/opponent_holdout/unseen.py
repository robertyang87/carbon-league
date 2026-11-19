"""Whole-map replay check against opponent registration names unseen in the named training lineage."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from statistics import mean

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
ROOT=ACTOR.parents[2]
sys.path.insert(0,str(ACTOR/"blended_actor_dev/serving"))
import probe


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def write(path,value):
    data=json.dumps(value,allow_nan=False,sort_keys=True,separators=(",",":")).encode()
    if path.suffix==".gz":
        data=gzip.compress(data,mtime=0)
    if path.exists() and data!=path.read_bytes():
        raise ValueError(f"evidence mismatch {path}")
    path.write_bytes(data)


def block_identity(value,ids,seeds,terrain):
    if isinstance(value,list):
        for item in value:
            block_identity(item,ids,seeds,terrain)
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ("id","gid","match_id") and isinstance(v,str):
                ids.add(v)
            if k in ("seed","random_seed") and isinstance(v,int):
                seeds.add(v)
            if k in ("seeds","train_seeds","validation_seeds") and isinstance(v,list):
                seeds.update(x for x in v if isinstance(x,int))
            if k in ("terrain","terrain_sha256","initial_terrain_u8_sha256") and isinstance(v,str):
                terrain.add(v)
            block_identity(v,ids,seeds,terrain)


def freeze():
    broad=load(HERE.parent/"panel.json")
    top=ACTOR.parent/"all_top5"
    elite={a["name"] for a in load(top/"manifest.json")["top5"]}
    names=set(broad["prior_train_registrations"]) | {r["opponent"] for r in broad["games"]} | elite
    ids,seeds,terrain=set(),set(),set()
    block_identity(broad["games"],ids,seeds,terrain)
    for filename,expected in broad["exclusions"].items():
        path=ROOT/filename
        assert digest(path)==expected
        block_identity(load(path),ids,seeds,terrain)
    cards={}
    for filename,expected in broad["source_refs"].items():
        path=ROOT/filename
        if path.name!="manifest.json":
            continue
        assert digest(path)==expected
        for g in load(path)["games"]:
            if not(g["clean_terminal"] and g["frame_verified"] and g["cohort"]=="external"):
                continue
            s=next(i for i,n in enumerate(g["seat_names"]) if n in elite)
            opp=g["seat_names"][1-s]
            if opp in names or g["id"] in ids or g["seed"] in seeds:
                continue
            src=probe.CACHE/(g["id"]+".json.gz")
            assert digest(src)==g["sha256"]
            original=json.loads(gzip.decompress(src.read_bytes()))
            loc=hashlib.sha256(bytes(original["steps"][0][0]["observation"]["full_carbon"])).hexdigest()
            if loc not in terrain:
                cards[g["id"]]=dict(id=g["id"],sha256=g["sha256"],terrain_sha256=loc,
                    seed=g["seed"],seat=s,teacher=g["seat_names"][s],opponent=opp,split="registration_holdout")
    rows=sorted(cards.values(),key=lambda g:g["id"])
    assert len(rows)==len({r["seed"] for r in rows})==len({r["terrain_sha256"] for r in rows}) and rows
    write(HERE/"panel.json",dict(games=rows,broad_panel=digest(HERE.parent/"panel.json"),
          parent=load(HERE.parent/"protocol.json")["parent_deployed_sha256"],
          steps=load(HERE.parent/"protocol.json")["eval_steps"],
          design="All available clean maps against unrepresented registration names, including all scores; same-name exclusion does not resolve common implementations. Results descriptive. No checkpoint selection."))
    print(len(rows),len({r["opponent"] for r in rows}))


def measure(candidate):
    plan=load(HERE/"panel.json")
    parent=ACTOR/"actor_partner_dev/candidate.py"
    assert digest(parent)==plan["parent"] and digest(HERE.parent/"panel.json")==plan["broad_panel"]
    arms={"parent":probe.package(parent,"unseen_parent"),"candidate":probe.package(candidate,"unseen_candidate")}
    rows=[]
    for card in plan["games"]:
        episode=probe.run_game(arms,card,plan["steps"])
        count=Counter()
        for root in episode["roots"]:
            count["roots"]+=1
            count["changed"]+=root["arms"]["candidate"]["actual_command"]!=root["arms"]["parent"]["actual_command"]
            for slot in root["slots"].values():
                values=slot["outputs"]
                if any(len(v["legal"])<=1 for v in values.values()):
                    continue
                count["choices"]+=1
                for name,x in values.items():
                    count[name]+=x["actual"]==slot["demo_canonical"]
        rows.append(dict(**card,counts=count))
    assert all(r["counts"]["choices"]>0 for r in rows)
    by_name={}
    for name in sorted({g["opponent"] for g in rows}):
        scores=[(r["counts"]["candidate"]-r["counts"]["parent"])/r["counts"]["choices"] for r in rows if r["opponent"]==name]
        by_name[name]=dict(maps=len(scores),map_mean_agreement_gain=mean(scores))
    total=sum((r["counts"] for r in rows),Counter())
    result=dict(candidate=digest(candidate),panel=digest(HERE/"panel.json"),totals=total,by_opponent=by_name,rows=rows,
                equal_registration_delta=mean(x["map_mean_agreement_gain"] for x in by_name.values()))
    write(HERE/"result.json",result)
    print(total,result["equal_registration_delta"])


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--candidate",type=Path)
    args=cli.parse_args()
    if args.candidate:
        measure(args.candidate)
    else:
        freeze()
