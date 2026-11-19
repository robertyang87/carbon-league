"""Freeze a seat- and player-balanced public game panel disjoint from all declared ancestor data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
import sys

HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent;BASE=ACTOR.parent;REPO=ACTOR.parents[2]
sys.path.insert(0, str(ACTOR));sys.path.insert(0, str(BASE))
import data_build
from map_identity import local_board

SHARDS=Path("/workspace/sapling-broader-elite-v1-shards")
START=REPO/"data/weights/sapling_1200k/master6_actor_v1.npz"
PARENT=ACTOR/"actor_partner_dev/candidate.py"
SEED=7924059
EXCLUDE=(BASE/"elite_bc_data.json",BASE/"top5_h2h_data.json",BASE/"cash_dagger_data.json",
         BASE/"tree_dagger_data.json",BASE/"soft_it50_data_manifest.json",ACTOR/"data_split.json",ACTOR/"data_loss_split.json",
         ACTOR/"blended_actor_dev/panel.json",ACTOR/"elite_role_mix_dev/protocol.json",ACTOR/"elite_role_mix_dev/conditional/protocol.json",
         ACTOR/"elite_role_mix_dev/recruit_only_dev/protocol.json")
def sha(v): return hashlib.sha256(v).hexdigest()
def load(path):return json.loads(path.read_bytes())
def save(path,value):
    with path.open("x") as f:json.dump(value,f,ensure_ascii=False,indent=1,allow_nan=False);f.write("\n")

def freeze():
    top=BASE/"all_top5"; order=[top,*sorted(p for p in top.glob("appendix_*") if p.is_dir())];sources={};games={};terrain_by_id={}
    for folder in order:
        p=folder/"manifest.json"; v=folder/"validation_maps.jsonl.gz"
        if not p.exists() or not v.exists():raise ValueError("manifest without maps")
        for path in (p,v):sources[str(path.relative_to(REPO))]=sha(path.read_bytes())
        for row in map(json.loads,gzip.decompress(v.read_bytes()).splitlines()):
            if row["id"] in terrain_by_id and terrain_by_id[row["id"]]!=row["initial_terrain_u8_sha256"]:raise ValueError("map hash conflict")
            terrain_by_id[row["id"]]=row["initial_terrain_u8_sha256"]
        doc=load(p)
        for g in doc["games"]:
            if g["id"] in games and (g["seed"],g["sha256"])!=(games[g["id"]]["seed"],games[g["id"]]["sha256"]):raise ValueError("duplicate card conflict")
            games[g["id"]]=g
    elite=[r["name"] for r in load(top/"manifest.json")["top5"]]
    blocked_ids=set();blocked_seeds=set();blocked_terrain=set();prior_names=set();exclusion_receipts={}
    def block(seed,terrain):
        blocked_seeds.add(seed);blocked_terrain.add(terrain if terrain else local_board(seed))
    for p in EXCLUDE:
        d=load(p); exclusion_receipts[str(p.relative_to(REPO))]=sha(p.read_bytes())
        for g in d.get("games",d.get("external_games",[])):
            blocked_ids.add(g["id"]);block(g["seed"],g.get("terrain_sha256",terrain_by_id.get(g["id"])))
            if g.get("split")=="train":
                if g.get("opponent"):prior_names.add(g["opponent"])
                if g["id"] in games:prior_names.update(games[g["id"]]["seat_names"])
    for p in (BASE/"training.json",ACTOR/"data_prior_local_seeds.json"):
        d=load(p);exclusion_receipts[str(p.relative_to(REPO))]=sha(p.read_bytes())
        for s in d.get("seeds",d.get("train_seeds",[])+d.get("validation_seeds",[])):block(s,None)
    other_panels=(ACTOR/"elite_role_mix_dev/conditional/fresh_paired_dev/protocol.json",ACTOR/"elite_role_mix_dev/conditional/match_protocol.json")
    for p in other_panels:
        d=load(p);exclusion_receipts[str(p.relative_to(REPO))]=sha(p.read_bytes())
        for s in d.get("seeds",[]):block(s,None)
        for g in d.get("exposures",[]):block(g["seed"],None)
    if any(v is None for v in blocked_terrain):raise ValueError("absent terrain in excluded ancestry")
    blocked_counts=dict(ids=len(blocked_ids),seeds=len(blocked_seeds),terrains=len(blocked_terrain))
    pool=[]
    for k,g in games.items():
        if (not g["clean_terminal"] or not g["frame_verified"] or g["self_play"] or g["cohort"]!="external" or
            g["seed"] in blocked_seeds or k in blocked_ids or terrain_by_id[k] in blocked_terrain or
            20291101<=g["seed"]<=20292100 or not (data_build.CACHE/(k+".json.gz")).is_file()): continue
        seat=next((i for i,n in enumerate(g["seat_names"]) if n in elite),None)
        if seat is None:continue
        margin=g["score_by_seat"][seat]-g["score_by_seat"][1-seat]
        pool.append(dict(id=k, seed=g["seed"], seat=seat, terrain_sha256=terrain_by_id[k], sha256=g["sha256"],teacher= g["seat_names"][seat],
                         opponent=g["seat_names"][1-seat], opponent_new_to_ancestor=g["seat_names"][1-seat] not in prior_names,turns=g["transitions"],
                         margin=margin,wdl="W" if margin>0 else "L" if margin<0 else "D"))
    print("eligible model/seat", Counter((x["teacher"],x["seat"]) for x in pool),flush=True)
    selected=[]
    for model in elite:
        for seat in (0,1):
            remaining=[x for x in pool if x["teacher"]==model and x["seat"]==seat]
            counts=Counter()
            for split,size in (("holdout",2),("val",2),("train",22 if model==elite[1] else 26 if model==elite[0] else 24)):
                for _ in range(size):
                    c=min(remaining,key=lambda x:(counts[x["opponent"]],sha(f"{SEED}|{x['id']}".encode())),default=None)
                    if c is None:raise ValueError(f"insufficient games for {model} seat {seat}, {split}")
                    selected.append(dict(c,split=split,gid=len(selected)))
                    counts[c["opponent"]]+=1
                    remaining=[x for x in remaining if x["terrain_sha256"]!=c["terrain_sha256"] and x["seed"]!=c["seed"]]
    if any(len({g[k] for g in selected})!=len(selected) for k in ("seed","terrain_sha256","id")):raise ValueError("selected splits overlap")
    save(HERE/"panel.json",dict(source_refs=sources,exclusions=exclusion_receipts,blocked=blocked_counts,
        prior_train_registrations=sorted(prior_names),eligible_cards=len(pool),games=selected))
    save(HERE/"protocol.json",dict(panel_sha256=sha((HERE/"panel.json").read_bytes()), parent_npz_sha256=sha(START.read_bytes()),
         parent_deployed_sha256=sha(PARENT.read_bytes()), freeze_sha256=sha(Path(__file__).read_bytes()), builder_sha256=sha(Path(data_build.__file__).read_bytes()),
         seed=SEED,batch=16,train_epochs=1,lr=.00009,grad_clip=1.,optimizer="AdamW, weight_decay .001",
         sample="Each game contributes up to 8 roots in each of 4 consecutive 75-step bands, without replacement. One full predeclared pass, narrow residuals not used.",
         loss="Map-phase balanced action CE (0.75 normalized target-role groups plus 0.25 all actionable slots) on student plus 0.5 deployed CE on the frozen .75 expert + .25 student blend; .16 KL(frozen parent student || student). Action inputs only, illegal/forced slots omitted. Value and aux heads frozen.",
         eval_steps=[0,20,50,80,110,140,170,200,230,260,285,297],gate="One final checkpoint. Report per-role/early/mid metrics and actual deployed command agreement versus original on every held map; true strength requires independent responding-opponent games."))
    print("maps",dict(Counter(g["split"] for g in selected)),"eligible",len(pool),"blocked",blocked_counts,flush=True)


def build():
    d=load(HERE/"panel.json");proto=load(HERE/"protocol.json")
    if sha((HERE/"panel.json").read_bytes())!=proto["panel_sha256"] or sha(Path(data_build.__file__).read_bytes())!=proto["builder_sha256"]:raise ValueError("frozen inputs changed")
    for p,h in d["source_refs"].items():
        if sha((REPO/p).read_bytes())!=h:raise ValueError("corpus changed at "+p)
    SHARDS.mkdir(exist_ok=True); receipts=[]
    for i,g in enumerate(d["games"]):
        r=data_build.prep(g,SHARDS)
        receipts.append(dict(gid=g["gid"],id=g["id"],rows=r["rows"],shard_sha256=r["sha256"],rejected=r["rejected"],phase_roots=r["phase_roots"]))
        if (i+1)%20==0:print("built",i+1,"/",len(d["games"]),flush=True)
    save(HERE/"shards.json",dict(games=receipts,builder=proto["builder_sha256"],panel=proto["panel_sha256"]))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","build"]);a=p.parse_args()
    freeze() if a.stage=="freeze" else build()
