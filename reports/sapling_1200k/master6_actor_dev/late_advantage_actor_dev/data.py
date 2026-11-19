from pathlib import Path
import gzip,hashlib,json,sys,os
import numpy as np
from pydantic_core import from_json
HERE=Path(__file__).resolve().parent; ACTOR=HERE.parent; BASE=ACTOR.parent; ROOT=BASE.parents[1]; TOP=BASE/"all_top5"
sys.path.insert(0,str(ACTOR));sys.path.insert(0,str(TOP))
import data_build as build
import cnet2_feat as F
from defense import DIST
from lots import measure, FATES
CACHE=Path("/tmp/sapling-late-adv-v1"); OLD=Path("/workspace/sapling-master6-actor-data")
def sha(x):return hashlib.sha256(x).hexdigest()
def dump(p,x):
    raw=(json.dumps(x,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix==".gz" else raw)
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())
def panel():
    all_old=read(ACTOR/"data_split.json")["games"]; known=set(read(ACTOR/"economy_aux_dev/result.json")["sources"]["train"]["replays"])
    old=[r for r in all_old if r["id"] in known];assert len(old)==216
    allowed=[r for r in old if int(sha(("late-rival-1|"+r["opponent"]).encode()),16)%4]
    train=sorted(allowed,key=lambda r:(r["wdl"]!="L",sha(("late-actor-1|"+r["id"]).encode())))[:48]
    test=[]
    for tag in ("97de1ebc","2cc7c358","b5e6a7f8","aca9b8a3"):
        p=TOP/("appendix_"+tag); m=read(p/"manifest.json")
        terrains={r["id"]:r["initial_terrain_u8_sha256"] for r in map(json.loads,gzip.decompress((p/"validation_maps.jsonl.gz").read_bytes()).splitlines())}
        for r in m["games"]:
            if "claude-AlphaCarbon-Master6" not in r["seat_names"]:continue
            s=r["seat_names"].index("claude-AlphaCarbon-Master6")
            mrg=r["score_by_seat"][s]-r["score_by_seat"][1-s]
            test.append({"gid":10000+len(test),"id":r["id"],"opponent":r["seat_names"][1-s],"seat":s,
                         "turns":r["transitions"],"seed":r["seed"],"sha256":r["sha256"],"terrain_sha256":terrains[r["id"]],
                         "wdl":"W" if mrg>0 else "L" if mrg<0 else "D","margin":mrg,"split":"held"})
    for k in ("terrain_sha256","id","seed"):
        if {g[k] for g in old}&{g[k] for g in test} or len({g[k] for g in test})!=len(test):raise ValueError("overlap")
    return train+test

STATE=("steps_left","cargo_log","distance_home","current_visible_carbon","enemy_nearest","enemy_lighter_nearest",
       "own_collectors","enemy_collectors","tree_cell","hazard_trees","recent_home_within","cash_log","cargo_slack")
def ownmap(pos,side):return int(F.CANON[pos]) if side else int(pos)
def row_features(obs,s,uid,cfg):
    pl=obs["players"];me,foe=pl[s],pl[1-s]
    pos,c,_=me[2][uid];ownc=[w for w in me[2].values() if w[2]=="COLLECTOR"]; enemy=[w for w in foe[2].values() if w[2]=="COLLECTOR"]
    remain=cfg["episodeSteps"]-1-obs["step"];dist=min(DIST[pos][b] for b in me[1].values());trees={q for p in pl for q,_ in p[3].values()}
    y,x=divmod(pos,15)
    near=lambda p: max(min(abs(y-p//15),15-abs(y-p//15)),min(abs(x-p%15),15-abs(x-p%15)))<=1
    return [remain/74,np.log1p(c),dist,obs["carbon"][pos]/100,min([DIST[pos][w[0]] for w in enemy] or [15]),
            min([DIST[pos][w[0]] for w in enemy if w[1]<=c] or [15]),len(ownc),len(enemy),float(pos in trees),
            sum(near(z) for z,a in foe[3].values()),float(dist<=2),np.log1p(me[0]),min(20,remain-dist)]
def prepare(g):
    path=(OLD if g["split"]=="train" else CACHE)/(g["id"]+".npz")
    if g["split"]!="train":build.prep(g,CACHE)
    info=read(path.with_suffix(".json"))
    if sha(path.read_bytes())!=info["sha256"] or info["source_sha256"]!=g["sha256"]:raise ValueError("unverified shard")
    packed=(build.CACHE/(g["id"]+".json.gz")).read_bytes()
    if sha(packed)!=g["sha256"]:raise ValueError("replay identity")
    doc=from_json(gzip.decompress(packed)); fs=doc["steps"];s=g["seat"]
    if sha(bytes(fs[0][0]["observation"]["full_carbon"]))!=g["terrain_sha256"]:raise ValueError("wrong board")
    lots,audit=measure(doc,s); samples=[]; blocked_until=225
    with np.load(path,allow_pickle=False) as d:
        for row,t in enumerate(d["step"].astype(int)):
            if t<blocked_until:continue
            end=min(t+4,g["turns"])
            blocked_until=end
            shared=fs[t][0]["observation"];obs={k:shared[k] for k in ("players","trees","step")};obs.update(carbon=fs[t][s]["observation"]["carbon"],player=s)
            units=obs["players"][s][2];uid_at={ownmap(w[0],s):u for u,w in units.items()}
            for slot in np.flatnonzero((d["tok_kind"][row,:10]==0)&(d["y"][row,:10]>=0)&(d["legal"][row,:10].sum(1)>1)):
                uid=uid_at[int(d["tok_pos"][row,slot])]
                if units[uid][2]!="COLLECTOR":raise ValueError("slot ownership")
                acts=fs[t+1][s]["action"]
                a=F.MOVES.index(acts.get(uid) or "NONE");a=4 if s and a==2 else 2 if s and a==4 else a
                if a!=d["y"][row,slot]:raise ValueError("action id mismatch")
                item=[lots[h,uid] for h in range(t,end) if (h,uid) in lots]
                if not item:raise ValueError("collector absent")
                fates=np.sum([v["new_fates"] for v in item],axis=0)
                samples.append({"t":int(t),"end":int(end),"slot":int(slot),"uid":uid,"recorded_action":int(a),
                                "observed":row_features(obs,s,uid,doc["configuration"]),"old_cargo":units[uid][1],
                                "old_cargo_direct":item[0]["existing_direct"],"new_gross":sum(x["gross"] for x in item),
                                "new_settlement":fates.tolist()})
    if sum(r["new_settlement"][0] for r in samples)>audit["new_fates"][0]+1e-4:raise ValueError("duplicate selected lot")
    return {"card":g,"shard_sha256":info["sha256"],"all_late":audit,"sample":samples}
def main():
    CACHE.mkdir(exist_ok=True)
    cards=panel();dump(HERE/"panel.json",cards); payload=[]
    for i,g in enumerate(cards):
        p=CACHE/(g["id"]+".late.json")
        fingerprint=sha(Path(__file__).read_bytes()+(HERE/"lots.py").read_bytes())
        stored=read(p) if p.exists() else {}
        if stored.get("build")!=fingerprint or stored.get("card")!=g:
            stored=prepare(g);stored["build"]=fingerprint;dump(p,stored)
        payload.append(stored)
        if (i+1)%10==0:print(i+1,len(cards),flush=True)
    dump(HERE/"settlements.json.gz",{"state_names":STATE,"fate_names":FATES,"games":payload})
if __name__=="__main__":main()
