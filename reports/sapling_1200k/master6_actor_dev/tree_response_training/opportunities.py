"""Five physical options per living planter before each decision, replaying the true submitted prefix."""
import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(ROOT / "code"))
import cnet2_feat as F
from root_recovery_branches.branches import begin, public
from planter_routes_dev.features import PlanterRouteState, route_value, _ROUTE_DEST, _ROUTE_DIST

MOVES = ("NONE","UP","RIGHT","DOWN","LEFT")
BANK = ("RECCOLLECTOR","RECPLANTER")
MAXDIST = 15   # 15x15 toroidal Manhattan distance is at most 14
PARENT_SHA = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
FEATURE_SHA = "7a3f1aca5d967e79d90eb6f7414a73c7a8623ed7c792eed3c0fcd0a78214ff80"
HISTORY = (3,5,10)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(command):
    return command or "NONE"


def neighbors(cell):
    return [int(c) for c in _ROUTE_DEST[cell,1:]]


def nearest(cell, positions):
    return min((int(_ROUTE_DIST[cell,p]) for p in positions),default=MAXDIST)


def state(obs, cfg):
    """Use only fields supplied to the acting seat. Hidden carbon is never read."""
    if int(cfg["size"])!=15:
        raise ValueError("this actor only supports its frozen 15x15 geometry")
    seat,t=obs["player"],obs["step"]
    own, enemy = obs["players"][seat],obs["players"][1-seat]
    bases={p:(s,bid) for s,pl in enumerate(obs["players"]) for bid,p in pl[1].items()}
    trees={p:(s,tid,age) for s,pl in enumerate(obs["players"]) for tid,(p,age) in pl[3].items()}
    visible=set()
    for pos,_,typ in own[2].values():
        if typ=="COLLECTOR":visible.add(pos)
        elif typ=="PLANTER":visible.update(neighbors(pos))
        else:raise ValueError(f"unrecognized worker: {typ}")
    opened={p:float(obs["carbon"][p]) for p in sorted(visible)}
    life=cfg["treeLifespan"]
    price=float(cfg["plantCost"])
    if len(own[3])>cfg["playerTreeProtectiveNumber"]:
        price+=cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**sum(age<life for _,_,age in trees.values())
    footprint={q for p,(s,_,age) in trees.items() if s==seat and life-age>=10 for q in neighbors(p) if q not in trees and q not in bases}
    collectors={u:(p,c) for u,(p,c,typ) in enemy[2].items() if typ=="COLLECTOR"}
    enemy_bases=set(enemy[1].values())
    routes={u:{"cell":p,"cargo":float(c),"home_distance":nearest(p,enemy_bases)} for u,(p,c) in collectors.items()}
    exposed=sum(r["cargo"] for r in routes.values() if r["home_distance"]<=4)
    return {"seat":seat,"t":t,"cash":float(own[0]),"opponent_cash":float(enemy[0]),"bases":bases,"trees":trees,"visible":opened,
            "footprint":sorted(footprint),"enemy_return_routes":routes,"heavy_return":float(exposed),
            "own":own,"enemy":enemy,"new_price":price,"life":life,"cfg":cfg,
            "ec":list(p for p,c in collectors.values()),"ep":[p for p,_,typ in enemy[2].values() if typ=="PLANTER"]}


def quote(st,cell):
    """Standing still there *now*. Age>=lifespan disappears before the payment phase."""
    tree=st["trees"].get(cell)
    if cell in st["bases"]:return "base",None
    if tree and tree[2]<st["life"]:
        if tree[0]==st["seat"]:return "own_tree",None
        return "seize",float(st["cfg"]["seizeCost"])
    return "plant",st["new_price"]


def commitments(st,commands):
    own, enemy, cfg = st["own"],st["enemy"],st["cfg"]
    counts=Counter(w[2] for pl in (own,enemy) for w in pl[2].values())
    seats_left=max(0,cfg["workerLimit"]-len(own[2]))
    liquid=st["cash"]
    hires={}
    for uid,cell in own[1].items():
        action=norm(commands.get(uid))
        if action not in BANK:continue
        typ="Collector" if action==BANK[0] else "Planter"
        cost=cfg["rec"+typ+"Cost"]+counts[typ.upper()]*cfg["rec"+typ+"IncreaseCost"]
        if seats_left and liquid>=cost:
            hires[uid]=(cell,float(cost))
            liquid-=cost
            seats_left-=1
    dest={}
    requests={}
    for uid,(pos,_,typ) in own[2].items():
        command=norm(commands.get(uid))
        if command not in MOVES:raise ValueError(f"unsupported worker action {command!r}")
        dest[uid]=int(_ROUTE_DEST[pos,MOVES.index(command)])
        if typ=="PLANTER" and command=="NONE":
            _, cost=quote(st,pos)
            if cost is not None: requests[uid]=cost
    return hires,dest,requests


def snapshot(st,uid):
    worker=st["own"][2].get(uid)
    return {"t":st["t"],"worker_present":worker is not None,"position":None if worker is None else worker[0],
            "own_cash":st["cash"],"opponent_cash":st["opponent_cash"],"own_workers":len(st["own"][2]),
            "opponent_workers":len(st["enemy"][2]),"own_trees":len(st["own"][3]),"opponent_trees":len(st["enemy"][3]),
            "productive_area":len(st["footprint"]),"heavy_return":st["heavy_return"],
            "root_visible_enemy_collector_routes":st["enemy_return_routes"],
            "nearest_enemy_planter":None if worker is None else nearest(worker[0],st["ep"])}


def options(st,commands,uid,prefix,history):
    own=st["own"]
    root=own[2][uid][0]
    registered=norm(commands.get(uid))
    hires,dest,requests=commitments(st,commands)
    reserved_hires=sum(cost for _,cost in hires.values())
    reserved_workers=sum(cost for w,cost in requests.items() if w!=uid)
    available=max(0.,st["cash"]-reserved_hires-reserved_workers)
    all_actions={}
    for ix,cmd in enumerate(MOVES):
        cell=int(_ROUTE_DEST[root,ix])
        kind,price=quote(st,cell)
        conflicts=sorted([w for w,c in dest.items() if w!=uid and c==cell]+[w for w,(c,_) in hires.items() if c==cell])
        coll,plan=nearest(cell,st["ec"]),nearest(cell,st["ep"])
        nb=neighbors(cell)
        grow=[c for c in nb if c not in st["trees"] and c not in st["bases"]]
        visible_carbon=[st["visible"][c] for c in grow if c in st["visible"]]
        tree=st["trees"].get(cell)
        known=sum(visible_carbon)
        all_actions[cmd]={
            "cell":cell,"parent_selected":cmd==registered,"entry_legal":prefix["legal"][ix],"friendly_conflict":bool(conflicts),
            "friendly_competitors":conflicts,"collision_free_given_own_orders":not conflicts,
            "prefix_predecessors_reserved_here":prefix["reserved"][ix],
            "enemy_collector_distance":coll,"enemy_planter_distance":plan,"possible_enemy_contact_next":min(coll,plan)<=1,
            "current_site_type":kind,"payable_site":price is not None,"price":price,"available_cash":available,
            "requested_other_planter_reserve":reserved_workers,"approved_own_hire_reserve":reserved_hires,
            "site_affordable_on_present_budget":price is not None and available>=price,
            "can_pay_now":cmd=="NONE" and price is not None and available>=price and not conflicts,
            "current_turn_intended_cost":price if cmd=="NONE" and price is not None else 0.,
            "live_tree_now":None if tree is None else {"owner":tree[0],"id":tree[1],"age":tree[2]},
            "adjacent_carbon":[{"cell":c,"carbon":st["visible"].get(c),"resource_square":c in grow} for c in nb],
            "observable_resource_sum":known,"known_resource_cells":len(visible_carbon),"possible_resource_cells":len(grow),
            "known_fraction":sum(c in st["visible"] for c in nb)/4,
            "remaining_transitions":st["cfg"]["episodeSteps"]-1-st["t"],
            "friendly_productive_edge_cells":sum(c in st["footprint"] for c in nb),
            "geometric_move_sort":[int(bool(conflicts)),int(min(coll,plan)<=1),int(plan<=2),-len(grow),-round(known,6),ix]}
    hist=[]
    if any(h["seat"]!=st["seat"] or h["t"]>=st["t"] for h in history):
        raise ValueError("lookbacks require earlier same-seat public observations")
    past={s["t"]:s for s in history}
    if len(past)!=len(history):
        raise ValueError("ambiguous duplicate historical turn")
    for lag in HISTORY:
        wanted=st["t"]-lag
        h=past.get(wanted)
        old=None
        if h is not None:
            old=snapshot(h,uid)
            old["own_tree_entities"]=h["own"][3]
            old["opponent_tree_entities"]=h["enemy"][3]
            old["own_worker_entities"]=h["own"][2]
            old["opponent_worker_entities"]=h["enemy"][2]
            old["seat_visible_carbon"]=[[p,c] for p,c in h["visible"].items()]
        hist.append({"lag":lag,"t":wanted,"observed":h is not None,"predecision":old})
    ep=nearest(root,st["ep"])
    return {"t":st["t"],"seat":st["seat"],"uid":uid,"position":root,"parent_command":registered,
            "heavy_return":st["heavy_return"],"productive_area":len(st["footprint"]),"enemy_planter_distance":ep,
            "stratum_bits":[int(st["heavy_return"]>=160),int(len(st["footprint"])>=12),int(ep<=2)],
            "current":snapshot(st,uid),"productive_cells":st["footprint"],"enemy_planter_positions":st["ep"],
            "tree_entities":[[p,*info] for p,info in sorted(st["trees"].items())],
            "base_entities":[[p,*info] for p,info in sorted(st["bases"].items())],
            "worker_entities":[[s,w,*v] for s,pl in [(st["seat"],own),(1-st["seat"],st["enemy"])] for w,v in pl[2].items()],
            "acting_prefix":prefix["prior"],"slot":prefix["slot"],"actions":all_actions,"history":hist}


def iter_game(path):
    """Yield every live own planter turn from a full v1 tape. No opponent order enters an action row."""
    path=Path(path)
    packed=path.read_bytes()
    raw=json.loads(gzip.decompress(packed))
    if raw["pack_sha256"] != PARENT_SHA or digest(Path(F.__file__).read_bytes()) != FEATURE_SHA:
        raise ValueError("unregistered Sapling actor or prefix source")
    seed,seat=raw["seed"],raw["candidate_seat"]
    cfg=raw["game_configuration"]
    if raw.get("errors") or raw.get("steps")!=len(raw["actions"]):
        raise ValueError("incomplete or faulted source action tape")
    env=begin(raw)
    feat=F.Feat(seat)
    hist=deque(maxlen=max(HISTORY))
    route=PlanterRouteState()
    model_path=HERE.parent/"planter_option/value_weights.json"
    predictor=json.loads(model_path.read_bytes())
    model_digest=digest(model_path.read_bytes())
    if model_digest!="59726c09984f6d2210d28ee90712cf8f0c37033ff9635bd272bdc0ca6e3edaf7":
        raise ValueError("unregistered route-value head")
    tape_hash=digest(packed)
    for t,joint in enumerate(raw["actions"]):
        if env.done or env.states[0].observation.step!=t:raise ValueError("inconsistent source step")
        obs=public(env,seat)
        if "full_carbon" in obs or "reveal" in obs:raise ValueError("privileged field")
        st=state(obs,cfg)
        route.observe(obs,seat,obs["carbon"],cfg,t)
        x=feat.observe(obs);pref=F.Prefix(x,feat.aux)
        labels=feat.labels(joint[seat]);prior=[]
        if feat.to_actions(labels)!={k:v for k,v in joint[seat].items() if norm(v)!="NONE"}:
            raise ValueError(f"source action outside actor representation at {t}")
        for slot,uid in enumerate(feat.own_uid):
            if uid is None:continue
            _,_,legal=pref.feats(slot)
            actual=int(labels[slot])
            if not legal[actual]:raise ValueError(f"logged action illegal at {t}/{uid}")
            if uid in st["own"][2] and st["own"][2][uid][2]=="PLANTER":
                pos=st["own"][2][uid][0]
                canonical=F.CANON_MOVE if seat else tuple(range(5))
                detail={"slot":slot,"prior":list(prior),"legal":[bool(legal[i]) for i in canonical],
                        "reserved":[int(pref.reserved[feat.cpos(int(_ROUTE_DEST[pos,i]))]) for i in range(5)]}
                row=options(st,joint[seat],uid,detail,hist)
                for act in row["actions"].values():
                    scored=route.features(act["cell"])
                    if scored is None:
                        act["frozen_route_estimate"]=None
                    else:
                        weighted,prob,net=route_value(predictor,scored[0])
                        act["frozen_route_estimate"]={"estimated_first_tenure_net_if_paid":net,
                            "estimated_purchase_probability":prob,"weighted_net":weighted,
                            "model_root_site_price":scored[1]}
                yield {"seed":seed,"tape_sha256":tape_hash,"route_weights_sha256":model_digest,**row}
            pref.advance(slot,actual)
            prior.append([slot,uid,norm(joint[seat].get(uid))])
        hist.append(st)
        env.step(joint)
    if not env.done or [s.reward for s in env.states]!=raw["official_score_by_seat"]:
        raise ValueError("source did not reproduce the official terminal")


def save(destination):
    plan=json.loads((HERE/"protocol.json").read_bytes())
    expected=[(seed,seat) for seed in plan["train_maps"] for seat in plan["physical_seats"]]
    paths=[HERE/"source"/f"raw_{s}_{p}.json.gz" for s,p in expected]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError("all four registered training games must exist")
    receipt=json.loads((HERE/"source_receipt.json").read_bytes())
    if receipt["protocol_sha256"]!=digest((HERE/"protocol.json").read_bytes()):
        raise ValueError("source games belong to a different training protocol")
    officials={(r["seed"],r["seat"]):r for r in receipt["games"]}
    if len(receipt["games"])!=len(expected) or set(officials)!=set(expected):
        raise ValueError("full registered map/seat set missing")
    counter=Counter()
    with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as temp:
        try:
            with gzip.GzipFile(fileobj=temp,mode="wb",filename="",mtime=0) as stream:
                for (seed,seat),path in zip(expected,paths):
                    if digest(path.read_bytes())!=officials[seed,seat]["raw_sha256"]:
                        raise ValueError("source tape differs from official immutable receipt")
                    for row in iter_game(path):
                        if (row["seed"],row["seat"])!=(seed,seat):raise ValueError("source seat mismatch")
                        counter[(seed,seat)]+=1
                        stream.write((json.dumps(row,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode())
            temp.flush()
            if destination.exists() and digest(destination.read_bytes())!=digest(Path(temp.name).read_bytes()):
                raise ValueError("immutable opportunity evidence differs")
            os.replace(temp.name,destination)
        finally:
            Path(temp.name).unlink(missing_ok=True)
    print("rows by map and seat",[(k,counter[k]) for k in expected],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"opportunities.jsonl.gz")
    save(p.parse_args().out)
