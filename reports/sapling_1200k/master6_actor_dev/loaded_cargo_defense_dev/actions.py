"""One loaded Sapling collector may choose its next-best safe action on the original production prefix."""
import ast
import hashlib
from math import isfinite
import sys

MOVES = ("NONE","UP","RIGHT","DOWN","LEFT")
OFFSETS = ((0,0),(-1,0),(0,1),(1,0),(0,-1))
MIN_CARGO = 80.
PARENT_SHA = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
FEATURES = ("cargo","strict_lighter","equal","second_attacker","alternative_safe_routes",
            "additional_home_distance","remaining_slack","parent_rank","parent_logit_gap","teammate_route_cost")


def pos_after(p, action, n):
    y,x=divmod(p,n)
    dy,dx=OFFSETS[MOVES.index(action)]
    return ((y+dy)%n)*n+(x+dx)%n


def distance(a,b,n):
    ay,ax=divmod(a,n);by,bx=divmod(b,n)
    dy,dx=abs(ay-by),abs(ax-bx)
    return min(dy,n-dy)+min(dx,n-dx)


def public(obs):
    return {k:obs[k] for k in ("step","player","players","trees","carbon")}


def install_parent_trace(actor, frozen_parent_bytes, validate=None):
    """Record the five physical logits from the exact guarded pass returned by the two-network actor.

    Instrument the original decoder at its argmax only. Both passes otherwise execute as written,
    and the return guard still chooses the array. Reference identity selects the record even when
    both passes happened to decode identical joint orders with different scores.
    """
    if hashlib.sha256(frozen_parent_bytes).hexdigest()!=PARENT_SHA:
        raise ValueError("not the registered Sapling combined actor")
    if validate is not None:
        actor._local_edit = validate
    if getattr(actor,"_loaded_trace_installed",False):
        return
    source=frozen_parent_bytes.decode()
    node,=[d for d in ast.parse(source).body if isinstance(d,ast.FunctionDef) and d.name=="_return_decode"]
    decode_source=ast.get_source_segment(source,node)
    anchor="        action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    if decode_source.count(anchor)!=1:
        raise ValueError("production decoder capture point differs")
    code=decode_source.replace(anchor,"        _cargo_record(slot,combined,legal)\n"+anchor)
    if not hasattr(actor,"_local_edit"):
        # Existing physical occupancy and full-prefix validator can be attached by the native
        # runner/packager. The proposal path additionally checks each protected slot directly.
        actor._local_edit=None
    production_mix=actor._mixed_actions
    pending=[]
    current={}
    def record(slot, scores, legal):
        current[int(slot)] = tuple(float(s) if allowed else None for s,allowed in zip(scores,legal))
    actor._cargo_record=record
    exec(compile(code,"<frozen Sapling _return_decode with score capture>","exec"),vars(actor))
    traced_decode=actor._return_decode
    def capture_decode(x,aux,encoded,apply_residual):
        current.clear()
        labels=traced_decode(x,aux,encoded,apply_residual)
        pending.append((labels,dict(current)))
        return labels
    def capture_mix(x,aux):
        pending.clear()
        labels=production_mix(x,aux)
        scores,=[s for y,s in pending if y is labels]
        actor._loaded_trace=(x,aux,labels.copy(),scores)
        pending.clear()
        return labels
    actor._return_decode=capture_decode
    actor._mixed_actions=capture_mix
    actor._loaded_trace_installed=True
    actor._loaded_trace=None


def common_enemy_reach(obs,cfg,n):
    """One common action set for all candidates. Ties and individually optional overlaps are retained."""
    them=obs["players"][1-obs["player"]]
    mult=1-cfg.get("moveCost",0.)
    reach={}
    for u,(p,c,role) in sorted(them[2].items()):
        if role=="COLLECTOR":
            for a in MOVES:
                target=pos_after(p,a,n)
                reach.setdefault(target,[]).append((u,c if a=="NONE" else c*mult,a))
    count=sum(w[2]=="COLLECTOR" for t in obs["players"] for w in t[2].values())
    cost=cfg.get("recCollectorCost",30.)+count*cfg.get("recCollectorIncreaseCost",0.)
    if len(them[2])<cfg.get("workerLimit",10) and them[0]>=cost:
        for u,p in them[1].items():
            reach.setdefault(p,[]).append(("spawn:"+u,0.,"RECCOLLECTOR"))
    return {p:sorted(rows,key=lambda r:(r[1],r[0],r[2])) for p,rows in reach.items()}


def exposure(reach, p, carried):
    attackers=[r for r in reach.get(p,()) if r[1]<=carried]
    smaller=[r[0] for r in attackers if r[1]<carried]
    equal=[r[0] for r in attackers if r[1]==carried]
    # If multiple collectors commit to the same tile, all die when minimum is tied.
    # Every one of those collectors may independently take a different simultaneous action.
    min_tied=len(attackers)>1 and attackers[0][1]==attackers[1][1]
    return {"strict":smaller,"equal":equal,"commitments":attackers,"minimum_tied_if_all_arrive":min_tied}


def _downhill(p,bases,n):
    dist=min(distance(p,b,n) for b in bases)
    neighbors = (pos_after(p, a, n) for a in MOVES[1:])
    return {q for q in neighbors if min(distance(q,b,n) for b in bases)<dist}


def ranked_options(obs,cfg,parent,score_by_uid):
    """Public geometry and recorded logits. Return at most one highest-scoring safe alternative per unit."""
    cfg=cfg or {}
    n=cfg.get("size",15);left=cfg.get("episodeSteps",300)-1-obs["step"]
    if n!=15 or left<=0:
        return []
    own,enemy=obs["players"][obs["player"]],obs["players"][1-obs["player"]]
    homes=set(own[1].values()); hostile=set(enemy[1].values())
    if not homes:
        return []
    mine=own[2];mult=1-cfg.get("moveCost",0.)
    ends={u:pos_after(w[0],parent.get(u,"NONE"),n) for u,w in mine.items()}
    hire={p for u,p in own[1].items() if parent.get(u) in ("RECPLANTER","RECCOLLECTOR")}
    reach=common_enemy_reach(obs,cfg,n)
    results=[]
    for uid,(p,c,role) in sorted(mine.items()):
        original=parent.get(uid,"NONE")
        if role!="COLLECTOR" or c<MIN_CARGO or uid not in score_by_uid:
            continue
        start=ends[uid];amount=c if original=="NONE" else c*mult
        threat=exposure(reach,start,amount)
        if not threat["commitments"]:
            continue
        dist0=min(distance(start,b,n) for b in homes)
        blocked=hire|{d for u,d in ends.items() if u!=uid}
        scores=score_by_uid[uid]
        ordered=sorted((a for a in MOVES if scores.get(a) is not None),key=lambda a:(-scores[a],MOVES.index(a)))
        if not ordered or original not in ordered or scores[original]!=scores[ordered[0]]:
            raise ValueError("scores are not from the actual parent decision")
        legal_order=[original]+[a for a in ordered if a!=original]
        saved=[]
        for rank,action in enumerate(legal_order[1:],2):
            dest=pos_after(p,action,n);amount1=c if action=="NONE" else c*mult
            if dest in blocked|hostile or exposure(reach,dest,amount1)["commitments"]:
                continue
            d=min(distance(dest,b,n) for b in homes)
            if d>left-1 or d>dist0+2:
                continue
            # A free downhill neighbor is a current-board proxy, not a claim about next-turn rivals.
            clear={q for q in _downhill(dest,homes,n) if q not in blocked|hostile and not exposure(reach,q,amount1*mult)["commitments"]}
            if d and not clear:
                continue
            lost_routes=0.
            veto=False
            for friend,(fpos,fc,fr) in mine.items():
                ep=ends[friend]
                if friend==uid or fr!="COLLECTOR" or fc<=0 or ep in homes:
                    continue
                # Compare the same friend's next return menu with each position provisionally occupied.
                # Only count a lost corridor when the newly occupied cell really was available originally.
                other_blocked=hire|{v for u,v in ends.items() if u not in (uid,friend)}
                safe={q for q in _downhill(ep,homes,n) if q not in other_blocked|hostile and
                      not exposure(reach,q,fc*mult)["commitments"]}
                before=safe-{start}
                after=safe-{dest}
                if len(after)<len(before):
                    lost_routes+=fc/max(1,len(before))
                    if fc>=MIN_CARGO and not after:
                        veto=True
            if veto or lost_routes>c:
                continue
            gap=scores[original]-scores[action]
            if not isfinite(gap):
                raise ValueError("nonfinite parent logit gap")
            features=[float(c),len(threat["strict"]),len(threat["equal"]),int(len(threat["commitments"])>=2),
                      len(clear),d-dist0,left-1-d,rank,float(gap),float(lost_routes)]
            priority=c/(1.+.1*gap) - .15*lost_routes
            saved.append({"uid":uid,"original":original,"action":action,"kind":"loaded_evade",
                          "features":features,"priority":priority,"threat":threat,"safe_routes":sorted(clear),
                          "destination":dest,"parent_destination":start,"parent_rank":rank,
                          "parent_logit":scores[original],"alternative_logit":scores[action],"carried":c,
                          "teammate_route_cost":lost_routes})
        results.extend(saved)
    # Candidate validation may disqualify the nominal runner-up. Preserve all safe backups for that step.
    return sorted(results,key=lambda q:(-q["carried"],q["uid"],q["parent_rank"],MOVES.index(q["action"])))


def candidates(obs,cfg,parent,actor,validate=None):
    """Strictly replay all slots on the adjusted prefix; return each threatened unit's best valid action."""
    validator=validate or getattr(actor,"_local_edit",None)
    if validator is None:
        raise ValueError("supply the frozen full-prefix validator")
    x,aux,labels,raw=actor._loaded_trace
    feat=actor._FEAT
    if feat.step!=obs["step"] or feat.seat!=obs["player"] or feat.to_actions(labels)!=parent:
        raise ValueError("stale guarded parent prefix")
    f=sys.modules["cnet2_feat"]
    order={u:{a:raw[s][int(f.CANON_MOVE[i]) if feat.seat else i] for i,a in enumerate(MOVES)}
           for s,u in enumerate(feat.own_uid[:10]) if u is not None}
    seen=set(); result=[]
    cfg={"recCollectorCost":30,"recCollectorIncreaseCost":0,"recPlanterCost":30,
         "recPlanterIncreaseCost":0,"workerLimit":10,**(cfg or {})}
    for item in ranked_options(obs,cfg,parent,order):
        u=item["uid"]
        if u in seen:
            continue
        out,reason=validator(obs,cfg,parent,{u:item["action"]},x,aux,feat)
        if reason!="accepted":
            continue
        if {k for k in parent.keys()|out.keys() if parent.get(k)!=out.get(k)}!={u}:
            raise ValueError("nonlocal collector intervention")
        result.append(item)
        seen.add(u)
    return sorted(result,key=lambda q:(-q["priority"],q["uid"],q["parent_rank"]))


def choose(obs,cfg,parent,actor,validate=None):
    options=candidates(obs,cfg,parent,actor,validate)
    return options[0] if options else None
