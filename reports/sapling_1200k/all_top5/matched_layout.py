"""Same-opponent layout comparisons on the frozen Top Five corpus, one row per map."""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
STEP=60
CASH_LIMIT=300.
FACTORS={"workers":("collectors","planters"),"tree_turns":("remaining_tree_turns_30",)}


def mean(values):
    return sum(values)/len(values)


def describe(games):
    return dict(maps=len(games),wins=sum(r["outcome"]==1 for r in games),
                draws=sum(r["outcome"]==.5 for r in games),losses=sum(r["outcome"]==0 for r in games),
                average_outcome=mean([r["outcome"] for r in games]) if games else None)


def compare(rows):
    """Standardize the sign within each exact matchup. Every included map gets equal weight."""
    eligible=[g for g in rows if g["factor_difference"]!=0]
    pairs=defaultdict(list)
    for g in eligible:
        pairs[(g["player"],g["opponent"])].append(g)
    strata=[];matched=[];contributions=[];reverse=[]
    for names,part in sorted(pairs.items()):
        lead=[g for g in part if g["factor_difference"]>0]
        trail=[g for g in part if g["factor_difference"]<0]
        if not lead or not trail:continue
        l,t=describe(lead),describe(trail);delta=l["average_outcome"]-t["average_outcome"]
        weight=len(part)
        strata.append(dict(player=names[0],opponent=names[1],lead=l,trail=t,maps=weight,difference=delta,
                           ids=[r["id"] for r in part]))
        matched.extend(part);contributions.append((weight,delta))
        # Strong counterexample for this matchup: the same player both lost
        # with the proposed advantage and won with the proposed disadvantage.
        lost=[g for g in lead if g["outcome"]==0]
        won=[g for g in trail if g["outcome"]==1]
        if lost and won:
            reverse.append((weight,names,sorted(lost,key=lambda r:r["id"])[0],sorted(won,key=lambda r:r["id"])[0]))
    def leader_score(rs):
        return mean([g["outcome"] if g["factor_difference"]>0 else 1-g["outcome"] for g in rs]) if rs else None
    count=len(matched)
    if not count:raise ValueError("no contrastable matchup")
    weighted=sum(n*d for n,d in contributions)
    delta=weighted/count
    remove=[(weighted-n*d)/(count-n) for n,d in contributions if n<count]
    # Overlap weighting corresponds to the slope of a linear model with separate
    # intercepts by matchup. Report it as a distinct sensitivity analysis.
    overlap=[(s["lead"]["maps"]*s["trail"]["maps"]/s["maps"],s["difference"]) for s in strata]
    equal=mean([s["difference"] for s in strata])
    overlap_delta=sum(w*d for w,d in overlap)/sum(w for w,_ in overlap)
    left=[g for g in matched if g["factor_difference"]>0]
    right=[g for g in matched if g["factor_difference"]<0]
    rd=sorted(reverse,key=lambda x:(-x[0],x[1]))
    return dict(cash_close_maps=len(rows),factor_tied_maps=len(rows)-len(eligible),unmatched_favored_score=leader_score(eligible),
                informative_maps=len(eligible),all_matchup_count=len(pairs),bidirectional_matchups=len(strata),
                maps_from_one_sided_matchups=len(eligible)-count,matched_maps=count,matched_favored_score_before_balancing=leader_score(matched),
                matched_reference_leads=describe(left),matched_reference_trails=describe(right),
                within_matchup_difference=delta,balanced_favored_score=.5+.5*delta,
                equal_matchup_weight_difference=equal,overlap_weight_difference=overlap_delta,
                leave_one_matchup_out_difference_range=[min(remove),max(remove)] if remove else None,
                matchups_with_one_map_on_the_scarcer_side=sum(min(s["lead"]["maps"],s["trail"]["maps"])==1 for s in strata),
                within_matchup_positive_zero_negative=[sum(s["difference"]>0 for s in strata),sum(s["difference"]==0 for s in strata),
                                                       sum(s["difference"]<0 for s in strata)],
                paired_reverse_case_count=len(rd),illustration=dict(lead_loss=rd[0][2],trail_win=rd[0][3]) if rd else None,
                strata=strata)


def run():
    manifests=(HERE/"manifest.json").read_bytes(); data=(HERE/"layout.json.gz").read_bytes()
    m=json.loads(manifests); l=json.loads(gzip.decompress(data))
    mm={g["id"]:g for g in m["games"]};rows=l["games"]
    if (m["state"]!="complete" or l["git_revision"]!=m["revision"] or set(mm)!={g["id"] for g in rows}
            or len(mm)!=len(rows) or len({g["seed"] for g in rows})!=len(rows)):
        raise ValueError("incomplete maps or different sources")
    rank={s["name"]:s["rank"] for s in m["top5"]};fi={name:i for i,name in enumerate(l["field_order"])}
    groups=defaultdict(list);flow=defaultdict(Counter)
    for g in rows:
        c=mm[g["id"]];co=c["cohort"];names=c["seat_names"]
        f=flow[co];f["complete_replays"]+=1
        if (g["sha256"]!=c["sha256"] or g["scores"]!=c["score_by_seat"] or g["seed"]!=c["seed"]
                or g["statuses"]!=c["terminal"] or len(set(names))!=2):raise ValueError(g["id"])
        if g["statuses"]!=["DONE","DONE"]:f["nonstandard_terminal"]+=1;continue
        if str(STEP) not in g["snapshots"]:f["ended_before_t60"]+=1;continue
        snapshot=g["snapshots"][str(STEP)]
        # Orient mutual games to the better pre-frozen rank; never count both perspectives.
        seat=min(range(2),key=lambda p:rank.get(names[p],9999))
        cash=[p[fi["cash"]] for p in snapshot]
        if abs(cash[0]-cash[1])>CASH_LIMIT:f["cash_gap_above_300"]+=1;continue
        f["cash_close_maps"]+=1
        scr=g["scores"];outcome=1 if scr[seat]>scr[1-seat] else 0 if scr[seat]<scr[1-seat] else .5
        for fac,fields in FACTORS.items():
            ft=[sum(p[fi[k]] for k in fields) for p in snapshot]
            groups[co,fac].append(dict(id=g["id"],map_seed=g["seed"],player=names[seat],opponent=names[1-seat],seat=seat,
                                       focal_cash=cash[seat],opponent_cash=cash[1-seat],focal_quantity=ft[seat],
                                       opponent_quantity=ft[1-seat],factor_difference=ft[seat]-ft[1-seat],outcome=outcome,
                                       focal_final=scr[seat],opponent_final=scr[1-seat]))
    result=dict(revision=m["revision"],input_sha256={"layout.json.gz":hashlib.sha256(data).hexdigest(),
                  "manifest.json":hashlib.sha256(manifests).hexdigest()},criterion="t60, absolute cash gap <=300.",
                estimand="Within each exact pre-frozen-rank focal player/opponent pairing, average focal WDL (draw=0.5) when focal leads minus when focal trails. Both signs required. Weight each contrast by eligible matchup maps / all matched maps. Balanced favored-player WDL = (1 + difference)/2. Baseline = unadjusted WDL of whichever player has more. One independent row per map.",
                reverse_case_selection="For each cohort/factor choose matchup with most maps observing both lead-loss and trail-win. Ties by names; show smallest alphabetical game ID of each type. Other strata and all member IDs are retained.",
                flow={k:dict(v) for k,v in flow.items()},metrics={})
    for co in ("mutual","external"):
        result["metrics"][co]={fac:compare(groups[co,fac]) for fac in FACTORS}
    return result


if __name__=="__main__":
    d=run()
    (HERE/"matched_layout.json").write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
