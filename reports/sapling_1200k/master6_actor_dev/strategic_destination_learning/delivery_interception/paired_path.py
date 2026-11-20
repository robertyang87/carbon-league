"""Trace the two registered workers, their deposits, and both complete teams' rent on sealed A/B continuations."""
from collections import Counter
import gzip
import json
from analyze import HERE, NATIVE, CashEvents, FastCarbon, board, read, sha, write

D = HERE / "native_response"


def tape(path, card):
    raw = path.read_bytes()
    assert sha(raw) == card["turns_sha256"]
    return [json.loads(x) for x in gzip.decompress(raw).splitlines()]


def difference(a,b):
    return {k:round(a.get(k,0)-b.get(k,0),4) for k in sorted(a.keys()|b.keys())}


def track(folder, tag, prefix, checked):
    arm = read(folder/(tag + ".json"))
    records = tape(folder/"prefix.jsonl.gz",prefix) + tape(folder/(tag+".jsonl.gz"),arm)
    m = prefix["root_menu"]
    s, t0 = prefix["seat"], prefix["root_step"]
    ids = {"target":m["target_uid"],"focal":m["focal_uid"]}
    sides = {"target":1-s,"focal":s}
    raw = read(NATIVE.parent/"direct_original/games/candidate/raw_621758346_1.json.gz")
    sim = FastCarbon(raw["game_configuration"]).reset(seed=prefix["seed"])
    assert sha(bytes(map(int,sim.carbon)))==prefix["terrain"]
    sim.cash=CashEvents(sim.cash)
    move,dist=board(sim.size)
    histories={k:[] for k in ids}
    totals={n:[Counter(),Counter()] for n in ("first10","suffix")}
    for t,row in enumerate(records):
        assert t==row["step"]==sim.step_count
        if t>=t0:
            before={}
            for k,uid in ids.items():
                w=sim.workers.get(uid)
                if w is None:continue
                issued=row["orders"][sides[k]].get(uid,"NONE")
                before[k]=dict(step=t,pre_pos=w.pos,pre_cargo=w.carbon,order=issued,arrived=move[w.pos][issued],
                               pre_nearest_own_base=min(dist[w.pos][pos] for _,pos in sim.bases[w.player]))
        sim.cash.events.clear()
        sim.step(*row["orders"])
        if t<t0:continue
        for kind, owner, change, *parts in sim.cash.events:
            for h in (("suffix","first10") if t<t0+10 else ("suffix",)):
                wallet=totals[h][owner]
                if kind=="tree_absorption":
                    wallet["ground_rent"]+=change-parts[2];wallet["enemy_cargo_tax"]+=parts[2]
                else:wallet[kind]+=change
        for key,r in before.items():
            uid=ids[key]; w=sim.workers.get(uid); owner=sides[key]
            r.update(post_pos=w.pos if w else None,post_cargo=w.carbon if w else None,
                     deposits=[e for e in sim.cash.events if e[0] in ("own_delivery","opponent_worker_delivery_to_this_base") and e[3]==uid],
                     collision=[e for e in sim.last_collisions if e[0]==uid],
                     immediate_cash_at_death=[e for e in sim.cash.events if e[0] in ("base_collision_cash","tree_collision_cash") and e[3]==uid])
            if r["collision"]:
                recipient=[u.id for u in sim.workers.values() if u.pos==r["arrived"]]
                r["surviving_worker_at_death_cell"]=recipient
            histories[key].append(r)
    assert list(sim.cash)==arm["terminal_economy"]["cash"]
    assert checked["official_scores"]==arm["official_scores_by_seat"]
    def receipt(role):
        seq=histories[role]; start=seq[0]["pre_pos"]
        banked=[(r,e) for r in seq for e in r["deposits"] if e[0]=="own_delivery" and e[1]==sides[role] and e[2]>0]
        first=None
        if banked:
            r,e=banked[0]; elapsed=r["step"]-t0+1
            walked=sum(v["pre_pos"]!=v["arrived"] for v in seq[:elapsed])
            first=dict(step=r["step"],amount=e[2],cell=e[4],elapsed=elapsed,moves=walked,stays=elapsed-walked,
                       shortest_to_actual_depot=dist[start][e[4]],extra_moves_over_shortest_to_actual_depot=walked-dist[start][e[4]],
                       extra_elapsed_over_initial_nearest=elapsed-seq[0]["pre_nearest_own_base"],within10=elapsed<=10)
        death=next((v for v in seq if v["post_pos"] is None),None)
        return dict(uid=ids[role],timeline_first10=seq[:10],first_positive_own_bank_after_root=first,
                    all_positive_own_banking=[[r["step"],e[2],e[4]] for r,e in banked],
                    actual_own_deposits_first10=sum(e[2] for r,e in banked if r["step"]<t0+10),
                    actual_own_deposits_suffix=sum(e[2] for _,e in banked),death=death,
                    elapsed_extra_note="Elapsed minus initial nearest distance includes waiting and route choice. Extra travelled moves compare the actual deposit cell, and only exist after a deposit.")
    return dict(arm=tag,official_margin=arm["margin"],u=arm["u"],official_scores=arm["official_scores_by_seat"],
                target=receipt("target"),focal=receipt("focal"),by_physical_seat=totals,
                root_effect=checked["root_effect"])


def review(folder):
    result=read(folder/"verified.json"); prefix=read(folder/"prefix.json")
    if result["status"]!="paired_raw":
        return dict(seed=result["seed"],status=result["status"],paired=False)
    offers=prefix["root_menu"]; s=prefix["seat"]; i=offers["intercept_arm"]; j="B" if i=="A" else "A"
    arms={k:track(folder,k,prefix,result["branch"][k]) for k in (i,j)}
    delta={}
    for period in ("first10","suffix"):
        v,w=arms[i]["by_physical_seat"][period],arms[j]["by_physical_seat"][period]
        delta[period]=dict(own=difference(v[s],w[s]),opponent=difference(v[1-s],w[1-s]))
        for role in ("focal","target"):
            key="actual_own_deposits_"+period
            delta[period][role+"_bank"]=round(arms[i][role][key]-arms[j][role][key],4)
    return dict(seed=result["seed"],seat=s,step=prefix["root_step"],paired=True,target_uid=offers["target_uid"],
                focal_uid=offers["focal_uid"],intercept=i,avoid=j,arms=arms,delta_intercept_minus_avoid=delta,
                delta_margin=result["intercept_minus_avoid_margin"],delta_u=result["intercept_minus_avoid_u"],
                verifier_sha256=sha((folder/"verified.json").read_bytes()))


def main():
    p=read(D/"protocol.json")
    roots=[]
    for r in p["fixtures"]:
        roots.append(review(D/"outputs"/f"{r['seed']}_{r['seat']}"))
    write(HERE/"native_paths.json",dict(protocol_sha256=sha((D/"protocol.json").read_bytes()),roots=roots,
        scope="One matched original response at the root, then independent full original responses. Root-opportunity attribution differs from subsequent stochastic trajectories. Total team rent cannot be assigned to a particular collector."))
    for r in roots:
        if r["paired"]:
            print(r["seed"],r["delta_margin"],r["delta_intercept_minus_avoid"]["first10"])
if __name__=="__main__":main()
