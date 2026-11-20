"""Verify the single DS v2 versus ClaudeFable5.2-v7 replay and reconcile every cash transition."""
from collections import Counter
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
ROOT = LADDER.parents[2]
sys.path[:0] = [str(ROOT / "reports/sapling_1200k/all_top5"),
                str(ROOT / "reports/master_family/deep_review"), str(LADDER / "tree_payback")]
import economy_extract as economy
from carbon_deep_events import _snapshot, _actions, _reconstruct
from labels import tree_rows
from validation import official

MATCH = "bdbe59b04f71e43e53696874c8fd5958"
BLOB = "9d98e3a8afa6f793ea38485a3f2032bb1837c810eef93f6df3b1b38e87984772"
SEATS = (1, 0)
CUTS = (0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299)


def require(value, message):
    if not value:
        raise ValueError(message)


def at_enemy_base(frames, cfg):
    """A lighter interceptor can reach 160 on the scoring turn; check which treasury gets paid."""
    from zerosum_env.envs.carbon import idgen
    from zerosum_env.envs.carbon.helpers import Board

    root, uid = 272, "player-1-worker-62"
    result = {}
    for move in ("RIGHT", "UP"):
        idgen.reset()
        state = copy.deepcopy(frames[0][0]["observation"])
        state["carbon"] = state["full_carbon"]
        sim = economy.FastCarbon(cfg).load_observation(frames[0][0]["observation"])
        sim.cash = economy.CashEvents(sim.cash)
        for t in range(root+1):
            commands = [dict(r["action"] or {}) for r in frames[t+1]]
            if t==root:
                require(commands[1][uid]=="RIGHT","historical mover mismatch")
                commands[1][uid]=move
            state=Board(state,cfg,commands).next().observation
            sim.cash.events.clear()
            sim.step(*commands)
        post=sim.obs(0)
        require(state["players"]==post["players"] and state["trees"]==post["trees"] and
                list(sim.carbon)==state["carbon"],"branch engines diverge")
        if move=="RIGHT":
            require(state["players"]==frames[root+1][0]["observation"]["players"],"actual orders differ")
        result[move]=dict(post_cash=[p[0] for p in state["players"]],worker=state["players"][1][2].get(uid),
                          opponent_carrier=state["players"][0][2].get("player-0-worker-42"),cash_events=sim.cash.events)
    return result


def run():
    path = LADDER / "replays" / (MATCH + "-1.json.gz")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BLOB, "Different raw replay")
    env = json.loads(gzip.decompress(raw))
    frames, cfg = env["steps"], env["configuration"]
    roster = [f[0]["observation"]["players"] for f in frames]
    score = [s[0] for s in roster[-1]]
    require(len(frames) == 300 and env["statuses"] == ["DONE","DONE"] and score == env["rewards"] == [7179.18,7121.18],
            "Incorrect result, steps or seat")
    replayed = official({"path":path.name},path.parent)
    require(replayed == {"matched":299,"first_divergence":None}, "Official board mismatch")
    flows, intercepted, deposits, enemy_tax = economy.reconstruct(frames,cfg)
    dist = economy.distances(cfg["size"])
    margin = [round(p[1][0]-p[0][0],2) for p in roster]
    commands = [[s["action"] or {} for s in f] for f in frames[1:]]
    tree_log = tree_rows(env)

    def interval(start,end):
        sides = []
        for s in SEATS:
            vals={k:sum(q[s][j] for q in flows[start:end]) for j,k in enumerate(economy.KEYS)}
            require(abs(sum(vals.values())-(roster[end][s][0]-roster[start][s][0]))<.015,"Cash conservation")
            vals.update(tree_net=vals["tree"]+vals["plant"]+vals["seize"], collected_from_enemy=sum(q[s] for q in intercepted[start:end]),
                        tree_tax_on_enemy=sum(q[s] for q in enemy_tax[start:end]))
            sides.append({k:round(v,2) for k,v in vals.items()})
        return dict(start=start,end=end,begin_margin=margin[start],end_margin=margin[end],ds=sides[0],fable=sides[1],
                    state_at_end=[economy.profile(roster[end],s,dist) for s in SEATS])

    def track(uid,seat,t):
        entries=[]
        for k in sorted(set([max(0,t-d) for d in [10,5,3,1,0]])):
            worker=roster[k][seat][2].get(uid);other=roster[k][1-seat][2]
            if not worker:entries.append(dict(t=k,absent=True));continue
            pos,cargo,role=worker; action=commands[k][seat].get(uid,"NONE") if k<299 else None
            nxt=roster[k+1][seat][2].get(uid) if k<299 else None
            reach=[] if nxt is None else [other_uid for other_uid,w in other.items() if w[2]=="COLLECTOR"
                      and w[1]<cargo and dist[w[0]][nxt[0]]<=1]
            entries.append(dict(t=k,pos=pos,cargo=round(cargo,4),role=role,issued=action,
                                opponent_lighter_collectors_that_can_reach_actual_next_cell=reach))
        return entries

    snapshots = [_snapshot(f,cfg["size"]) for f in frames]
    per_turn=[]; receipts=[]; first=[{},{}]; collisions=[]; transfers=[]
    for t in range(299):
        info=[]
        for seat in SEATS:
            events=_reconstruct(snapshots[t],snapshots[t+1],_actions(frames[t+1]),cfg,seat)
            info.extend([dict(reporting_seat=seat,event=e) for e in events if e["kind"] not in ("collision","delivery")])
            for e in events:
                if e["kind"] in ("recruit","tree_planted","tree_captured") and e["kind"] not in first[seat]:
                    first[seat][e["kind"]]=e
                if e["kind"]=="collision" and (e["own_cargo_to_opponent"]>0 or
                                                 e["own_removed_cargo_gross"]>0 and e["own_cargo_to_ground"]>0):
                    collisions.append(dict(seat=seat,event=e))
        for s,uid,amount in deposits[t]:
            event=dict(arrival=t+1,seat=s,worker=uid,paid=round(amount,4))
            receipts.append(event);first[s].setdefault("delivery",event)
        before={uid:(s,tr) for s,p in enumerate(roster[t]) for uid,tr in p[3].items()}
        for s,p in enumerate(roster[t+1]):
            for uid,(cell,age) in p[3].items():
                if uid in before and before[uid][0]!=s:
                    transfers.append(dict(t=t,from_seat=1-s,to_seat=s,tree=uid,cell=cell,age=age,
                                     occupant=[wid for wid,w in p[2].items() if w[0]==cell and w[2]=="PLANTER"]))
        per_turn.append(dict(t=t,margin_before=margin[t],margin_after=margin[t+1],issued=commands[t],cash=flows[t],
                             deposits=deposits[t],collected_from_enemy=intercepted[t],events=info))
    streaks=[]
    for sign,ts in itertools.groupby(range(1,300), key=lambda k:(margin[k]>0)-(margin[k]<0)):
        rr=list(ts);streaks.append(dict(sign=sign,start=rr[0],end=rr[-1],length=len(rr)))
    windows={}
    for width in (3,5,10):
        valid=range(300-width)
        for word,fn in (("gain",max),("loss",min)):
            start=fn(valid,key=lambda j: margin[j+width]-margin[j])
            windows[f"{word}_{width}"]=interval(start,start+width)
    # First sustained lead/deficit and the final flips are observational boundaries, not decision labels.
    cuts=[(61,62),(61,75),(61,100),(127,137),(175,177),(250,260),(253,263),(262,269),
          (269,275),(275,285),(285,290),(290,295),(295,296),(295,299),(269,299)]
    key_windows=[]
    for start,end in cuts:
        ordered=[r for r in receipts if start<r["arrival"]<=end]
        nominated=[]
        for s in SEATS:
            rs=[r for r in ordered if r["seat"]==s]
            if rs:
                top=max(rs,key=lambda r:r["paid"])
                nominated.append(dict(**top,history=track(top["worker"],s,top["arrival"])))
        key_windows.append(dict(cash=interval(start,end),deposits=ordered,largest_deposit_per_side=nominated))
    for e in collisions:
        t=e["event"]["step_to"]; s=e["seat"]
        e["paths"]={worker:track(worker,s,t) for worker in e["event"]["deleted_worker_ids"] if worker in roster[t-1][s][2]}
    owners={}
    for s in SEATS:
        trees=[x for x in tree_log if x['seat']==s]
        by_period=[]
        for start,end in ((0,100),(100,200),(200,299),(269,299)):
            chosen=[x for x in trees if start<=x["t"]<end]
            by_period.append(dict(start=start,end=end,count=len(chosen),repaid=sum(x["repaid"] for x in chosen),
                                  expense=round(sum(x["cost"] for x in chosen),2),
                                  original_tenure_cash=round(sum(x["gross"] for x in chosen),2)))
        owners[str(s)]=dict(planted=len(trees),original_tenure_repaid=sum(x["repaid"] for x in trees),
                            seized_before_repaid=sum(x["seized_unrepaid"] for x in trees),
                            first_tenure_exit=dict(Counter(x["exit"] for x in trees)),cohorts=by_period,
                            trees=[{k:v for k,v in x.items() if k!="cash"} for x in trees])
    out=dict(match=MATCH,replay_sha256=BLOB,official=replayed,fast_verified=299,physical_seat_order=["ClaudeFable5.2-v7","ds-pro-v2"],
             cash_field_order=economy.KEYS,record="actions indexed by pre-state t, paid arrivals by new state t+1",
             physical_geometry_sha256=hashlib.sha256(bytes(frames[0][0]["observation"]["full_carbon"])).hexdigest(),
             first_by_physical_seat=first,whole=interval(0,299),periods=[interval(a,b) for a,b in zip(CUTS,CUTS[1:])],
             streaks=streaks,win_tally={str(s):sum(r["length"] for r in streaks if r["sign"]==s) for s in (-1,0,1)},
             largest_windows=windows,major_windows=key_windows,all_deposits=receipts,all_seizures=transfers,
             collisions_with_cargo_loss=collisions,trees_by_physical_seat=owners,turns=per_turn,
             single_root_comparison_272=at_enemy_base(frames,cfg))
    encoded=json.dumps(out,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    (HERE/"ledger.json.gz").write_bytes(gzip.compress(encoded,mtime=0))


if __name__=="__main__":
    run()
