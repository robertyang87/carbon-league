"""Replay the seven actual departures toward home and value one extra mine before cashing out."""
import argparse
from collections import Counter
from pathlib import Path
import gzip
import json
import sys

from cash_practice import audit, np, ROOT
sys.path.insert(0, str(ROOT/"code"))
import cnet2_feat as F


def continue_to_home(obs, uid, our_orders):
    me = obs["player"]
    own, opp = obs["players"][me],obs["players"][1-me]
    pos,cargo,_=own[2][uid]
    homes=list(own[1].values())
    end=F.DEST[pos]
    other_dest={int(F.DEST[v[0],F.MOVES.index(our_orders.get(u) or "NONE")]) for u,v in own[2].items() if u!=uid}
    other_dest.update(b for bid,b in own[1].items() if our_orders.get(bid))
    hostile=[v[0] for v in opp[2].values() if v[2]=="COLLECTOR" and v[1]<=cargo]
    threat=(F.DIST[end][:,hostile].min(axis=1)<=1 if hostile else np.zeros(5,bool))
    trees={v[0] for v in opp[3].values()}
    def key(a):
        d=int(end[a])
        toll=sum(v in trees for v in [d,*F.NB8[d]]) if d not in homes else 0
        return (d in other_dest or d in opp[1].values(), bool(threat[a]),int(F.DIST[d,homes].min()),toll,a)
    return F.MOVES[min(range(5),key=key)]


def measure(state,cfg,steps,row,wait):
    sim=audit.FastCarbon(cfg).set_state(state)
    observer=audit.Observer()
    t,s,uid=row["t"],row["seat"],row["uid"]
    start_cash=sim.cash[s]
    advance=0
    for turn in range(10):
        j=t+turn+1
        if j>=len(steps) or uid not in sim.workers:break
        commands=[dict(side.get("action") or {}) for side in steps[j]]
        if turn==0:
            a=row["parent"]
            action="NONE" if wait else F.MOVES[F.CANON_MOVE[a] if s else a]
        else:
            action=continue_to_home(sim.obs(s),uid,commands[s])
        commands[s][uid]=action
        observer.step(sim,commands)
        advance+=1
        receipts=[e["amount"] for e in observer.events if e["event"]=="deposit" and e["worker"]==uid and e["player"]==s and e["amount"]>0]
        if receipts:
            return dict(turns=advance,credited=sum(receipts),status="own_deposit")
        if any(audit._extinct(sim.get_state(),i) for i in (0,1)):break
    return dict(turns=advance,credited=0.,status="disappeared" if uid not in sim.workers else "no_traced_deposit")


def main(check,corpus,out):
    check_rows=json.loads(check.read_bytes())
    inputs=[r for r in check_rows["changes"] if r["parent_d"]<r["source_d"]]
    games=sorted(set(r["id"] for r in inputs))
    rows=[]
    for gid in games:
        replay=json.loads(gzip.decompress((corpus/(gid+".json.gz")).read_bytes()))
        steps,cfg=replay["steps"],replay["configuration"]
        sim=audit.FastCarbon(cfg).load_observation(steps[0][0]["observation"])
        selected={r["t"]:[] for r in inputs if r["id"]==gid}
        for r in inputs:
            if r["id"]==gid:selected[r["t"]].append(r)
        for t in range(max(selected)+1):
            if t in selected:
                p=sim.full_observation()["players"]
                x=steps[t][0]["observation"]["players"]
                cash_error=max(abs(p[s][0]-x[s][0]) for s in (0,1))
                if cash_error>.02 or any(set(p[s][2])!=set(x[s][2]) for s in (0,1)):
                    raise ValueError((gid,t,"factual state mismatch",cash_error))
                for r in selected[t]:
                    exact=sim.get_state()
                    wid,seat=r["uid"],r["seat"]
                    got=p[seat][2][wid];want=x[seat][2][wid]
                    if got[0]!=want[0] or abs(got[1]-want[1])>.01:
                        raise ValueError("target factual state mismatch")
                    original=measure(exact,cfg,steps,r,False)
                    extra=measure(exact,cfg,steps,r,True)
                    rows.append(dict(**r,immediate=original,mine_once=extra,
                                     extra_receipt=extra["credited"]-original["credited"]))
            sim.step(*[(side.get("action") or {}) for side in steps[t+1]])
    diffs=[r["extra_receipt"] for r in rows]
    audit.shared.atomic_json(out,dict(scope="Only 10-map holdout roots where parent moved nearer home. Other workers replay archived orders; one collector follows common public safe-then-short return logic after either first action. Up to 10 turns, first normal own-base deposit counted, missing receipt zero. No estimate of foregone new work after the earlier arrival, reactive opponent response or win rate.",
              input_sha256=audit.digest(check),n=len(rows),positive=sum(v>1e-4 for v in diffs),
              negative=sum(v< -1e-4 for v in diffs),mean_delta=float(np.mean(diffs)),rows=rows))
    print(len(rows),diffs)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("check","corpus","out"):p.add_argument(n,type=Path)
    a=p.parse_args();main(a.check,a.corpus,a.out)
