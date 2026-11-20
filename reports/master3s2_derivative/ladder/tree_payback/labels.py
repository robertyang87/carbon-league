"""Realized cash labels for new trees' original ownership tenures in a completed ladder replay."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"reports/sapling_1200k/all_top5"))
import economy_extract as economy


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_replay(replay):
    """Accept a ladder environment dictionary, or a local JSON/JSON.GZ path with that payload."""
    if isinstance(replay, dict):
        return replay
    raw = Path(replay).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def tree_rows(replay):
    """Return one row per successful plant ordered by actual payment within the action.

    t and end_t are zero-based action indices (action t leads to state t+1).
    Entries in cash are [t, rent_from_ground, rent_from_enemy_cargo, direct_collision_cash].
    A tree starts earning for the buyer upon planting. An incoming owner earns absorption on
    the seizure action. Earlier-in-action collision cash still belongs to the then owner.
    net excludes the original planter's labor or alternative moves. A later recapture starts
    a new ownership interval, and is not part of any new-plant label. Data after t are labels.
    """
    data = load_replay(replay)
    frames,cfg = data["steps"],data["configuration"]
    require(len(frames)>=2 and data["statuses"]==["DONE","DONE"], "Expected a completed two-sided official replay")
    require(frames[0][0]["observation"]["step"]==0 and all(f["status"]=="DONE" for f in frames[-1]),
            "Partial, truncated or execution-fault replay")
    # This verifies both official cash balances and every worker and tree identity/owner/age at every transition.
    authoritative,_,_,_ = economy.reconstruct(frames,cfg)
    sim = economy.FastCarbon(cfg).load_observation(frames[0][0]["observation"])
    sim.cash = economy.CashEvents(sim.cash)
    active, rows = {},[]
    for t,after in enumerate(frames[1:]):
        obs = after[0]["observation"]
        require(obs["step"]==t+1,"Nonconsecutive actions")
        src = frames[t][0]["observation"]["players"]
        pre = {tid:s for s,player in enumerate(src) for tid in player[3]}
        post = {tid:(s,tree[0]) for s,player in enumerate(obs["players"]) for tid,tree in player[3].items()}
        new = {(s,cell):tid for tid,(s,cell) in post.items() if tid not in pre}
        sim.cash.events.clear()
        sim.step(*[side["action"] or {} for side in after])
        writes = sim.cash.events
        cash = {}
        for kind,owner,value,*detail in writes:
            if kind=="tree_collision_cash" and value>0:
                uid=detail[2]
                if uid in active:
                    require(active[uid]["seat"]==owner,"Unexpected pre-transfer collision beneficiary")
                    cash.setdefault(uid,[0.,0.,0.])[2]+=value
        ending=[]
        for uid,r in list(active.items()):
            if uid not in post or post[uid][0]!=r["seat"]:
                r.update(end_t=t,exit="seized" if uid in post else "withered")
                ending.append(uid)
        # CashEvents omits zero writes. The platform defaults pay positive planting prices.
        # If planting is free, authoritative tree.birth/actor and engine worker iteration order
        # are still recoverable from the official post-action tree roster and pre-payment roster.
        payments=[(s,detail[0],detail[1],-value,i) for i,(kind,s,value,*detail) in enumerate(writes) if kind=="plant"]
        recorded={(s,cell) for s,_,cell,_,_ in payments}
        if recorded != set(new):
            require(recorded<=set(new), "A plant debit has no corresponding new tree")
            workers_order={worker_id:i for i,worker_id in enumerate(sim.workers)}
            for (s,cell),uid in new.items():
                if (s,cell) in recorded:
                    continue
                birth_worker=obs["trees"][uid][0]
                require(cfg["plantCost"]==0 and birth_worker in src[s][2] and src[s][2][birth_worker][0]==cell,
                        "New tree has no individually attributable plant debit")
                payments.append((s,birth_worker,cell,0.,None))
            payments.sort(key=lambda p:workers_order[p[1]])
        require(len(payments)==len(new),"Mismatch between original plants and debits")
        for i,(s,worker,cell,cost,cash_index) in enumerate(payments):
            uid=new[s,cell]
            require(uid not in active and src[s][2][worker][0]==cell and src[s][2][worker][2]=="PLANTER",
                    "Planting actor or cell mismatch")
            r=dict(t=t,seat=s,tree=uid,cell=cell,planter=worker,cost=float(cost),plant_order=i,
                   cash_write_index=cash_index,break_even_t=t if cost==0 else None,cash=[],gross=0.)
            active[uid]=r
            rows.append(r)
        by_tree = Counter()
        for kind,owner,value,*detail in writes:
            if kind=="tree_absorption":
                uid,cell,enemy=detail
                require(post[uid]==(owner,cell),"Absorption attributed to wrong owner or cell")
                by_tree[uid]+=value
                if uid in active and active[uid]["seat"]==owner:
                    slot=cash.setdefault(uid,[0.,0.,0.])
                    slot[0]+=value-enemy
                    slot[1]+=enemy
        for uid in post:
            require(abs(by_tree[uid]-obs["trees"][uid][1])<1e-6,"Per-tree posted receipt and simulation disagree")
        for uid,components in cash.items():
            r=active[uid]
            r["cash"].append([t,*components])
            r["gross"]+=sum(components)
            if r["break_even_t"] is None and r["gross"]+1e-7>=r["cost"]:
                r["break_even_t"]=t
        for uid in ending:
            del active[uid]
    last=len(frames)-2
    for r in active.values():
        r.update(end_t=last,exit="horizon" if len(frames)==cfg["episodeSteps"] else "early_terminal")
    for s in (0,1):
        require(abs(sum(r["cost"] for r in rows if r["seat"]==s)+sum(t[s][5] for t in authoritative))<1e-5,
                "Summed individual plant debits disagree with the cash ledger")
    for r in rows:
        r.update(net=r["gross"]-r["cost"],repaid=r["break_even_t"] is not None,
                 seized_unrepaid=r["exit"]=="seized" and r["break_even_t"] is None)
    return rows
