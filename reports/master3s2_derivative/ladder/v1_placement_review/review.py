"""Reconcile every frozen v1 rated placement result and independently replay every losing game."""
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
sys.path.insert(0, str(LADDER / "losses_nov20"))
from compare import economy, summarize
import validation

BASE = "b99742a4a510a4de5dc99639b3c85604"
FREEZE = "805f6ab04fd81959bf43fd1db00d1d7724ffa5e5082c7a1251e6c6405bf2b338"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cash_accounts(rows, seat):
    order = (seat, 1-seat)
    totals = {name: [sum(row[s][i] for row in rows) for s in order]
              for i, name in enumerate(economy.KEYS)}
    totals["net_tree"] = [sum(totals[name][j] for name in ("tree", "plant", "seize"))
                          for j in range(2)]
    differences = {name: values[0]-values[1] for name,values in totals.items()}
    return ({name:[round(v,2) for v in values] for name,values in totals.items()},
            {name:round(v,2) for name,v in differences.items()},
            sum(differences[name] for name in economy.KEYS))


def losing_game(card, path):
    game = summarize(card["opponent"], card["match_id"], card["seat"])
    official=validation.official({"path":path.name},path.parent)
    require(official==dict(matched=card["steps"],first_divergence=None), "official replay differs")
    states=game["steps"]
    receipts=game["receipts_by_transition"]
    require(len(states)==card["steps"]+1==len(receipts)+1,"omitted states")
    replay = json.loads(gzip.decompress(path.read_bytes()))
    public = replay["steps"]
    timeline, _, _, _ = economy.reconstruct(public, replay["configuration"])
    for interval in [game["whole"], *game["phases"]]:
        rows = timeline[interval["start"]:interval["end"]]
        income, diff, unrounded_change = cash_accounts(rows, card["seat"])
        require(abs(unrounded_change-interval["margin_change"])<1e-6, "phase cash imbalance")
        interval["income"].update(income)
        interval["difference"].update(diff)
    worst={}
    for width in (3,5,10):
        a=min(range(300-width),key=lambda t:states[t+width]["margin"]-states[t]["margin"])
        b=a+width
        returns=[dict(t=t+1,**e) for t in range(a,b) for e in receipts[t]]
        for e in returns:
            uid,seat,t=e["worker"],e["seat"],e["t"]
            seen=[]
            for k in sorted(set(max(0,t-j) for j in (10,5,3,1,0))):
                players=public[k][0]["observation"]["players"]
                worker=players[seat][2].get(uid)
                action=(public[k+1][seat]["action"] or {}).get(uid,"NONE") if k<299 else None
                seen.append(dict(t=k,state=worker,next_if_present=action if worker else None))
            e["history"]=seen
        worst[str(width)]=dict(start=a,end=b,margin_from=states[a]["margin"],margin_to=states[b]["margin"],returns=returns)
    negative=[s for s in game["all_streaks"] if s["sign"]==-1]
    firsts={str(n):next((s for s in negative if s["frames"]>=n),None) for n in (5,10,20)}
    last_lead=max((s["t"] for s in states if s["margin"]>0),default=None)
    game.update(card=card,official=official,fast_verified=card["steps"],first_sustained_deficits=firsts,
                last_strict_lead=last_lead,worst_windows=worst,
                phases_at={str(t):states[t] for t in (0,50,100,150,200,250,275,299)},
                full_joint_commands=[[s["action"] or {} for s in f] for f in public[1:]])
    return game


def main():
    frozen=(HERE/"snapshot.json").read_bytes()
    require(sha(frozen)==FREEZE,"different frozen public page set")
    snap=json.loads(frozen)
    seen=[m for p in snap["pages"] for m in p["ids"]]
    matches=snap["official_matches_unfiltered"]
    require(len(set(seen))==len(seen) and set(seen)==set(matches),"incomplete or duplicate page data")
    exposure={(r["match_id"],r.get("game",1)):r for r in json.loads((LADDER/"exposed_ladder_maps.json").read_bytes())}
    register=[];losses=[];pending=[];win_economies=[]
    for m in sorted(matches.values(),key=lambda r:(r.get("finished") or float("inf"),r["id"])):
        require(BASE in (m["a"],m["b"]),"wrong strategy")
        if m["kind"]!="newrank-placement" or not m["rated"]:
            continue
        if m["state"]!="done" or not m.get("result"):
            pending.append(dict(id=m["id"],state=m["state"]));continue
        own="a" if m["a"]==BASE else "b";opp="b" if own=="a" else "a"
        for g in m["result"]:
            key=(m["id"],g["game"])
            data=exposure[key];path=LADDER/"replays"/f"{key[0]}-{key[1]}.json.gz"
            raw=path.read_bytes();env=json.loads(gzip.decompress(raw))
            digest=sha(bytes(env["steps"][0][0]["observation"]["full_carbon"]))
            require(sha(raw)==data["replay_gzip_sha256"] and digest==data["opening_carbon_bytes_sha256"]
                    and g["map_sha256"]==data["scorecard_map_sha256"],"wrong map or replay bytes")
            seat=g["order"].index(own)
            require(env["rewards"]==[g["scores"][s] for s in g["order"]] and
                    env["statuses"]==[g["statuses"][s] for s in g["order"]] and
                    len(env["steps"])-1==g["steps"],"official scores, statuses or transitions disagree")
            margin=float(Decimal(str(g["scores"][own]))-Decimal(str(g["scores"][opp])))
            outcome="W" if g["winner"]==own else ("D" if g["winner"]=="draw" else "L")
            require(outcome==("W" if margin>0 else "L" if margin<0 else "D"),"winner conflicts with both normal scores")
            card=dict(match_id=key[0],game=key[1],opponent=m["name_"+opp],opponent_id=m[opp],seat=seat,
                      score=[g["scores"][own],g["scores"][opp]],margin=margin,outcome=outcome,
                      statuses=env["statuses"],steps=g["steps"],wall_seconds=g["duration_seconds"],
                      replay_sha256=sha(raw),terrain_sha256=digest)
            register.append(card)
            if outcome=="L":
                losses.append(losing_game(card,path))
            if outcome=="W":
                timeline, _, _, _ = economy.reconstruct(env["steps"], env["configuration"])
                income, differences, unrounded_change = cash_accounts(timeline, seat)
                require(abs(unrounded_change-margin)<1e-6, "win cash imbalance")
                win_economies.append(dict(match_id=key[0],game=key[1],opponent=card["opponent"],
                                          seat=seat,margin=margin,verified_steps=len(timeline),
                                          income=income,difference=differences))
    require(len(register)==snap["settled"] and len(losses)==snap["losses"] and not pending,"settled denominator changed")
    result=dict(frozen_utc=snap["utc"],snapshot_sha256=FREEZE,strategy_id=BASE,selection=snap["official_filter"],
                registered_games=register,outcomes=dict(Counter(r["outcome"] for r in register)),pending=pending,
                seat_outcomes={str(s):dict(Counter(r["outcome"] for r in register if r["seat"]==s)) for s in (0,1)},
                distinct_terrains=len(set(r["terrain_sha256"] for r in register)),losses=losses,
                wins_economy=win_economies,
                win_comparison={key:sum(r["difference"][field]<0 for r in win_economies) for key,field in
                                (("greater_plant_spend","plant"),("lower_net_tree","net_tree"),("lower_bank","bank"))},
                orientation="monetary comparisons below are candidate minus opponent; full_joint_commands and receipt seat use physical seats")
    data=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    (HERE/"ledger.json.gz").write_bytes(gzip.compress(data,mtime=0))


if __name__=="__main__":
    main()
