"""Recompute eight official games, seatwise physical ledgers and search/latency from the hosted artifacts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
sys.path.insert(0, str(LANE))
from verify_pair import verify_game, live

sys.path.insert(0, str(live.ROOT/"reports/sapling_1200k"))
from turning_point_ledger import FastCarbon, CashEvents

PERIODS = ("whole", "0_99", "100_199", "200_298", "last_99_observed")
MONEY = ("own_delivery", "opponent_worker_delivery_to_this_base", "ground_rent", "cargo_tax",
         "tree_collision_cash", "base_collision_cash", "plant", "seize", "hire_collector", "hire_planter", "rounding")


def read(p):
    return json.loads(p.read_bytes())


def store(p, data):
    raw = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    raw = gzip.compress(raw, mtime=0) if p.suffix==".gz" else raw+b"\n"
    p.write_bytes(raw)


def balance(ledger, seat):
    return {k: ledger[seat].get(k,0.)-ledger[1-seat].get(k,0.) for k in MONEY}


def search(rows, op):
    counts = Counter()
    depth = Counter()
    candidates = Counter()
    cold = []
    trace = []
    prior = Counter()
    for row in rows:
        call = row["calls"][op]
        if call["status"] != "ok":
            continue
        answer = call["response"]
        new = Counter(answer["search_counts"])
        jump = {k:new[k]-prior[k] for k in ("searched","single","aborted","changed")}
        assert all(v in (0,1) for v in jump.values())
        prior = new
        last = answer.get("search_last", {})
        if jump["single"]:
            kind = "single"
        elif jump["searched"]:
            kind = "searched"
        elif last.get("depth")==0 and last.get("candidates",0)>1:
            kind = "zero"
        else:
            kind = "unresolved_no_search"
        counts[kind] += 1
        if last:
            assert kind!="searched" or last["depth"]>0
            depth[last["depth"]] += 1
            candidates[last["candidates"]] += 1
        if kind=="zero":
            cold.append(row["step"])
        trace.append(dict(t=row["step"],kind=kind,last=last,increment=jump,
                          rpc_wall=call["elapsed"],public_wall=answer.get("public_wall_seconds"),public_cpu=answer.get("public_cpu_seconds")))
    spans = []
    for t in cold:
        if spans and t == spans[-1][1]+1:
            spans[-1][1] = t
        else:
            spans.append([t,t])
    return dict(calls=len(trace),branches=dict(counts),depth=dict(depth),candidates=dict(candidates),
                zero_spans=spans,end_counters=dict(prior),timeline=trace)


def settle_game(game, directory):
    receipt = read(directory/"receipt.json")
    assert live.sha(directory/"receipt.json") == game["receipt_sha256"]
    rpath=directory/"turns.jsonl"
    rows = [json.loads(line) for line in rpath.read_bytes().splitlines()] if rpath.exists() else []
    if rows:
        assert live.sha(rpath)==receipt["turns_sha256"]
    seat = game["seat"]
    env,cfg = live.official() # registration selected by verify_game.
    sim = FastCarbon(cfg).reset(seed=game["seed"])
    assert list(sim.carbon)==list(env.states[0].observation.full_carbon)
    sim.cash=CashEvents(sim.cash)
    initial = list(sim.cash)
    ledger = {k:[Counter(),Counter()] for k in PERIODS}
    steps = {k:0 for k in PERIODS}
    turns = []
    n = game["official_transitions"]
    for row in rows:
        t = len(turns)
        assert row["step"]==t and not env.done
        if any(c["status"]!="ok" for c in row["calls"]):
            break
        joint = [c["response"]["action"] for c in row["calls"]]
        before_players = json.loads(json.dumps(env.states[0].observation.players))
        before = list(sim.cash)
        sim.cash.events.clear()
        sim.step(*joint)
        env.step(joint)
        physical = [p[0] for p in env.states[0].observation.players]
        assert all(math.isclose(x,y,abs_tol=1e-6,rel_tol=0) for x,y in zip(physical,sim.cash))
        events = list(sim.cash.events)
        detail = [Counter(),Counter()]
        for kind,side,amount,*rest in events:
            if kind=="tree_absorption":
                detail[side]["ground_rent"] += amount-rest[2]
                detail[side]["cargo_tax"] += rest[2]
            else:
                assert kind in MONEY
                detail[side][kind] += amount
        periods = ["whole",("0_99" if t<100 else "100_199" if t<200 else "200_298")]
        if t>=max(0,n-99):
            periods.append("last_99_observed")
        for k in periods:
            steps[k]+=1
            for s in (0,1):
                ledger[k][s].update(detail[s])
        assert all(abs((physical[s]-before[s])-sum(detail[s].values()))<1e-6 for s in (0,1))
        turns.append(dict(t=t,players_before=before_players,joint_orders=joint,cash_after=physical,
                       margin_change=physical[seat]-physical[1-seat]-before[seat]+before[1-seat],money=detail,
                       events=events))
    assert len(turns)==n
    scores=[x.reward for x in env.states] if env.done else None
    if game["status"]=="complete":
        assert scores==game["official_scores_by_seat"]
    assert all(math.isclose(sum(ledger["whole"][s].values()),sim.cash[s]-initial[s],abs_tol=1e-5) for s in (0,1))
    timing = []
    for s in (0,1):
        times=sorted(row["calls"][s]["elapsed"] for row in rows)
        if not times:
            timing.append(dict(s=s,requests=0))
            continue
        ref=receipt["timing_by_seat"][s]
        per = {k:times[math.ceil(q*len(times))-1] for k,q in (("p95",.95),("p99",.99),("max",1))}
        assert all(per[k]==ref["observed_wait_seconds_"+k] for k in per)
        extra=sum(max(0,x-1) for x in times)
        assert math.isclose(extra,ref["observed_excess_seconds"]) and ref["observed_waits_over_one_second"]==sum(x>1 for x in times)
        timing.append(dict(**per,seat=s,requests=len(times),init=ref["init_seconds"],slow=sum(x>1 for x in times),
                    excess=extra,peak_rss_kib=ref["reported_peak_rss_kib"],cgroup_peak_bytes=ref["reported_peak_cgroup_bytes"],
                    timeouts=ref["timed_out_requests"],errors=ref["error_requests"]))
    pivots={}
    if turns:
        pivot=min(turns,key=lambda t:t["margin_change"])
        t=pivot["t"]
        pivots=dict(t=t,change=pivot["margin_change"],from_t_minus_10=turns[max(0,t-10):t+1],
                    lookbacks={str(h): dict(first_t=max(0,t-h),prior_steps=min(t,h),includes_pivot_t=t,
                        cumulative_margin_change=sum(v["margin_change"] for v in turns[max(0,t-h):t+1]))
                        for h in (3,5,10)})
    losing_spans = {}
    for width in (3,5,10):
        if len(turns)>=width:
            candidates=((sum(turns[i+j]["margin_change"] for j in range(width)),i) for i in range(len(turns)-width+1))
            change,i=min(candidates)
            span=turns[i:i+width]
            both=[Counter(),Counter()]
            for turn in span:
                for s in (0,1):
                    both[s].update(turn["money"][s])
            losing_spans[str(width)]=dict(first_t=i,last_t=i+width-1,net_change=change,component_diff=balance(both,seat),
                full_command_and_state=span)
    official_margin = None if game["official_scores_by_seat"] is None else scores[seat]-scores[1-seat]
    physical_margin = sim.cash[seat]-sim.cash[1-seat]
    return dict(game=game,status=receipt["status"],winner=receipt.get("winner_seat"),scored=receipt["counted"],
        official_margin=official_margin,physical_margin=physical_margin,physical_start=initial,physical_end=list(sim.cash),
        official_minus_physical=None if official_margin is None else official_margin-physical_margin,
        money_by_period_by_seat=ledger,margin_components={k:balance(z,seat) for k,z in ledger.items()},period_steps=steps,
        largest_observed_cash_loss=pivots,largest_losing_spans=losing_spans,original_search=search(rows,1-seat),ladder_clock_by_seat=timing,
        whole_wall_seconds=receipt.get("whole_wall_seconds"),turns_sha256=receipt.get("turns_sha256"))


def unpack_archives(root, scratch):
    dirs=list(root.rglob("registration.json"))
    zips=list(sorted(root.rglob("*.zip"))) if root.is_dir() else [root]
    for i,z in enumerate(zips):
        with zipfile.ZipFile(z) as archive:
            regs=[n for n in archive.namelist() if Path(n).name=="registration.json"]
            for j,reg in enumerate(regs):
                dest=scratch/f"{i}-{j}"
                dest.mkdir()
                prefix=reg.removesuffix("registration.json")
                for name in ("registration.json","attempt.json","receipt.json","turns.jsonl"):
                    if prefix+name in archive.namelist():
                        (dest/name).write_bytes(archive.read(prefix+name))
                dirs.append(dest/"registration.json")
    return dirs


def analyze(root):
    matrix_file=LANE/"protocol.json"
    matrix=read(matrix_file)
    expected={(seed,s,a) for seed in matrix["seeds"] for s in (0,1) for a in ("student","parent")}
    games={}
    with tempfile.TemporaryDirectory() as temp:
        for reg in unpack_archives(root,Path(temp)):
            ticket=read(reg)
            if ticket.get("matrix_sha256")!=live.sha(matrix_file):
                continue
            key=(ticket["seed"],ticket["candidate_seat"],ticket["arm"])
            assert key in expected
            directory=reg.parent
            if key in games:
                assert live.sha(directory/"receipt.json")==games[key]["game"]["receipt_sha256"]
                continue
            v=verify_game(directory,*key,matrix_file)
            games[key]=settle_game(v,directory)
    if set(games)!=expected:
        raise ValueError(f"matrix incomplete, missing {sorted(expected-set(games))}")
    store(HERE/"settled_games.json.gz",dict(matrix_sha256=live.sha(matrix_file),program=live.sha(Path(__file__)),games=list(games.values())))
    return games


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("artifacts",type=Path)
    args=p.parse_args()
    out=analyze(args.artifacts)
    print([(k,r["official_margin"]) for k,r in sorted(out.items())])
