"""Reconcile the midgame checkpoints of each registered ticket against its signed joint actions."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
from itertools import islice
import json
from pathlib import Path
from statistics import mean
import zipfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "opponent_matched_economy/sapling.py"
spec = importlib.util.spec_from_file_location("matched_sapling_ledger", SOURCE)
sapling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sapling)
check = sapling.check
ARMS = ("parent", "uniform", "weighted")
PAIRS = (("weighted", "uniform"), ("weighted", "parent"), ("uniform", "parent"))
TICKS = (60, 100)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(g):
    if not g["counted"]:
        return "unresolved"
    if g["winner_seat"] is None:
        return "draw"
    return "win" if g["winner_seat"] == g["seat"] else "loss"


def state(players, totals):
    result = []
    for p, ledger in zip(players, totals, strict=True):
        result.append(dict(cash=p[0], trees=len(p[3]), normal_delivery=ledger["own_delivery"],
                           ground_rent=ledger["ground_rent"], tree_cargo_tax=ledger["cargo_tax"],
                           net_tree_operations=ledger["ground_rent"] + ledger["cargo_tax"] + ledger["plant"] + ledger["seize"]))
    return result


def profile_game(g, tape):
    n = g["official_transitions"]
    terminal = g["status"] == "complete"
    check(type(n) is int and 0 <= n <= len(tape), "Recorded transitions exceed actions")
    if terminal:
        check(n == len(tape) and n > 0, "Incomplete terminal action tape")
    replay = sapling.transitions(g, tape[:n])
    totals = [Counter(), Counter()]
    ticks = {str(t): None for t in TICKS}
    opening = None
    last = None
    try:
        # The complete path verifies the official terminal and both rewards at iterator exhaustion.
        # A failed request is outside the executed prefix and must never advance the simulator.
        for t, before, _, flow, after in replay if terminal else islice(replay, n):
            if opening is None:
                opening = [p[0] for p in before]
            if t in TICKS:
                ticks[str(t)] = state(before, totals)
            for s in (0, 1):
                sapling.ledger.check_equal(sum(flow[s].values()), after[s][0] - before[s][0], "One-turn bank flow")
                totals[s].update(flow[s])
            last = after
    finally:
        replay.close()
    if not terminal and n in TICKS and len(tape) > n and last is not None:
        # A signed but unexecuted next request attests that the checkpoint was reached alive.
        check(tape[n]["step"] == n, "Unexpected unexecuted request offset")
        ticks[str(n)] = state(last, totals)
    if last is not None:
        for s in (0, 1):
            sapling.ledger.check_equal(opening[s] + sum(totals[s].values()), last[s][0], "Full bank flow")
    scores = g["official_scores_by_seat"]
    adjustment = [scores[s] - last[s][0] for s in (0,1)] if scores is not None and terminal else None
    return dict(checkpoints=ticks, reached_transitions=n, entire_official_trajectory=terminal,
                last_executed_state=state(last, totals) if last is not None else None,
                signed_flows_by_seat=[dict(a) for a in totals] if n else None,
                official_minus_terminal_bank=adjustment)


def load_archive(path, block, games, plan, role_order, protocol_sha, run):
    check(digest(path) == block["sha256"], "Hosted artifact digest mismatch")
    seed, seat = block["seed"],block["seat"]
    eligible = [games[(seed, seat, a)] for a in role_order if games[(seed, seat, a)].get("receipt_sha256")]
    if not eligible:
        return
    with zipfile.ZipFile(path) as z:
        sequence = [json.loads(line) for line in z.read("execution.jsonl").splitlines()]
        check(sequence == block["host_sequence"] and [r["arm"] for r in sequence] == role_order,
              "Host execution order differs from registration")
        for g in eligible:
            a = g["arm"]
            seq = next(r for r in sequence if r["arm"] == a)
            raw_reg, raw_rec = [z.read(a+"/"+name) for name in ("registration.json","receipt.json")]
            reg, rec = json.loads(raw_reg), json.loads(raw_rec)
            check(seq["receipt_sha256"] == sapling.sha(raw_rec) == g["receipt_sha256"] and
                  seq["arm"] == reg["arm"] == a and reg["matrix_sha256"] == protocol_sha and
                  sapling.sha(raw_reg) == rec["protocol_sha256"] and str(reg["workflow_run"]) == str(run) and
                  (reg["seed"],reg["candidate_seat"],reg["opponent_seat"]) == (seed,seat,1-seat) and
                  (rec["seed"],rec["candidate_seat"]) == (seed,seat), "Unbound registration or game")
            check(reg["physical_map_sha256"] == g["terrain"] == plan["physical_boards"][str(seed)] and
                  reg["platform"] == plan["platform"], "Wrong platform or terrain")
            for role, file_key in (("candidate",a),("opponent","opponent"),("candidate_weights",a+"_weights")):
                check(reg["files"][role] == plan["files"][file_key], "Candidate, weight or opponent identity differs")
            check(all(rec.get(k) == g[k] for k in ("status","counted","winner_seat","official_scores_by_seat","official_transitions")),
                  "Receipt outcome differs from audited settlement")
            if "turns_sha256" in rec:
                raw_tape = z.read(a+"/turns.jsonl")
                check(sapling.sha(raw_tape) == rec["turns_sha256"], "Joint action tape differs")
                tape = [json.loads(line) for line in raw_tape.splitlines()]
            else:
                check(rec["official_transitions"] == 0 and a+"/turns.jsonl" not in z.namelist(), "Unbound partial action tape")
                tape = []
            g.update(profile_game(g,tape), archive_id=block["id"], archive_sha256=block["sha256"],
                     turns_sha256=rec.get("turns_sha256"), candidate_sha256=reg["files"]["candidate"]["sha256"])


def endpoints(g):
    seat = g["seat"]
    scores = g["official_scores_by_seat"]
    values = dict(official_margin=scores[seat]-scores[1-seat] if scores is not None else None,
                  won=None if not g["counted"] else int(g["winner_seat"] == seat))
    for t in TICKS:
        pair = g["checkpoints"][str(t)]
        for field in ("cash", "normal_delivery", "ground_rent", "trees", "net_tree_operations", "tree_cargo_tax"):
            values[f"t{t}_{field}_margin"] = pair[seat][field]-pair[1-seat][field] if pair else None
    return values


def complete_means(rows):
    return {k: mean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None for k in rows[0]}


def wdl(rows):
    counts=Counter(outcome(g) for g in rows)
    return dict(registered=len(rows),win=counts["win"],draw=counts["draw"],loss=counts["loss"],unresolved=counts["unresolved"],
                observed_win_fraction_over_registered=counts["win"]/len(rows))


def settlement(hosted, protocol):
    path = hosted/"settlement.json"
    doc, plan = [json.loads(p.read_bytes()) for p in (path,protocol)]
    protocol_sha=digest(protocol)
    schedule={(p["seed"],p["seat"]):p["order"] for p in plan["schedule"]}
    seeds=plan["seeds"]
    check(tuple(plan["arms"]) == ARMS and len(seeds)==len(set(seeds))==2 and set(schedule)=={(s,i) for s in seeds for i in (0,1)}
          and len(plan["schedule"]) == 4 and all(sorted(p)==sorted(ARMS) for p in schedule.values()),"Bad schedule")
    check(plan["files"]["opponent"]["sha256"] == sapling.NATIVE_SHA and
          doc["protocol_sha256"]==protocol_sha and doc["registered_games"]==plan["expected_games"]==12,"Unbound settlement")
    rows={}
    for g in doc["games"]:
        key=(g["seed"],g["seat"],g["arm"])
        check(key not in rows,"Duplicate ticket")
        rows[key]={k:g.get(k) for k in ("seed","seat","arm","terrain","counted","winner_seat","status","official_scores_by_seat",
                                         "official_transitions","receipt_sha256","reason")}
        rows[key].update(checkpoints={str(t):None for t in TICKS},entire_official_trajectory=False,official_minus_terminal_bank=None)
    check(set(rows)=={(s,i,a) for (s,i),arms in schedule.items() for a in arms},"Missing or extra registered row")
    seen=set()
    for block in doc["archives"]:
        key=(block["seed"],block["seat"])
        check(key in schedule and key not in seen,"Repeated or invalid block")
        seen.add(key)
        load_archive(hosted/(str(block["id"])+".zip"),block,rows,plan,schedule[key],protocol_sha,doc["workflow_run"])
    check(all(not g["receipt_sha256"] or "archive_id" in g for g in rows.values()),"Missing source receipt for a verified ticket")
    arms={}
    for a in ARMS:
        games=[g for g in rows.values() if g["arm"]==a]
        arms[a]=dict(all=wdl(games),by_seat={s:wdl([g for g in games if g["seat"]==s]) for s in (0,1)})
    comparisons={}
    for a,b in PAIRS:
        tickets={}
        for s in seeds:
            tickets[s]={}
            for seat in (0,1):
                one,two=[endpoints(rows[s,seat,k]) for k in (a,b)]
                tickets[s][seat]={k:one[k]-two[k] if one[k] is not None and two[k] is not None else None for k in one}
        maps={s:complete_means(list(tickets[s].values())) for s in seeds}
        comparisons[f"{a}_minus_{b}"]=dict(by_map_seat=tickets,by_map=maps,
            by_physical_seat={p:complete_means([tickets[s][p] for s in seeds]) for p in (0,1)},
            equal_map_mean=complete_means(list(maps.values())))
    return dict(schema="sapling-midgame-economic-development/accounting-v1",protocol_sha256=protocol_sha,
                settlement_sha256=digest(path),analyzer_sha256=digest(Path(__file__)),transitions_sha256=digest(SOURCE),
                workflow_run=doc["workflow_run"],commit=doc["commit"],records_reconciled=doc["records_reconciled"],
                missing_tickets=doc["missing_tickets"],verification_errors=doc["verification_errors"],
                scope="Two development maps, paired by physical seat. An absent checkpoint, official score or undecided outcome yields null for its incomplete contrast.",
                registered=12,formal_games=0,arms=arms,
                games=[rows[s,p,a] for s in seeds for p in (0,1) for a in schedule[s,p]],comparisons=comparisons)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hosted",type=Path,default=HERE/"hosted")
    ap.add_argument("--protocol",type=Path,default=HERE/"protocol.json")
    ap.add_argument("--out",type=Path)
    options=ap.parse_args()
    data=settlement(options.hosted,options.protocol)
    out=options.out or options.hosted/"economy.json.gz"
    raw=gzip.compress(json.dumps(data,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":")).encode(),mtime=0)
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(dir=out.parent,delete=False) as temp:
        temp.write(raw)
    Path(temp.name).replace(out)
    print(digest(out), data["arms"])
