"""Reconcile the four cold-repair development games against the V1 official cash ledger."""
from collections import Counter
import gzip
import importlib.util
import json
import zipfile

import matrix as m

spec = importlib.util.spec_from_file_location("dev1_accounts", m.FAMILY / "dev1/economy.py")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


def collision_flows(rows):
    totals = [Counter(), Counter()]
    for row in rows:
        direct = {event[3]: event[0] for event in row["cash_events"]
                  if event[0] in ("tree_collision_cash", "base_collision_cash") and event[2] > 0}
        for wid, lost, cargo, recipient, survivor in row["collisions"]:
            totals[lost]["worker_deaths"] += 1
            if cargo <= 0:
                continue
            totals[lost]["laden_deaths"] += 1
            if recipient is None:
                destination = "ground_before_cap"
            else:
                relation = "friendly" if recipient == lost else "enemy"
                destination = relation + "_" + ("collector" if survivor is not None else direct[wid])
            totals[lost][destination] += cargo
    return [{key:round(value, 5) for key,value in sorted(counter.items())} for counter in totals]


def late_request_costs(archive, member):
    turns = [json.loads(line) for line in archive.read(member + "/turns.jsonl").splitlines()]
    rows = {}
    for root in (279, 282, 285):
        calls = [call for turn in turns[root:] for call in turn["calls"]]
        rows[root] = dict(joint_transitions=len(turns)-root, public_requests=len(calls),
                          both_requests_serial_seconds=sum(call["elapsed"] for call in calls),
                          both_public_cpu_seconds=sum(call["response"]["public_cpu_seconds"] for call in calls))
    return rows


def main():
    hosted = m.HERE / "hosted"
    settlement = m.read(hosted / "settlement.json")
    m.require(settlement["records_reconciled"] and settlement["protocol_sha256"] == m.sha(m.PLAN), "unknown archive")
    schedule = {(ticket["seed"], ticket["seat"]) for ticket in m.read(m.PLAN)["schedule"]}
    games = settlement["games"]
    m.require(len(schedule) == len(games) == 4 and {(g["seed"], g["seat"]) for g in games} == schedule, "incomplete draw")
    digests = {a["id"]: a["sha256"] for a in settlement["archives"]}
    cfg, books = v1.ledger.config(), []
    for game in games:
        archive = hosted / game["archive_path"]
        m.require(m.sha(archive) == digests[game["archive_id"]], "archive hash")
        with zipfile.ZipFile(archive) as z:
            book = v1.ledger.replay(game, z, cfg)
            book["archived_suffix_cost_by_root"] = late_request_costs(z, game["arm"])
        rows = book.pop("transitions")
        seat = book["seat"]
        accounts = book["signed_account_by_seat"]
        book["cash_margin_components"] = {
            name: sum(accounts[seat].get(k,0)-accounts[1-seat].get(k,0) for k in fields)
            for name,fields in v1.GROUPS.items()}
        v1.ledger.check_equal(sum(book["cash_margin_components"].values()), book["official_margin"], "terminal score")
        book["collision_flows_by_physical_seat"] = collision_flows(rows)
        used = set()
        book["turn_references"] = v1.prior.trace_pivots(book["cash_margin_by_state"], {m.ARM: used})
        book["selected_turns"] = {t:rows[t] for t in sorted(used)}
        book["archive_sha256"] = digests[game["archive_id"]]
        books.append(book)
    output = dict(schema="cold-repair-actual-cash/1", settlement_sha256=m.sha(hosted/"settlement.json"),
                  protocol_sha256=m.sha(m.PLAN), games=books)
    target = hosted / "economy.json.gz"
    temp = target.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(json.dumps(output,sort_keys=True,allow_nan=False,separators=(",",":")).encode(),mtime=0))
    temp.replace(target)


if __name__ == "__main__":
    main()
