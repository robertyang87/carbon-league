"""Independently resolve chosen actions and historical returns, including separate repeat streams."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "joint_response_critic_dev"


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def margin_change(row):
    s = row["seat"]
    old, new = (row[k] for k in ("parent_scores","alternative_scores"))
    return (new[s] - new[1-s]) - (old[s] - old[1-s])


def check(test, detail):
    if not test:
        raise ValueError(detail)


def main():
    bank = read(SOURCE / "signed_roots.json.gz")
    scored = read(HERE / "native_scores.json.gz")
    by_case = {r["case"]:r for r in bank["rows"]}
    by_identity = {r["identity"]:r for r in bank["rows"]}
    scores = {r["case"]:r for r in scored["rows"]}
    check(scores.keys() == by_case.keys() and len(scores) == 108, "registration mismatch")
    native_result = read(HERE/"result.json")
    by_split = defaultdict(lambda: defaultdict(list))
    for case, r in scores.items():
        s = by_case[case]
        limit = min(8, 299 - s["t"])
        full = min(r["depth"]) == limit and max(r["depth"]) == limit
        choose = bool(full and r["native_q"][1] > r["native_q"][0])
        check(r["full"] == full and r["choose"] == choose and r["public_root_sha256"] == s["public_root_sha256"], case)
        for q, terminal in zip(r["native_q"],r["terminal"]):
            m = terminal["simulated_margin"]
            exact = min(2.5, max(-2.5, m / 3000)) + 0.5 * ((m>0)-(m<0))
            check(abs(q-exact)<1e-7, case + " terminal value")
        by_split[r["split"]][s["terrain"]].append((choose, margin_change(s)))
    for split, maps in by_split.items():
        got = mean(mean(a*b for a,b in rows) for rows in maps.values())
        check(abs(got - native_result["split"][split]["mean_per_map"])<1e-7, split)
    groups = defaultdict(list)
    for rep in bank["repeated_streams"]:
        primary = by_identity[rep["identity"]]
        check(rep["public_root_sha256"] == primary["public_root_sha256"] and
              rep["parent"] == primary["parent"] and rep["alternative"] == primary["alternative"] and
              rep["root_opponent"] == primary["root_opponent"], "repeat root identity")
        groups[primary["case"]].append(rep)
    records=[]
    for case, repeats in groups.items():
        a = by_case[case]; scored_row = scores[case]
        deltas = [margin_change(r) for r in [a,*repeats]]
        records.append(dict(seed=a["seed"],t=a["t"],seat=a["seat"],family=a["family"],case=case,
                            primary=deltas[0],repeats=deltas[1:],opposite_sign=min(deltas)<0<max(deltas),
                            native_choice=scored_row["choose"],q=scored_row["native_q"]))
    receipt={"scores_sha256":hashlib.sha256((HERE/"native_scores.json.gz").read_bytes()).hexdigest(),
             "registered_cases":len(scores),"verified_terminal_value_cases":sum(bool(r["terminal"]) for r in scores.values()),
             "repeat_sets":records,"opposite_sign_count":sum(v["opposite_sign"] for v in records),
             "scope":"Retrospective selected roots with common simultaneous orders and subsequent natural responses. Exclude repeat streams from primary metric."}
    print(json.dumps(receipt, indent=2,allow_nan=False))


if __name__=="__main__":
    main()
