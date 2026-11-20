"""Summarize matched work and observed search behavior without assigning game strength."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ARMS = ("parent", "candidate")


def totals(rows):
    report = {}
    for unit in ("wall_s", "cpu_s"):
        times = {a: [median(x[unit] for x in r["fixed"]["times"][a]) for r in rows] for a in ARMS}
        sums = {a: sum(v) for a,v in times.items()}
        report[unit] = dict(sum_medians=sums,
                            fraction_saved=1-sums["candidate"]/sums["parent"],
                            faster_roots=sum(c<p for c,p in zip(times["candidate"],times["parent"])))
    return report


def adaptive(rows):
    summary = {}
    for arm in ARMS:
        cases=[seq[-1] for row in rows for seq in row["adaptive"][arm]]
        summary[arm]=dict(n=len(cases), mean_candidates=sum(x["candidates"] for x in cases)/len(cases),
                          mean_depth=sum(x["depth"] for x in cases)/len(cases),
                          full_depth=sum(x["depth"]==8 for x in cases),zero_depth=sum(x["depth"]==0 for x in cases),
                          searched_changes=sum((x["best"] or 0)!=0 for x in cases),
                          external_wall_max=max(x["wall_s"] for x in cases),
                          root_median_wall=median(x["wall_s"] for x in cases),
                          own_repeats_disagree=sum(r["adaptive"][arm][0][-1]["issued"]!=r["adaptive"][arm][1][-1]["issued"] for r in rows))
    stable=[]
    by_root=[]
    for r in rows:
        finals={a:[s[-1] for s in r["adaptive"][a]] for a in ARMS}
        different=(all(x[0]["issued"]==x[1]["issued"] for x in finals.values())
                   and finals["parent"][0]["issued"]!=finals["candidate"][0]["issued"])
        if different:stable.append([r["seed"],r["seat"],r["t"]])
        by_root.append([r["seed"],r["seat"],r["t"],{a:[[x["candidates"],x["depth"],x["best"]] for x in seqs] for a,seqs in finals.items()},different])
    return dict(arms=summary,stable_changed=stable,by_root=by_root)


def hosted():
    source=HERE.parent/"ladder/cargo_defense/development/hosted"
    paths=sorted(source.glob("*.zip"))
    if len(paths)!=4:raise ValueError("Expected four prior hosted receipts")
    stats=[];records=[];missing_parent=0;candidate_times=[]
    for p in paths:
        with ZipFile(p) as z:
            rec=json.load(z.open("parent/receipt.json"))
            seat=rec["candidate_seat"]
            records.append(dict(archive=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seed=rec["seed"],candidate_seat=seat))
            assert rec["official_transitions"]==299
            for raw in z.open("parent/turns.jsonl"):
                row=json.loads(raw)
                parent=row["calls"][seat]["response"]
                candidate_times.append(parent["public_wall_seconds"])
                missing_parent+=int("search_last" not in parent)
                stats.append((row["step"],row["calls"][1-seat]["response"]["search_last"]))
    n=sum(step<=291 and s["candidates"]>1 for step,s in stats)
    return dict(source="Four parent-v-original games, both on separately applied 1.5 CPU/512 MiB. Search work telemetry only for original examiner.",
                archives=records,all_calls=len(stats),parent_calls_without_search_last=missing_parent,parent_public_wall_median=median(candidate_times),
                deadline_eligible_non_single=n,deadline_eligible_full=sum(s["depth"]==8 and step<=291 and s["candidates"]>1 for step,s in stats),
                true_zero_after_admission=sum(s["depth"]==0 and s["candidates"]>1 for _,s in stats),
                candidate_count=dict(Counter(s["candidates"] for _,s in stats)),depth=dict(Counter(s["depth"] for _,s in stats)))


def main():
    with gzip.open(HERE/"measurements.json.gz") as data:rows=json.load(data)["rows"]
    assert len(rows)==12 and all(r["fixed"]["bitwise_equal"] for r in rows)
    for row in rows:
        assert all(len(row["adaptive"][a])==2 and len(row["fixed"]["times"][a])==2 for a in ARMS)
    other=[r for r in rows if not r["pilot"]]
    report=dict(all_fixed=totals(rows),nonpilot_fixed=totals(other),all_adaptive=adaptive(rows),nonpilot_adaptive=adaptive(other),historical_host=hosted(),
                neural_passes=sum(len(r["fixed"]["features"]) for r in rows),neural_input_rows=sum(n for r in rows for n,_ in r["fixed"]["features"]))
    if (HERE/"repeat.json.gz").exists():
        with gzip.open(HERE/"repeat.json.gz") as data:second=json.load(data)
        assert len(second["rows"])==12 and second["first_study_sha256"]==hashlib.sha256((HERE/"measurements.json.gz").read_bytes()).hexdigest()
        assert all(len(r["fixed"]["times"][a])==5 for r in second["rows"] for a in ARMS)
        report["second_fixed_all"]=totals(second["rows"])
        report["second_fixed_nonpilot"]=totals([r for r in second["rows"] if not r["pilot"]])
    (HERE/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report["all_fixed"],report["nonpilot_fixed"],report["all_adaptive"]["arms"],report["all_adaptive"]["stable_changed"],flush=True)
if __name__=="__main__":
    main()
