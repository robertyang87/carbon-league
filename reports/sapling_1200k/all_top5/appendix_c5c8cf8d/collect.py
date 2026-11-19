"""Archive the frozen score-index delta and independently replay all genuinely new Top Five trajectories."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from appendix_batch import prepare
from appendix_32b555ee.collect import BATCHES as EARLIER, REV as OLD, scores
from validation_delta import run as validate
from appendix_f7109df6.report import build

REV="c5c8cf8dab041844d0d790c91ea9166c89c9e1d2"
BATCHES=(*EARLIER,"appendix_32b555ee")


def all_sources(rev):
    return {p for p in fetch.git("ls-tree","-r","--name-only",rev).decode().splitlines() if fetch.sourcefile(p)}


def read_score(path,rev,paths):
    return scores(fetch.git("show",rev+":"+path)) if path in paths else {}


def score_delta(manifests):
    prior={}
    for p in manifests:
        for g in fetch.load(p)["games"]:
            if g["id"] in prior:raise ValueError("overlapping prior game")
            prior[g["id"]]=(p.relative_to(TOP).parent.as_posix(),g)
    if len(prior)!=1421:raise ValueError("unexpected predecessor")
    paths={rev:all_sources(rev) for rev in (OLD,REV)}
    changes=set(fetch.git("diff","--name-only","--no-renames",OLD,REV).decode().splitlines()) & (paths[OLD]|paths[REV])
    details={};refs=defaultdict(list);cards={}
    for p in sorted(changes):
        a,b=[read_score(p,rev,paths[rev]) for rev in (OLD,REV)]
        if a.keys()-b.keys() or any(fetch.identity(a[k])!=fetch.identity(b[k]) for k in a.keys()&b.keys()):
            raise ValueError("a prior score was changed in "+p)
        added=b.keys()-a.keys()
        if added:details[p]={"before":len(a),"after":len(b),"added_references":len(added)}
        for k in sorted(added):
            if k in cards and fetch.identity(cards[k])!=fetch.identity(b[k]):raise ValueError("contradictory new score")
            cards[k]=b[k];refs[k].append(p)
    prior_sources=defaultdict(list);scored_top=set()
    for p in sorted(paths[OLD]):
        for k,c in read_score(p,OLD,paths[OLD]).items():
            if c["name_a"] in fetch.NAMES or c["name_b"] in fetch.NAMES:scored_top.add(k)
            if k in refs:
                prior_sources[k].append(p)
                if fetch.identity(c)!=fetch.identity(cards[k]):raise ValueError("previous mirror differs")
    if scored_top!=set(prior):raise ValueError("previous all-index census and archival membership differ")
    rows=[]
    for k,c in sorted(cards.items()):
        names=[c["name_"+s] for s in c["order"]];points=[c["scores"][s] for s in c["order"]]
        ranks=[fetch.NAMES.get(n) for n in names];eligible=any(ranks)
        if k in prior and fetch.identity(c)!=fetch.identity(prior[k][1]["card"]):raise ValueError("archived card differs")
        rows.append({"id":k,"physical_seats":names,"physical_top5_ids":ranks,"score_by_seat":points,"added_sources":refs[k],
                     "previous_score_sources":prior_sources[k],"previous_raw_batch":prior[k][0] if k in prior else None,
                     "class_of_id":("previous_top5" if k in prior else "new_top5") if eligible else "outside_top5",
                     "cohort":"mutual" if all(ranks) else "external" if eligible else "outside"})
    unique=Counter(r["class_of_id"] for r in rows)
    reference=Counter(c for r in rows for c in [r["class_of_id"]]*len(r["added_sources"]))
    report={"previous_score_revision":OLD,"revision":REV,"source":"发生新增的全仓成绩索引合计","index_sources":details,
            "old_reference_count":sum(d["before"] for d in details.values()),"new_reference_count":sum(d["after"] for d in details.values()),
            "distinct_ids_by_class":dict(unique),"source_references_by_class":dict(reference),"added_source_references":sum(reference.values()),
            "distinct_added_reference_ids":len(rows),"within_delta_repeated_references":sum(reference.values())-len(rows),
            "previous_distinct_games":len(prior),"new_distinct_games":unique["new_top5"],"existing_references":reference["previous_top5"],
            "non_top_five_refs":reference["outside_top5"],"new_mutual_games":sum(r["class_of_id"]=="new_top5" and r["cohort"]=="mutual" for r in rows),
            "union_games":len(prior)+unique["new_top5"],"added":rows}
    fetch.atomic(HERE/"score_delta.json",(json.dumps(report,ensure_ascii=False,indent=1)+"\n").encode())
    print({k:v for k,v in report.items() if k not in ("index_sources","added")},flush=True)
    return report


def enrich(m,d):
    build(HERE)
    old=fetch.load(TOP/"appendix_32b555ee/validation.json")
    totals=Counter()
    members=Counter()
    for batch in BATCHES:
        for g in fetch.load(TOP/batch/"manifest.json")["games"]:totals[g["cohort"]]+=1
    for g in m["games"]:
        totals[g["cohort"]]+=1
        for s,reg in enumerate(g["seat_strategy_ids"]):
            if reg is None:continue
            x,y=g["score_by_seat"][s],g["score_by_seat"][1-s]
            members[g["seat_names"][s],s,"W" if x>y else "L" if x<y else "D"]+=1
    if sum(totals.values())!=d["union_games"] or old["union_distinct"]["id"]!=d["previous_distinct_games"]:
        raise ValueError("scope mismatch")
    rule=[f"本批 {name} 席 {s} 为 {members[name,s,'W']} 胜、{members[name,s,'D']} 平、{members[name,s,'L']} 负。" for name in sorted({n for n,s,w in members}) for s in (0,1)]
    p=HERE/"README.md";lines=p.read_text().splitlines();bins=d["distinct_ids_by_class"]
    lines[2]=(f"从 `{OLD[:8]}` 的 {d['previous_distinct_games']:,} 局查到 `{REV[:8]}`。"
              f"扫过全部冻结索引，{len(d['index_sources'])} 份成绩索引产生 {d['added_source_references']} 条新增引用，"
              f"其中 {d['within_delta_repeated_references']} 条在同一增量内重复。按比赛 ID 与局号去重后，"
              f"{bins.get('previous_top5',0)} 个已归档，{bins.get('outside_top5',0)} 个双方均在冻结前五之外，新增前五 {len(m['games'])} 局。"
              f"累计归档 {d['union_games']:,} 局，前五互打 {totals['mutual']} 局、对外 {totals['external']:,} 局。"
              "[完整逐条去重与来源](score_delta.json) · [物理席与原件身份](manifest.json)")
    lines[3:3]=["","".join(rule)]
    p.write_text("\n".join(lines)+"\n")


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("census","collect","analyze","report"))
    stage=parser.parse_args().stage
    manifests=[TOP/b/"manifest.json" for b in BATCHES];m=HERE/"manifest.json"
    if stage in ("census","collect"):
        d=score_delta(manifests)
        if stage=="collect":
            prepare(REV,manifests,HERE,(d["new_distinct_games"],d["new_mutual_games"]))
        return
    d=fetch.load(HERE/"score_delta.json");batch=fetch.load(m)
    if {g["id"] for g in batch["games"]}!={r["id"] for r in d["added"] if r["class_of_id"]=="new_top5"}:raise ValueError("independent census mismatch")
    if stage=="analyze":
        parents=[]
        for b in BATCHES:
            path=TOP/b/"validation_maps.jsonl.gz";witness=fetch.git("show",REV+":"+path.relative_to(fetch.ROOT).as_posix())
            if witness!=path.read_bytes():raise ValueError("prior terrain ledger changed")
            parents.append((b,fetch.sha(witness)))
        validate(fetch.CACHE,HERE,REV,OLD,parents)
        def run(p,*args):subprocess.run([sys.executable,str(p),*map(str,args)],check=True)
        run(TOP/"appendix_499886f1/cases.py","--manifest",m,"--cache",fetch.CACHE)
        run(TOP/"appendix_0137949d/cash.py","--manifest",m,"--cache",fetch.CACHE)
        idx=TOP/f"context_{HERE.name}_index.tsv.gz"
        run(TOP/"context_catalog.py","--manifest",m,"--output",idx)
        run(TOP/"context_audit.py","--index",idx,"--validation",HERE/"validation_games.jsonl.gz",
            "--out",TOP/f"context_{HERE.name}_receipt.json","--jobs","1")
    enrich(batch,d)


if __name__=="__main__":main()
