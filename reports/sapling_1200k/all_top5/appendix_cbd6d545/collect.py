"""Supplement the 1,440 archived Top Five games at the frozen November 19 index revision."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from appendix_batch import prepare
from appendix_39a745e0.collect import BATCHES as EARLIER, REV as OLD, all_sources, read_score
from appendix_f7109df6.report import build
from validation_delta import run as validate

REV="cbd6d545491bc4f6f8fcbb3174628bdef66245f4"
BATCHES=(*EARLIER,"appendix_39a745e0")


def census():
    archives={}
    for b in BATCHES:
        for g in fetch.load(TOP/b/"manifest.json")["games"]:
            if g["id"] in archives:raise ValueError("old archive overlap")
            archives[g["id"]]=(b,g)
    if len(archives)!=1440:raise ValueError("expected 1440 historical maps")
    trees={r:all_sources(r) for r in (OLD,REV)}
    changed=set(fetch.git("diff","--name-only","--no-renames",OLD,REV).decode().splitlines())
    changed&=trees[OLD]|trees[REV]
    changes={};added=defaultdict(list);cards={}
    for p in sorted(changed):
        before,after=[read_score(p,r,trees[r]) for r in (OLD,REV)]
        # Unchanged out-of-cohort diagnostics can contain a null penalty score.
        # Preserve them byte-for-field; all newly referenced cards still need valid seat scores.
        if not before.keys()<=after.keys() or any(after[k] != v and fetch.identity(after[k]) != fetch.identity(v) for k,v in before.items()):
            raise ValueError(f"previous score disappeared or changed: {p}")
        keys=after.keys()-before.keys()
        if not keys:continue
        changes[p]={"before":len(before),"after":len(after),"added":len(keys)}
        for k in keys:
            if k in cards and fetch.identity(cards[k])!=fetch.identity(after[k]):raise ValueError("new cross-index mirror mismatch")
            cards[k]=after[k];added[k].append(p)
    old_at=defaultdict(list);oldtop=set()
    for p in sorted(trees[OLD]):
        for k,c in read_score(p,OLD,trees[OLD]).items():
            if k in added:
                if fetch.identity(c)!=fetch.identity(cards[k]):raise ValueError("old mirror mismatch")
                old_at[k].append(p)
            if {c["name_a"],c["name_b"]}&fetch.NAMES.keys():oldtop.add(k)
    if oldtop!=archives.keys():raise ValueError("mismatch between all prior scores and sealed batch")
    rows=[]
    for k,c in sorted(cards.items()):
        names=[c["name_"+s] for s in c["order"]];points=[c["scores"][s] for s in c["order"]]
        identities=[fetch.NAMES.get(n) for n in names]
        if k in archives and fetch.identity(c)!=fetch.identity(archives[k][1]["card"]):raise ValueError("old archive mismatch")
        rows.append(dict(id=k,physical_seats=names,physical_top5_ids=identities,score_by_seat=points,added_sources=added[k],
                         previous_score_sources=old_at[k],previous_raw_batch=archives[k][0] if k in archives else None,
                         class_of_id="old_top5" if k in archives else "new_top5" if any(identities) else "outside_top5",
                         cohort="mutual" if all(identities) else "external" if any(identities) else "outside"))
    bins=Counter(r["class_of_id"] for r in rows);mentions=Counter()
    for r in rows:mentions[r["class_of_id"]]+=len(r["added_sources"])
    doc=dict(previous_score_revision=OLD,revision=REV,previous_distinct_games=len(archives),index_sources=changes,
             old_reference_count=sum(d["before"] for d in changes.values()),new_reference_count=sum(d["after"] for d in changes.values()),
             source="发生新增的全仓成绩索引",added_source_references=sum(mentions.values()),distinct_added_reference_ids=len(rows),
             within_delta_repeated_references=sum(mentions.values())-len(rows),distinct_ids_by_class=dict(bins),
             source_references_by_class=dict(mentions),existing_references=mentions["old_top5"],non_top_five_refs=mentions["outside_top5"],
             new_distinct_games=bins["new_top5"],new_mutual_games=sum(r["cohort"]=="mutual" and r["class_of_id"]=="new_top5" for r in rows),
             union_games=len(archives)+bins["new_top5"],added=rows)
    fetch.atomic(HERE/"score_delta.json",(json.dumps(doc,indent=1,ensure_ascii=False)+"\n").encode())
    print({k:v for k,v in doc.items() if k not in ("added","index_sources")},flush=True)
    return doc


def render(d,m):
    build(HERE)
    groups=Counter(g["cohort"] for b in BATCHES for g in fetch.load(TOP/b/"manifest.json")["games"])
    names=defaultdict(lambda:[Counter(),Counter()])
    for g in m["games"]:
        groups[g["cohort"]]+=1
        for s,reg in enumerate(g["seat_strategy_ids"]):
            if not reg:continue
            x,y=g["score_by_seat"][s],g["score_by_seat"][1-s]
            names[g["seat_names"][s]][s]["W" if x>y else "D" if x==y else "L"]+=1
    p=HERE/"README.md";lines=p.read_text().splitlines();bins=d["distinct_ids_by_class"]
    lines[2]=(f"从 `{OLD[:8]}` 的 {d['previous_distinct_games']:,} 局查到 `{REV[:8]}`。{len(d['index_sources'])} 份成绩索引新增 "
              f"{d['added_source_references']} 次引用，本批跨索引重报 {d['within_delta_repeated_references']} 次。"
              f"合并后有 {d['distinct_added_reference_ids']} 个对局 ID，其中 {bins.get('old_top5',0)} 个此前已归档，"
              f"{bins.get('outside_top5',0)} 个双方均在冻结前五之外，实增前五 {len(m['games'])} 局。"
              f"累计 {sum(groups.values()):,} 局，互打 {groups['mutual']} 局、对外 {groups['external']:,} 局。"
              "[逐条去重与索引](score_delta.json) · [原始回放身份](manifest.json)")
    lines[3:3]=["","；".join(n+" "+"，".join(f"席 {s} 为 {row['W']} 胜、{row['D']} 平、{row['L']} 负" for s,row in enumerate(seats))
                            for n,seats in sorted(names.items()))+"。"]
    p.write_text("\n".join(lines)+"\n")


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("census","collect","analyze"));stage=parser.parse_args().stage
    manifests=[TOP/b/"manifest.json" for b in BATCHES];m=HERE/"manifest.json"
    if stage in ("census","collect"):
        d=census()
        if stage=="collect":prepare(REV,manifests,HERE,(d["new_distinct_games"],d["new_mutual_games"]))
        return
    d=fetch.load(HERE/"score_delta.json");batch=fetch.load(m)
    if {g["id"] for g in batch["games"]}!={r["id"] for r in d["added"] if r["class_of_id"]=="new_top5"}:raise ValueError("independent census mismatch")
    parents=[]
    for b in BATCHES:
        p=TOP/b/"validation_maps.jsonl.gz";blob=fetch.git("show",REV+":"+p.relative_to(fetch.ROOT).as_posix())
        if p.read_bytes()!=blob:raise ValueError("historical map ledger changed")
        parents.append((b,fetch.sha(blob)))
    validate(fetch.CACHE,HERE,REV,OLD,parents)
    def run(path,*args):subprocess.run([sys.executable,str(path),*map(str,args)],check=True)
    run(TOP/"appendix_499886f1/cases.py","--manifest",m,"--cache",fetch.CACHE)
    run(TOP/"appendix_0137949d/cash.py","--manifest",m,"--cache",fetch.CACHE)
    idx=TOP/f"context_{HERE.name}_index.tsv.gz"
    run(TOP/"context_catalog.py","--manifest",m,"--output",idx)
    run(TOP/"context_audit.py","--index",idx,"--validation",HERE/"validation_games.jsonl.gz","--out",TOP/f"context_{HERE.name}_receipt.json","--jobs","1")
    render(d,batch)
    run(HERE / "contrasts.py")
    run(TOP / "future_economy_check.py", "--replay-cache", fetch.CACHE / "replays",
        "--expect-games", d["union_games"], "--out", HERE / "future_economy_check.json")
    page = HERE / "README.md"
    with page.open("a") as stream:
        stream.write("\n同名 Krypton-Pro 在两张地图分别得 6,087.49 和 5.50，Master6 另以 238.71 分负于 it720。三局并列核对首工、实收、招工和关键时段的双方命令。[逐局现金与动作对照](contrasts.md) · [未来 10/30 步实收标签核验](future_economy_check.json)\n")


if __name__=="__main__":main()
