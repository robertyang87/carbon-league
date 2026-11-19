"""Reconcile changed score appearances, seal unique Top 5 games and check both simulators for this cutoff."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
sys.path.insert(0, str(TOP))
import fetch
from appendix_batch import prepare
from appendix_1fdacb1b.collect import BATCHES as BEFORE_PREVIOUS, REV as PREVIOUS
from validation_delta import run as validate

REV = "32b555ee1e31c8dbdccf81edc8d84c2f3e6cf965"
BATCHES = (*BEFORE_PREVIOUS, "appendix_1fdacb1b")


def scores(raw):
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    seen = {}
    for r in fetch.walk_matches(json.loads(raw)):
        if r.get("state","done") != "done":
            continue
        k = fetch.key_of(r)
        if k in seen and fetch.identity(seen[k]) != fetch.identity(r):
            raise ValueError(f"contradiction within one index: {k}")
        seen[k] = r
    return seen


def census():
    known = {}
    for b in BATCHES:
        for g in fetch.load(TOP / b / "manifest.json")["games"]:
            if g["id"] in known:
                raise ValueError("prior archives overlap")
            known[g["id"]] = (b, g)
    if len(known) != 1407:
        raise ValueError("1407 prior distinct maps required")
    changed = fetch.git("diff","--name-status","--no-renames",PREVIOUS,REV).decode().splitlines()
    refs, counts, cards = defaultdict(list), {}, {}
    for line in changed:
        status, path = line.split("\t")
        if not fetch.sourcefile(path):
            continue
        old = {} if status == "A" else scores(fetch.git("show", PREVIOUS + ":" + path))
        new = {} if status == "D" else scores(fetch.git("show", REV + ":" + path))
        if old.keys()-new.keys() or any(fetch.identity(old[k]) != fetch.identity(new[k]) for k in old.keys() & new.keys()):
            raise ValueError(f"a prior identity was removed or changed: {path}")
        added = sorted(new.keys()-old.keys())
        if not added and not old and not new:
            continue
        counts[path] = dict(before=len(old),after=len(new),new_references=len(added))
        for k in added:
            if k in cards and fetch.identity(cards[k]) != fetch.identity(new[k]):
                raise ValueError(f"contradictory mirror: {k}")
            cards[k] = new[k]
            refs[k].append(path)
    # Check whether an apparently new reference already existed in another of the repository's source indexes.
    # Read historical Git blobs once. Never use a current working tree to derive the score cut.
    oldsources = defaultdict(list)
    seen_top = set()
    for path in fetch.git("ls-tree","--name-only","-r",PREVIOUS).decode().splitlines():
        if not fetch.sourcefile(path):
            continue
        for k,r in scores(fetch.git("show",PREVIOUS+":"+path)).items():
            if {r["name_a"],r["name_b"]} & set(fetch.NAMES):
                seen_top.add(k)
            if k in refs:
                if fetch.identity(cards[k]) != fetch.identity(r):
                    raise ValueError(f"old index mirror mismatch: {k}")
                oldsources[k].append(path)
    if seen_top != set(known):
        raise ValueError("before snapshot and archived identities differ")
    rows = []
    for k,c in sorted(cards.items()):
        seats = [c["name_"+s] for s in c["order"]]
        bank = [c["scores"][s] for s in c["order"]]
        ranks = [fetch.NAMES.get(n) for n in seats]
        eligible = any(ranks)
        if eligible and bool(oldsources[k]) != (k in known):
            raise ValueError("archival mirror mismatch")
        if k in known and fetch.identity(known[k][1]["card"]) != fetch.identity(c):
            raise ValueError("prior physical record mismatch")
        label = "new_top5" if eligible and k not in known else "prior_top5_mirror" if eligible else "other_match"
        rows.append(dict(id=k,physical_seats=seats,physical_strategy_ids_if_top5=ranks,score_by_seat=bank,
                         added_sources=refs[k],before_sources=oldsources[k],previously_archived_top5_batch=known[k][0] if k in known else None,
                         class_of_id=label,also_referenced_in_new_sources=len(refs[k])-1,cohort="mutual" if all(ranks) else "external" if eligible else "other",
                         master6_result=next((("W" if bank[s]>bank[1-s] else "L" if bank[s]<bank[1-s] else "D")
                            for s,n in enumerate(seats) if n=="claude-AlphaCarbon-Master6"),None)))
    new = [r for r in rows if r["class_of_id"]=="new_top5"]
    bins = Counter(r["class_of_id"] for r in rows)
    before_refs=sum(t["before"] for t in counts.values())
    after_refs=sum(t["after"] for t in counts.values())
    doc=dict(previous_score_revision=PREVIOUS,revision=REV,previous_distinct_games=len(known),
             index_sources=counts,source="各有新增的成绩索引合计",old_reference_count=before_refs,
             new_reference_count=after_refs,added_source_references=sum(len(r["added_sources"]) for r in rows),
             distinct_added_reference_ids=len(rows),distinct_ids_by_class=dict(bins),
             added_references_by_class=dict(sum((Counter({r["class_of_id"]:len(r["added_sources"])}) for r in rows),Counter())),
             mirror_extras_within_interval=sum(r["also_referenced_in_new_sources"] for r in rows),
             existing_references=sum(len(r["added_sources"]) for r in rows if r["class_of_id"]=="prior_top5_mirror"),
             non_top_five_refs=sum(len(r["added_sources"]) for r in rows if r["class_of_id"]=="other_match"),
             new_distinct_games=len(new),new_mutual_games=sum(r["cohort"]=="mutual" for r in new),union_games=len(known)+len(new),
             new_master6_results=dict(Counter(r["master6_result"] for r in new if r["master6_result"])),added=rows)
    fetch.atomic(HERE / "score_delta.json",(json.dumps(doc,ensure_ascii=False,indent=1)+"\n").encode())
    return doc


def report(manifests,m,delta):
    aggregate=Counter()
    master="claude-AlphaCarbon-Master6"
    for manifest in [*manifests,m]:
        for g in fetch.load(manifest)["games"]:
            if master not in g["seat_names"]:continue
            s=g["seat_names"].index(master)
            ours,theirs=g["score_by_seat"][s],g["score_by_seat"][1-s]
            aggregate["W" if ours>theirs else "L" if ours<theirs else "D"]+=1
    new=fetch.load(m)["games"]
    physical=[Counter(),Counter()]
    for g in new:
        if master not in g["seat_names"]:continue
        seat=g["seat_names"].index(master)
        a,b=g["score_by_seat"][seat],g["score_by_seat"][1-seat]
        physical[seat]["W" if a>b else "L" if a<b else "D"]+=1
    if physical[0]+physical[1]!=Counter(delta["new_master6_results"]):raise ValueError("seatwise WDL mismatch")
    by_seat="，".join(f"席 {seat} 有 {row['W']} 胜、{row['D']} 平、{row['L']} 负" for seat,row in enumerate(physical))
    report=HERE/"README.md"
    lines=report.read_text().splitlines()
    lines[2]=(f"固定上一批的 1,407 张独立地图，核对到 `{REV[:8]}` 时改变的 {len(delta['index_sources'])} 份成绩索引。"
              f"新增 {delta['added_source_references']} 次引用，合并本批 {delta['mirror_extras_within_interval']} 次跨索引重报后有 "
              f"{delta['distinct_added_reference_ids']} 个对局 ID。其中 {delta['existing_references']} 个已归档，"
              f"{delta['non_top_five_refs']} 个双方均在冻结前五之外，实际新归档 {len(new)} 局。"
              "累计前五 1,421 局，其中互打 115 局、对外 1,306 局。[逐条去重与各索引路径](score_delta.json) · [原始回放身份](manifest.json)")
    lines[3:3]=["",f"本批 Master6 {by_seat}。"
                 f"合并历史，它共 {sum(aggregate.values())} 场，{aggregate['W']} 胜、{aggregate['D']} 平、{aggregate['L']} 负。"
                 "这些成绩来自实际天梯配对，不能当作 Sapling 对齐强度的直接评估。"]
    report.write_text("\n".join(lines)+"\n")



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("census","collect","analyze","context"))
    stage=parser.parse_args().stage
    manifests=[TOP / b / "manifest.json" for b in BATCHES]
    if stage in ("census","collect"):
        delta=census()
        print({k:v for k,v in delta.items() if k not in ("added","index_sources")},flush=True)
        if stage=="collect":
            prepare(REV,manifests,HERE,expected=(delta["new_distinct_games"],delta["new_mutual_games"]))
    else:
        delta=fetch.load(HERE/"score_delta.json")
        m=HERE/"manifest.json"
        if {r["id"] for r in fetch.load(m)["games"]}!={r["id"] for r in delta["added"] if r["class_of_id"]=="new_top5"}:
            raise ValueError("full-index census does not agree with delta")
        def run(path,*flags):
            subprocess.run([sys.executable,str(path),*map(str,flags)],check=True)
        if stage=="analyze":
            parents=[]
            for b in BATCHES:
                f=TOP/b/"validation_maps.jsonl.gz"
                if b=="appendix_1fdacb1b":
                    prev=fetch.load(TOP/b/"validation.json")
                    if prev["union_distinct"]["id"]!=1407 or prev["revision"]!=PREVIOUS:
                        raise ValueError("1407 predecessor not verified")
                    digest=prev["map_registry_sha256"]
                else:
                    digest=hashlib.sha256(fetch.git("show",REV+":"+str(f.relative_to(fetch.ROOT)))).hexdigest()
                if fetch.sha(f.read_bytes())!=digest:
                    raise ValueError("previous map witness changed")
                parents.append((b,digest))
            validate(fetch.CACHE,HERE,REV,PREVIOUS,parents)
            run(TOP/"appendix_499886f1/cases.py","--manifest",m,"--cache",fetch.CACHE)
            run(TOP/"appendix_0137949d/cash.py","--manifest",m,"--cache",fetch.CACHE)
        if stage=="context":
            index=TOP/f"context_{HERE.name}_index.tsv.gz"
            run(TOP/"context_catalog.py","--manifest",m,"--output",index)
            run(TOP/"context_audit.py","--index",index,"--validation",HERE/"validation_games.jsonl.gz",
                "--out",TOP/f"context_{HERE.name}_receipt.json","--jobs","1")
            run(TOP/"appendix_f7109df6/report.py",HERE)
            report(manifests,m,delta)



if __name__=="__main__":
    main()
