"""Freeze the next nonoverlapping Top Five replay and public-context appendix."""
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from appendix_336e8b8c.collect import BATCHES, REV as OLD, all_sources, read_score
from appendix_batch import prepare
from validation_delta import run as validate
from appendix_f7109df6.report import build

REV = "97de1ebc1834785a6648f5876ce803311354a63b"
PAST = (*BATCHES,"appendix_336e8b8c")


def save(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


def inventories(previous=None,revision=None,batches=None):
    old_revision = previous or OLD
    new_revision = revision or REV
    prior_batches = PAST if batches is None else batches
    sources = {r:all_sources(r) for r in (old_revision, new_revision)}
    changed = set(fetch.git("diff","--name-only","--no-renames",old_revision,new_revision).decode().splitlines()) & set.union(*sources.values())
    known = {g["id"] for b in prior_batches for g in fetch.load(TOP/b/"manifest.json")["games"]}
    additions = {}
    counts = Counter()
    for path in sorted(changed):
        before, after = [read_score(path,r,sources[r]) for r in (old_revision,new_revision)]
        if before.keys()-after.keys():
            raise ValueError(f"removed scores in {path}")
        counts["before"]+=len(before); counts["after"]+=len(after)
        for gid in sorted(after.keys()-before.keys()):
            game=after[gid]
            ids=[fetch.NAMES.get(game["name_"+s]) for s in game["order"]]
            category="old" if gid in known else ("top" if any(ids) else "other")
            entry=additions.setdefault(gid,{"class":category,"physical_names":[game["name_"+s] for s in game["order"]],
                                             "mutual":all(ids),"paths":[]})
            partial = any(game["scores"].get(s) is None for s in game["order"])
            if entry.setdefault("unscored",partial) != partial:
                raise ValueError("numeric and unscored mirror disagree")
            entry["paths"].append(path)
            counts[category]+=1
    new = {k:v for k,v in additions.items() if v["class"]=="top"}
    partial={k:v for k,v in new.items() if v["unscored"]}
    return {"unscored_new_ids":sorted(partial),"indexed_distinct":len(known)+len(new),"source":"本次有更新的成绩索引","previous_score_revision":old_revision,"revision":new_revision,"old_reference_count":counts["before"],
            "new_reference_count":counts["after"],"existing_references":counts["old"],"non_top_five_refs":counts["other"],
            "union_games":len(known)+len(new)-len(partial),"new_count":len(new)-len(partial),"new_mutual":sum(r["mutual"] for r in new.values()),
            "all_added_references_by_id":additions}


def main():
    d=inventories(); save(HERE/"score_delta.json",d)
    print("score delta",d["new_count"],d["new_mutual"],d["union_games"],flush=True)
    mf=HERE/"manifest.json"
    if not mf.exists():
        missing={}
        prepare(REV,[TOP/b/"manifest.json" for b in PAST],HERE,(d["new_count"],d["new_mutual"]),unscored=missing)
        if set(missing) != set(d["unscored_new_ids"]):
            raise ValueError("unscored source mismatch")
        save(HERE/"unscored/cards.json", missing)
    expected={i for i,r in d["all_added_references_by_id"].items() if r["class"]=="top" and not r["unscored"]}
    if expected != {g["id"] for g in fetch.load(mf)["games"]}:
        raise ValueError("full catalog and score diff disagree")
    prior=[]
    for b in PAST:
        p=TOP/b/"validation_maps.jsonl.gz"
        if p.read_bytes()!=fetch.git("show",REV+":"+p.relative_to(fetch.ROOT).as_posix()):
            raise ValueError("historical report changed")
        prior.append((b,fetch.sha(p.read_bytes())))
    subprocess.run([sys.executable,str(HERE/"unscored/audit.py")], check=True)
    validate(fetch.CACHE,HERE,REV,OLD,prior,unscored=d["unscored_new_ids"])
    def run(script,*args):
        subprocess.run([sys.executable,str(TOP/script),*map(str,args)],check=True)
    run("appendix_499886f1/cases.py","--manifest",mf,"--cache",fetch.CACHE)
    run("appendix_0137949d/cash.py","--manifest",mf,"--cache",fetch.CACHE)
    ix=TOP/f"context_{HERE.name}_index.tsv.gz"
    run("context_catalog.py","--manifest",mf,"--output",ix)
    run("context_audit.py","--index",ix,"--validation",HERE/"validation_games.jsonl.gz",
        "--out",TOP/f"context_{HERE.name}_receipt.json","--jobs","1")
    mutual=sum(g["cohort"]=="mutual" for b in PAST for g in fetch.load(TOP/b/"manifest.json")["games"])+d["new_mutual"]
    build(HERE)
    page=HERE/"README.md"
    generated=page.read_text().replace("互打 115 张",f"互打 {mutual} 张").replace(
      "本批在独立考试排除集中逐图绑定真实 225 格开局。","本批真实 225 格开局已逐图留证，须与正式考试排除清单共同核验。")
    page.write_text(generated+"\n本口径累计 1,487 局有双方完整分数，另有 1 局执行出错。含这场共有 1,488 个不同比赛 ID。该局官方状态与退出前的真实过程单独存档。[异常局回放及动作前缀](unscored/README.md)\n")
    run("future_economy_check.py","--replay-cache",fetch.CACHE/"replays","--expect-games",d["union_games"],
        "--out",HERE/"future_economy_check.json")


if __name__ == "__main__":
    main()
