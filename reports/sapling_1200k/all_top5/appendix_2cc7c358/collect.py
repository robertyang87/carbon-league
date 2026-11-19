"""Archive the remainder of the index update at 15:26, with the prior score-fault map retained."""
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
from appendix_97de1ebc import collect as prior
from appendix_batch import prepare
from validation_delta import run as validate
from appendix_f7109df6.report import build
import fetch

OLD=prior.REV
REV="2cc7c3588279454823a2fd816e6d8ad593166125"
PAST=(*prior.PAST,"appendix_97de1ebc")
FAULT=TOP/"appendix_97de1ebc/unscored/manifest.json"


def main():
    # Use the same score-delta parser as the preceding cohort; the archive verifier independently scans all sources.
    prior.OLD,prior.REV,prior.PAST=OLD,REV,PAST
    d=prior.inventories()
    # The previous fault's archived scorecard can itself acquire an index reference. Count that reference once, keep its board historical.
    prior_fault_ids={g["id"] for g in fetch.load(FAULT)["games"]}
    for gid in d["unscored_new_ids"]:
        if gid not in prior_fault_ids:
            raise ValueError("new score fault requires a separate raw-replay audit")
        entry=d["all_added_references_by_id"][gid]
        entry["class"]="old_fault"
        d["existing_references"]+=len(entry["paths"])
    d["previous_fault_references"]=[g for g in d["unscored_new_ids"]]
    d["unscored_new_ids"]=[]
    if (d["new_count"],d["new_mutual"],d["union_games"])!=(15,0,1502):
        raise ValueError("unexpected score update")
    prior.save(HERE/"score_delta.json",d)
    m=HERE/"manifest.json"
    if not m.exists():
        partial={}
        prepare(REV,[TOP/b/"manifest.json" for b in PAST],HERE,(15,0),unscored=partial)
        if set(partial)!={g["id"] for g in fetch.load(FAULT)["games"]}:
            raise ValueError("unexpected fault cohort")
    expected={k for k,r in d["all_added_references_by_id"].items() if r["class"]=="top" and not r["unscored"]}
    if expected!={g["id"] for g in fetch.load(m)["games"]}:
        raise ValueError("independent catalog delta mismatch")
    registries=[]
    for b in PAST:
        p=TOP/b/"validation_maps.jsonl.gz"
        if p.read_bytes()!=fetch.git("show",REV+":"+p.relative_to(fetch.ROOT).as_posix()):
            raise ValueError("historical registry differs")
        registries.append((b,fetch.sha(p.read_bytes())))
    validate(fetch.CACHE,HERE,REV,OLD,registries,previous_fault_ledgers=[FAULT])
    def run(script,*args):
        subprocess.run([sys.executable,str(TOP/script),*map(str,args)],check=True)
    run("appendix_499886f1/cases.py","--manifest",m,"--cache",fetch.CACHE)
    run("appendix_0137949d/cash.py","--manifest",m,"--cache",fetch.CACHE)
    ix=TOP/f"context_{HERE.name}_index.tsv.gz"
    run("context_catalog.py","--manifest",m,"--output",ix)
    run("context_audit.py","--index",ix,"--validation",HERE/"validation_games.jsonl.gz",
        "--out",TOP/f"context_{HERE.name}_receipt.json","--jobs","1")
    build(HERE)
    p=HERE/"README.md"
    text=p.read_text().replace("互打 115 张","互打 117 张").replace(
        "本批在独立考试排除集中逐图绑定真实 225 格开局。","本批记录逐场绑定真实 225 格开局，供独立考试排除。")
    p.write_text(text+"\n含上批单列的执行故障局，本次截止共 1,503 张已见初盘。1,502 场有双方完整官方分数。[故障场次与真实地图](../appendix_97de1ebc/unscored/README.md)\n")
    run("future_economy_check.py","--replay-cache",fetch.CACHE/"replays","--expect-games",1502,
        "--out",HERE/"future_economy_check.json")


if __name__=="__main__":
    main()
