"""Apply the core decision, economy and defense extractors to the eight later arrivals."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from defense import audit
from economy_endgame import last_assets
from economy_extract import analyze
from economy_publish import case_page
from layout import layout
from plant_timing_extract import extract as planting
from raid_timing import game_rows, sample_stats


def main(cache):
    manifest=json.loads((HERE/"manifest.json").read_bytes())
    if manifest["state"]!="complete":raise ValueError("incomplete manifest")
    data=[];risks=[]
    for i,card in enumerate(manifest["games"]):
        f=cache/card["path"]
        packed=f.read_bytes()
        if hashlib.sha256(packed).hexdigest()!=card["sha256"]:raise ValueError("invalid source")
        game=json.loads(gzip.decompress(packed))
        cash=analyze(packed)
        if cash["method"]!="simulator_verified" or cash["rewards"]!=card["score_by_seat"]:
            raise ValueError("economic reconstruction or scores differ")
        cash["terminal_assets"]=last_assets(f)
        defense=audit(game)
        events=game_rows(i,game,defense)
        risks.append(events)
        purchases=planting(packed)
        record={"id":card["id"],"source_sha256":card["sha256"],"layout":layout(card,f),
                "economy":cash,"defense":defense,"planting":purchases,
                "raid_opportunities":sample_stats(events,np.ones(len(events),dtype=bool))}
        data.append(record)
        md=case_page(card,cash,{"replay_sha256":card["sha256"]})
        md+=["","## 截击与种树时机","",
             f"行动前达到重货截击几何条件的工人对／回合共 {len(events):,} 个。两席种树工的行动前机会、选择与所有权账分别写入本批证据的 planting 字段。此局收据保留零事件和实际支付。",
             "","[八局逐局背景及分支时机数据](cases.json.gz) · [截击机会定义与字段顺序](../raid_timing.py)"]
        (HERE/(card["id"]+".md")).write_text("\n".join(md)+"\n")
    np.savez_compressed(HERE/"raid_rows.npz",rows=np.concatenate(risks))
    output={"rows":data,"revision":manifest["revision"],"steps":sum(r["defense"]["transitions"] for r in data)}
    (HERE/"cases.json.gz").write_bytes(gzip.compress(json.dumps(output,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    print(len(data),output["steps"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    main(p.parse_args().cache)
