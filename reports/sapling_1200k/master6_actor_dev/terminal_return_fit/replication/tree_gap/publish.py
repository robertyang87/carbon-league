"""Render all physical-seat/phase accounts alongside the frozen eight-game diagnosis."""
from pathlib import Path
import json,gzip
from collections import Counter
from statistics import mean

HERE=Path(__file__).resolve().parent
G=json.loads((HERE/"games.json").read_bytes())
S=json.loads(gzip.decompress((HERE/"spells.json.gz").read_bytes()))
P=("0–99","100–199","200–298","all")

def totals(rows,phase,side):
    ledger=Counter()
    for r in rows:ledger.update(r["phase"][phase][side])
    return ledger

def sh(x):return f"{x:,.2f}"
def avg(rows,phase,k):return mean(r["phase"][phase]["decomposition"][k] for r in rows)
def sign(x):return f"{x:+,.2f}"
def armname(r):return "v1" if r=="learned" else "亲本"
def fmtcell(a,b):return f"{a} / {b}"

def main():
    details=["# 八场逐段树权、资金与种植工", "", "从本方视角，每个双数栏依次为本方 / 考官。每树步收益是该段吸收÷期末持有树步；买入、被夺按实际付款次统计。所有区间均按发令前的步编号。", ""]
    for arm in ("parent","learned"):
        rr=[r for r in G if r["arm"]==arm]
        details+=["## "+armname(arm),"","| 图 / 席 | 段 | 持有树步 | 每树步收息 | 树息 | 自种 | 夺树 | 被夺 | 可付自种格 | 已有种植工步 | 优先观察位置 / 已买（本方；考官） |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in rr:
            for ph in P:
                a,b=(Counter(r["phase"][ph][v]) for v in ("own","examiner"))
                f=lambda k: fmtcell(a[k],b[k])
                details.append(f'| {r["seed"]} / {r["seat"]} | {ph} | {f("held_tree_turns")} | {a["absorption"]/a["held_tree_turns"]:.3f} / {b["absorption"]/b["held_tree_turns"]:.3f} | {sh(a["absorption"])} / {sh(b["absorption"])} | {f("plant_count")} | {f("seize_count")} | {f("lost_to_seizure")} | {f("plant_affordable")} | {f("planter_turns")} | {a["covered_uncontested_plant_opportunities"]}/{a["covered_uncontested_planted"]}；{b["covered_uncontested_plant_opportunities"]}/{b["covered_uncontested_planted"]} |')
        details+=["","| 图 / 席 | 本方实际失树 | 对手接手后树步 / 吸收 | 对手失树 | 本方接手后树步 / 吸收 | 本方被夺前自己持有步（只含被夺） |", "|---|---:|---:|---:|---:|---:|"]
        for r in rr:
            a,b=(r["loss"][k] for k in ("own","examiner"))
            details.append(f'| {r["seed"]} / {r["seat"]} | {a["after_seizure_opponent_spells"]} | {a["subsequent_opponent_ticks"]} / {sh(a["subsequent_opponent_absorption"])} | {b["after_seizure_opponent_spells"]} | {b["subsequent_opponent_ticks"]} / {sh(b["subsequent_opponent_absorption"])} | {a["before_loss_ticks"]} |')
        details+=[""]
    (HERE/"PHASES.md").write_text("\n".join(details).rstrip()+"\n")

if __name__=="__main__":main()
