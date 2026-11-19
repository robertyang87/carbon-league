"""Summarize the registered original matches after all four official receipts exist."""
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent
def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)

plan = read(here / "protocol.json")
rows = read(here / "scores.json")
seed, = plan["seeds"]
assert sorted((r["arm"], r["seat"]) for r in rows) == [(a,s) for a in ("candidate","parent") for s in (0,1)]
lines = [f"# 开发图 {seed} · Sapling 对仓内原版 Master3s2", "",
         "从开局自然采样到官方终局。新版用 Sapling 与自有 it50 联合出令，训练前后的两包共用伙伴与部署规则。原版公开入口保持温度 1、0.58 秒墙钟。", "",
         "|参赛包|席位|本方分|Master3s2 分|官方分差|胜负|原版调用|原版调用墙钟合计|完成深度 8 的拍数|",
         "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
for arm, label in [("candidate","新版"),("parent","父代")]:
    subset = sorted((r for r in rows if r["arm"] == arm),key=lambda r:r["seat"])
    for r in subset:
        seat = r["seat"]
        path = here / "games" / arm / f"wall_{seed}_{seat}.json.gz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["wall_sha256"]
        wall = read(path)
        calls = wall["calls"]
        assert r["seed"] == seed and r["actual_opponent_calls"] == r["steps"] == len(calls)
        assert r["status"] == ["DONE","DONE"] and wall["identity"]["binding"]["original"] == plan["original"]
        duration = sum(c["public_wall_seconds"] for c in calls)
        d8 = sum(c.get("max_depth")==8 for c in calls)
        result = {"win":"胜","loss":"负","draw":"平"}[r["result"]]
        lines.append(f"|{label}|{seat}|{r['own']:.2f}|{r['opponent']:.2f}|{r['margin']:+.2f}|{result}|{len(calls)}|{duration:.2f} 秒|{d8}|")
lines.append("")
for arm, label in [("candidate","新版"),("parent","父代")]:
    sub=[r for r in rows if r["arm"]==arm]
    lines.append(f"{label}两席取得 {sum(r['result']=='win' for r in sub)} 胜、{sum(r['result']=='draw' for r in sub)} 平、{sum(r['result']=='loss' for r in sub)} 负，平均官方分差 {mean(r['margin'] for r in sub):+.2f}。")
lines.extend(["","本次只有一个独立地形，无法估计跨图不确定度。该图已从正式考试排除。","",
              "[预登记](protocol.json) · [地形核验](clearance.json) · [整场结果](scores.json) · [双方动作、官方结算与原版逐拍墙钟](games/)"])
(here/"README.md").write_text("\n".join(lines)+"\n")
