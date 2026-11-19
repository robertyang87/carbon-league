"""Read every registered full-game result and verify its official score and original clock."""
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(directory):
    plan = read(directory / "protocol.json")
    rows = read(directory / "scores.json")
    expected = [(s, a, i) for s in plan["seeds"] for a in ("candidate", "parent") for i in (0,1)]
    assert sorted(expected) == sorted((r["seed"],r["arm"],r["seat"]) for r in rows)
    checked = []
    for r in rows:
        s, arm, seat = r["seed"], r["arm"], r["seat"]
        prefix = directory / "games" / arm
        receipt = prefix / f"receipt_{s}_{seat}.json"
        clock = prefix / f"wall_{s}_{seat}.json.gz"
        assert digest(receipt)==r["receipt_sha256"] and digest(clock)==r["wall_sha256"]
        signed = read(receipt)
        official = signed["official"]
        wall = read(clock)
        assert signed["binding"]["original"]==wall["identity"]["binding"]["original"]==plan["original"]
        assert official["statuses"]==["DONE","DONE"] and official["official_reward_by_seat"]==r["score_by_seat"]
        assert official["steps"]==len(wall["calls"])==r["actual_opponent_calls"]
        assert wall["summary"]["native_budget_seconds"]==.58
        own, enemy = (Decimal(str(r["score_by_seat"][i])) for i in (seat,1-seat))
        margin = own-enemy
        outcome = "胜" if margin>0 else "负" if margin<0 else "平"
        assert outcome=={"win":"胜","loss":"负","draw":"平"}[r["result"]]
        checked.append(dict(seed=s,arm=arm,seat=seat,own=own,enemy=enemy,margin=margin,outcome=outcome,
                            steps=len(wall["calls"]),wall=sum(c["public_wall_seconds"] for c in wall["calls"]),
                            cpu=sum(c["process_cpu_seconds"] for c in wall["calls"]),
                            d8=sum(c.get("max_depth")==8 for c in wall["calls"])))
    return sorted(checked,key=lambda r:(r["seed"],r["arm"],r["seat"]))


def main():
    directory = Path(sys.argv[1])
    rows = scores(directory)
    labels = {"candidate":"新版","parent":"父代"}
    n = len({r["seed"] for r in rows})
    lines = ["# Sapling 对原版 Master3s2 的新图整场对照","",
             "同一开发图换席，Sapling 与自有 it50 联合出令。对手使用原版公开入口、温度 1 与 0.58 秒墙钟搜索。","",
             "|地图|参赛包|席位|本方官方分|原版官方分|分差|结果|调用数|原版墙钟秒|原版 CPU 秒|完成深度 8 的拍数|",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(('|{seed}|'+labels[r["arm"]]+'|{seat}|{own:.2f}|{enemy:.2f}|{margin:+.2f}|{outcome}|{steps}|{wall:.2f}|{cpu:.2f}|{d8}|').format(**r))
    lines.append("")
    for arm,label in labels.items():
        sub = [r for r in rows if r["arm"]==arm]
        counts = Counter(r["outcome"] for r in sub)
        m = mean(r["margin"] for r in sub)
        lines.append(f"{label} {counts['胜']} 胜、{counts['平']} 平、{counts['负']} 负，平均官方分差 {m:+.2f}。")
    lines.extend(["",f"此次覆盖 {n} 个独立地形，同图两席配对。样本量不足以推断总体胜率，所有参赛地图都排除出正式考试。","",
                  "[预登记](protocol.json) · [官方比分](scores.json) · [完整动作、计分与原版逐拍墙钟](games/)"])
    (directory / "README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
