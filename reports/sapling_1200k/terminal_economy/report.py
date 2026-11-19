"""Produce a map-level readout for proxy generalization and the two original-examiner controls."""
import json
from pathlib import Path
import run


def load(path):
    return json.loads(path.read_bytes())


def main():
    doc=load(run.HERE/"results.json")
    if doc.get("status")!="complete": raise ValueError("incomplete proxy phase")
    freeze=doc["selection"]
    cand=freeze["chosen"]
    attr=load(run.HERE/"attribution.json")["games"]
    original=[]
    for name in ("parent","candidate"):
        directory=run.HERE/("original_"+name)
        games=load(directory/"results.json")
        ver=load(directory/"official_replay.json")["runs"][directory.name]["games"]
        if (games.get("status")!="complete" or len(ver)!=2 or
            {g["seed"] for g in ver}!={run.REAL} or {g["seat"] for g in ver}!={0,1}):
            raise ValueError("unverified original-examiner results")
        original.append(sorted(games["games"],key=lambda g:g["candidate_seat"]))
    lines=["# 经济状态混合权重训练与原版对照","",
           "候选在训练阶段按每图两席平均 U 的改善地图数选定，训练结束即冻结。风险场景以双方可见的树吸收历史、附近敌树、运货位置及敌方采集工的拦截范围识别。其余局面沿用 75%% it50、25%% Sapling V3。此次选择的两个 Sapling 系数分别为树经济 %.0f%%、受威胁运货 %.0f%%。"%(100*run.ARMS[cand][0],100*run.ARMS[cand][1]),"",
           "## 自家 it45，逐图完整对局","",
           "| 阶段 | 地图 | 父代席 0 / 席 1 分差 | 候选席 0 / 席 1 分差 | 平均 ΔU | 平均分差变化 |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label,data in (("训练",freeze["training"]),("独立留图",doc["holdout"])):
        for a,b in zip(data["parent"]["by_map"],data[cand]["by_map"],strict=True):
            lines.append("| %s | %s | %.2f / %.2f | %.2f / %.2f | %+.4f | %+.2f |"%
                          (label,a["seed"],*a["seat_margins"],*b["seat_margins"],b["U"]-a["U"],b["margin"]-a["margin"]))
    delta=doc["holdout_delta"]
    worst=min(delta["by_map"],key=lambda d:d["margin"])
    worst_seat=max(abs(d["seat_margins"][0]-d["seat_margins"][1]) for k in ("parent",cand)
                   for d in doc["holdout"][k]["by_map"])
    lines.extend(["",
                  f"三张独立留图中 {delta['maps_positive_U']} 张 ΔU 为正、{delta['maps_negative_U']} 张为负。"
                  f"平均 ΔU {delta['U']:+.4f}、平均分差变化 {delta['margin']:+.2f}。"
                  f"最差地图 {worst['seed']} 相对父代 {worst['margin']:+.2f}。"
                  f"留图中同策略同一图两席最大分差极差为 {worst_seat:.2f}，样本量按地图数计。","",
                  f"留图独立重算了 {sum(len(r['audited']) for r in attr)} 个网络动作，并保留所有中间步骤的公开观察。"
                  f"相对原父代改变 {sum(len(r['changed']) for r in attr)} 个动作，去掉 Sapling 后改变 {sum(len(r['student_sensitive']) for r in attr)} 个。"
                  "在相同公开状态插入伪地图种子、对手名字、伪全图碳量，所有抽查动作与原动作相同。","",
                  "## 原版 Master3s2，独立开发图","",
                  "| 地图 | 席 | 父代分差 | 候选分差 | 配对变化 | 候选结果 |",
                  "| ---: | ---: | ---: | ---: | ---: | --- |"])
    for a,b in zip(*original,strict=True):
        lines.append(f"| {run.REAL} | {b['candidate_seat']} | {a['margin']:+.2f} | {b['margin']:+.2f} | {b['margin']-a['margin']:+.2f} | "+{"win":"胜","loss":"负","draw":"平"}[b["result"]]+" |")
    w=sum(g["result"]=="win" for g in original[1])
    m=[sum(g["margin"] for g in gs)/2 for gs in original]
    lines+=["",f"候选 {w}/2 胜，平均官方分差 {m[1]:+.2f}，同一地图父代 {m[0]:+.2f}。"
            "双方全程使用原版 Master3s2 搜索和温度；对手独立采样，两组对局共享地图与席位，动作轨迹可能不同。"
            "只有一张原版开发地图，无法从中估计跨图不确定度。正式题没有打开。","",
            "本目录的 [冻结协议](PROTOCOL.md)、[训练及代理留图](results.json)、[选参检查点](selection.json)、"
            "[可见信息核验](attribution.json)、[父代实战](original_parent/results.json) 和 [候选实战](original_candidate/results.json) 给出各项数字的来源。"]
    all_changes=load(run.HERE/"original_candidate/all_changes.json")["games"]
    checks=load(run.HERE/"original_candidate/attribution.json")["games"]
    lines.insert(-1, "在候选自己的原版轨迹上逐步重算，冻结包的 598 个实际动作全部复现。原父代在同样的公开历史下，"
                 + "、".join(f"席{g['seat']}改变 {len(g['changed'])} 根" for g in all_changes)
                 + "。关闭 Sapling 后，52 个抽查根中有 "
                 + str(sum(len(g["student_sensitive"]) for g in checks)) + " 个改变，伪造元数据与隐藏碳图均未改变抽查动作。")
    (run.HERE/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__": main()
