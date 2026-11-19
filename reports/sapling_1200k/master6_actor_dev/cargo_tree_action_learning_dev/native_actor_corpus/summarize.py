"""Render the registered map-equal outcomes and verify all published model identities."""
from collections import Counter
from common import *

ARMS = ("parent", "head", "full")
NAMES = {"parent":"首胜父包", "head":"仅动作头", "full":"全策略网络"}
def pct(x):return f"{100*x:.2f}%"
def pp(x):return f"{100*x:+.2f}"
def main():
    result=read(HERE/"held.json");fit=read(HERE/"fit.json");w=read(HERE/"weight_audit.json")
    sample=read(HERE/"data.json");roots=read(HERE/"training_roots.json.gz")
    if result["held_commands_sha256"]!=sha(HERE/"held_commands.json.gz") or w["fit_sha"]!=sha(HERE/"fit.json"):
        raise ValueError("receipt changed")
    for k,h in fit["signature"].items():
        if h!=sha(HERE/k):raise ValueError(k)
    for a in ("full","head"):
        r=fit["arms"][a]
        if r["package_sha256"]!=sha(HERE/(a+".py")) or r["weights_sha256"]!=sha(HERE/(a+".npz")):
            raise ValueError("release changed: "+a)
    test=result["results"]["test"];qual=result["qualification"]["full"]["passed"]
    diff=test["paired_map_bootstrap"]["full_minus_parent"]["worker"];head_diff=test["paired_map_bootstrap"]["full_minus_head"]["worker"]
    cmp=test["micro"]["pairs"]["full_minus_parent"]["worker"]
    score=sample["counts"];by_seat=Counter(r["expert_seat"] for r in roots);by_phase=Counter(r["t"]//100 for r in roots)
    lines=["# 用原版 Master3s2 的双席实发命令训练 Sapling","",
           f"全策略网络在六张诊断图的工人动作一致率为 {pct(test['map_equal_accuracy']['full']['worker'])}，"
           f"相对首胜父包变化 {pp(diff['mean_difference'])} 个百分点，相对等预算动作头变化 {pp(head_diff['mean_difference'])} 个百分点。"
           f"本轮自由入口动作门{'通过' if qual else '未通过'}。本次评估沿用记录中的双方轨迹，未产生新的原版对局或胜率估计。[按图、角色、阶段和席位的完整结果](held.json)","",
           "| 数据划分 | 策略 | 工人 | 采集工 | 种树工 | 基地 | 完整联合命令 |","|---|---|---:|---:|---:|---:|---:|"]
    for part,label in (("val","验证，3 图"),("test","诊断，6 图")):
        for arm in ARMS:
            s=result["results"][part]["map_equal_accuracy"][arm]
            lines.append("| "+" | ".join([label,NAMES[arm],*[pct(s[k]) for k in ("worker","collector","planter","base","whole_joint")]])+" |")
    lo,hi=diff["paired_percentile_95"];lh,hh=head_diff["paired_percentile_95"]
    lines.extend(["",f"同一张地图两席和各步一起重抽 4,000 次。全策略网络对父包的工人一致率配对差 95% 区间为 [{pp(lo)}, {pp(hi)}] 个百分点，对动作头的区间为 [{pp(lh)}, {pp(hh)}]。"
                  f"在 {cmp['entities']:,} 个工人命令中，实际改令 {cmp['changed']} 次，其中新改对 {cmp['gained']} 次、原对改错 {cmp['lost']} 次、双方都错但选了不同动作 {cmp['both_wrong_changed']} 次。[每个工号的三包实际指令](held_commands.json.gz)","",
                  "## 训练数据与参数","",
                  "冻结的历史清单有 29 张地图、58 条完整原版对局记录。逐一校验原始文件和官方收据，并重放双方实际联合命令，17,342 次推进与 58 场官方终分相符。"
                  "示范者是每条记录的物理席 1−candidate_seat。原清单按地形分为 19/4/6，隔离先前首胜对照和这轮碳预测解码开发用过的七张地图后，实用 13/3/6 张。最新开发和正式保留区间均排除。"
                  "诊断图 621497120、621497121 曾进入另一项旧蒸馏训练，621495120 曾作该旧项目的留图，H24 也核验过部分历史留图。这六图只衡量本轮拟合以外的历史行为匹配。[原始对局 SHA、物理席及地形分区](sources.json)","",
                  f"训练使用 26 条完整轨迹，各轨分 96 段并在每段预选一拍，获得 {len(roots):,} 个不同真实局面，两物理席各 {by_seat[0]:,}、{by_seat[1]:,} 拍，"
                  f"前中后三段分别为 {by_phase[0]}、{by_phase[1]}、{by_phase[2]} 拍。{score['active_slots']:,} 个有归属的单位/基地槽中有 {score['representable_actionable']:,} 个可选专家动作、"
                  f"{score['constrained_single_legal']:,} 个只有单一合法令，遇到首个不可表示前缀 {score.get('roots_with_bad_prefix',0)} 拍。每席通过所有公开观察维护自己的特征历史，目标只进入当前槽损失和后续槽前缀。"
                  "训练时父包的自由解码先决定修正支路，随后固定 75% it50 项并对 25% Sapling 反传。验证和诊断由三个发布包各自自由解码并执行自己的守卫。[采样目录](training_roots.json.gz) · [冻结规则](protocol.json)","",
                  f"全策略网络和动作头各做 312 次更新，每次使用相同的 8 个局面。直接对比部署与 Torch 的 78 个训练根，实发条件下合法动作相对打分最大误差 {fit['parent_logit_parity']['max_centered_error']:.2g}。"
                  f"CPU0 单线程完成缓存提取约 131 秒，两臂优化、检查点核验及成包共 {fit['seconds']:.1f} 秒。第 78 步已在训练轨迹复核正常入口，两臂分别改变了 6 和 2 个工人指令。[中期检查点](checkpoint_78.json) · [训练过程](loss.json.gz) · [模型与软件身份](fit.json)","",
                  "| 参数 | 全策略网络 | 仅动作头 |","|---|---:|---:|"])
    for title,name in (("允许训练","trainable_parameters"),("发布后数值有变化","quantized_changed")):
        lines.append(f"| {title} | {fit['arms']['full'][name]:,} | {fit['arms']['head'][name]:,} |")
    for label,k in (("训练权重档中的 FP32 改动","source_fp32_parameters_changed"),("INT8 整数码改动","int8_codes_or_values_changed"),("INT8 行标度改动","row_scales_changed")):
        totals=[sum(g.get(k,0) for g in w["groups"][a].values()) for a in ("full","head")]
        lines.append(f"| {label} | {totals[0]:,} | {totals[1]:,} |")
    gate=read(HERE/"gate_audit.json")
    if gate["commands_sha"]!=result["held_commands_sha256"]:raise ValueError("gate command digest mismatch")
    before=gate["original_pooled"];after=gate["final_equal_terrain"]
    lines.extend(["","## 角色护栏的两种复算","",
                  "原评估代码 d74e33630 将所有地图的同类工人合并计算 2 个百分点的退化阈值。该代码冻结后、三包首次留出自由入口调用以前，改为每张图等权计算。采用两种口径分别重算同一份实际命令，全策略网络均通过，动作头均未通过；动作头在主指标上低于全策略网络。[两版历史评估逻辑与判读](gate_audit.json)","",
                  "| 相对父包，百分点 | 采集工，合并 | 采集工，地图等权 | 种树工，合并 | 种树工，地图等权 |","|---|---:|---:|---:|---:|"])
    for a in ("full","head"):
        g=before[a]["pooled_role_deltas_vs_parent"];h=after[a]["map_equal_role_deltas_vs_parent"]
        lines.append("| "+" | ".join([NAMES[a],pp(g["collector"]),pp(h["collector"]),pp(g["planter"]),pp(h["planter"])])+" |")
    lines.extend(["","INT8 行的缩放系数改变会同时改变这一行多个解量化权重。两个发布包的固定伙伴、动作守卫和价值头保持同一身份。[分层权重审核](weight_audit.json) · [全策略部署包](full.py) · [动作头部署包](head.py)","",
                  "## 重建","",
                  "在含历史原件的检出中从本目录运行以下命令。缓存与优化器断点写入 work/，中断后从上一条完整轨迹或最近 26 次更新处继续。","",
                  "```bash","export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1",
                  "for script in audits.py prepare_data.py fit.py check_weights.py evaluate.py check_gate.py summarize.py; do",
                  "  taskset -c 0 /workspace/carbon-review-venv/bin/python \"$script\" || exit 1","done","```",""])
    (HERE/"README.md").write_text("\n".join(lines))
if __name__=="__main__":main()
