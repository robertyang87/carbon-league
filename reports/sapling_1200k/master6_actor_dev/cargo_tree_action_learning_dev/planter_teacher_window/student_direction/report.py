"""Summarize test maps, trainer example coverage, and same-prefix constraints without interpreting a label as a win."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from dataset import HERE,SEQ,read,write,sha


def main():
    q=read(HERE/"qualification.json");dat=read(HERE/"data.json");fit=read(HERE/"fit.json");inv=read(HERE/"invariants.json")
    examples=read(HERE/"teacher_examples.json")
    rows=read(HERE/"rows.json.gz");issued=read(HERE/"held_actions.json.gz");loss=read(HERE/"conditional_losses.json.gz")
    key=lambda r:(r["map"],r["seat"],r["t"],r["uid"])
    # Different tapes can reach the same physical map/seat/step/uid. Always join with the tape identity where available.
    test=[r for r in rows if r["split"]=="test"];assert len(test)==len(issued)==len(loss)
    assert list(map(key,loss))==list(map(key,test))
    official={(c["game"],c["seat"]):c["official_margin"] for c in read(HERE/"official_scores.json")["maps"]}
    by={(r["game"],r["t"],r["uid"],r["seat"]):r for r in issued}
    results={}
    groups={"距最近敌工增加":lambda r:r["after_enemy"]>r["before_enemy"],"距最近敌工不增加":lambda r:r["after_enemy"]<=r["before_enemy"],
            "示范者最终败局":lambda r:official[r["game"],r["seat"]]<0,"示范者胜或平":lambda r:official[r["game"],r["seat"]]>=0}
    for group,pred in groups.items():
        selected=[(r,l,by[(r["game"],r["t"],r["uid"],r["seat"])]) for r,l in zip(test,loss) if pred(r)]
        result=dict(terrains=len({r["map"]for r,_,_ in selected}),rows=len(selected),arms={})
        for arm in ("parent","control","candidate"):
            maps=defaultdict(list)
            for r,l,a in selected:
                maps[r["map"]].append((l[arm],1/r["nplanters"],a[arm]==r["action"]))
            result["arms"][arm]=dict(nll=float(np.mean([np.average([x[0] for x in pop],weights=[x[1]for x in pop]) for pop in maps.values()])),
                hit=float(np.mean([np.mean([x[2] for x in pop]) for pop in maps.values()])))
        results[group]=result
    write(HERE/"slices.json",results)
    t=dat["splits"]["train"];m=q["physical_deployment_metrics"]
    w=inv["emission_by_role"];qual=q["qualitative_matches"]
    body=["# Sapling 种植工移动方向的条件策略","",
          f"从首胜包 8da4b465 冻结全部已有参数，在 Sapling 的 96 维循环隐藏状态与四个移动方向的局面特征之间，训练了 {fit['trained_parameters']:,d} 个方向参数。最终单文件为 {q['candidate'][:12]}，大小 {fit['arms']['candidate']['bytes']:,d} 字节。它可以逐步独立执行；训练和日志使用的原版及前五动作不会随包部署。","",
          f"训练使用 {t['maps']} 张地图上的 {t['decisions']:,d} 条实际种植工移动。每条在同一拍已经执行的示范前序单位动作上重新计算隐藏状态和占位；输入只取当时该席可见信息。{t['losing_map_decisions']} 条来自示范者的败局，{t['moving_farther_from_nearest_rival']:,d} 条移动后距当拍最近敌工更远，这个数不能识别选手的主观意图。原版使用历史上搜索后实际提交的命令，天梯使用已登记前五队员的真实命令。每份留图的两席和轨迹始终归同一物理地形。","",
          "候选按实际后续 24 步中的选址周边净地租、投入及双方入库差，对已执行的方向做加权模仿。主动对照在同一批次、同样结构和更新次数下仅模仿已执行的方向。只有行动带上的方向有这些后续收益标签，其他方向没有反事实价值。","",
          "| 留图 | 地图／方向决策 | 父代方向 NLL／实令命中 | 同预算 BC | 24 步候选 |","|---|---:|---:|---:|---:|"]
    for grp,label in (("all","全部"),("Master3s2","原版"),("Top5","天梯前五")):
        vals=m[grp]
        fields=[f"{vals[k]['conditional_nll']:.5f} / {100*vals[k]['published_hit_rate']:.2f}%" for k in ("parent","control","candidate")]
        body.append("| "+ " | ".join([label,f"{vals['parent']['maps']} / {vals['parent']['decisions']:,d}",*fields])+" |")
    body.extend(["",f"留图有 {q['actual_changed_directions']} 个种植工从一个实际移动方向改为另一个，覆盖 {q['changed_direction_terrains']} 张地形。候选在前五来源的条件概率改善，在原版来源上退步。整队自由解码后，两个来源的实际方向命中均低于父包。两张早期教师介入图中 12 个曾实际换方向的机会，父包、BC 和候选分别复现 {qual['parent']}、{qual['control']}、{qual['candidate']} 个。两图已曝光，该核对只记录能否重现旧行为。","",
          f"在同一前缀，先固定原包给原地停留和四向移动的总概率，再调整四个方向内部的条件分布。实际贪心执行保留原先停或走的选择。两份已封包的模型留图概率质量误差上限为 {max(max(v.values()) for v in q['preservation'].values()):.2g}。独立执行器对 66 个旧观察复现未启用方向头的父包，核查 {w['planter']} 次种植工槽位都保持同前缀停／走，{w['collector']} 次采集工、{w['base']} 次基地均使用原先的同前缀动作。上游种植方向发生变化时，后续工人和基地仍按真实新前缀重新出令。新增同队占位冲突会退回完整父令。","",
          "328 席 0、330 席 1 的开发复验另行登记。两图地形与建集全部来源隔离，父代和学生从相同旧图根接受原版真实应对。局中窗口检验与标准从开局到终局的整场成绩分别计量。","",
          "天梯样本有一场灭绝。地形种子 236516046 的终点现金为 10101.14／18.11，官方终分为 10081.15／−7。双方排序相同。全样本依官方分复算后，胜负分组变动为零。训练的 24 步局部回报独立于终局胜负。[逐场官方分与现金](official_scores.json)","",
          "[方案与冻结](protocol.json) · [来源与分组](../../territory_sequence/sources.json) · [全部训练行](rows.json.gz) · [训练及导出](train.py) · [单文件](candidate.py) · [方向参数](candidate.npz) · [逐步实令](held_actions.json.gz) · [反例分组](slices.json) · [不变量复验](invariants.json) · [原版复验协议](response/protocol.json)"])
    if (HERE/"response"/"results.json").is_file():
        body.extend(["", "## 原版独立应对与单步耗时", "",
            "328 席 0 改了两步方向，学生窗口的官方终差比对照少 1.78。330 席 1 改了三步方向，学生后缀 9,089.87 比 8,886.51，比对照终差多 871.92。该学生臂的原版考官在 171 次调用里有 158 次零深回退，父臂为 0 次；两臂考官实际耗时分别为 10.49 和 77.59 秒。328 图考官又先于学生改令发生分叉。两条响应均入账，该包还没有从开局在新图对原版取得胜场。[四条真实轨迹与官方分](response/README.md)",
            "", "绑定同一包，又在一张已用开发图两席完成 598 个公开请求的本地复演。P95 68.14 ms，P99 124.93 ms，最大 176.29 ms，超过一秒 0 次，冷启最长 5.02 秒，进程峰值 90.76 MiB。实测主机仍是共享 4 核、16 GiB，采用单核亲和性和 512 MiB 地址空间限制，正式资源认证待该包在真实限制下独立执行。[单步计时与资源读回](timing/summary.json)"])
    (HERE/"README.md").write_text("\n".join(body)+"\n")


if __name__=="__main__":main()
