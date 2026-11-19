"""Reconcile both sides' income, then tabulate fixed paired external-opponent results."""
from collections import Counter
from pathlib import Path
import sys
from run import BASE, HERE, read, sha, write
from root_recovery_branches.branches import begin
from all_top5.economy_extract import KEYS, POS, FastCarbon, CashEvents


def books(arm, row):
    tape = HERE / arm / row["raw_file"]
    if sha(tape) != row["raw_sha256"]:
        raise ValueError("tape corrupted")
    raw = read(tape)
    env = begin(raw)
    sim = FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    sim.cash = CashEvents(sim.cash)
    periods = []
    balances = [list(sim.cash)]
    deltas = []
    for t, joint in enumerate(raw["actions"]):
        sim.cash.events.clear()
        sim.step(*joint)
        env.step(joint)
        cash = [p[0] for p in env.states[0].observation.players]
        banks = [Counter(), Counter()]
        for key, s, change, *_ in sim.cash.events:
            banks[s][KEYS[POS[key]]] += change
        if any(abs(cash[s]-sim.cash[s]) > .011 or abs(cash[s]-balances[-1][s]-sum(banks[s].values())) > .011 for s in (0,1)):
            raise ValueError(f"unbalanced at {t}")
        balances.append(cash)
        deltas.append([dict(k) for k in banks])
    if not env.done or [s.reward for s in env.states] != row["official_score_by_seat"]:
        raise ValueError("terminal score mismatch")
    def period(first, last):
        return {"start_t": first, "end_t": last, "by_physical_seat": [
            {k:round(sum(b[s].get(k,0) for b in deltas[first:last]),4) for k in KEYS}
            for s in (0,1)]}
    end = len(deltas)
    bounds = [0] + [b for b in (60,100,200,260) if b<end] + [end]
    periods = [period(a,b) for a,b in zip(bounds,bounds[1:])]
    s = row["candidate_seat"]
    m = [c[s]-c[1-s] for c in balances]
    w = min(end,10)
    a = min(range(end - w + 1), key=lambda i:(m[i+w]-m[i],i))
    return {"arm": arm,"candidate_seat":s,"seed":row["seed"],"raw_file":str(tape.relative_to(HERE)),
            "tape_sha256":sha(tape),"score_by_seat":row["official_score_by_seat"],"terminal":end,
            "totals":period(0,end)["by_physical_seat"],"phases":periods,
            "largest_margin_decline_10_turns":{**period(a,a+w),"delta":round(m[a+w]-m[a],2),
                                                "cash_before":{str(max(0,a-k)):balances[max(0,a-k)] for k in (10,5,3,0)}},
            "physical_cash":balances,"cash_sources":deltas}


def render():
    result = read(HERE/"results.json")
    econ = {}
    for arm, rows in result["games"].items():
        econ[arm] = [books(arm,r) for r in rows]
    write(HERE/"income.json.gz", {"source_codes":list(KEYS),"games":econ})
    lines = ["# 双网终局回报 v1 对独立规则对手的推广实战", "",
             "候选与父代分别对仓内 v0.8 补员规则策略打两张预先固定的新图，每图换席。两个神经网络在两臂中都按正常入口推理。v1 使用原版 Master3s2 在一张图上的四个完整终局回报拟合四项动作特征的一个更新方向。",
             "","| 地图 | 本方席 | 父代己方／对手 | v1 己方／对手 | Δ分差 | ΔU | 首次改令步 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cluster in result["map_clusters"]:
        for pair in cluster["pairs"]:
            s = pair["seat"]
            def by_arm(name):
                r = next(x for x in result["games"][name] if (x["seed"],x["candidate_seat"]) == (cluster["seed"],s))
                score = r["official_score_by_seat"]
                return f"{score[s]:,.2f}／{score[1-s]:,.2f}"
            lines.append(f"| {cluster['seed']} | {s} | {by_arm('parent')} | {by_arm('learned')} | {pair['delta_margin']:+,.2f} | {pair['delta_U']:+.4f} | {pair['first_joint_divergence_turn']} |")
    lines.extend(["", "| 两图、四局 | 胜／平／负 | 平均分差 | 平均 U |","|---|---:|---:|---:|"])
    for arm,label in (("parent","父代"),("learned","v1")):
        r=result["arms"][arm]
        lines.append(f"| {label} | {'／'.join(map(str,r['WDL']))} | {r['mean_margin']:+,.2f} | {r['mean_U']:.5f} |")
    lines.extend(["", "两张地图分别将两个席位的配对结果等权平均。"])
    for c in result["map_clusters"]:
        lines.append(f"- 图 {c['seed']} 平均 Δ分差 {c['mean_delta_margin']:+,.2f}，平均 ΔU {c['mean_delta_U']:+.4f}。")
    lines.extend(["", "| 地图 | 席 | v1 相对父代己方回库 | 对手回库 | 己方树息 | 对手树息 |","|---|---:|---:|---:|---:|---:|"])
    for p,n in zip(econ["parent"],econ["learned"]):
        seed,s=n["seed"],n["candidate_seat"]
        diff=lambda s,k: n["totals"][s][k]-p["totals"][s][k]
        lines.append(f"| {seed} | {s} | {diff(s,'bank'):+,.2f} | {diff(1-s,'bank'):+,.2f} | {diff(s,'tree'):+,.2f} | {diff(1-s,'tree'):+,.2f} |")
    lines.extend(["", "每局使用仓内原始解释器同步推进，保存双方指令，再用新解释器逐局复算终局。经济账另将每步两席现金、真实回库、双方树息、基地或树上碰撞、招工、种植及夺树支出核到分，同时存下各局十步最大回撤前十、五、三步的现金。对手通过自己的公开观察维护碳图与对手模型，按原实现补种植员。每个席位的真实命令次数和双网分歧选择计数均在答卷。",
                  "", "本次 218 张 Master6 对外示范涉及 51 个公开对手名，名册与可核的历史辅助训练清单都没有这个补员脚本。辅助轨迹实际用过另一个 raid_v0_1 规则对手。上游 it50 的全部历史对手和天梯匿名包的字节身份无法逐一验证。两个独立地图和一个脚本只支持这组外推诊断。对原版 Master3s2 的正式胜率另有独立赛制。","",
                  "[冻结设计](protocol.json) · [地图与训练名单核查](clearance.json) · [逐局成绩与官方复核](results.json) · [逐回合经济账](income.json.gz) · [父代指令](parent/) · [v1 指令](learned/) · [对手源码](../../../../../code/policy_v0_8_replace.py)"])
    (HERE/"README.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines[2:22]))


if __name__=="__main__":
    render()
