"""Report the registered eight development games and every associated channel movement."""
from collections import Counter
from statistics import mean
from verify import DEV,HERE,read,write,sha


CATEGORIES = {
    "回库": ("own_delivery",),
    "树息": ("tree_absorption",),
    "招工": ("hire_collector","hire_planter"),
    "种树、夺树": ("plant","seize"),
    "碰撞、额外交付及舍入": ("tree_collision_cash","base_collision_cash","opponent_worker_delivery_to_this_base","rounding"),
}


def side_cash(g, s):
    a=g["physical_cash_channels"][s]
    return {name:sum(a.get(k,0) for k in keys) for name,keys in CATEGORIES.items()}


def gap(g):
    s=g["seat"]; own=side_cash(g,s); opp=side_cash(g,1-s)
    return {k:own[k]-opp[k] for k in CATEGORIES}


def first_separate(seed,seat):
    raw={a:read(DEV/a/f"raw_{seed}_{seat}.json.gz")["actions"] for a in ("parent","candidate")}
    for t,(a,b) in enumerate(zip(raw["parent"],raw["candidate"],strict=True)):
        if a!=b:
            return {"pre":t,"candidate_differs":a[seat]!=b[seat],"examiner_differs":a[1-seat]!=b[1-seat]}
    return None


def middle_tempo(g,owner):
    cohort=[h for h in g["holdings"] if 50<h["start"]<=250 and h["owner"]==owner]
    return {"plant":sum(h["origin"]=="plant" for h in cohort),"seize":sum(h["origin"]=="seize" for h in cohort),
            "net_first_tenure":sum(h["gross"]+h["collision"]-h["price"] for h in cohort),
            "direct_first_mean_life":mean(h["end"]-h["start"] for h in cohort) if cohort else None}


def main():
    data=read(HERE/"official_accounts.json.gz")
    driver=read(DEV/"comparison.json")
    refs={(g["seed"],g["seat"]):g for g in driver["paired_map_seats"]}
    by={arm:{(r["seed"],r["seat"]):r for r in rows} for arm,rows in data["development"].items()}
    pair_results=[]
    for seed,s in sorted(by["parent"]):
        a=by["parent"][seed,s];b=by["candidate"][seed,s];oa,ea=a["physical_final_cash"][s],a["physical_final_cash"][1-s]
        ob,eb=b["physical_final_cash"][s],b["physical_final_cash"][1-s]
        delta={k:gap(b)[k]-gap(a)[k] for k in CATEGORIES}
        if max(abs(sum(delta.values())-(b["margin"]-a["margin"])),abs(b["margin"]-refs[(seed,s)]["candidate"]["margin"]))>1e-4:
            raise ValueError("registered margin mismatch")
        root=first_separate(seed,s)
        pair_results.append({"seed":seed,"seat":s,"official_margin_change":b["margin"]-a["margin"],
                             "own_score_change":ob-oa,"opponent_score_change":eb-ea,"by_channel_change":delta,
                             "parent_gap":gap(a),"candidate_gap":gap(b),"first_joint_divergence":root,
                             "parent_largest_jump":a["largest_cash_jump"]["own_minus_enemy_gain"],
                             "candidate_largest_jump":b["largest_cash_jump"]["own_minus_enemy_gain"],
                             "middle_ownership":{arm:{"sapling":middle_tempo(g,s),"opponent":middle_tempo(g,1-s)}
                                                 for arm,g in (("parent",a),("candidate",b))}})
    trainings=data["training_independent_checks"]
    result={"source_sha256":sha(HERE/"official_accounts.json.gz"),"pairs":pair_results,
            "training_verified_games":len(trainings),"training_verified_map_count":len({r["seed"] for r in trainings}),
            "training_verified_transitions":sum(r["steps"] for r in trainings),
            "training_verified_tenures":sum(r["holdings"] for r in trainings),
            "development_verified_transitions":sum(g["steps"] for rows in data["development"].values() for g in rows)}
    write(HERE/"economic_comparison.json",result)
    render(result,by)


def render(report,by):
    p=report["pairs"]
    lines=["# 四视角开发对局的现金差与中盘种植",
           "",f"两张地图换席，两个冻结包各打四场。官方游戏规则重放 {report['development_verified_transitions']:,} 次联合推进，八场全部打满、全部负。下表按该游戏实际现金写入点将同图同席的终局分差变化分解。正数表示候选相对父代在该项拉近分差。双方每场独立出令并采样。"
           "[原始配对结果](../comparison.json) · [双方逐阶段、持树期和现金峰值](official_accounts.json.gz)",
           "","| 图 / 席 | 终局分差变化 | 我方终分变化 | 对手终分变化 | 回库净差变化 | 树息净差变化 | 招工支出差变化 | 种树夺树支出差变化 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in p:
        c=r["by_channel_change"]
        nums=(r["official_margin_change"],r["own_score_change"],r["opponent_score_change"],c["回库"],c["树息"],c["招工"],c["种树、夺树"])
        lines.append(f"| {r['seed']} / {r['seat']} | "+" | ".join(f"{x:+,.2f}" for x in nums)+" |")
    lines+=["","各候选场与考官相差的实际收入和中盘持有行为如下。种植和夺树按结果态步号 51–250 计，每次所有权转移另起一段。","",
            "| 图 / 席 | 候选官方分差 | 回库净差 | 树息净差 | 第 51–250 步新种，我 / 敌 | 同期夺树，我 / 敌 |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in p:
        k=r["seed"],r["seat"];h=r["middle_ownership"]["candidate"];u,e=h["sapling"],h["opponent"]
        lines.append(f"| {k[0]} / {k[1]} | {by['candidate'][k]['margin']:+,.2f} | {r['candidate_gap']['回库']:+,.2f} | "
                     f"{r['candidate_gap']['树息']:+,.2f} | {u['plant']} / {e['plant']} | {u['seize']} / {e['seize']} |")
    diffs=[r["first_joint_divergence"] for r in p]
    arms=Counter("双边" if d["candidate_differs"] and d["examiner_differs"] else
                 "我方" if d["candidate_differs"] else "考官" for d in diffs)
    plant=[r["middle_ownership"]["candidate"] for r in p]
    lines+=["",f"首次联合动作分叉在第 {', '.join(str(d['pre']) for d in diffs)} 步（依表排序）。其中 {arms['考官']} 对首先仅考官改令，"
            f"{arms['我方']} 对首先仅我方改令，{arms['双边']} 对双方同时分叉。整场分数变化混有温度 1 考官的不同反应。"
            "要验证某次种植能否反转终局分差，须在同一真实局面比较两种可执行命令，再让两边继续响应。",
            "",f"四盘候选在中段共种 {sum(h['sapling']['plant'] for h in plant)} 棵，对手种 {sum(h['opponent']['plant'] for h in plant)} 棵；"
            f"中段新种数较少的对局占 {sum(h['sapling']['plant']<h['opponent']['plant'] for h in plant)}/4。"
            "这两张地图已经作为开发样本使用。","",
            f"另按同一套官方 Board 余额写入对照已归档十张训练图二十局，共复算 {report['training_verified_transitions']:,} 步，"
            f"逐项核对 {report['training_verified_tenures']:,} 段持树记录的所有者、起止、购价、树息与碰撞现金；每局六个阶段的双方现金分类和工树暴露步数均吻合。"
            "[十图审计](../../tree_response_training/ten_map_economy/README.md) · [官方重放实现](verify.py) · [逐局对账](economic_comparison.json)"]
    detail=read(HERE/"cohort.json")
    current=detail["counts_and_realized_returns"]["51_250"]["plant"]
    own=current["sapling"];opp=current["opponent"]
    c=detail["per_game_mid_new_tree_profit_difference"]
    paymaps=list(detail["per_map_mid_new_plant"].values())
    fewer=sum(m["sapling"]["purchases"] < m["opponent"]["purchases"] for m in paymaps)
    lower=sum(m["sapling"]["mean_direct_profit"] < m["opponent"]["mean_direct_profit"] for m in paymaps)
    late=detail["counts_and_realized_returns"]["285_294"]["plant"]["sapling"]
    lines.extend(["", "## 中盘每笔新植的收益",
                  "",f"十图的第 51–250 步，Sapling 实付新种 {own['purchases']} 笔，平均成本 {own['mean_price']:.2f}，首次连续持有 {own['mean_held_turns']:.2f} 步，每笔直接净收入 {own['mean_direct_profit']:.2f}。"
                  f"考官新种 {opp['purchases']} 笔，对应为 {opp['mean_price']:.2f}、{opp['mean_held_turns']:.2f} 步、{opp['mean_direct_profit']:.2f}。"
                  f"双方在同图换席再汇总，{fewer}/{len(paymaps)} 图我方新种较少，{lower}/{len(paymaps)} 图每棵净收入较低。我方 {own['later_seized_by_opponent']}/{own['purchases']} 新树后来被夺，考官为 {opp['later_seized_by_opponent']}/{opp['purchases']}。",
                  "",f"按每局两边直接净收入分解，我方中盘新树差 {c['total']:+.2f}。数量差按双方每棵平均净收入折算为 {c['count_component']:+.2f}，每棵净收入差按双方平均数量折算为 {c['mean_realized_profit_component']:+.2f}。"
                  "这是对已发生收益的算术分解，额外种植后考官可以改路线、夺树或交货。",
                  "",f"第 285–294 步我方实际新种 {late['purchases']} 笔，{late['direct_profit_positive']} 笔在终局前直接回本，合计 {late['direct_profit_sum']:+.2f}；"
                  "过早统一关闭新植会撤掉部分实际赚钱的购买。"
                  "[购入与再失逐段分层](cohort.json) · [分解算法](cohort.py)"])
    (HERE/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
