"""Render the complete per-game and late tree tables from the independently reconciled ledger."""
from collections import Counter
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
NAMES={"basecal":"招工校准 V4","it50":"原生 it50","soft_it50_deep214":"214 万单网",
       "soft_it50_v5":"120 万 V5","soft_it50_v5_guard":"120 万 V5 + 守卫","tree_raid":"夺树双网",
       "v4r":"保护现金 V4r","win_cash":"胜负 PPO"}
COMPONENTS={"own_delivery":"常规交货","plant":"栽树支出","tree_absorption":"树吸收","seize":"夺树支出",
            "hire_collector":"招采集工","hire_planter":"招种植工","rounding":"舍入",
            "opponent_worker_delivery_to_this_base":"对方工人在本基地交货",
            "tree_collision_cash":"树格碰撞入账","base_collision_cash":"基地格碰撞入账"}


def name(game):
    return NAMES.get(game["run"].removeprefix("tree_exam_"),game["run"])


def known(x):
    return "—" if x is None else str(x)


def example(game, purchase):
    x=purchase
    return (f'{name(game)}，地图 {game["seed"]} 席 {game["seat"]}，结算第 {x["buy_state"]} 步'
            f'在格 {x["position"]} {"栽树" if x["kind"]=="plant" else "夺树"}，花 {x["cost"]:,.2f}。'
            f'首次持有期截至第 {x["end_state"]} 步，吸收地碳 {x["terrain_absorption"]:,.2f}，'
            f'吸收敌方工人载碳 {x["enemy_cargo_absorption"]:,.2f}，格上碰撞现金 {x["collision_cash"]:,.2f}，'
            f'实际到账 {x["gross_received"]:,.2f}，扣购买费后 {x["net_received_less_purchase"]:+,.2f}。')


def render(games):
    assert len(games)==len(set((g["run"],g["seed"],g["seat"]) for g in games))
    count=sum(g["turns"] for g in games)
    out=["# 原版 Master3s2 逐步经营账","",
         f'这些开发对局共有 {len(games)} 局、{count:,} 次真实联合推进。每一局都重放双方原动作，逐笔勾稽两个现金账户，再与原存终局状态 SHA-256 对照。'
         f'累计存入和树吸收与原考场快照核对 {sum(g["external_snapshot_comparisons"] for g in games)} 项。'
         f'每步每方现金的最大未解释残差为 {max(g["max_abs_unexplained_cash_residual"] for g in games):.2g}。'
         '每局明细含初始地碳与基地、双方每步指令、工人位置和载货、到最近自家基地的环面距离、碰撞受益方、种植与树权变化、逐笔现金流。'
         '同图换席或多版本的读数彼此相关。本表只覆盖归档了完整动作的对局。'
         ' [逐局窗口、开局与末段投资明细](turning_point_ledger/summary.json) · [复算程序](turning_point_ledger.py)',
         "","## 对手何时建立领先","",'表中的对手领先等于考官现金减候选现金。结算步 t 使用前一状态 t−1 各自提交的动作。'
         '连续领先须连续 20 个结算点严格为正。最大单步与最大 20 步窗口按各局全部经营现金状态独立扫描；大小相同取最早。'
         '“最大分项”取该段双方现金差中绝对额最大的来源，符号表示对领先增长的贡献。'
         '每局还存了对手首次领先 500、1,000 及各自持续 10 步的时点、最大 10 步窗口、窗口前 10 步布局、正负现金全分项、两方最重三名工人的载货与回家距离，以及最早的真实招工、交货、种夺树和碰撞死亡事件。',
         "","| 候选 | 图／席 | 步数 | 首先领先方 | 对手首次连续领先起点 | 对手至局末持续领先起点 | 最大单步（结算步／增量） | 最大 20 步（状态／增量） | 最大分项 |",
         "|---|---|---:|---|---:|---:|---|---|---|"]
    for g in games:
        p=g["lead_periods"]; w=g["windows"]["20"]; q=g["windows"]["1"]
        leader={"opponent":"考官","candidate":"候选"}.get(p["first_to_lead"],"平局")
        term=known(p["final_unbroken_opponent_lead_from"])
        k,v=next(iter(w["opponent_minus_candidate_components"].items()))
        out.append(f'| {name(g)} | {g["seed"]}／{g["seat"]} | {g["turns"]} | {leader} t{p["first_non_tied_turn"]} | {known(p["first_20_consecutive_opponent_leads"])} | {term} | t{q["state_end"]}／{q["growth"]:+,.0f} | {w["state_start"]}→{w["state_end"]}／{w["growth"]:+,.0f} | {COMPONENTS.get(k,k)} {v:+,.0f} |')
    out+=["","## 最后 30 次推进如何结现","",'此段从状态 269 到状态 299，共收录第 270–299 步结算。'
          '只对打满的对局比较。碰撞与交货归属按实际入账席位：敌方载货落到本方基地、树或碰撞获胜的采集工有不同去向。'
          '落到存活工人身上的货，会继续暴露给碰撞和树的吸收，后续交付时才形成现金。'
          '栽种和夺取均按当步实付；只记连续持有阶段真实收到的地碳、敌工载碳与树格碰撞钱。树权易手或枯萎时结清这段持有期，'
          '期末无树木出售收入。每席每步现金舍入另列，不摊派到树。表中合计覆盖所有打满的候选及参照局，未列反事实威慑收益。',
          "","| 版本 | 到达步限／全局 | 候选栽树回本／笔数 | 考官栽树回本／笔数 | 候选夺树回本／笔数 | 考官夺树回本／笔数 |",
          "|---|---:|---:|---:|---:|---:|"]
    late_items=[]
    for run in sorted(set(g["run"] for g in games)):
        members=[g for g in games if g["run"]==run]
        full=[g for g in members if g["last_30"]["applicable"]]
        totals=Counter()
        for g in full:
            for x in g["last_30"]["purchases"]:
                role="cand" if x["owner"]==g["seat"] else "opp"
                totals[role,x["kind"],"count"]+=1
                totals[role,x["kind"],"paid"]+=int(x["repaid"])
                if role=="cand":
                    late_items.append((g,x))
        columns=[f"{totals[role,kind,'paid']}/{totals[role,kind,'count']}" for kind in ("plant","seize") for role in ("cand","opp")]
        out.append(f'| {name(members[0])} | {len(full)}/{len(members)} | '+" | ".join(columns)+" |")
    if late_items:
        out+=["","逐笔记录中的两个栽种案例：",""]
        planted=[(g,x) for g,x in late_items if x["kind"]=="plant"]
        for fn in (max,min):
            g,x=fn(planted,key=lambda e:e[1]["net_received_less_purchase"])
            out.append("- "+example(g,x))
    for g in games:
        if not g["last_30"]["applicable"]:
            out+=["",f'{name(g)} 在地图 {g["seed"]} 席 {g["seat"]} 于 t{g["turns"]} 提前终止。结算后两席的经营现金为 {g["last_cash_by_seat"]}，'
                  f'官方终分为 {g["official_score_by_seat"]}。终局规则使考官比分领先相对经营现金额外变化 {g["terminal_scoring_adjustment_to_opponent_lead"]:+,.2f}，'
                  '这笔变化只进入官方计分行。']
    out+=["","## 复算","","在仓库根目录运行。命令读取原有逐步动作，不调用考官或训练模型。","","```sh",
          "cd reports/sapling_1200k",
          "python turning_point_ledger.py --out turning_point_ledger " + " ".join(sorted(set(g["run"] for g in games))),
          "python turning_point_tables.py > TURNING_POINT_LEDGER.md","```"]
    return "\n".join(out)+"\n"


if __name__=="__main__":
    records=json.loads((HERE/"turning_point_ledger/summary.json").read_bytes())
    print(render(records["games"]),end="")
