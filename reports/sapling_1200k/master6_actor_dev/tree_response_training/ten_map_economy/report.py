"""Summarize the group of ten training maps, without choosing games by final outcome."""
from collections import Counter
from statistics import mean, median
from census import ACCOUNTS, HERE, read


def main():
    games = read(HERE/"games.json.gz")["games"]
    cohorts = read(HERE/"tenures.json.gz")
    margin = mean(g["official_margin"] for g in games)
    n = len(games)
    mean_channels = {k:mean(g["components"][k] for g in games) for k in ACCOUNTS}
    totals_by_phase = []
    for phase in range(6):
        totals = [Counter(), Counter()]
        for g in games:
            p = g["phases"][phase]
            for side, v in enumerate(totals):
                i = side ^ g["seat"]
                v.update(p["accounts"][i])
                v.update(p["turn_exposure"][i])
        rows = []
        for v in totals:
            rows.append((v["tree_turns"]/n,v["tree_absorption"]/v["tree_turns"],v["tree_absorption"]/n))
        totals_by_phase.append(rows)
    lines = ["# Sapling 何时失去树收入",
             "",
             f"五批已登记训练图共 {n//2} 张、{n} 局。冻结父代对完整原版 Master3s2 全负，平均官方终局分差 {margin:,.2f}。每局双方都完成 299 步。十张图和二十场双席对局都纳入分解。[逐局独立官方重放与逐阶段现金](games.json.gz) · [逐次持有期](tenures.json.gz)",
             "",
             f"每局正常入库相差 {mean_channels['own_delivery']:+,.2f}，树息相差 {mean_channels['tree_absorption']:+,.2f}。我方种树开支较少，抵回 {mean_channels['plant']:+,.2f}；夺树开支差 {mean_channels['seize']:+,.2f}，招工合计差 {mean_channels['hire_planter']+mean_channels['hire_collector']:+,.2f}。完整现金流同官方终分逐局核平。这些是同局发生的收入和成本，提前种一棵树的实际收益仍须比较两种选择后双方继续决策的终局。",
             "",
             "| 已完成步段 | 我方 / 对手持树步数，每局 | 我方 / 对手每持树步的收入 | 我方相对树收入，每局 |",
             "|---|---:|---:|---:|"]
    for i,(a,b) in enumerate(totals_by_phase):
        lines.append(f"| {i*50+1}–{min((i+1)*50,299)} | {a[0]:.1f} / {b[0]:.1f} | {a[1]:.2f} / {b[1]:.2f} | {a[2]-b[2]:+.2f} |")
    lines += ["","持树步数逐次推进前按所有者计，含到龄退出的最后一步。每持树步的收入为同一段全部树息除以持树步数。最后阶段覆盖 49 次推进。",
              "", "| 取得所有权的步段 | 新种，我方 / 对手 | 夺树，我方 / 对手 | 首次连续持有步数，我方 / 对手 | 该批树本局直接净收入，每局，我方 / 对手 |",
              "|---|---:|---:|---:|---:|"]
    for lo,hi in [(0,50),(50,250),(250,299)]:
        summaries = []
        for side in (0,1):
            hs = [h for g in cohorts for h in g["holdings"] if h["owner"]==g["seat"]^side and lo<h["start"]<=hi]
            sort = Counter(h["origin"] for h in hs)
            summaries.append((sort["plant"],sort["seize"],mean(h["preturns"] for h in hs),
                              sum(h["gross"]+h["collision"]-h["price"] for h in hs)/n))
        a,b=summaries
        lines.append(f"| {lo+1}–{hi} | {a[0]} / {b[0]} | {a[1]} / {b[1]} | {a[2]:.1f} / {b[2]:.1f} | {a[3]:,.2f} / {b[3]:,.2f} |")
    lines += ["","同一棵树被夺走时结束上一段持有，再为新所有者另起一段。净收入只计此次持有产生的树息、树上碰撞现金以及实付，不含对手可能因此损失的未来树息、工人移动成本和这笔钱的其他用途。",
              "",f"20 局中，从某一步起现金持续落后直至终局，该步的中位数为 {median(g['first_state_from_which_cash_lead_stays_negative'] for g in games):.0f}。只看终盘会错过第 51–250 步中我方少种的 200 棵树。两边这段夺树次数几乎相同；末段我方反而多种 47 棵。值得检验的改动，是在中段还有时间回本、地点安全且买得起时提早付款，并在尾段评估剩余收益。逐局真实持有收益不能直接给未选行动贴上收益标签。[逐步核账](census.py) · [逐树核账](tenures.py)"]
    (HERE/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
