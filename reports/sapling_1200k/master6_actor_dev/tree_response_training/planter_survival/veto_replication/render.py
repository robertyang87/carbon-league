"""Render the frozen veto results. Each row refers to the original v1 trajectory."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render(r):
    totals = r["total"]
    fit = totals["policies"]["fitted"]
    by_map = list(r["maps"].values())
    lines = [
        "# 付款前撤走种树工：两张新来源图的逐步复验",
        "",
        f"在 {len(by_map)} 张图双席的 {totals['all_source_planter_roots']:,} 个种树工决策中，"
        f"{totals['eligible_payment_intents']} 个同时满足三个条件：v1 原令停留、付得起树价、存在合法且与友军无冲突的移动。"
        f"冻结风险头会改其中 {fit['edits']} 条，真实配对后救回 {fit['saved']} 个原本会死的工人，"
        f"另使 {fit['harmed']} 个原本存活的工人死亡。它放弃了 {fit['foregone_purchases']} 笔本来会成交的种植或夺树，"
        f"对应当拍实际支出 {fit['foregone_debit']:.2f}。",
        "",
        "## 按原图与物理席核账",
        "",
        "| 地图 | 席 | 可付停令 | 原停阵亡 | 改走可救 | 触发 / 实改 | 救回 | 反伤 | 可救漏掉 | 两臂皆死 | 放弃实付笔数 / 金额 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, group in r["physical_map_seats"].items():
        seed, seat = key.split("/")
        a = group["policies"]["fitted"]
        lines.append(
            f"| {seed} | {seat} | {a['eligible']} | {a['v1_deaths']} | {a['savable']} | "
            f"{a['all_isolated_triggers']} / {a['edits']} | {a['saved']} | {a['harmed']} | "
            f"{a['unrescued_preventable']} | {a['both_die']} | {a['foregone_purchases']} / {a['foregone_debit']:.2f} |"
        )
    lines += [
        "",
        "## 同一机会与同一个移动的对照",
        "",
        "| 地图 | 策略 | 改令 | 救回 | 反伤 | 原停阵亡仍未救 | 放弃实付笔数 / 金额 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "v1": "原 v1",
        "fitted": "冻结概率头",
        "geometry": "几何可达",
        "contact_rate": "训练图内统一可达阵亡率",
    }
    for seed, group in r["maps"].items():
        for policy, label in labels.items():
            a = group["policies"][policy]
            lines.append(
                f"| {seed} | {label} | {a['edits']} | {a['saved']} | {a['harmed']} | "
                f"{a['unrescued_v1_deaths']} | {a['foregone_purchases']} / {a['foregone_debit']:.2f} |"
            )
    key = lambda event: tuple(event[:4])
    geo_keys = {key(event) for event in r["edits"]["geometry"]}
    only = [e for e in r["edits"]["fitted"] if key(e) not in geo_keys]
    lines += [
        "",
        "## 概率头额外触发的局面",
        "",
        "| 地图 / 席 / 回合 | 工人 | 移动 | 预测阵亡概率 停 / 移 | 次步实际存活 停 / 移 | 放弃实付 |",
        "|---|---|---|---:|---|---:|",
    ]
    for seed, seat, t, uid, move, hp, mp, hdead, mdead, kind, paid in only:
        lines.append(
            f"| {seed} / {seat} / {t} | {uid} | {move} | {hp:.3f} / {mp:.3f} | "
            f"{'死' if hdead else '活'} / {'死' if mdead else '活'} | {paid:.2f} |"
        )
    scores = r["equal_map_brier"]

    def val(v):
        return f"{v:.5f}" if v is not None else "未定义"

    lines.extend(
        [
            "",
            f"在有合格机会的 {r['equally_weighted_maps_with_eligible_roots']} 张图上，按图、同图按根、同根按臂等权，"
            f"下一步阵亡的 Brier 分数为：冻结概率头 {val(scores['fitted'])}、几何零一分 {val(scores['geometric'])}、"
            f"仅使用训练图可达范围内统一阵亡率 {val(scores['contact_rate'])}。分数越小表示概率误差越低。",
            "",
            "原 v1 命令和现金前缀来自归档源局。替代移动由冻结的采样器选取；删去敌方同拍命令、结局及超出合法性所需的其余字段，四局所有采样记录仍选出相同移动。"
            "到标签计算时，双方其余命令沿用该拍已提交指令；官方 Board 与 FastCarbon 必须对每个分支的工人、树及碳结果逐项一致。"
            "同一席同一回合有多个触发时，只对公开 UID 排序第一人计改令。逐工身份、物理命令、双方死亡以及本应归属该工人的支出保存在证据中。",
            "",
            "这些数字覆盖既有 v1 轨迹上的单步干预。目标是核实能否救回当拍的工人；转移后接下来的双方行动、树的未来收入、整局比分均未测量。"
            "所列金额是没有发生的购树支出。地图数只有两张，保留每图方向，不据此估计胜率或签发新参赛包。",
            "",
            "[预注册](protocol.json) · [逐根两臂汇总、改令与哈希](results.json) · [复算脚本](run.py)",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (HERE / "README.md").write_text(render(json.loads((HERE / "results.json").read_bytes())))
