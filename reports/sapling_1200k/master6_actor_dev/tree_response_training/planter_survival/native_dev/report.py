"""Write the development decision and every registered native outcome from sealed evidence."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_bytes())


def render():
    gate = load("gate.json")
    lines = [
        "# 公开撤离选择器的开发结果",
        "",
        "| 来源图 | 当拍改令 | 救回 | 反伤 | 漏救 | 放弃已成交树 | 对应支出 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in ("03", "04"):
        d = load("transfer" + b + ".json")
        for s, r in d["maps"].items():
            lines.append(
                f"| {s} | {r['edits']} | {r['saved']} | {r['harmed']} | "
                f"{r['missed_rescues']} | {r['foregone_payment']} | {r['foregone_debit']:.2f} |"
            )
    lines += [
        "",
        "03 是已见图的工程核验。04 在候选字节冻结后首次测公开选择器的替代动作。两批四图均在双方既有同时命令下只分支一步。",
    ]
    if gate["decision"] == "do_not_run_native":
        lines += ["", "逐图净救或零反伤条件未全部达到。八局开发赛取消，正式考卷未开。"]
    else:
        r = load("comparison.json")
        wins = r["wins"]
        lines += [
            "",
            f"八局整场开发完成。父代、候选各打 4 局，分别赢 {wins['parent']} 局、{wins['candidate']} 局。"
            f"按图等权的平均分差变化为 {r['equal_map_delta_margin']:+.2f}，平均 U 变化为 {r['equal_map_delta_U']:+.5f}。"
            f"候选实际改了 {r['candidate_actual_edits']} 条工人命令。",
            "",
            "| 地图 / 席 | 父代 本方－考官 | 候选 本方－考官 | 分差变化 | ΔU | 候选改令 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for pair in r["pairs"]:
            s, p = pair["seed"], pair["seat"]
            a, b = pair["parent"], pair["candidate"]
            sa, sb = a["official_reward_by_seat"], b["official_reward_by_seat"]
            lines += [
                f"| {s} / {p} | {sa[p]:.2f} − {sa[1-p]:.2f} | {sb[p]:.2f} − {sb[1-p]:.2f} | "
                f"{pair['delta_margin']:+.2f} | {pair['delta_U']:+.5f} | {b['attribution']['actual_edits']} |"
            ]
        status = "满足" if r["directional_followup_signal"] else "未满足"
        lines += [
            "",
            f"此批{status}预注册的继续验证条件。所有席位均计入。样本只有两张图，原版搜索使用真实墙钟和未固定温度采样，"
            "同图两臂的分差包含考官路径波动。一次救工改变后续局面，不能用单步获救数换算整局收益。",
            "",
            "[全局得分、双方资金、搜索调用与真实改令](comparison.json)",
        ]
    lines += [
        "",
        "[预注册](protocol.json) · [03 四席](transfer03.json) · [04 四席](transfer04.json) · [开场裁决](gate.json)",
    ]
    if (HERE / "actual_edits.json").exists():
        actual = load("actual_edits.json")
        totals = actual["totals"]
        lines.extend(
            [
                "",
                "## 候选实际改令的当拍结果",
                "",
                f"在候选自己的真实轨迹上，逐一恢复原来的停留命令，其余工人和对手保留当拍已经提交的动作。在 {totals['changed']} 次实际撤离中，"
                f"{totals['saved']} 次从当拍阵亡转为存活，{totals['harmed']} 次从存活转为阵亡；其余命令不影响当拍生死。"
                f"放弃原可成交 {totals['foregone_purchases']} 笔树，原付款合计 {totals['foregone_debit']:.2f}。这些是决策已经改变后又遇到的真实局面，不能将救回的次数解释为同一批独立工人，或将它们相加作为胜率增益。",
                "",
                "| 地图 / 席 | 实际改令 | 救回 | 反伤 | 放弃已成交树 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for g in actual["games"]:
            lines.append(
                f"| {g['seed']} / {g['seat']} | {g['changed']} | {g['saved']} | {g['harmed']} | {g['foregone_purchases']} |"
            )
        lines += [
            "",
            "[逐步工号、物理命令、死亡与真实付款](actual_edits.json) · [当拍反事实复算](actual_edits.py)",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (HERE / "RESULTS.md").write_text(render())
