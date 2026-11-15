#!/usr/bin/env python3
"""State update for newface-watch 10:56 run 2026-11-15. Preserves all fields."""
import json, datetime

P = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/newface_watch_state.json"
s = json.load(open(P))

s["board_top12"] = [
    {"name": "RyanX-AIPowercore-M9XJ-630", "rank": 1, "rating": "2857 ± 49", "total": 47, "wins": 44},
    {"name": "claude-AlphaCarbon-Master3s3z", "rank": 2, "rating": "2844 ± 29", "total": 195, "wins": 180},
    {"name": "RyanX-AIPowercore-M9XJ-600", "rank": 3, "rating": "2839 ± 38", "total": 78, "wins": 70},
    {"name": "RyanX-AIPowercore-M9XJ-it570", "rank": 4, "rating": "2830 ± 28", "total": 182, "wins": 169},
    {"name": "小肥鱼v5", "rank": 5, "rating": "2776 ± 46", "total": 41, "wins": 34},
    {"name": "claude-AlphaCarbon-Master3s2", "rank": 6, "rating": "2736 ± 22", "total": 281, "wins": 231},
    {"name": "哈基米 v3pro", "rank": 7, "rating": "2645 ± 21", "total": 243, "wins": 192},
    {"name": "claude-AlphaCarbon-Master3s", "rank": 8, "rating": "2636 ± 19", "total": 328, "wins": 252},
    {"name": "哈基米 v3", "rank": 9, "rating": "2601 ± 20", "total": 278, "wins": 212},
    {"name": "stephen-K4TacSearch-v3", "rank": 10, "rating": "2549 ± 19", "total": 287, "wins": 199},
    {"name": "AgyCarbon-P2C-Godslayer-CVI", "rank": 11, "rating": "2525 ± 19", "total": 290, "wins": 184},
    {"name": "Fable5.2-Max-V6", "rank": 12, "rating": "2507 ± 18", "total": 297, "wins": 189},
]
s["harvest"] = {
    "clone": "ok",
    "snapshot_entries": 400,
    "snapshot_matches": 73,
    "new_harvested": 10,
    "accumulated_files": 3208,
    "index_entries": 3207,
    "clone_note": "一次成功（HEAD ea5fe7f277333286f843bd5c7bfeb8b5f0c5306a，2026-11-15 10:50:58 +0800）；归集 snapshot 400、matched 73、新增入账 10，accumulated 3208 文件（3207 回放＋harvest.log）、index 3207 条。",
}

upd = {
 "RyanX-AIPowercore-M9XJ-630": (1, 49, 45,
   "第 1 名（2857±49，榜单 44/47；117 端 49 局 45 胜 91.8%、最近 10 局 9 胜；先手 21/22、后手 24/27——双手均已验证；10:26→本趟 42→49，约 14 局/小时，估计距 50 局刻度约 5 分钟）。清单 42→49 已刷新入仓并读回核实，commit c8f683d47994；语料 0 件（缺口 49；工作流仍堵，请作战室排入 DGX 直拉队列）"),
 "小肥鱼v5": (5, 43, 35,
   "第 5 名（2776±46，榜单 34/41；117 端 43 局 35 胜 81.4%、最近 10 局 8 胜；先手 18/24、后手 17/19；10:26→本趟 31→43，约 24 局/小时，估计距 50 局刻度约 18 分钟）。清单 31→43 已刷新入仓并读回核实，commit 6edfbfacd98f；语料 0 件（缺口 43；工作流仍堵，请作战室排入 DGX 直拉队列）"),
 "RyanX-AIPowercore-M9XJ-it570": (4, 182, 169,
   "第 4 名（2830±28，榜单 169/182；117 端 182 局 169 胜，无变化）；冻结件 data/m9xj570_matches.json 89 局读回核实未动（sha 9d69b8472d70）；语料 data/m9xj570_corpus 90 件"),
 "Krypton-Cloud500": (29, 227, 119,
   "第 29 名（board 30 直读，榜单 119/227；117 端 227 局，无变化）；清单 227 已在仓无变化；语料 data/krypton_corpus 206 件（缺口 21 局，刚过 20 局派发线；工作流仍堵，未派发）"),
 "RyanX-AIPowercore-Lite-V8": (15, 301, 198,
   "第 15 名（light 榜第 1，2494±18，榜单 198/301；117 端 301 局 198 胜，+1）；清单 300→301 已刷新入仓并读回核实，commit ad24920ab1bf；语料 data/lite-v8_corpus 268 件（缺口 33 局；工作流仍堵，未派发）"),
 "哈基米 v3.1": (None, 96, 65,
   "仍从榜单消失（board 30 直读不在列）；117 端 96 局无变化；终局清单 96 局已在仓并读回核实（sha bfb35d528305）；语料 data/hakimi31_corpus 96 件（终局缺口 0，已完整）；未触发、无复刻线在跑；留册盯梢"),
 "claude-AlphaCarbon-Master3s3": (None, 20, 16,
   "仍从榜单消失（board 30 直读不在列）；117 端 20 局无变化；仓内终局清单 data/m3s3_matches.json 20 局无变化（sha 1ba53fb9fa15）；语料 data/m3s3_corpus 20 件（终局缺口 0，已完整）；留册盯梢"),
 "Krypton-Search256-v1": (None, 176, 97,
   "仍从榜单消失（board 30 直读不在列）；117 端 176 局无变化（已在仓，sha 23257d53711e）；语料 data/krypton-search256_corpus 136 件（缺口 40 局；工作流仍堵，未派发）；留册盯梢"),
 "Krypton-ProSearch-v1": (None, 216, 136,
   "仍从榜单消失（board 30 直读不在列）；117 端 216 局无变化（已在仓，sha 4c73f06e9e20）；语料 data/krypton-prosearch_corpus 158 件（缺口 58 局；工作流仍堵，未派发）；留册盯梢"),
 "Krypton-Search256-v2": (None, 125, 63,
   "仍从榜单消失（board 30 直读不在列）；117 端 125 局无变化（已在仓，sha c6184d34dff0）；语料 data/krypton-search256-v2_corpus 105 件（缺口 20 局，刚达派发线；工作流仍堵，未派发）；留册盯梢"),
 "Krypton-CloudSearch-v1": (20, 255, 160,
   "第 20 名（board 30 直读，榜单 160/255；117 端 255 局 160 胜，+2）；清单 253→255 已刷新入仓并读回核实，commit d5a1202c7d43；语料 data/krypton-cloudsearch_corpus 125 件（缺口 130 局；工作流仍堵，未派发）"),
 "Krypton-CS-Plant-v1": (23, 234, 148,
   "第 23 名（board 30 直读，榜单 148/234；117 端 234 局，无变化）；清单 234 已在仓无变化；语料 data/krypton-csplant_corpus 192 件（缺口 42 局；工作流仍堵，未派发）"),
 "Fable5.2-Max-V6": (12, 297, 189,
   "第 12 名（2507±18，榜单 189/297；117 端 297 局 189 胜，+1）；清单 296→297 已刷新入仓并读回核实，commit c6f8762c770f；语料 data/fable6_corpus 256 件（缺口 41 局；工作流仍堵，未派发）"),
 "Krypton-M128Search-v1": (19, 199, 127,
   "第 19 名（board 30 直读，榜单 126/198；117 端 199 局 127 胜，+2）；清单 197→199 已刷新入仓并读回核实，commit 2a8f6d9cf108；语料无目录（缺口 199；工作流仍堵，未派发）"),
 "RyanX-AIPowercore-M9XJ-600": (3, 79, 70,
   "第 3 名（2839±38，榜单 70/78；117 端 79 局 70 胜 88.6%、最近 10 局 9 胜；先手 34/39、后手 36/40——双手均已验证；10:26→本趟 75→79，约 8 局/小时，估计距 80 局刻度约 8 分钟）。清单 75→79 已刷新入仓并读回核实，commit 09cc4b872be3；语料 data/m9xj600_corpus 53 件（缺口 26 局，距上次拉取增量 26，已达派发线；工作流仍堵，未派发）"),
}
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")
for w in s["watchlist"]:
    if w["name"] in upd:
        r, t, wn, note = upd[w["name"]]
        w["prev_at"] = w.get("prev_at")
        w["prev_rank"] = w.get("rank")
        w["prev_total"] = w.get("total")
        w["rank"] = r
        w["total"] = t
        w["wins"] = wn
        w["trigger_needs"] = note

entry = ("｜11:00 本趟：前十与上趟完全相同，无新面孔、无刻度、无触发——但两名成员距刻度仅差 1 局：M9XJ-630 第 1 名（2857±49；117 端 49 局 45 胜 91.8%、最近 10 局 9 胜；先手 21/22、后手 24/27；10:26→本趟 42→49，约 14 局/小时，估计距 50 局刻度约 5 分钟）、M9XJ-600 第 3 名（2839±38；117 端 79 局 70 胜 88.6%、最近 10 局 9 胜；先手 34/39、后手 36/40；10:26→本趟 75→79，约 8 局/小时，估计距 80 局刻度约 8 分钟）；小肥鱼v5 第 5 名（2776±46；117 端 43 局 35 胜 81.4%、最近 10 局 8 胜；先手 18/24、后手 17/19；10:26→本趟 31→43，约 24 局/小时，估计距 50 局刻度约 18 分钟）。三人均未越线，本趟不报作战室，下趟必见至少两起刻度事件。"
 "it570 第 4（117 端 182 局 169 胜，无变化），冻结件 89 未动（sha 9d69b8472d70）；it320 live 153 局 146 胜无变化，已接手且已从榜单消失，不发提醒。"
 "清单刷新 11 份并读回核实：m9xj630 42→49（c8f683d47994）、m3s3z 194→196（817b56b44fa6）、m9xj600 75→79（09cc4b872be3）、xiaofeiyu5 31→43（6edfbfacd98f）、hakimi3pro 241→242（90ed7677aa6f）、k4v2 372→374（100af63fa662）、fable6 296→297（c6f8762c770f）、lite-v8 300→301（ad24920ab1bf）、krypton-cloudsearch 253→255（d5a1202c7d43）、krypton-m128 197→199（2a8f6d9cf108）、fable 218→209（2e69998ff2d2，见下）；k4v3 287、krypton-csplant 234、krypton 227、m9xj(it260) 108、hakimi31 96、m3s3 20、krypton-search256 176、krypton-prosearch 216、krypton-search256-v2 125 无变化未提交；名字守卫 21 份清单全过（0 坏）；id 集单调只增、零删减，唯 Fable V5 例外（见下）。"
 "⚠️Fable V5 清单删减继续：117 端 218→209（+0 −9），已按 117 现状提交（commit 2e69998ff2d2，读回 209，117 端 100 胜）；榜单总数 336（178 胜）不变 vs 117 端 209 差 127；自 332 起累计 −123；语料 313 件现比清单多 104 件；榜单侧清退/重索引之疑未消，anomaly 事件已记（今日已记，不重复）。"
 "语料缺口：m9xj630 49（无目录）、xiaofeiyu5 43（无目录）、krypton-m128 199（无目录）、krypton-cloudsearch 130、krypton-prosearch 58、m3s3z 63、hakimi3pro 61、k4v2 48、krypton-csplant 42、fable6 41、krypton-search256 40、k4v3 36、lite-v8 33、m9xj600 26、krypton 21、krypton-search256-v2 20（均达/越过 20 局派发线）、fable −104；ladder-replay-pull 仍堵（末 run 43334886254、2026-11-13 00:10Z failure，本趟已复核、此后无新 run）且 Carbon 线绕开 Actions，本趟不派发，请作战室排入 DGX 直拉队列（m9xj630、xiaofeiyu5、krypton-m128、krypton-cloudsearch 优先）。"
 "回放快照：克隆一次成功（HEAD ea5fe7f2，2026-11-15 10:50:58 +0800），snapshot 400、matched 73、新增入账 10，accumulated 3208 文件、index 3207 条。"
 "board 30 动向：Krypton-M144Search-v1 第 17（66/109，+6，仍在前十外，100 局事件已于 09:56 记）；Echo3 第 14（195/310，无变化）；安生N23 第 22（38/64，light）；豆包0.1 第 30（52/112，light）；哈基米 v3.1、M3s3 与三名消失 Krypton 成员 board 30 直读仍不在列，117 端局数均无变化。")
s["last_trigger"] = entry + s.get("last_trigger", "")

st = s.setdefault("standing", {}).get("Fable5.2-Max-V5", {})
st.update({
    "best_rank": 12,
    "rank_now": 31,
    "record": "178/336",
    "status": "grandfathered",
    "note": "用户 2026-11-12 15:01 裁定在册资格保留；语料已入库（data/fable_matches.json 209 条、data/fable_corpus 313 文件，语料现比清单多 104 件）；复刻线作战室已接手（takeover 事件已记），排队归作战室；本趟清单 218→209（+0 −9，commit 2e69998ff2d2，读回核实 209，117 端 100 胜），榜单总数 336（178 胜）不变 vs 117 端 209 差 127；自 332 起累计 −123；榜单侧清退/重索引之疑未消，anomaly 事件已记；在册资格不受影响",
})
s["standing"]["Fable5.2-Max-V5"] = st
s["updated_at"] = now
json.dump(s, open(P, "w"), ensure_ascii=False, indent=1)
print("state updated at", now)
print("board_top12:", len(s["board_top12"]), "watchlist:", len(s["watchlist"]),
      "events:", len(s["reported_events"]))
