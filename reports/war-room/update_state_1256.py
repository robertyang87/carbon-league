#!/usr/bin/env python3
"""newface-watch 12:56 run: write back state file (board_top12, harvest,
watchlist, standing/Fable-V5, last_trigger prepend, updated_at).
Preserves all other fields (notes, reported_events, known_old_faces, run)."""
import json, datetime

P = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/newface_watch_state.json"
s = json.load(open(P))
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
ts = now.strftime("%Y-%m-%dT%H:%M:%S+08:00")

s["board_top12"] = [
    {"name": "claude-AlphaCarbon-Master4s", "rank": 1, "rating": "2889 ± 57", "total": 40, "wins": 38},
    {"name": "RyanX-AIPowercore-M9XJ-630", "rank": 2, "rating": "2857 ± 41", "total": 63, "wins": 55},
    {"name": "claude-AlphaCarbon-Master3s3z", "rank": 3, "rating": "2830 ± 27", "total": 202, "wins": 182},
    {"name": "小肥鱼v5", "rank": 4, "rating": "2824 ± 37", "total": 64, "wins": 50},
    {"name": "RyanX-AIPowercore-M9XJ-it570", "rank": 5, "rating": "2823 ± 28", "total": 187, "wins": 171},
    {"name": "RyanX-AIPowercore-M9XJ-600", "rank": 6, "rating": "2819 ± 35", "total": 87, "wins": 74},
    {"name": "claude-AlphaCarbon-Master3s2", "rank": 7, "rating": "2727 ± 21", "total": 284, "wins": 231},
    {"name": "哈基米 v3pro", "rank": 8, "rating": "2636 ± 21", "total": 249, "wins": 192},
    {"name": "claude-AlphaCarbon-Master3s", "rank": 9, "rating": "2629 ± 19", "total": 333, "wins": 254},
    {"name": "哈基米 v3", "rank": 10, "rating": "2594 ± 20", "total": 283, "wins": 214},
    {"name": "stephen-K4TacSearch-v3", "rank": 11, "rating": "2543 ± 19", "total": 292, "wins": 201},
    {"name": "Echo4", "rank": 12, "rating": "2536 ± 50", "total": 24, "wins": 16},
]

s["harvest"] = {
    "clone": "ok",
    "snapshot_entries": 400,
    "snapshot_matches": 104,
    "new_harvested": 26,
    "accumulated_files": 3278,
    "index_entries": 3277,
    "clone_note": "一次成功（HEAD 43c4578）；归集 snapshot 400、matched 104、新增入账 26，accumulated 3278 文件（3277 回放＋harvest.log）、index 3277 条。",
}

UPD = {
    "claude-AlphaCarbon-Master4s": (1, 1, 24, 41, 38,
        "第 1 名（2889±57，榜单 38/40；117 端清单 41 局 38 胜 92.7%、场均净胜 +1833.0、中位 +1907.3；先手 25/28、后手 13/13——双手均已验证；最近 10 局 9 胜 +2692.3；对 小肥鱼v5 1/3 为唯一明显短板——本趟新负 2 局，对 it570 1/2；首局 11:38，约 32 局/小时，估计距 50 局刻度约 17 分钟（约 13:15，估计）、距 100 局触发约 1.8 小时（约 14:50，估计）；清单 24→41 已刷新入仓读回核实 commit 56f2c93822b7；语料 0 件（缺口 41；工作流仍堵，请作战室排入 DGX 直拉队列）；Master 系第四代，路线待验证，威胁评级高"),
    "RyanX-AIPowercore-M9XJ-630": (2, 2, 62, 63, 55,
        "第 2 名（2857±41，榜单 55/63；117 端 63 局 55 胜 87.3%、场均净胜 +1379.9、中位 +1349.4；先手 28/31、后手 27/32——双手均已验证；最近 10 局 7 胜 +829.4；对 M3s3z 2/6 仍是唯一短板，对 小肥鱼v5 5/7；末局 12:25、近 35 分钟无新局疑似停打，80 局刻度顺延（此前估计 15:30–17:00，估计）；清单 62→63 已刷新入仓读回核实 commit 7f380a173fc1；语料 0 件（缺口 63；工作流仍堵，请作战室排入 DGX 直拉队列）"),
    "小肥鱼v5": (4, 5, 60, 65, 51,
        "第 4 名（2824±37，榜单 50/64；117 端 65 局 51 胜 78.5%、场均净胜 +828.0、中位 +740.3；先手 25/32、后手 26/33——双手均已验证；最近 10 局 5 胜 +253.3（上趟 6 胜 −18.1）；对 630 2/7、对 600 3/6 偏弱，对 M3s3z 5/6、对 Master4s 2/3——本趟连赢榜首 2 局；近速约 9 局/小时放缓，估计距 80 局刻度约 1.5 小时（约 14:30，估计）；清单 60→65 已刷新入仓读回核实 commit 46d3941027e3；语料 0 件（缺口 65；工作流仍堵，请作战室排入 DGX 直拉队列）；50 局刻度已报"),
    "RyanX-AIPowercore-M9XJ-it570": (5, 4, 187, 187, 171,
        "第 5 名（2823±28，榜单 171/187；117 端 187 局 171 胜，无变化）；冻结件 data/m9xj570_matches.json 89 局读回核实未动（sha 9d69b8472d70，frozen_sha_ok）；语料 data/m9xj570_corpus 90 件（冻结）；已触发、不再推进"),
    "Krypton-Cloud500": (32, 31, 227, 227, 119,
        "第 32 名（board 60 直读，榜单 119/227；117 端 227 局无变化）；清单 227 已在仓无变化；语料 data/krypton_corpus 206 件（缺口 21 局，刚过 20 局派发线；工作流仍堵，未派发）"),
    "RyanX-AIPowercore-Lite-V8": (18, 17, 304, 306, 198,
        "第 18 名（light 榜第 1，2486±18，榜单 198/306；117 端 306 局 198 胜，+2）；清单 304→306 已刷新入仓读回核实 commit 4170071d9171；语料 data/lite-v8_corpus 268 件（缺口 38 局；工作流仍堵，未派发）"),
    "哈基米 v3.1": (None, None, 96, 96, 65,
        "仍从榜单消失（board 30 直读不在列）；117 端 96 局无变化；终局清单 96 局已在仓并读回核实（sha bfb35d528305）；语料 data/hakimi31_corpus 96 件（终局缺口 0，已完整）；未触发、无复刻线在跑；留册盯梢"),
    "claude-AlphaCarbon-Master3s3": (None, None, 20, 20, 16,
        "仍从榜单消失（board 30 直读不在列）；117 端 20 局无变化；仓内终局清单 data/m3s3_matches.json 20 局无变化（sha 1ba53fb9fa15）；语料 data/m3s3_corpus 20 件（终局缺口 0，已完整）；留册盯梢"),
    "Krypton-Search256-v1": (None, None, 176, 176, 97,
        "仍从榜单消失（board 30 直读不在列）；117 端 176 局无变化；仓内清单 176 局读回核实无变化（sha 23257d53711e）；语料 data/krypton-search256_corpus 136 件（缺口 40 局；工作流仍堵，未派发）；留册盯梢"),
    "Krypton-ProSearch-v1": (None, None, 216, 216, 136,
        "仍从榜单消失（board 30 直读不在列）；117 端 216 局无变化；仓内清单 216 局读回核实无变化（sha 4c73f06e9e20）；语料 data/krypton-prosearch_corpus 158 件（缺口 58 局；工作流仍堵，未派发）；留册盯梢"),
    "Krypton-Search256-v2": (None, None, 125, 125, 63,
        "仍从榜单消失（board 30 直读不在列）；117 端 125 局无变化；仓内清单 125 局读回核实无变化（sha c6184d34dff0）；语料 data/krypton-search256-v2_corpus 105 件（缺口 20 局；工作流仍堵，未派发）；留册盯梢"),
    "Krypton-CloudSearch-v1": (23, 22, 258, 261, 160,
        "第 23 名（board 30 直读，榜单 160/260；117 端 261 局，+3）；清单 258→261 已刷新入仓读回核实 commit 650ed9f04af3；语料 data/krypton-cloudsearch_corpus 125 件（缺口 136 局；工作流仍堵，未派发）"),
    "Krypton-CS-Plant-v1": (25, 24, 235, 238, 148,
        "第 25 名（board 30 直读，榜单 148/237；117 端 238 局，+3）；清单 235→238 已刷新入仓读回核实 commit b735d4c89026；语料 data/krypton-csplant_corpus 192 件（缺口 46 局；工作流仍堵，未派发）"),
    "Fable5.2-Max-V6": (15, 14, 299, 300, 190,
        "第 15 名（board 30 直读，榜单 190/300；117 端 300 局，+1）；清单 299→300 已刷新入仓读回核实 commit 36ae57e0c2d7；语料 data/fable6_corpus 256 件（缺口 44 局；工作流仍堵，未派发）"),
    "Krypton-M128Search-v1": (22, 19, 201, 203, 128,
        "第 22 名（board 30 直读，榜单 128/203；117 端 203 局，+2）；清单 201→203 已刷新入仓读回核实 commit bd77238a43b7；语料 0 件（缺口 203；工作流仍堵，请作战室排入 DGX 直拉队列）"),
    "RyanX-AIPowercore-M9XJ-600": (6, 6, 87, 87, 74,
        "第 6 名（2819±35，榜单 74/87；117 端 87 局 74 胜 85.1%、场均净胜 +1265.3、中位 +1419.1；先手 37/44、后手 37/43——双手均已验证；最近 10 局 5 胜 +54.4 走冷；对 M3s3z 3/9 仍是唯一短板，对 小肥鱼v5 3/6；末局 12:07、近 55 分钟无新局疑似停打，距 100 局触发 13 局、时间未定（此前估计 3–6 小时，估计）；清单 87 无变化未提交；语料 53 件（缺口 34；距上次拉取增量 34 局已达 20 局线，工作流仍堵，请作战室排入 DGX 直拉队列）；80 局刻度已报"),
}
for w in s["watchlist"]:
    u = UPD.get(w["name"])
    assert u is not None, f"missing update for {w['name']}"
    rank, prev_rank, prev_total, total, wins, needs = u
    assert w["rank"] == prev_rank and w["total"] == prev_total, (
        f"stale base for {w['name']}: state rank/total {w['rank']}/{w['total']} != expected prev {prev_rank}/{prev_total}")
    w["prev_rank"], w["prev_total"] = prev_rank, prev_total
    w["prev_at"] = ts
    w["rank"], w["total"], w["wins"] = rank, total, wins
    w["trigger_needs"] = needs

s["standing"]["Fable5.2-Max-V5"] = {
    "best_rank": 12,
    "note": "用户 2026-11-12 15:01 裁定在册资格保留；语料已入库（data/fable_matches.json 173 条、data/fable_corpus 313 文件，语料现比清单多 140 件）；复刻线作战室已接手（takeover 事件已记），排队归作战室；本趟清单 183→173（+1 −11，commit 844d3aaaa663，读回核实 173，117 端 75 胜），榜单总数 336（178 胜）不变 vs 117 端 173 差 163；自 332 起累计 −159；榜单侧清退/重索引之疑未消，anomaly 事件今日已记，不重复；在册资格不受影响；board 60 直读第 33 名（178/336）",
    "rank_now": 33,
    "record": "178/336",
    "status": "grandfathered",
}

note = ("｜13:00 本趟：前十与上趟完全相同，无新面孔、无刻度、无触发，本趟不报作战室。名次微调：小肥鱼v5 升至第 4（2824±37，50/64）、it570 退至第 5（2823±28，171/187）。"
"Master4s 第 1 名（2889±57；117 端 41 局 38 胜 92.7%、场均净胜 +1833.0、中位 +1907.3；先手 25/28、后手 13/13；最近 10 局 9 胜 +2692.3；对 小肥鱼v5 1/3 为唯一明显短板——本趟新负 2 局，对 it570 1/2；约 32 局/小时，估计距 50 局刻度约 17 分钟（约 13:15，估计）、距 100 局触发约 1.8 小时（约 14:50，估计）；清单 24→41 已刷新入仓读回核实 commit 56f2c93822b7；语料 0 件（缺口 41，刚过 20 局派发线））。"
"小肥鱼v5 第 4（2824±37；117 端 65 局 51 胜 78.5%、场均净胜 +828.0、中位 +740.3；先手 25/32、后手 26/33；最近 10 局 5 胜 +253.3；对 630 2/7、对 600 3/6 偏弱，对 M3s3z 5/6、对 Master4s 2/3——本趟连赢榜首 2 局；近速约 9 局/小时放缓，估计距 80 局刻度约 1.5 小时（约 14:30，估计）；清单 60→65 commit 46d3941027e3；语料 0 件缺口 65）。"
"M9XJ-630 第 2（2857±41；117 端 63 局 55 胜 87.3%、场均净胜 +1379.9、中位 +1349.4；先手 28/31、后手 27/32；最近 10 局 7 胜 +829.4；对 M3s3z 2/6 仍是唯一短板；末局 12:25、近 35 分钟无新局疑似停打，80 局刻度顺延；清单 62→63 commit 7f380a173fc1；语料 0 件缺口 63）。"
"M9XJ-600 第 6（2819±35；117 端 87 局 74 胜 85.1%、场均净胜 +1265.3、中位 +1419.1；先手 37/44、后手 37/43；最近 10 局 5 胜 +54.4 走冷；对 M3s3z 3/9 仍是唯一短板；末局 12:07、近 55 分钟无新局疑似停打，距 100 局触发 13 局、时间未定；清单 87 无变化；语料 53 件缺口 34）。"
"it570 第 5（117 端 187 局 171 胜，无变化），冻结件 89 未动（sha 9d69b8472d70，frozen_sha_ok）；it320 live 153 局 146 胜无变化，已接手且已从榜单消失，不发提醒。"
"清单刷新 10 份并读回核实：master4s 24→41（56f2c93822b7）、m9xj630 62→63（7f380a173fc1）、xiaofeiyu5 60→65（46d3941027e3）、fable6 299→300（36ae57e0c2d7）、k4v2 379→380（6093bf126527）、lite-v8 304→306（4170071d9171）、krypton-cloudsearch 258→261（650ed9f04af3）、krypton-csplant 235→238（b735d4c89026）、krypton-m128 201→203（bd77238a43b7）、fable 183→173（844d3aaaa663，见下）；m9xj600 87、m3s3z 202、hakimi3pro 248、k4v3 292、krypton 227、m9xj(it260) 108、m9xj320 153、hakimi31 96、m3s3 20、krypton-prosearch 216、krypton-search256 176、krypton-search256-v2 125 无变化未提交；名字守卫全部通过（0 坏）；id 集单调只增、零删减，唯 Fable V5 例外（见下）。"
"⚠️Fable V5 清单删减继续：117 端 183→173（+1 −11），已按 117 现状提交（commit 844d3aaaa663，读回 173，117 端 75 胜）；榜单总数 336（178 胜）不变 vs 117 端 173 差 163；自 332 起累计 −159；语料 313 件现比清单多 140 件；榜单侧清退/重索引之疑未消，anomaly 事件今日已记，不重复。"
"语料缺口：krypton-m128 203（无目录）、krypton-cloudsearch 136、krypton-prosearch 58、m3s3z 69、hakimi3pro 67、xiaofeiyu5 65、m9xj630 63、k4v2 54、krypton-csplant 46、fable6 44、k4v3 41、lite-v8 38、master4s 41（无目录，刚过 20 局线）、m9xj600 34、krypton 21、krypton-search256-v2 20（均达/越过 20 局派发线）、fable −140；ladder-replay-pull 仍堵（末 run 43334886254、2026-11-13 00:10Z failure，本趟已复核、此后无新 run）且 Carbon 线绕开 Actions，本趟不派发，请作战室排入 DGX 直拉队列（master4s、m9xj630、xiaofeiyu5、krypton-m128、krypton-cloudsearch 优先；m9xj600 距上次拉取增量 34 已达线）。"
"回放快照：克隆一次成功（HEAD 43c4578），snapshot 400、matched 104、新增入账 26，accumulated 3278 文件、index 3277 条。"
"board 30 动向：Echo4 新名字第 12/13 位游走（board 12 直读第 12、2536±50 16/24；board 30 直读第 13、2537±50；117 端刚拉 30 局 21 胜 70.0%、场均净胜 +425.0、中位 +600.1；先手 13/19、后手 8/11；最近 10 局 8 胜 +348.8；首局 12:35，约 64 局/小时爬升极快，对手多为中游、未碰顶级强家；未进前十，按规矩不做新面孔分析，下趟留意是否进前十）；Fable5.2-Max-V7 第 12（2538±39，30/44，+10）；Godslayer-CVI 第 14（187/298）；Fable6 第 15（190/300）；Echo3 第 16（196/314）；K4v2 第 17（235/380）；Lite-V8 第 18（198/306）；哈基米 v2 pro 第 19（248/372）；K4v1 第 20（242/395）；Krypton-M144 第 21（67/113）；Krypton-M128 第 22（128/203）；Krypton-CloudSearch 第 23（160/261）；Carago-V2微调 第 24（229/395）；Krypton-CS-Plant 第 25（148/238）；安生N23 第 26（41/73）；Echo2 第 27（230/397）；Master2s 第 28（355/518）；Godslayer-CV 第 29（199/379）；哈基米 v2 第 30（87/171）；哈基米 v3.1、M3s3 与三名消失 Krypton 成员 board 30 直读仍不在列，117 端局数均无变化。")
s["last_trigger"] = note + s["last_trigger"]
s["updated_at"] = ts
json.dump(s, open(P, "w"), ensure_ascii=False)
print("updated_at", ts)
print("watchlist entries:", len(s["watchlist"]))
print("reported_events:", len(s["reported_events"]))
print("last_trigger head:", s["last_trigger"][:120])
