#!/usr/bin/env python3
"""Newface watch 23:56 run — state file update. Preserves reported_events,
last_trigger, known_old_faces, standing keys; rewrites board_top12, watchlist,
harvest, standing Fable V5 note, appends notes, bumps updated_at."""
import json

P = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/newface_watch_state.json"
s = json.load(open(P))

s["board_top12"] = [
{"name": "RyanX-AIPowercore-M9XJ-it570", "rank": 1, "rating": "2875 ± 34", "total": 110, "wins": 100},
{"name": "claude-AlphaCarbon-Master3s3z", "rank": 2, "rating": "2872 ± 40", "total": 76, "wins": 69},
{"name": "claude-AlphaCarbon-Master3s2", "rank": 3, "rating": "2782 ± 25", "total": 218, "wins": 185},
{"name": "哈基米 v3pro", "rank": 4, "rating": "2712 ± 32", "total": 87, "wins": 64},
{"name": "claude-AlphaCarbon-Master3s", "rank": 5, "rating": "2678 ± 21", "total": 260, "wins": 204},
{"name": "哈基米 v3", "rank": 6, "rating": "2637 ± 26", "total": 132, "wins": 96},
{"name": "哈基米 v3.1", "rank": 7, "rating": "2630 ± 29", "total": 95, "wins": 65},
{"name": "AgyCarbon-P2C-Godslayer-CVI", "rank": 8, "rating": "2580 ± 23", "total": 171, "wins": 107},
{"name": "stephen-K4TacSearch-v3", "rank": 9, "rating": "2552 ± 27", "total": 106, "wins": 61},
{"name": "Fable5.2-Max-V6", "rank": 10, "rating": "2551 ± 27", "total": 107, "wins": 63},
{"name": "stephen-K4TacSearch-v2", "rank": 11, "rating": "2547 ± 19", "total": 284, "wins": 181},
{"name": "Echo3", "rank": 12, "rating": "2542 ± 21", "total": 202, "wins": 124},
]

s["harvest"] = {
"accumulated_files": 2095,
"clone": "ok",
"index_entries": 2094,
"new_harvested": 14,
"snapshot_entries": 400,
"snapshot_matches": 90,
"clone_note": "本趟克隆一次成功（CLONE_EXIT=0）；归集 snapshot 400、matched 90、新增入账 14，accumulated 2095 文件、index 2094 条（index 比文件少 1 条为此前已有的缺口，本趟新增 14 件全部入 index）",
}

s["watchlist"] = [
{
"name": "RyanX-AIPowercore-M9XJ-it570",
"prev_at": "2026-11-13T23:56:00+08:00",
"prev_rank": 1,
"prev_total": 110,
"rank": 1,
"total": 110,
"wins": 100,
"trigger_needs": "第 1 名；117 端 110 局 100 胜 90.9%（榜单 110 一致），场均 +1480.2、中位 +1556.8，先手 48/53、后手 52/57——后手已验证；冻结件 data/m9xj570_matches.json 89 局读回核实未动（sha 9d69b847）；BC 线已收口，只盯梢；最近 10 局 4 胜 +184.5，对 M3s3z 3/10",
},
{
"name": "Krypton-Cloud500",
"prev_at": "2026-11-13T23:56:00+08:00",
"prev_rank": 20,
"prev_total": 161,
"rank": 20,
"total": 161,
"wins": 84,
"trigger_needs": "第 20 名（board 30 直读，在册资格保留）；117 端清单 161 局 84 胜 52.2%（榜单直读 161 一致），场均 +97.9、中位 +93.6，先手 41/76、后手 43/85；清单 160→161 已刷新入仓并读回核实（commit d3dcb693、sha e87a5946）；最近 10 局 2 胜 −762.4，势头仍差；语料 0 件（仓内无 krypton_corpus 目录），ladder-replay-pull 仍堵未派发；继续观察",
},
{
"name": "RyanX-AIPowercore-Lite-V8",
"prev_at": "2026-11-13T23:56:00+08:00",
"prev_rank": 15,
"prev_total": 160,
"rank": 15,
"total": 160,
"wins": 103,
"trigger_needs": "第 15 名（board 30 直读）；117 端清单 160 局 103 胜 64.4%（榜单直读 160 一致），场均 +342.2、中位 +360.9，先手 48/74、后手 55/86；清单 159→160 已刷新入仓并读回核实（commit 0660eda6、sha f0d77421）；最近 10 局 6 胜 +151.1；已过 100 局但不在前十、按规矩不触发；语料 data/lite-v8_corpus 152 件（距上次拉取增量 8 局，按规矩跳过）；继续观察",
},
{
"name": "哈基米 v3pro",
"prev_at": "2026-11-13T23:56:00+08:00",
"prev_rank": 4,
"prev_total": 86,
"rank": 4,
"total": 86,
"wins": 63,
"trigger_needs": "第 4 名；117 端清单 86 局 63 胜 73.3%（榜单直读 87，榜单领先 1 局、容差内，下一趟复核），场均 +465.8、中位 +622.2，先手 29/47、后手 34/39；清单 83→86 已刷新入仓并读回核实（commit b4902382、sha c32ecd76）；80 局刻度事件已在账（不重复报）；语料 data/hakimi3pro_corpus 70 件（增量 16 局，按规矩跳过）；距触发差 14 局",
},
{
"name": "哈基米 v3.1",
"prev_at": "2026-11-13T23:56:00+08:00",
"prev_rank": 7,
"prev_total": 96,
"rank": 7,
"total": 96,
"wins": 65,
"trigger_needs": "第 7 名；117 端清单 96 局 65 胜 67.7%（榜单直读 95，117 端领先 1 局），场均 +440.5、中位 +473.3，先手 29/41、后手 36/55；清单 90→96 已刷新入仓并读回核实（commit 2c732a86、sha bfb35d52）；50/80 局刻度均已报；语料 data/hakimi31_corpus 65 件（增量 31 局，已达派发线，但 ladder-replay-pull 仍堵未派发）；距触发差 4 局，按近速约 16.3 局/小时估计约 15 分钟到 100 局（估计）；最近 10 局 7 胜 +274.5",
},
{
"name": "claude-AlphaCarbon-Master3s3z",
"prev_at": "2026-11-13T23:56:00+08:00",
"prev_rank": 2,
"prev_total": 76,
"rank": 2,
"total": 76,
"wins": 69,
"trigger_needs": "第 2 名；117 端清单 76 局 69 胜 90.8%（榜单 76 一致），场均 +1489.9、中位 +1631.6，先手 32/37、后手 37/39；清单 73→76 已刷新入仓并读回核实（commit 98831140、sha 982dd25a）；50 局刻度已报；语料 data/m3s3z_corpus 45 件（增量 31 局，已达派发线，但 ladder-replay-pull 仍堵未派发）；距 80 局刻度差 4 局，按近速约 14.8 局/小时估计约 16 分钟（估计）；最近 10 局 9 胜 +1057.6；对 it570 7/10",
},
{
"name": "claude-AlphaCarbon-Master3s3",
"prev_at": "2026-11-13T23:56:00+08:00",
"prev_rank": None,
"prev_total": 20,
"rank": None,
"total": 20,
"wins": 16,
"trigger_needs": "仍从榜单消失（board 30 复核不在前 30）；仓内终局清单 data/m3s3_matches.json 20 局（sha 1ba53fb9，与上趟一致）无变化；117 端无法重拉（已从榜单 payload 移除、策略 ID 无法解析），已下线策略无法新增对局，diff 0 结论成立；语料 0 件，ladder-replay-pull 仍堵未派发，请作战室排入 DGX 直拉队列",
},
]

s["standing"]["Fable5.2-Max-V5"] = {
"best_rank": 12,
"note": "用户 2026-11-12 15:01 裁定在册资格保留；语料已入库（data/fable_matches.json 290 条、data/fable_corpus 182 文件）；复刻线作战室已接手（takeover 事件已记），排队归作战室；本趟 117 端清单 290 条（158 胜）289→290 已刷新入仓并读回核实（commit bb5301a6、sha e4fd7943）；23:56 board 直读第 23 名、158/290；语料增量 108 局已达派发线，但 ladder-replay-pull 仍堵未派发；在册资格不受影响",
"rank_now": 23,
"record": "158/290（54.5%）",
"status": "grandfathered",
}

note = (
"｜23:56 本趟：无新面孔（前十 10 名与上趟完全一致）、无刻度、无触发。数据保险：6 份清单刷新入仓并读回核实"
"（krypton 160→161 d3dcb693、lite-v8 159→160 0660eda6、hakimi3pro 83→86 b4902382、hakimi31 90→96 2c732a86、"
"m3s3z 73→76 98831140、fable 289→290 bb5301a6，全部走 main、纯追加零删减；hakimi3 132 无变化未提交；"
"it570 110 局但冻结件 89 未动（sha 9d69b847）、VERIFY-ONLY；it320 终局 153 局、仓内与本地 final 逐 id 比对 diff 0，"
"takeover 在账，不提醒、不执行第 4 步；M3s3 仍消失、终局 20 局 sha 1ba53fb9 未变）。榜单与 117 端差 1 局两处："
"v3pro 榜单 87 vs 117 端 86、v3.1 榜单 95 vs 117 端 96，容差内、下一趟复核。回放语料：m3s3z/hakimi31/hakimi3/krypton/fable"
" 五家增量均达 20 局派发线（31/31/27/161/108），fable6 缺口 22 件亦达线，但 ladder-replay-pull 仍堵（末 run 43334886254、"
"2026-11-13 00:10Z failure，此后无新 run），本趟无派发；另发现 k4v3/fable6 语料缺口在缩小（k4v3 26→6、fable6 27→22），"
"疑似 DGX 直拉在补，请作战室确认；其余成员增量不足 20 局（hakimi3pro 16、lite-v8 8、k4v3 6）按规矩跳过。刻度临近："
"v3.1 96 局距触发差 4 局（估计约 15 分钟），M3s3z 76 局距 80 差 4 局（估计约 16 分钟），v3pro 86 局距触发差 14 局。"
"回放快照：克隆一次成功（CLONE_EXIT=0），snapshot 400、matched 90、新增入账 14，accumulated 2095 文件、index 2094 条。"
"我方最高仍 Echo3 第 12（124/202）；Echo2 第 17（180/306）、Grove1 第 25（191/332）。"
)
s["notes"] = s.get("notes", "") + note
s["updated_at"] = "2026-11-13T23:56:00+08:00"

json.dump(s, open(P, "w"), ensure_ascii=False, indent=1)
print("state updated; reported_events:", len(s["reported_events"]),
"watchlist:", len(s["watchlist"]),
"board12:", [(b["name"], b["total"]) for b in s["board_top12"][:3]])
