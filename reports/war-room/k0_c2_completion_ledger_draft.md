# K0 前置④ C2 分区完工总账（草稿，完工时填实）

## 运行身份
- 机器：spot-31（i-uf6023700gydxbxk4zl6，100.98.67.20）
- 分区：selfplay-994508 … selfplay-994599，共 92 局（gid 0–91，局单序）
- 局单：/root/carbon/k0pkg/games_c2_full.json，md5 87c948fa02d9a98c4d31e9f9d97fb78a
- 启动：2026-11-14T19:34:02+08:00，MAIN_PID = PGID = 45462，workers 62
- config_sha256：9a2b2440e9267680c9b27a1d51a50db9cacdbd04582ae101ef54abaa86d8d10c
- weights_md5：06a03d9cfeec239af3065a49fefec382
- games_sha256（分区）：2c87b4956617ec98862514e173b7441476c9a5b929d54907dd057b47a58c0c76
- selected_sha256（分区）：088899c3501f1071142980aa785feb8655cd701025c293132449af8034e7d0f9
- code_md5s（7 件）：k0_annotate 7f567a2e / gate e922d9d8 / teacher c6ade5ce / feat 2a45b08b / search 94f7fb64 / np d8c15ad4 / fastsim aefe6c83
- 窗口 [200,299)，seed 12345，pi 1.0/1.0，train_temp 1.0

## 首片（标定点＋第二波闸）——已闭环
- 首片落盘时刻：2026-11-14 21:52:39 CST（selfplay-994569，mtime 直读）→ 单局墙时约 2 小时 18.6 分，真校准约 43–45 秒/根（30.0 与 57–60 两模型均作废）
- DGX 验收：selfplay-994510 身份读回全合（code 七件/config 全值/weights 06a03d9c/games_sha 2c87b495/seed 12345/window [200,299]/π 1.0/n_roots 198/pass A），抽样 994508/994510/994569 字节 sha 与 progress 合，replay_sha256 与分发清单合
- 第一波 62/62 落盘（21:54:38 progress games=62）并经同步环全数回传 DGX 确认（末发 21:56:21）
- **第二波闸：PASS，无停令**；第二波 30 局已自动接续（21:57 直读 PGID 45462 在册、负载 ~30）
- 完工估计：2026-11-15 ~00:15（估计，按首波墙时外推）

## 完工（2026-11-15 00:04 CST 直读闭环）
- 末片落盘：2026-11-14 23:57:57（progress.json mtime）；末四片 994596–994599 于 23:59:38–23:59:40 回传 DGX
- 落盘分片：**92/92**（shards/ 清单与局单 92 局逐字合）
- progress.json games 计数：**92，done 92**（每局 n_roots 198、pass A、shard_sha256 在册）
- k0_sync_sent.txt 条数：**92**（与分片数一致）
- 标注发布：generation **gen-9c084fec553aee18**，total roots **18,216**，files_verified 14，weights 06a03d9c / config 9a2b2440 全值合；日志 0 Traceback/ERROR
- 同步环停环：stop 文件落盘 2026-11-15 ~00:02 CST → 进程验证不在册（00:04:06 直读 ps rc=1，日志 "STOP seen, exiting" 00:03:41）
- 标注进程组收尾：PGID 45462 **自然退出**（00:01:37 直读不在册、pgrep 无 k0_annotate、负载 0.35，机器怠速）
- DGX 侧：92/92 分片＋progress.json 已在 returns/c2/（末发 23:59:40）；**200 局并集核对＋生产 join 归 DGX**（C1 分区另计）

## 备注
- 计时片 OUT 备份保留：/root/carbon/data/k0_annot_c2_timing_bak/
- 本总账发出对象：DGX 分战区（合并＋200 局并集核对归 DGX）＋作战室
