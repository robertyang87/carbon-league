# 终局 pilot 阶段 1 — 开跑记录（Studio 线执行，2026-11-14 07:31 CST）

派单：作战室 2026-11-14 07:19，唯一执行人＝Studio 线；依据 spec §17 规模锁定注（策略组 02:55）：阶段 1 计数已锁，不必再等策略组另令。

设计：cap2 = 200 局 × 2 根 = 400 根。gate 每（局，层）只抽 1 根，故分两段子跑：
- A：--step-min 200 --step-max 250，窗口 [200,250)
- B：--step-min 250 --step-max 299，窗口 [250,299)
同种子 scene 995300 / teacher 7000 / cont 996100 / anchor 995310；顺序跑；总墙时上限 4.5h（16200s）写进 launcher，超时自动 SIGTERM 子跑进程组并停（不闷烧）。

输入验证（开跑前，verify_p1_input.py hard_ok）：
- DGX 直拉 tar md5 dc0746aa89ee1c85044ec42d1c141fa5（45,715,311 B），Studio 侧 200/200 md5 全合
- steps 条目 300/局；identity weights md5 全 06a03d9c（正典）
- 终局可抽位 39,600（99 步 × 2 席 × 200 局），与 DGX 验收逐字合
- gate aae25fdad5c376b0d4b907ace0f1e31f；weights 06a03d9c；ds f86be41341a9e2222c49985e9af169f0；split 200 du / 0 q；开跑前 spawn_main 0
- 口径差（备查、非阻断）：回放 steps 300 条/局（状态数 120,000）vs index 299/局（DGX 119,600 按转移数计）；U distinct 回放 result 计 400 vs DGX 验收 384（定义未对齐，待核）

开工（07:31:07 CST）：launcher PID/PGID 15696；A 段 gate PID/PGID 15697（独立 session），12 workers，load 9.83 上行。
A 段 PREREG：run_id 67d7f44efe0d05ad，窗口 [200,250)，200 局/200 根，RUN disjoint，canonical＋formal，reasons 空，教师 v3，belief sample。
开工三件套回执已发作战室（submission c9f86d96-11a0-4659-8ffc-e5b1ccdf8126）。

估计：A 约 1.6h（完约 09:05）、阶段 1 合计约 3.2h（B 完约 10:40），均按冒烟实测 347.7 CPU-s/根 ÷ 12 workers 推（估计）。
完工回执待报：σ̂_game、du 均值＋step 三分带（200–229/230–259/260–299）、零行占比（teacher_calls=0 / du=0 的 padding 根单独列数、判读只用真实根）、wall/根耗时、sidecar 绑定。合并分析脚本 merge_p1.py 已备（复刻 gate 的 cluster_stats/cluster_bootstrap_ci 定义）。

已知 launcher 缺陷：p1_launch.py 未把 run 记录 append 进 rec["runs"]（runs 恒为空），A 段 gate PID/PGID 已手工补进 p1_launch.json，B 段切换时同样手工补；下次 launcher 须修 append。
监视：watch_p1.sh（本 VM，每 60s 轮询，最长 ~4.7h）＋ 15 分钟定时趟兜底。
