# v3 全量 pilot 完工回执 — 2026-11-14（Studio 线）

## 完工事实
- 开跑 2026-11-14 12:10:49 CST，工件 15:33 落盘；launcher wall 12,154.0 秒，gate wall_s 12,136.5 秒；rc=0，capped=false（4.5h 墙帽未触）。
- 零残留已验：launcher 89589 / gate 89592 进程组不在册，括号口径 spawn_main = 0（watcher 完工趟＋本线直读双验）。
- 工件：Studio `data/stage1_formal/pilot_du_v3.json`（36,160 B，md5 a63c55428935f5f1408de07eb944e866）＋ sidecar `.pilot.json`（2,963 B，md5 ec5e804c960264e6f9f84b46e3f7dff8）；本地副本在 ~/workspace/v3_pilot/receipt/（v1.5 工件 md5 eb4f1972… 与在册值一致）。

## 绑定（sidecar 读回）
- grade=formal，artifact_id d9b998b96ea43e2ab25133562b6c14460539e7d9a12c4fcc9c7ee14602807f95，run_id c49ccd42e06deafa
- weights 06a03d9cfeec239af3065a49fefec382；gate e922d9d890c3e1eff6643e947289acb0；教师 v3 stage0_teacher c6ade5ced42e2c54a77c1499163aa151；fastsim aefe6c83
- seeds：scene 995300 / teacher 7000 / cont 995400 / anchor 995310；canonical_config true；ds_overlap_check "RUN, disjoint (ds n=118)"

## 根集身份
- 与 v1.5 正式 pilot 逐根（game, step, me）比对：**113/113 全同**，new-only 0、old-only 0。34 局、每局 3–4 根。

## du（随报作记录，放行已由 stage-1 终判定案）
- join 口径与 FT-filtered 口径**完全一致**（原因见下节：全部行 step < ft，无行可滤）：du_mean（局聚合）**0.007124**，du_mean_roots 0.009137
- CI95 t-cluster [-0.0376, +0.0519]，bootstrap [-0.0360, +0.0491]（t_exact true）
- 旗标检查：CI 上界 0.0519 > 0.02，**未触发**「与 stage-1 终判实质矛盾」旗标，切分前不需回策略组复核本项。
- du 符号分布：正 64 / 负 49 / 零 0；du distinct 113/113、u_guided distinct 113/113（非退化）。

## σ̂（本线独立复算与 gate 值逐字一致）
- **σ̂_game = 0.128315**（逐局均值 ddof=1），**σ̂_root = 0.289439**（ddof=1）
- FT-filtered σ̂ 单独标明：与 join 相同（0.128315 / 0.289439）——本趟无 padding、无 supra-FT 行，过滤集 = 全集。

## zero / supra-FT 按终局类型分报（12:36 补条口径）
- k0 first_terminal 对 34 局逐局取 (step, cause) 并与 gate ft 交叉核对：**34/34 一致**（全部 ft=299、n_steps=300）。
- 终局类型：status 34 局 / extinction 0 局。
- zero 行：合计 **0**（status 0 / extinction 0）；teacher_calls=0 的 padding 行 0；gate 自报 n_padding_dropped = 0。
- supra-FT 行（step ≥ ft）：合计 **0**（status 0 / extinction 0），符号分布无行可报。
- 本条自动无事：全程无此类行。

## step 三分带诊断
- 根 step 范围 1–239（本池根集如此，非 [200,299) 窗口）：
  - step 1–199：79 根，du_mean +0.0392
  - step 200–229：22 根，du_mean −0.0470
  - step 230–259：12 根，du_mean −0.0857
  - step 260–299：0 根

## 教师成本
- 17,701 calls（与 v1.5 同数），156.65 calls/根；7.6506 s/call；教师总耗时 135,423.8 s；per-root 均值 1,198.4 s（中位 1,152.6、最大 2,295.2、最小 404.5）。

## N 重算全式（spec reports/STAGE1_TEACHER_GATES_20261113.md 修订口径原文）
- 公式：**N_games_raw = ceil((140·σ̂_game)²)**，其中 140 = (z_0.975 + z_0.80)/δ = (1.96 + 0.84)/0.02（δ=0.02 不动，α=0.05 双侧，power 0.80）；**钳制 [50, 200] 局**。
- 代入：σ̂_game = 0.128315 → 140·σ̂ = 17.9641 → 平方 322.71 → **raw = 323**（与工件自算 locked_N_games_raw 逐项一致；同式复算 v1.5：140×0.158344=22.168、平方 491.4 → 492，亦合）。
- **锁定 N = 200 局**（钳后值）。锁定值下功效：significance-only 0.5965、conjunction approx 0.50——按 spec 口径，钳制后的 N 不得说成功效达标。
- 决策输入（归策略组）：若照 v1.5 暂定先例把 cap 抬至 raw 323，池 1,525 局余量约 4.7 倍；du 成本估计（估计）：约 1,074 根 / 约 168,160 calls / Studio 12 工人墙时约 29.8 h（cap 200 则约 665 根 / 约 18.4 h，估计；锚点与续弈另计）。
- 本线建议值即上式结果：raw 323、钳后 200；**N 经策略组确认前 DGX 不切正式 split**，本线不自行进 split/q 闸。

## information-set 未跑标注（原样带上）
- test_teacher_information_set 未跑：需 fixture 权重 cnet2_b128.npz，生产 weights 目录没有此件（仅旧验收树 s1v2check/s1v3accept 有），未往生产目录掺副本，旧验收树可跑；作战室 2026-11-14 12:17 裁定不补跑、不在执行令开闸条件内。

## 接档
- m7bsr1 A4 选模按 15:17 策略组裁定接档开跑（a3ad86d4；开跑前未收到修复版验收通知）：另行开工回执。
