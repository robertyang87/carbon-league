# Stage-1 终局 pilot A+B 合并回执（Studio 直读＋本地合算）

- 时间：2026-11-14 11:14 CST 直读；B 段完工 11:06（endgame_p1b.json / .pilot.json / .log 落盘），launcher p1_launch.json done=true。
- 来源：Studio 直读（arm@100.78.149.20，标记 STUDIOSTAT1112 / STUDIOSTAT1112B）＋本地用仓内 merge_p1.py 同款定义合算（cluster_stats / cluster_bootstrap_ci，seed 996100）。本地副本：hidden_files/p1ab-20261114/（endgame_p1a/b.json、pilot.json、first_terminal_shard0.json、merge_p1_local.py、endgame_p1_merged.json）。

## 分段

| 段 | 窗口 | 根数 | du 均值 | σ̂_game | 95% CI t | 95% CI boot | 零行 | wall |
|---|---|---|---|---|---|---|---|---|
| A | [200,250) | 200/200 | 0.0453 | 0.2349 | [0.0125, 0.0781] | [0.0135, 0.0772] | 0 | 9786.6s（48.9s/根） |
| B | [250,299) | 200/200 | 0.00668 | 0.1671 | [-0.0166, 0.0300] | [-0.0160, 0.0302] | 18 | 3108.1s（15.5s/根） |

- B 段 18 零行全在 step 286–298，teacher_calls 均 >0（1–12 次），为真零值 du=0.0，非 join 填充。
- B 段 verdict 字段为 "PILOT (no verdict)"；teacher 调用 A 14807 次 / B 4905 次，合计 19622 次，teacher-ms 合计 148527s。

## 合并（双口径）

- 零行过滤（382 行 / 200 局）：game_mean 0.02799，CI t [0.00570, 0.05028]，boot [0.00574, 0.04984]，σ̂_game 0.1598（供 stage-2 定块），σ̂_root 0.2092，root_mean 0.02721。
- join 口径（全部 400 行，step < ft）：game_mean 0.02599，CI t [0.00422, 0.04775]，boot [0.00434, 0.04714]，σ̂_game 0.1561。
- 零行占比：18/400 = 4.5%（全在 B 段）。
- join 检查：step ≥ first_terminal 的行 0 行，400 行全接上；step < ft 但零值 18 行（即 B 段零行）。
- step 三分带（零行过滤）：200–229：0.0195（109 行，CI [-0.0318, 0.0708]，σ̂ 0.2702）；230–259：0.0400（132 行，CI [0.0031, 0.0769]，σ̂ 0.2144）；260–299：0.0212（141 行，CI [-0.0020, 0.0445]，σ̂ 0.1397）。
- wall/根计时：A+B 合计 12894.7s；launcher 总墙时 12934.3s（07:31:07–11:06），4.5 小时 cap（16200s）未触顶，提前约 55 分钟自然完工，非截断。
- sidecar 绑定：weights md5 两段一致 06a03d9cfeec239af3065a49fefec382；stage1_teacher_gate.py / fastsim.py（aefe6c83）等 code md5 两段一致；artifact_id 各段独立（A e0ba35d0…、B f1d754fe…）；pilot_run_id A 67d7f44e…、B cf195e0f…；grade 均为 formal；ds_overlap_check 均为 RUN, disjoint (ds n=118)；ds_manifest_md5 两段一致 f86be413…。

## U distinct 384-vs-400 对齐核对（按对齐后定义重算）

- 400 行中 du 不同值 383 个（382 个非零不同值＋18 行共享的 0.0）；零行过滤后 382 行各不相同；200 局局均值 200 个全不同；(game, step) 对 400 个全不同。
- 与作战室提到的 384 差 1：按原始值重算为 383，6 位小数取整为 382，未能复现 384。请策略组以 383 为准，或指明 384 的口径。

## 备注（不作裁决）

- 合并均值 0.0280 ≥ 0.02 的 business 门槛，但 CI 下界 0.0057 低于 0.02；CI 半宽约 0.0223（t）/ 0.0220（boot），已在 stage-2 预授权目标（半宽 ≤0.025）之内。是否过闸、stage-2 是否还补，由策略组裁决。
- 本回执只含直读与合算数字，未对 Studio 做任何写操作；合并计算在本地跑，未在 Studio 上写 endgame_p1_merged.json。

## Studio 空闲（同趟）

- 定性：正常空闲——pilot 已完成（done:true，B 段 GATE-RESULT 落盘 11:06）。
- 证据：11:14 直读 launcher 15696 与 gate 59060 均不在册，spawn_main worker 0 个，load 1.36/3.32/7.30（回落中），磁盘 414Gi 可用。
- Studio 当前可接 MPS 训练（约 350 万参数级小网络，50 秒/轮量级），请作战室派下一个活。
