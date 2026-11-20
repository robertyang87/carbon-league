# 价值谱 U′ 完训后处理 独立核验备忘（工程组主代理，2026-11-20）

核验对象：执行人终报 `reports/REPORT_POSTTRAIN_USPEC_20261120.md`
（本地 `~/workspace/valuespec/run/REPORT_POSTTRAIN_USPEC_20261120.md`）
与 DGX 原件、本地 per-game 数据。本备忘只记我本人独立复核的结果。

## 1. 文件与 md5

报告 62aca8f9、决斗备件 6db0e07c、trajectories b63bee31、
per_game csv ff808fa5、json cc859cf9、analyze b5fb780e——
与执行人终报逐字合 ✓。DGX 直读：包 uspec300_wall.py md5
**2ed4c1e6ec495ed85cb4656abbdc1de5**（2,867,315 B）✓；
训练产物 uspec300.npz md5 8879aaef ✓（2,101,008 B）。

## 2. 打包自验（DGX 日志直读）

uspec300_verify.log：前向 err 0.0000 PASS ✓；确定性 1 局×2 跑
IDENTICAL（终局现金 [8391.62, 7937.5]）✓；步时 mean 0.783s／
p95 1.666s／max 3.031s（0.75s 线）FAIL——只报数、不参闸，口径与
批 2 裁定 1a6e45f0 同（0.75s 是天梯单步时限，决斗走线下 bigeval）✓。

## 3. §7 复验（DGX 日志直读）

bridge_verify_packed.log：网格 2.626e-09／1e-3 ✓、孪生 2.020e-09／
1e-6 ✓、packed _u_of_margin 0.0／1e-12 ✓、逆残差 2.220e-16／1e-9 ✓、
单调双向／钳区饱和／零点跳变全过；饱和探针 B(_U_AT_CLAMP)＝B(±1)＝
±1.5 精确 ✓；退役点精确复合误差 0.0 ✓；末行
`BRIDGE VERIFY (f90af57d caliber): PASS` ✓。旧口径「saturation
values FAIL」行（B(0.9999999)＝1.3333）是退役点的预期行为，已由
新口径探针取代——与报告所述一致 ✓。③④ 两项见报告表，数值与
GATE_FINAL 逐值同（网格／孪生／逆残差三值与闸阶段逐字合）。

## 4. 次级读数（本地重算，per_game_uspec.json＋trajectories）

- 51 行；归因原始：对手得分主导 25／局面性重估 16／自身受损主导 8／
  混合 2 ✓；除 step-0：25／15／9／2 ✓。
- 与 gen3 逐局标签相同 31/51 ✓；塌方步同一步 20/51 ✓；
  |步差| 中位 7 ✓。
- 噪声带 B 由 trajectories 重算：p99 |Δw|＝**0.1286**（中位 0.0200、
  p90 0.0659、max 0.4185），与报告逐字合 ✓；margin_ok 51/51 ✓。
- 两局未看见：82cbda80 D＋0.0971＜B（最大跌幅 0.1804 @177）、
  1f765a0d D −0.0025 净升（最大跌幅 0.2368 @251）✓；净跌落 ≤0 的局
  1/51 ✓；原始 argmax 落 step-0 仅 1/51 ✓。
- 以上只作机制证据，不作判读（预注册 §2）——核验不改变此界线。

## 5. 决斗备件与 A 臂状态（核验时已过期，见下）

- 备件文件齐（DGX duel_prep／本地）；开场命令形与种子段
  968600–968799（side＝seed%2）如报告所述；本单未跑任何一局决斗 ✓。
- 执行人写备件时（13:19）“A 臂未落地”属实；但核验直读（13:22–13:24）
  显示状态已变（见 §6）。收据以核验时刻状态为准。

## 6. 核验时刻的 L20 新事实（不在执行人报告内，主代理直读）

- **Gen-5 训练已完训**：work0/gen5/ckpt.npz 已写（13:11，2,741,616 B，
  md5 **629a12b6dc82e241163e9b501f876ff6**），gen5_train.log 末行
  epoch 2 统计＋wrote；train2_uspec_w 进程已无。
- **NETEXP A/B 首跑 13:17 失败，已作废**：旧 ab_train.sh 直调
  train2_fixed.py（无 $PY 前缀、--data 空值），两臂均
  `Permission denied`、1 秒内双双失败，却因 echo 行写死 rc=0 而
  误印 AB_TRAIN_DONE（假完工戳；日志副本现存 DGX 镜像，L20 本地
  日志已被新跑覆盖）。
- **重跑 13:21:54 进行中**：L20 线已修正脚本（$PY＋$DATA＋rc=$?，
  与我备的修复件仅日期格式不同），armA 训练进程 PID 112361 在册、
  R 态、RSS 9.2 GB、45 秒内 CPU 时间推进约 48 秒——在跑，未挂死；
  日志尚无步行（数据装载／stdout 缓冲所致，未证为卡死）。
  残留：现行脚本末行 AB_TRAIN_DONE 仍无条件——若本轮任一臂失败
  仍会误印；**运行中不碰脚本**，此项留待下轮开跑前改。
- 我备的修复件 ~/workspace/netexp/ab_train_fixed.sh 已无用（L20
  脚本已修），留本地作记录，不上传、不覆盖运行中脚本。

## 7. 如实记（执行人已披露，核验不推翻）

- 打包首跑 extract_m3s2 导入失败、§7② 首跑漏 --packed——两次漏跑
  均不计入证据，重跑通过。
- analyze_uspec.py 三处文案残留（尾行 per_game_rejudge、new-judge
  pooled、m3s2 标签）——数字为 U′ 实测，文件内容无误。
- 运行记录遥测另有 1 次未在报告中逐项的 exec 失败（共 3 次失败记录，
  报告披露 2 次）；终报四项交付物已逐项独立复核，无缺项——记为
  开放注，不堵落仓。

## 判读与落仓口径

**四项核验通过。** 落仓：报告＋决斗备件＋次级读数 per-game csv/json＋
本备忘；包文件（2.87 MB）与 trajectories（9.4 MB）留 DGX 原件、
只记 md5（与 ① 轨迹文件留本地不入仓同口径）。决斗仍未开场——
唯一前置为 A 臂落地（armA 在跑中，armB 待后）。
