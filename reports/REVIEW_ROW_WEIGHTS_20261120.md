# 落仓说明 · 输局旁路 ② 行权重通道（工程组独立核验后落仓）

- 落仓时间：2026-11-20 ~10:48 CST
- 被审件：`tools/katago_m3s2/train2_w.py` md5 `cab1d22a20c34bfbe6d4e3b361a021b8`；
  `tools/lossprobe3/attach_row_weights.py` md5 `3483ef22e0b9e0e75ab1657bff68730f`
- 深审总判：审项 1–4 全部 PASS，审项 5 探针级 PASS，无作废级实质问题，**可落仓**。
- F1 核销：落仓前复核本地 `fixture/weighted/attach_report.json`——dst 字段已无 `echo10/` 层级，
  四件 md5 与磁盘实件逐字合（72b713ed…／081e0289…／d9f9f54f…／e2754884…），
  深审读到的滞后 dst 当时已不存在，无需修正。
- F2 如实记录：`attach_row_weights.py` 仅存本地，未暂存 DGX；转换侧在本地/仓侧运行，无功能影响。
- Q1 待留档：作者闪报含两次 exec 失败（exitCode 1，event 172948／172960），内容未说明；
  全部可验工件均复现通过，留档待作者说明。
- 待补核：作者正式报告 REPORT.md（C–B 完整数值对照表）尚未到；到后补核出最终总判。
  本落仓不以此为通过条件（探针级＋端到端复现已覆盖生效闸判定实质）。
- 现行件零改动（仓内 API 读回）：train2.py `06e3c87c…`、train.py `ab3953d2…`、
  设计件 `bf1e869d…`、分类表 `01c93475…`、处置表 `f9541305…`。
- 用途：Gen-5 训练用行权重通道（被压制局 policy ×0.5、value 不动；其余行全 1.0）。
  训练使用以 Gen-5 定版件全文核准为准。

---

# 深审报告 · 输局旁路 ② 行权重通道（非作者线）

- 深审人：工程组独立深审（非作者线）；日期 2026-11-20 ~10:37–10:45 CST
- 被审件：`tools/katago_m3s2/train2_w.py` md5 `cab1d22a20c34bfbe6d4e3b361a021b8`；
  `tools/lossprobe3/attach_row_weights.py` md5 `3483ef22e0b9e0e75ab1657bff68730f`
- 探针：`tests/probe_row_weights.py` md5 `32e0684c0872fffb041d85a76024d621`（仅供复跑，非交付件）
- 现行件（仓内 API 读回，未动一字）：train2.py `06e3c87c932a762dba6a09ee1bd06117`（`tools/katago_m3s2/train2.py`）；
  train.py `ab3953d2de270c3e89d657552b08856e`（`tools/katago_m3s2/train.py`，注：派单所写 `code/train.py` 路径不存在，实为 tools 下）；
  设计件 `bf1e869d623d8f09962a9fad4bbd56e5`；分类表 `01c93475298d5e5e9cbf5897202f118b`；
  处置表 `f9541305282c1e0544b4a7cff0c49589`（派单所写 f9541305 为其 8 位前缀，非不符）。
- 审后 md5 复核（本地＋DGX，全部未变）：train2_w `cab1d22a…`、attach `3483ef22…`、probe `32e0684c…`、
  分类表 `01c93475…`、DGX train2 `06e3c87c…`、layout `6877bc73…`、model `c102c297…`。
- 复跑环境：DGX（GB10），venv `/home/arm/junyang/carbon/.venv/bin/python`，device cuda；
  复跑工作区 `/home/arm/junyang/lp2_review/`（被审件 md5 验合后复制）；DGX 负载：探针时 ~0.8/6.2/8.5，
  四跑时 1.93→2.84（1 分钟均值），GPU 0%（跑前）；init＝gen2 镜像 ckpt `b944a2b9c96384187a22040a44dc8f98`（DGX 镜像直读）。

## 总判（审项 1–4；审项 5 探针级已复核，完整 C–B 对照表待作者 REPORT.md）

**无作废级实质问题。等价闸①的全部承重结论均由深审人独立复现，生效闸②探针级证据亦复现。**
审项 1–4 全部 PASS；审项 5 探针级 PASS，正式报告到后补核 C–B 全表再出最终总判。
发现两项低级记录问题（F1、F2）与一处待问（Q1），均不拦落仓，但应在落仓前修正/澄清。

## 审项 1 — train2_w 对 train2 的复用是否真实：PASS

- 逐行读毕＋`main()` 机械 diff（train2 仓内原版 vs train2_w）：除下述两处 delta 外，
  main() 其余逐行同：argparse 各默认值（epochs 3、batch 512、lr 1.5e-4→3e-5、wd 0.01、fpw 0.0、
  feed-workers 8、prefetch 4、feed-timeout 600、compile 默认关）、`torch.manual_seed`、`rng`、
  优化器、lr 余弦、BF16 autocast、backward/step、统计累加、epoch 打印、经 `layout.to_f16` 的导出、
  `--out-pt`，全部逐字同。仅描述串与注释不同（无数值影响）。
- 等价关键码全部 import 复用、未复制：`build_epoch_plan`、`load_dataset`、`OrderedFeeder`、
  `_to_device`、`materialize_numpy`（经 feeder）、`SIZE`/`N`（取自 train2）。`train2.py` 557 行中的
  V4 表、augment、feeder 均未在 train2_w 内重写。
- Delta (1) 权重抽取：`zip(plan, feeder.run_epoch(plan))`——`run_epoch` 逐字读过：内联路径每 plan 项产一
  批，池路径 `for want in range(len(plan))` 每步恰产一、且以 step 号重排后按序产出，故 zip 配对精确、
  无截断风险。`rows.arrays['wpw'/'wvw'][idx]` 与当批 materialize 所用 idx/gs 同源，augment 不碰权重
  （权重取自行数组、非 feeder 批内字段，与 `win`/`margin` 同类标量）；缺失键缺省全 1.0。
- Delta (2) 损失行：policy 两项把 `pw` 换成 `pw_eff = pw * wpw`（分子分母同步）；value 项改为
  wvw 加权均值；aux 三项（delta/fog/win）与组合系数（1.5/0.3/0.2/0.2）逐字未动。

## 审项 2 — 加权数学：PASS

- P1 全 1 恒等由深审人 DGX 独立复跑（probe，batch 128、seed 0、fpw 1.0、device cuda）：
  六项全部 bit-identical，数值与作者报告逐字同——lp_w 1.957751989364624、lp_b 1.3987722396850586、
  l_v 1.2154223918914795、l_d 1.1094348430633545、l_f 1.0855095386505127、l_win 0.7472715973854065，
  abs_diff 全 0.0。l_v 的位级同是实测结论（加权均值在全 1 时与 `F.mse_loss` mean 归约位级一致），
  本次复跑证实。
- 分母同步正确：policy 分子分母同用 `pw_eff`，value 分子分母同用 `wvw`；`clamp(min=1)` 为 train2 原样
  继承（train2 的 `pw.sum().clamp(min=1)` 同构）。
- 边界实测（DGX，合成张量）：全零权重批 lp_w＝0.0、l_v＝0.0、无 NaN；单行批 wpw 0.5 得 lp 为 wpw 1.0 的
  恰一半（0.023084212094545364 vs 0.04616842418909073）——系 clamp 将分母托底于 1 所致，与 train2 的
  分母处理同构，非缺陷；生产口径（batch 512、fpw 1.0）下 `pw_eff.sum()` 远大于 1，clamp 永不触发。

## 审项 3 — attach 脚本：PASS

- 分类表核对：58 行（MISTAKE 47／SUPPRESSED 11）；fixture 四局分类与派单一致——被压制局
  1c989743／234319e1＝SUPPRESSED，错输局 0e2cecb7／0f203344＝MISTAKE。
- 逐局模式（本地纯 numpy 重跑，`--input fixture/raw`）：rows_total 1196、rows_suppressed 598、
  unknown 空；四件输出与作者 `fixture/weighted/` 逐数组同（wpw/wvw 亦同），原键逐数组不变，
  wpw/wvw 均为 float32、形状 (n,)；被压制局 wpw 全 0.5／wvw 全 1.0，其余全 1.0/1.0。
  作者报告的逐件 md5 与磁盘实件逐字合（如 0e2cecb7：`72b713ed5ae119b8e0efb935f7cb72d0`）。
- 合并模式（`fixture/merged_check/rows_correction.npz`＋`lossprobe3/run/rows_correction_games.json`）：
  963 行、rows_suppressed 0，wpw/wvw 全 1.0，与作者合并输出逐数组同，原键逐数组不变。
- 未知局默认（父目录非 32-hex 的文件）：game 记 None、计入 unknown_games、权重 (1.0, 1.0)。
- 工具自带回读校验（逐局 verify_output、合并全量复核）在重跑中均通过；脚本从不就地改写、
  不碰 train.py/train2.py。

## 审项 4 — 闸口径与端到端复跑：PASS

- fpw=1.0 合理：`materialize_numpy` 中 `pw = searched + (1-searched)*fpw`；fpw=0 时 base 行
  （searched=0）pw 恒 0，wpw 效应在 lp 中不可见；fpw=1.0 时 pw 全 1，权重效应可测。fixture 的
  base 行 searched 全 0 已由 P4 的 step-0 混合（64 压制／64 全 1）佐证。
- P5 批序：probe 用 `generate_index_sequence(n, batch, 1, seed)`，其内部即以同种子 `RandomState`
  调 `build_epoch_plan`，与训练循环的 plan 生成同 RNG 消费顺序；深审人复跑得 step-0 哈希
  idx `b770941045bf753adca415a6102f7c36`、gs `c694ab680a9bccd612e0997487cc233c`，与作者报告逐字同。
- 端到端四跑（DGX，各 1 epoch/10 步，batch 128、seed 0、fpw 1.0、feed-workers 1、device cuda、
  init gen2 ckpt），深审人独立重跑，epoch 统计与作者逐字同：
  A1/A2/B 均为 loss 1.3058／pw 0.4782／pb 0.4926／v 0.0826／delta 0.147／fog 0.524／win 0.3106；
  C 为 loss 1.2998／pw 0.4786／pb 0.4853／v 0.0825／delta 0.1452／fog 0.5264／win 0.3164。
- state_dict 位级比对（torch.equal 逐张量）：
  自跑 A1 vs A2：**55/55 同**（确定性基线）；自跑 B vs A1：**55/55 同**（硬闸①复现）；
  自跑 C vs B：**0/55 同**（权重真实进入轨迹）；跨进程比对：自跑 A1/B/C vs 作者 A1/B/C 各 **55/55 同**。
- 导出 npz 比对：自跑 A1 vs B：**51/51 数组同**；自跑 A1/B/C vs 作者 A1/B/C 各 **51/51 同**；
  自跑 C vs B：48/51 数组不同（余 3 数组经 f16 导出舍入后同，.pt 全精度下 55 张量全不同——一致）。

## 审项 5 — 生效闸②探针级证据：探针级 PASS（完整 C–B 对照表待作者 REPORT.md）

- 深审人复跑 probe：P2 被压制行 policy 贡献相对差白/黑均为 **0.0**（＜1e-9），value 项 bit 不变
  （l_v 1.2154223918914795 两边同）；P3 梯度比 policy **0.5**、value **0.5**（恰值）；
  P4 step-0 权重抽取 64 压制／64 全 1；P5 见审项 4。全部与作者闪报逐字同。
- C 跑偏移已由深审人端到端复跑证实（epoch 统计见审项 4；C vs B 0/55 张量同、48/51 导出数组不同）。
- 待补：作者正式报告（REPORT.md）中的 C vs B 完整数值对照表——到后补核，届时出最终总判。
  本审件不以此为通过条件，因探针级＋端到端复现已覆盖生效闸的判定实质。

## 发现

- **F1（低）**：作者 `fixture/weighted/attach_report.json` 中各件 `dst` 字段带一段已不存在的
  `echo10/` 层级（如 `fixture/weighted/echo10/0e2cecb7…/rows_base.npz`），而磁盘实件在
  `fixture/weighted/<game>/rows_base.npz`。报告内 md5 与实件逐字合，说明哈希正确、只是 dst 字符串
  滞后于一次重排版（commonpath 口径所致）。不影响工件本身；落仓前应以实件重出报告或修正 dst 字段。
- **F2（低／记录）**：`attach_row_weights.py` 只存在于本地，未暂存 DGX
  （`/home/arm/junyang/lp2_weight/` 内无此件）；闪报「文件暂存 DGX … 与本地 …」仅对训练器侧成立。
  转换侧本就在本地/仓侧运行，无功能影响；落仓记录应如实写。
- **Q1（待问）**：作者闪报的 recorded_execution_results 含两次 exec 失败（exitCode 1，
  event 172948／172960），内容未在闪报中说明；深审人从现有证据无法识别其内容，且全部可验工件
  均复现通过。建议向作者问清该两次失败为何物、是否已处理，留档即可。

## 结论

审项 1–4 PASS，审项 5 探针级 PASS；两件新文件与等价闸证据成立，**可落仓**（待作者 REPORT.md 到后
补核 C–B 全表出最终总判；F1 的报告 dst 字段落仓前修正）。现行件零改动已由仓内 API 读回证实。
