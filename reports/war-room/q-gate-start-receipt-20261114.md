# 正式 q 闸开跑回执（Studio 线，2026-11-14）

派单：作战室 2026-11-14 16:04，唯一执行人 Studio。启动条件（m7bsr1 选模完工回执）已满足后接档开跑。

## 绑定（逐项直读验合）
- gate：code/stage1_teacher_gate.py e922d9d890c3e1eff6643e947289acb0
- 教师：stage0_teacher.py c6ade5ced42e2c54a77c1499163aa151（v3）；其余 code：cnet2_np d8c15ad4、cnet2_search 94f7fb64、cnet2_feat 2a45b08b、fastsim aefe6c83——6 件与 v3 pilot artifact 记录逐字合
- 权重：/Users/arm/s12test/heads_v2.npz，06a03d9cfeec239af3065a49fefec382
- test split：data/stageB_split_N323.json，md5 fd4cf14413be5273c131478cc1fdacc1，323 局（main 读回验合）
- 种子：scene 995300／teacher 7000／cont 995400／anchor 995310
- pilot 绑定：--pilot-artifact pilot_du_v3.json.pilot.json（grade formal，run_id c49ccd42e06deafa，locked 200／raw 323）；--expect-pilot-run-id c49ccd42e06deafa
- 判据：gate 内置 §13-A／§14 锁定口径（du game_mean ≥ 0.02 且 t-CI 下界 > 0）；不抬帽（不过无第二次抬帽）

## 开跑前准备（Studio 直读）
- 回放：323/323 自 DGX 局域网拉齐（77,716,571 B），逐件 md5 与 DGX 清单合；gate 清单 split_gate.json：q 0、du 323 唯一；与 pilot 34 局、ds118 均无交集（gate ds 检查再验）
- Fixture：test_gate_terminal_bound 12/12 PASS；test_teacher_gate_identity 4/4 OK。test_teacher_information_set 未跑（与 v3 pilot 同标注：需 fixture 权重 cnet2_b128.npz，生产 weights 目录没有；作战室 2026-11-14 12:17 已裁定不补跑、不掺副本）
- 开跑前硬闸：残留 0（a4 选模已清）；磁盘余量 413 Gi

## 开跑三件套（2026-11-14 16:11:24 CST）
- launcher PID/PGID 51489；gate PID/PGID 51491；在册（16:13 直读 etime 01:41）
- 12 spawn_main worker；负载 10.54（爬升中）
- PREREG 已落盘：q_du_formal.log 13,843 B，run_id c132bf81b0d70cde；n_scenes 1072；n_games 323；ds_overlap_check "RUN, disjoint (ds n=118)"；canonical_config true；formal true；diagnostic_reasons []
- 启动器：data/stageB_q/q_launch.py；日志 data/stageB_q/q_du_formal.log；产物 data/stageB_q/q_du_formal.json（完工落盘）；安全 cap 40h（仅防失控；策略组估计约 29.8h）
- 监视：本地 watcher（~/workspace/q_gate/q_watch.sh）＋15 分钟定时趟兜底

## 完工估计
约 29.8 小时（策略组估计），即约 2026-11-15 22:00 CST（估计）。完工回执按 v3 pilot 同格式（du_mean＋CI、σ̂_game/root、zero/supra 分报、join 核查、sidecar 绑定、step 带、N 全式、information-set 标注原样）。

## 补遗（策略组回执要求，2026-11-14 16:15 转交后补齐）
绑定六项（原文列明）：
1. gate stage1_teacher_gate.py = e922d9d890c3e1eff6643e947289acb0
2. 教师 stage0_teacher.py = c6ade5ced42e2c54a77c1499163aa151（v3）
3. 权重 heads_v2.npz = 06a03d9cfeec239af3065a49fefec382
4. test split stageB_split_N323.json = fd4cf14413be5273c131478cc1fdacc1（323 局）
5. seed-cont = 995400
6. grade = formal（PREREG formal=true、diagnostic_reasons=[]）

根集身份：
- gate 清单 split_gate.json md5 61de5c4c8f83c3869d00ca3cd084c993（q 0、du 323 唯一）
- 323 stems 规范序列化 sha256 = 088380184ef7f652a6daf6dccacb9062eb24b9dd7675faafb5dc5344fd2dd709；PREREG games 集合与清单 du 集逐项相同（同 sha256）
- 场景种子 995300；n_scenes 1072 根；run_id c132bf81b0d70cde
- 逐根 (game, step, me) 全清单在完工产物 rows 中，完工回执以其哈希对锁
