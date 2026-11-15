# 正式 q 闸加速重启开跑回执（2026-11-14，Studio 线）

依据：策略组裁定（批准 --workers 12→20、其余逐字不变重启；旧 run 登记 VOID；K0 前置 (3) 保持「执行中」）＋所有者拍板「批了就重启加速」。唯一执行人：Studio 线。

## 旧 run 作废处置

- 旧 run_id **c132bf81b0d70cde**（16:11:24 开跑）按裁定登记 **VOID**：执行参数重启取代，零结果被消费。
- 17:36 按进程组 TERM 清杀（PGID 51489／51491），零残留验收通过：双 PID 逐个 DEAD、spawn_main 计数 0、gate 进程计数 0（干净口径，非关键字 grep）。
- PREREG 留档：Studio `data/stageB_q/q_du_formal_run_c132bf81_void.log`（14,184 B，md5 `83c24d6b0863d6930b2d1e0a158b867b`）、`q_launch_run_c132bf81_void.json`（md5 `4ccf95ebc7b94e68b0456caf230b7fb5`）。
- **旧跑无分阶段产物**：无 q_du_formal.json、无任何 rows 落盘（gate 完工才出产物；归档日志只有 PREREG＋停机行）。裁定中「抽 ≥20 根重叠行比对」一条无对象可比，按裁定原文以代码级证明为据，特此注明。

## fixture 重跑

- `code/test_gate_terminal_bound.py`：**12/12 PASS**。
- `automation/acceptance_s1/test_teacher_gate_identity.py`：**4/4 OK**。
- `test_teacher_information_set`：仍按 2026-11-14 12:17 裁定不跑（缺 fixture 权重，不往生产目录掺副本），原样标注。

## 新 run 开跑（2026-11-14 17:36:48 CST）

- run_id **51972702c4f81c21**；launcher PID/PGID **72802**、gate PID/PGID **72804**；40h 安全 cap 按新 run 自身起算。
- 六绑定逐字不变（新 PREREG 读回）：gate `e922d9d890c3e1eff6643e947289acb0`、教师 `c6ade5ced42e2c54a77c1499163aa151`（v3）、权重 `06a03d9cfeec239af3065a49fefec382`、test split `fd4cf144`（323 局）、seed-cont **995400**、grade formal；scene 995300／teacher 7000／anchor 995310／--belief sample／pilot 绑定（工件 run c49ccd42e06deafa）不变；其余 code md5（cnet2_np d8c15ad4、cnet2_search 94f7fb64、cnet2_feat 2a45b08b、fastsim aefe6c83）与旧 PREREG 逐字合。
- **唯一变更：--workers 12→20**（launcher argv 读回验合），无任何夹带。
- **根集哈希锁硬闸 PASS**：stems sha256（紧凑 JSON 序列化，与首跑同口径）＝ `088380184ef7f652a6daf6dccacb9062eb24b9dd7675faafb5dc5344fd2dd709`，与钉死值全等；split_gate.json md5 `61de5c4c8f83c3869d00ca3cd084c993`；n_scenes **1072**、323 stems 全唯一；新旧两跑 stem 集逐项相同且顺序相同。
- 开跑三件套（17:38 直读）：双 PID 在册（etime 01:33）、spawn_main **20**（各 CPU 约 1:26、持续推进）、负载 18.61／13.51／12.94（1 分钟值正向 20 爬升）。
- K0 前置 (3)：按裁定保持「执行中」，不回退。
- 完工估计：约 22 小时（估计），约 2026-11-15 15:40 CST（估计）。监视：本地 watcher v3（盯 launcher 72802）＋15 分钟定时趟兜底。
