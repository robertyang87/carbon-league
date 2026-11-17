# 执行提案：小教师定标——换 trunk 重跑搜索比对（阶段 4D）

- 时间：2026-11-18 01:3x CST（只写提案，不开工）
- 裁定：`reports/RULING_SMALL_TEACHER_STAGE4D_RETRY_TRUNK_20261118.md`（commit `6388b27e`，API 读回 md5 `c52d44cd` 合）
- 前情：4C 直读口径已作废（`reports/eng_verify/STAGE4C_DIRECT_READ_REPORT.md`，commit `a03728d9`，md5 `ff3de9ea`）——教师每臂 q 是 rollout 聚合量，无法单次直读复现。本提案是裁定 §2 的主路：整个 `evaluate_root` 用学生 trunk 重跑，臂集钉死教师臂。
- **本提案只提案，不授权执行。策略组批后才开工；批前不写 harness、不动任何机器。**

## 0. 目标与口径

- 对附卷两局 994400／994401（窗 [200,299)，每局 198 根、共 396 根），用学生 trunk 重跑整个 `evaluate_root`，再与教师比对。
- 比对仍用 (game, step, action) 键、`shard_compare` v4（md5 `1e9ad085`，不改）出 Tier 1/2 四项；可比根数与可比动作条目数须明示。
- Tier 阈值（注册口径，不变）：Tier 1——flips = 0（硬项）／|signed mean bias| ≤ 1e-4／max|Δq| ≤ 1e-3／max|Δdu| ≤ 1e-3；Tier 2——flips ≤ 2%／|bias| ≤ 1e-3／max|Δq| ≤ 5e-2／max|Δdu| ≤ 5e-2。Tier 落点不由执行方判，只报数，裁定归策略组。

## 1. 机制：教师原值逐项钉死

生产调用（`k0_annotate.py` 7f567a2e，`_eval_root` L336–352）：`teacher=2`、`K=16`、`M=8`、`H=2`、`belief="sample"`、`seed_teacher=7000`、`seed_base = 7000 + step`（L347）、`guard_snaps=G._teacher_guard_snaps(scene, belief)`（L349）。重跑全程沿用同一组参数。

### 1.1 臂集与先验——取自教师分片，不重算

- 来源：教师侧标注输出 `annot_teacher/shards/selfplay-99440{0,1}.npz`，generation `gen-3f2c17f8b7beb5b6`，字段 `arms (198,16,12) int16`（12 维动作编码）、`priors (198,16) float64`、`n_arms`（全 16）。
- 重跑中 `gen_candidates`（`stage0_teacher.py` L209–287）**不调用**；臂集与先验逐根原样读入。聚合公式沿用 `evaluate_root` L494–496：`Q_mean = Q.mean(axis=1)`；`wq = exp((Q_mean − max Q_mean)/0.05) × sqrt(priors)`；`q = wq / wq.sum()`。

### 1.2 信念场景——同码同种复现（trunk 无关）

- 来源：`evaluate_root` L391–401。`belief_carbon(feats[0], _players, sim.config, rng=np.random.default_rng([int(seed_base)+m, 0xC0]), mode="sample")`（L398–400），再经 `_legal_root` 合法化（L401）；oracle 分支不走（生产 belief 恒为 sample）。
- `belief_carbon` 不取 `pol` 参数——信念世界与 trunk 无关，是回放状态＋种子的确定性函数。复现方式：harness 沿用钉版 `k0_annotate` 的场景供给（`G.extract_scene`／`_fresh_sim`，L339–341）重建每根的 scene／feats，再逐行执行 L391–401。`n_unknown` 同取该步返回值，与教师分片同值。
- 验证：不靠单项复核，靠 §2 对照闸（教师 trunk 走同一 harness 须逐位复现教师 q）端到端验证。

### 1.3 对手根动作——教师 trunk＋同种复现，冻结后使用

- 来源：`evaluate_root` L403–426。`rng_m = np.random.default_rng(seed_base + m)`（L408）；`act(model_x, model_aux, sample_rng=rng_m, temp=1.0)`（L419）＋ guard 修正（L420–423）；每场景一个，先于候选臂算定、与所选臂无关（L387–390 注释）。
- 关键：该动作由**教师 trunk**采样（temp=1.0、固定 rng），不是学生 trunk 的动作。复现方式：Phase 1 用教师 trunk 执行 L403–426，逐根逐场景算出 `opp_actions[m]`、`opp_feats[m]`、`opp_snaps[m]` 后冻结；Phase 2（学生 rollout）只取用冻结值，**不重采样**。
- 分片不存对手动作——无现成字段可读，必须按上述方式复现；复现正确性同样由 §2 对照闸验证。

### 1.4 Guard 快照——同路复现

- `guard_snaps` 取自 `k0_annotate` 的 `G._teacher_guard_snaps(scene, belief)`（L349），与生产同路。
- 根 guard 由此恢复（`evaluate_root` L373–380）；分支 guard 快照 `me_snap`／`opp_snap`（L431–432）与每场景 `opp_snaps[m]`（L426）在 Phase 1 取定；rollout 中按 L447–452 原样恢复（sample 信念下 `g_opp` 恢复 `opp_snaps[m]`）。

### 1.5 只换 trunk 的边界

换为学生 Policy 的调用（仅此三处，其余一律教师原值）：

| 位置 | 代码 | 学生 trunk 做的事 |
|---|---|---|
| 延续步双方贪心解码 | `evaluate_root` L467–468 `act(x0, f0.aux)`／`act(x1, f1.aux)` | 双方延续动作均由学生 trunk 解码 |
| 叶评估 | L492 `_leaf_u`（内为 `pol.act(..., want_u=True)`） | 叶局面 u 头由学生 trunk 出 |
| guard 内重解码 | `guard.apply` 内的 `pol` 调用（L479–484 经由） | guard 逻辑／配置不变，底层 policy 为学生 |

不换的部分：信念采样（§1.2）、先验（§1.1）、对手根动作（§1.3）、臂集（§1.1）、guard 配置与快照（§1.4）、终局打分 `_terminal_u`（L490，F01/F04 终局分、无 trunk 调用）、每步后终局检查（L459、L488）、聚合公式（L494–496）、`teacher=2`（风格号，生产值；学生即按 style 2 蒸馏）。两 trunk 均按 L358–363 包为 `CachePolicy`。

- 学生 Policy 构造：经钉版 plug-in `stage0_teacher_small.py`（5c0dc757）的 STUDENT_CFG（C tier）＋学生 npz `b22020ff`，走 `cnet2_np` Policy／CachePolicy——与学生侧标注跑同一构造，不另起加载路径。
- 教师 Policy 构造：`stage0_teacher.py`（c6ade5ce）＋权重 `06a03d9c`，同一构造（对照闸与 Phase 1 用）。

### 1.6 Harness 写法（执行时，不在本提案内写）

- 新文件，不改任何钉版文件。结构照 `evaluate_root` 一分为二：Phase 1（setup）逐行执行 L387–432（教师 pol）；Phase 2（rollout）逐行执行 L433–496（`pol` 参数化）。直接 import 钉版 helper（`belief_carbon`、`_legal_root`、`_leaf_u`、`_terminal_u`、`feat_copy`、`_is_terminal`）与 `k0_annotate` 的场景供给，不重实现。
- 逐根调用量核对：满臂每根 680 次 trunk 调用（设计公式 `8 + (1+D+A) + 5AM`，其中 `5AM=640` 为 rollout 调用）；harness 记 `counter` 实测值，偏离即查，不略过。

### 1.7 输出格式与预期可比数

- 重跑分片与教师分片同 schema（`k0_annotate._root_row` L355–372）：`arms`（钉死值）、`n_arms`（钉死值）、`priors`（钉死值）、`q`（学生）、`Q`＝Q_mean（学生）、`Q_std`（学生）、`belief` 码、`n_unknown`（与教师同值）。`_meta.identity` 的 `config_sha256`／`games_sha256`／`replay_sha256` 与教师件逐字同（v4 比对器 L93–99 的 identity 前检查据此）；`_meta` 另记口径 `换 trunk 重跑`、学生 npz md5、教师 generation id、harness md5。
- 预期（构造使然，实测为准、不预设）：臂集逐根相同 → `compared_roots` 应为 396／396；可比动作条目数 ＝ Σ 教师 `n_arms`（教师件 `n_arms` 全 16 → 6,336）。实际数字以比对件为准。

## 2. 对照验证计划（开工前硬闸）

1. **对照跑**：同一 harness、教师 trunk、8 根——每局 steps 200–201 × seats {0,1}。对照教师分片的 `q` 与 `Q_mean`，要求**逐位一致**（exact float equality）；`arms`／`priors` 透传逐值一致（assert）。
2. **通过** → 对照闸过，方可开全量重跑。**不过** → 停，不开工，按下述诊断，不擅自放宽：
   - (a) 同机再跑一遍对照：两次对照结果彼此不一致 → harness 非确定性 → 作废；
   - (b) 两次对照彼此一致、但与教师分片有差：若偏差均匀 ≤ 1e-12 且每根 argmax 相同 → 可能是跨机浮点噪声——报偏差数字、停，等裁定（不 proceed、不擅自作废）；
   - (c) 其余情形（偏差更大、或 argmax 变）→ harness 有误 → 重跑作废，报偏差分布。
3. **确定性复核**：学生重跑中抽 2 根重跑一次，须与自身逐位一致；不一致即停查。
4. 闸在全量之前，不许先开全量后补对照。

## 3. 座位与用量估计

### 3.1 座位

- **主座位：spot-36**。依据：小教师注册 §4 的 annotation-grade CPU 座位；座位原则——CPU 重活走机队；L20 不在选单内（见下）。
- **备选**：spot-37、spot-38，依次。开工前直读验座位（load、空闲内存、无冲突任务）；主座位被占即顺延备选，选定座位写入执行回执。
- **排除**：spot-34（考试座位，考试优先、可抢占）；L20（只担 GPU 活，且 leg-1 训练在即、§6.4 正式窗 11-20 L20 绝对优先）；DGX（协调／供件／合并，不跑重标注活）；Studio（约 3 分钟杀 detached 长进程，且在跑 test118）。

### 3.2 用量估计（估计，未校准）

- 工作量：396 根 × 680 次学生 trunk 调用 ≈ **269,280 次调用**。
- 速率依据（实测，非假设）：学生侧标注在 L20 上 396 根／2 worker／墙时约 9 分钟 → **2.7 秒/根／worker**（2026-11-17 23:03 落盘实测）。教师侧同机同窗约 16 秒/根。
- 机队换算：教师速率 L20 约 16 秒/根 vs 机队约 30–35 秒/根（C1/C2 2026-11-14 定盘，62 worker  contention 下）→ 座位因子约 2×。学生重跑在机队估计 **约 5.4 秒/根／worker**（区间 5–8 秒，含 contention 波动）。
- 墙时：2 worker（每局一 worker）约 **18 分钟**（区间 15–30 分钟，估计）；单 worker 约 36 分钟。对照跑：8 根教师 trunk，约 4–5 分钟（估计）。搭建＋验合＋比对约 15 分钟。
- 总墙时预算：**约 1 小时（含 overhead，估计，未校准）**——与裁定「机队并行小时级以内（估计，未校准）」一致。CPU 约 1 worker-hour（估计）。
- **明示：L20 的 2.7 秒/根不是机队速率；机队速率未经实测，本估计未校准。首分片落盘后以实测单价重钉 ETA（与此前各趟同一纪律）。**

## 4. 身份三件套（逐件：值、来源、核验方式）

| 件 | 值 | 来源 | 核验方式 |
|---|---|---|---|
| 学生 npz | `b22020ff71a17af8f45a3dd2659f24c8` | L20 `stage4/student.npz`（ckpt_final.pt 导出） | 座位上 md5 直读，开工前 |
| 教师件 | `stage0_teacher.py` md5 `c6ade5ce`（25,186 B） | 仓 `code/stage0_teacher.py` | API 读回＋座位 md5 直读 |
| config | `config_sha256 9a2b2440` | harness 按运行 config 重算 | 须逐字等于 9a2b2440 |
| 回放 v2 | 994400 sha256 `6ed2623e`（226,125 B）；994401 `6607fe5e`（229,043 B） | 登记册 v2＋stage4/ 副本 | 开工前 sha256 对登记册 |
| k0_annotate | md5 `7f567a2e`（49,151 B） | 仓 `code/k0_annotate.py` | API 读回＋座位 md5 直读；场景供给与生产调用参数所源 |
| 教师 generation | `gen-3f2c17f8b7beb5b6` | L20 `annot_teacher/shards/` | generation id＋逐分片 md5 对标注 manifest |
| 学生 plug-in | `stage0_teacher_small.py` md5 `5c0dc757` | 仓 `code/stage0_teacher_small.py` | API 读回；学生身份对（plug-in，npz） |
| 权重 | `06a03d9c` | 教师 heads 权重 | 座位 md5 直读 |
| 代码栈 | cnet2_np `d8c15ad4`、cnet2_search `94f7fb64`、cnet2_feat `2a45b08b`、fastsim `aefe6c83`、gate `e922d9d8` | 仓内钉版 | 座位 md5 全扫 |
| 比对器 | shard_compare v4 md5 `1e9ad085` | 仓 `reports/eng_verify/shard_compare_v4.py` | API 读回；比对时不改 |

任一核验不合即停，不开工。

## 5. 交付件清单

- Harness：`reports/eng_verify/retry_trunk_harness.py`（新文件；提交后 API 读回验 md5）。
- 重跑分片：与教师／学生／v4 输出分开存放的目录（座位上，generation 新编），口径标注 `换 trunk 重跑`；同 §1.7 schema；`_meta.identity` 与教师件逐字同。
- 比对结果：`reports/eng_verify/compare_stage4_retry_trunk_396.json`——shard_compare v4 原样跑（base＝教师 `gen-3f2c17f8b7beb5b6`，cand＝重跑 generation），提交后 API 读回验 md5。
- 执行报告：`STAGE4D_RETRY_TRUNK_REPORT.md`——对照闸结果、实测单价、Tier 1/2 四项实测值、compared_roots 与可比动作条目数；只报数，不判 Tier。
- 比对值引用：旧件 994401 单根数值已作废，不得引用；v4 读数（94 根／228 条）仍按原口径记录。

## 6. 保底条款（裁定 §2 原文复述）

> 若重跑后 compared 仍 <100，以 v4＋重跑两份读数加说明作结，不再追新口径；Tier 判读按两份读数的保守面——任一份未达 Tier 2 宽屏即判未达 Tier 2；Tier 1 须四项全过。

## 7. 执行程序与停表

1. 本提案报策略组批后才开工；批前不写 harness、不动任何机器。
2. 开工三件套：座位直读确认＋身份三件套核验＋对照闸通过，三者齐才放全量。
3. 停表：身份任一不合、对照闸不过、座位冲突（训练／考试占座）、L20／DGX 被误用——任一触发即停报，不硬跑。
