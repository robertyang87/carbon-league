# K0 第二轮 件 2 回执 — code/k0_distill.py（联合 q 蒸馏入口）

派单：策略组（设计之准 `reports/K0_ROUND2_DESIGN_DRAFT_20261114.md` §6/§10 ＋ §13-C/D；方向裁定 = steer 1）。
实施：工程组本线。状态：**已落 main，读回字节一致；合成 fixture 自验五项全过；未启动任何真实训练**。

## 落仓

| 项 | 值 |
|---|---|
| 路径 | `code/k0_distill.py`（新建，repo 此前无此文件） |
| md5 | `d5fc464fa3e8acafd28f675f3f281091` |
| 字节数 | 37,912 |
| commit | `b674db2c4d5632bccba19bf85d82325942360973` |
| blob sha | `609867d3e6a62157569c2fb32fb2a2bf09b7411b` |
| 读回验证 | API 读回 md5 与本地逐字节一致（BYTE_IDENTICAL True） |

只新建此一文件；k0_train、cnet2_exit_train、闸门、教师、PPO 均未触碰。

## KL 方向（方向裁定 steer 1，已落地）

- 锚点与监测统一为 **KL(π_student ‖ π_parent)**，与 k0_train 现行 screen 口径（`k0_train.py` ~:1012–1015）同方向，与 Stage-A 全部 KL 数（B0 0.9361 / A0 0.9481 / A1 0.9774）可比。
- 口径：逐槽分类 KL（categorical KL over the slot's action distribution），对每个候选的有效槽（arm ≥ 0）求和，再对根的候选取均值；监测按根报 mean / p99。
- 代码注释、模块 docstring 与产物 provenance 的 `KL_DEFINITION` 字段均明示此方向与求和口径（验收 t3 实测 provenance 中该字符串在位）。
- 反向 KL（parent‖student，原始简报口径）未采用——锚点是约束不是模仿，且须与监测同量。

## 消费格式契约（§13-C 的消费侧读法，格式名 `cnet2-k0-annotation/1`）

- 数据集只经 `current.json` 指针消费；直接指向 generation 目录即拒（与 exit 侧口径一致）。
- 验封顺序：指针 manifest 哈希 → manifest 自封（canonical JSON 去 `self_sha256` 后 sha256）→ meta 封 → 逐字段存在性/字节数/sha256/行数全验，任一不过即拒并点名。
- 特征字段 = cnet2_build `FIELDS` 的模型输入（去 y）＋ gid/step。
- 标签分文件：`policy_labels.npz` {game, step, seat, arms int [R,K,OWN], n_arms, priors, q, pi0, pi1, pi}；`value_labels.npz` {game, step, seat, Q}。manifest producer 须带 teacher_config_hash、teacher_weights_md5、belief。
- 载入门（全部按根/按字段点名拒收）：特征浮点字段有限；bundle 键互相对齐且与特征 gid/step 逐行一致；n_arms ∈ [1,K]；填充候选惰性（arms 全 −1、priors 0、q 0、Q NaN）；真实候选 priors 有限 >0、和为 1（tol 1e-5）；Q 在真实候选上有限；存档 q 有限 ≥0、和为 1；arm 值 ∈ [−1,4] 且 (arm ≥ 0) 与 tok_mask[:OWN] 逐位一致。
- **π 门（§13-C 合格性）**：pi0/pi1/pi 有限、pi0/pi1 ∈ [0,1]、pi ∈ (0,1]、|pi − (pi0+(1−pi0)·pi1)| ≤ 1e-6——缺 π 或不一致按根拒收。
- **存档 q 交叉检**：以 (Q, priors) 经 `construct_target` 重构 q，与存档 q 最大绝对差 ≤ 2e-6，否则按根拒收。
- **父代绑定闸**：manifest 的 teacher_weights_md5 必须等于 `--parent` 文件字节的 md5，不符在训练前拒收并点名。

## 配方定义

- **目标构造**：`construct_target(Q, priors, q_temp=0.05, flatten=False)` = 教师公式 q ∝ sqrt(max(P_A, 1e-12)) · exp((Q − maxQ)/0.05)，Q 为 U 尺度。S/C 两臂都经此同一函数在消费侧构造目标。
- **联合 CE**：−Σ_k q_k · joint_logp_k；joint_logp 为逐候选在其自身前缀下 teacher-forced 重放的槽位 logp 之和，**不在候选集内重新归一**。
- **锚点**：每根 = 其候选的 KL(student‖parent) 均值；批损失 = IPW 加权根均值 (CE + kl_beta · 锚点)，kl_beta 显式参数（默认 1.0）。
- **冻结清单（§6.3）**：前缀 `utility.`、`wdl.`、`margin.`、`aux.` → requires_grad 关、优化器排除、发布前与初始快照逐位核验；任一冻结参数移动即 exit(3) 不落盘并点名。可训 = 骨干 ＋ 策略头。
- **C 臂 flatten（§5/§6）**：`--flatten-q` 只覆盖 Q 一处——把每根的 Q 换成该根 Q 均值，exp 项消去 → q₀ ∝ P_A^0.5；其余（同根、同候选集、同 lr/步数/批序）与 S 逐项同构。批序由单一 RandomState(seed) 的逐 epoch 置换决定，flatten 开关不碰 RNG（验收 t3 实测 S/C 的 batch_sha256 序列一致）。
- **IPW（§6.6/§13-D 补注）**：1/π 仅作能力保留——只有当载入记录中存在登记在册的 π < 1−1e-12 时才激活（`ipw_active`），权重 = 1/π；默认全 π=1 时恒等（权重全 1），不存在未登记的默认 IPW 路径。
- **监测（§6.5）**：每步日志行带 `kl_monitor` {mean, p99}（定义同上）与 `denominators` {roots, candidates, batch_sha256}；只记录不裁决。
- **产物**：`cnet2_train.export` 标准 npz（可被既有工具/k0_duel 类驱动直接加载）＋ meta.provenance（臂别、父代 md5/sha256、数据指纹 generation/teacher_config_hash/teacher_weights_md5/belief、配方 q_temp/kl_beta/lr/wd/bs/seed、ipw 块、冻结清单与逐位核验结果、本文件自身 code_md5）＋ `<out>.complete.json` 完成标记＋ `<out>.log.jsonl`。非有限 loss/grad/param 触发熔断日志 exit(3) 不落盘。无 resume（单代短跑从父代重开，为设计决定）。

## 验收自验（合成 fixture，`test_distill.py` 全套 ALL_PASS）

fixture：小 cfg {C:32, NB:2, D:64, NH:4, L:1, E:16, G:8}；parent.npz（torch seed 1234，md5 `c67a3afb3496087db75cd16c5b54796b`）；ds_a（R=6 根、K=4、23 个真实候选、全 π=1）；ds_b（IPW：奇数根 π=0.5，根 0/1 内容相同仅 seat 不同）；ds_nanpi（根 2 π=NaN，封存完好）；ds_badq（根 0 存档 q 两项对调、和仍为 1，封存完好）。

| # | 验收项 | 结果 |
|---|---|---|
| 1 | 联合 logp 一致：模块路径 vs 逐槽独立重放参考（每槽新解码、y 自该槽起清零） | **PASS** — 8 个候选最大绝对差 **2.9968e-07**（容差 1e-5） |
| 2 | Q 置平梯度一致：flatten 目标与独立 √P_A 归一目标逐值同；一步梯度 vs 独立重放-autograd 参考（β=0） | **PASS** — 目标差 **0.0** 且跨 Q 数组恒等；梯度最大绝对差 **8.82e-06**（最差 stem.bias，容差 1e-4） |
| 3 | 冻结头逐位不变 ＋ 两臂端到端：S/C 各 6 步子进程跑，批序 batch_sha256 序列一致、冻结参数与父代逐位同、可训参数确有移动、provenance/完成标记齐全 | **PASS** — frozen_bitwise True、frozen_check True、moved_count **63**、parent_md5_ok、marker_sha_ok、kl_monitor 键齐 |
| 4 | 缺 π/坏记录/未验封被拒：ds_nanpi 按根拒（π 非有限）；ds_badq 按根拒（存档 q 与 §13-C 构造不符，差 0.709）；篡改副本拒（字段 'scal' 哈希不符）；generation 目录直指拒（须经 current 指针）；父代与 manifest teacher_weights_md5 不符拒（子进程 rc=1 并点名） | **PASS**（五路全拒且点名） |
| 5 | IPW：π=0.5 根权重恰为 π=1 根两倍；恒等路径权重全 1 | **PASS** — 权重 [1.0, 2.0]，比值恰 **2.0**；ds_a ipw_active False、ds_b True；ds_b 内容相同两根 CE 逐值相同（10.1815454935956） |

注记：t3/t4 首跑各有一处非模块问题已修并重跑全套通过——t3 为 tmp 文件名未带 `.npz` 后缀致 np.savez 自动补后缀后 os.replace 失败（模块 bug，已修：tmp 名以 .npz 结尾）；t4 为测试侧把拒绝信息截断到 80 字符后再做子串匹配（测试 bug，模块拒绝行为本身正确，已修测试）。

## 证据位置

- 开发副本与 fixture：`~/workspace/k0-round2/distill/`（`k0_distill.py`、`make_fixture.py`、`test_distill.py`、`fixture/`）
- 验收运行：`test_distill.py` 全套输出 ALL_PASS（t1/t2/t3_e2e/t4_negatives/t4_parent_mismatch/t5_ipw）
