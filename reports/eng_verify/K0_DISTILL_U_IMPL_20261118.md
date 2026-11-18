# K0_DISTILL_U_IMPL_20261118 — k0_distill 改造版（终局 U 价值半环）实施回执

执行人：工程组（唯一执行人）。日期：2026-11-18。
依据：裁定 `reports/RULING_KATAGO_CORE_U_HEAD_20261118.md`（commit `12b4a792`，方案①改造核心另立新件）、设计 `reports/DESIGN_K0_DISTILL_U_20261118.md`（commit `5d11dbce`，md5 `4f186a44`，规格以它为准）、接口回执 `reports/eng_verify/KATAGO_LOOP_INTERFACE_CHECK_20261118.md` §2/§5。

## 1. 隔离纪律执行

- 钉版 `code/k0_distill.py`（`2728fee6`）与 `code/k0_join.py`（仓内 `fac55acb`）一字未改：实施全程在本机副本上以「断言式逐处替换」派生新文件，收尾复读两原件 md5 仍为 `2728fee63e16f4321c5639273a62631c` / `fac55acbcbffdfd6582c9449e535aa2f`。BC 线不受影响。
- 派生脚本留本机 `~/workspace/katago-loop-smoke/u_impl/make_distill_u.py` / `make_join_u.py`（每处替换均断言出现次数，对不上即整件不写）。

## 2. 新文件（最终名 + md5，均已入仓 main 并 API 读回逐字验）

| 文件 | md5 | 字节 | 入仓 commit |
|---|---|---|---|
| `code/k0_distill_u.py` | `06826f5926c97ecccda8b8cbd18ff020` | 67,392 | `59fe7b4b54c0` |
| `code/k0_join_u.py` | `9054aadca62a9770ea5b7df90d46cb73` | 40,262 | `b58653bec6b9` |

## 3. 头形态核对（不猜形状）

钉版 `cnet2_model.CNet2.heads(ctx)` 本体直读：`utility = sigmoid(Linear(D,1))` → [B]（[0,1]，= E[U]）；`wdl = Linear(D,3)` logits [B,3]（列序 win/draw/loss）；`margin = Linear(D,1)` → [B]（= E[margin]/1000）。`encode()` 不读候选轴（y 与逐候选重算特征不进 ctx），故同一根的 K 份展开行 ctx 恒等，三头取每根候选槽 0 读数。损失挂接按此形态，未出现对不齐。

## 4. /2 建集（join 改造版）

- 输入：接口冒烟既有产物（build gen `gen-4f6b7ce57c46bbfa`、annot gen `gen-3bd8b0585fce9f1f`）+ sidecar `terminal_u.json`（md5 `54f901afca6b90249f02c22a5c580876`），未重跑 annotate。
- 产物：/2 代 `gen-1e1864a45077767e`，60 根 / 960 候选，本机 `~/workspace/katago-loop-smoke/u_impl/joined2`；发布前 consumer preflight 经 **k0_distill_u** 本人加载通过（含三终局字段在册）。join 全程 RC=0。
- **seat↔side 映射（本趟关键发现，防静默翻转）**：sidecar 以产局方 side 名 `a`/`b` 记值，与标注 seat 空间不是同一命名。实证锚（本语料 selfplay-987001）：build 代 meta 的逐 seat 原始标签（`raw_u`/`raw_margin`，按 sides 序）为 seat 0 → u 0.192279 / margin −470.25、seat 1 → u 0.807721 / margin +470.25；sidecar 为 a → +470.25 / 0.807721、b → −470.25 / 0.192279（scores 亦同向：a 9290.22 = seat 1 的分数）。即 **sidecar `a` = seat 1、`b` = seat 0**。`k0_join_u` 不单独信任此映射：对每一局逐 seat 与 build meta 的 `raw_u`/`raw_margin` 交叉核对（容差 1e-9），任一不合即 die 点名该局该 seat；build meta 缺这两锚即 die 拒建（不可验的映射按拒处理，不猜）。另逐 seat 断言 sidecar 的 u 与其自身 margin 的 `0.5·W + 0.5·sigmoid(M/1000)` 一致（1e-6）。
- 落盘终局标签（逐 seat 直读验）：seat 0（30 根）：terminal_u 0.1922785368343741、terminal_margin −470.25、terminal_wdl 2；seat 1（30 根）：0.8077214631656259、+470.25、0。
- **wdl 编码映射（明示）**：按设计列序 **胜 0 / 平 1 / 负 2**，由本根 seat 视角终局真实分差符号定（margin > 0 → 0、= 0 → 1、< 0 → 2）。terminal_u 取 sidecar 逐 side 原值（产局方按每局真实终局分差先算的 U），禁止由 E[M] 反代——训练器侧另有逐根一致性闸复核此恒等式（见 §6 负例之外的数据闸）。

## 5. 开训前坏根预扫（改造版训练器本人判据）

以 `k0_distill_u.expand_candidate_batch` 对 /2 集逐根全扫：**全根 60 / 问题根 0**。

## 6. 设计 §6 四项验收

**(1) 全链 RC**：join_u 建集 RC=0（含 k0_distill_u preflight）→ 预扫 60/0 → 双目标训练 10 步 RC=0、权重落盘 + `.complete.json` 标记在册。训练日志 `student_u_smoke.npz.log.jsonl` 与权重同目录留本机。（produce/build/annotate 三环为接口冒烟既有产物，按派单复用。）

**(2) 三头 loss 分项（逐项非零且有限；全 10 步均在日志中逐项记录）**：

| 步 | total | ce | utility_mse (λu=1.0) | wdl_ce (λw=0.5) | margin_mse (λm=0.5) | terminal_weighted |
|---|---|---|---|---|---|---|
| 1 | 4.029460 | 3.120475 | 0.112448 | 0.968336 | 0.624740 | 0.908986 |
| 10 | 5.884953 | 4.073229 | 0.039409 | 0.492156 | 0.213029 | 0.392002 |

旁证：第 1 步 policy CE 3.120475 与钉版冒烟同数据同步数首步 3.12047 一致（初始 policy 侧未变，终局项为纯增量，总 loss = CE + KL + terminal_weighted 逐项可加总复核）。

**(3) 解冻对账**（训练前快照 vs 训练后；且已从落盘 npz 对 parent 独立复核，非仅训练器自报）：

| 头 | max\|Δ\| | 已更新 |
|---|---|---|
| utility | 0.002459932118654251 | 是 |
| wdl | 0.0029196515679359436 | 是 |
| margin | 0.00224362313747406 | 是 |
| aux | — | `np.array_equal` 逐位不变 = True（aux.weight / aux.bias 分别验） |

policy 侧：`trainable_moved` 含三头全部 6 个参数名及 trunk/decoder/policy 侧参数（与钉版同口径移动）。冻结集实测仅 `aux.weight`、`aux.bias`。

**(4) 新版文件 + 产物册登记**：新文件见 §2。冒烟权重 `student_u_smoke.npz`：md5 `0833d42979ec7c49eced04bfabd4034d`、sha256 `4bc56812b75865242e3234e8d88fa64a82af937cab889c6ef0b172b419ff03b2`、13,997,300 字节，本机 `~/workspace/katago-loop-smoke/u_impl/student_u_smoke.npz`。产物册 `data/artifact_registry.json` 已按既有格式追加条目 `katago_loop_u_smoke_student_10upd`（commit `633c81aba211`），put 后 API 读回验：注册表新 md5 `1fd25be32f61ab16180c5cad92706dd1`、条目在册、条目内 md5 与落盘一致。

## 7. 必测负例（防假跑，报错原文）

(a) /1 代数据集（接口冒烟 joined `gen-fc347a6811bb972e`）喂 k0_distill_u → RC=1，原文：

```
k0_distill_u data: generation format 'cnet2-k0-annotation/1' != 'cnet2-k0-annotation/2'; refusing to start
```

(b) /2 集删 `terminal_wdl` → RC=1，原文：

```
k0_distill_u data: value label bundle lacks terminal field 'terminal_wdl' required by format 'cnet2-k0-annotation/2'; a /2 generation without terminal labels must not train (the U half of the loop would silently not run); refusing to start
```

注：(b) 的测试集是把 /2 代的 value bundle 删字段后按 manifest 自封印口径合法重封的副本（本机 `u_impl/negb_missing_wdl`），使触发的是字段闸本身、而非文件哈希闸——模拟「有 /2 版式、无终局标签」的假跑形态。

## 8. λ 与版本戳样例

λ 为可配置常量（CLI `--lambda-u/--lambda-w/--lambda-m`），本趟取设计起点 **λu 1.0 / λw 0.5 / λm 0.5**，写入产物版本戳。权重 npz 的 meta 键为 `k0_distill_u`，样例（节录）：

```json
{"format": "k0-distill-u/1",
 "trainer": {"file": "k0_distill_u.py", "version": "k0-distill-u/1", "md5": "06826f5926c97ecccda8b8cbd18ff020"},
 "annotation_format": "cnet2-k0-annotation/2",
 "recipe": {"kl_beta": 1.0, "lambda_u": 1.0, "lambda_w": 0.5, "lambda_m": 0.5, "lr": 0.0003, "wd": 0.0001, "bs": 8, "seed": 7, "frozen_prefixes": ["aux."]},
 "unfreeze_reconciliation": {"heads": {"utility": {"max_abs_delta": 0.0024599, "updated": true}, "wdl": {"max_abs_delta": 0.0029197, "updated": true}, "margin": {"max_abs_delta": 0.0022436, "updated": true}}, "aux_bitwise_identical": true}}
```

训练配方其余口径与接口冒烟同：parent = 06a03d9c 热启动（`--parent` md5 核在启动闸内通过）、bs 8、seed 7、lr/wd/kl_beta 取训练器默认（3e-4 / 1e-4 / 1.0）。

## 9. 同机共跑情况

开训前 ps/load 直读：本机已无 `k0_annotate` 进程在跑（钉版对账任务已退场；当时 uptime 显示本机约 5 分钟前重启过——对账 annotate 的退场原因属其执行线，本线训练不依赖其产物，未深究），负载仅 runtime 守护进程。故未发生重活叠跑、无需等待，训练直接启动（墙时 239.15 秒 / 10 步）。

## 10. 未尽 / 存疑

- 本趟为冒烟规模（1 局 / 60 根）：正式环开启仍受裁定 Q2 约束——钉版六件对账（另一执行人）未齐前不开正式环，本回执不构成开环件。
- seat↔side 映射的运行时交叉核对依赖 build meta 携带 `raw_u`/`raw_margin`；未来语料若 build 代不带这两锚，join_u 会 die 拒建（设计意图：宁拒不猜），届时需先补锚再建集。
- /2 集与冒烟权重只在本机 muse 落盘（产物册已登记路径）；如需跨机消费再按登记册转运。
