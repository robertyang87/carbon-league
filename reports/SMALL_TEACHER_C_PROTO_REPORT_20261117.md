# 小教师 C 档延迟原型报告（2026-11-17，工程组执行）

承：策略组裁定 `reports/RULING_SMALL_TEACHER_C_PROTOTYPE_20261117.md`（commit c378f981）§2——B 档不过定案，按设计 §7 #10 先降 C 档原型；口径、阈值一字未动。本件只做原型构建＋测速：未提交 GitHub、未跑训练、未跑真实生成、未触生产进程，DGX 工作只在 `small_teacher_scratch/` 内。

## 1. 原型件

- cfg（裁定钉死）：C32／NB3／TA1／D64／L1／NH4／TNH4／E16／G4，约 21M MACs（裁定口径）
- 构建：`build_student_prototype.py`（md5 838e3270…，cfg 参数化）以同种子 20261117 随机初始化，经生产 cnet2_np.Policy 同路加载（bench 内 CachePolicy/load_weights 即生产路径）
- 参数量：**187,943**（≈0.19M，与裁定预估合）
- 原型 npz：DGX `small_teacher_scratch/harness/student_C_prototype.npz`，md5 **`c0fa300a4e731d9302558d4e1198210a`**（94 arrays + meta）
- 构建驱动：DGX `small_teacher_scratch/harness/build_c_proto.py`（仅传 cfg 调 build()，未改构建器本体）

## 2. 测速口径（与 B 轮逐项相同）

DGX 座、封口 994400 回放（md5 6ed2623e…，双份在册副本逐字同）单趟重放取 18 个真实局面（步 0＋200–207、双座）、batch=1、单进程、nice -n 10、每政策 288 次计时调用（16 cycles）；教师基线本轮同机复测，未沿用 B 轮数字下结论。脚本为仓内同件（bench_latency 35374c33…／bench_encode_only 1f45956a…，DGX 副本 md5 逐字合）。

一处执行说明：scratch `stack/` 缺 env_src 子树，eval_v53 按自身目录拼路径 import 失败，故 --code-dir 改用 `/home/arm/junyang/carbon/code`；已验该目录与 stack/、k0_dist/pkg 三处六件模块＋eval_v53（45792dfb…）＋carbon.json（cd55531b…）md5 逐字全等，测量代码与 B 轮同一，非口径变更。

注意：两件 bench 脚本输出 JSON 内学生键名为写死的 `student_B_random`，实测对象以 latency JSON 内 `student_npz_md5`＝c0fa300a… 为准（C 原型）；encode 脚本不记 npz md5，靠同轮命令记录对应同一 npz。

## 3. 读数

### act 口径（bench_latency_c_dgx.json）

| 政策 | mean | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 教师（本轮复测） | 33.58 ms | 33.71 | 35.56 | 28.85 | 37.66 |
| 学生 C（随机权重） | **3.50 ms** | 3.54 | 3.76 | 2.78 | 4.16 |

比值：**mean 9.58× / p95 9.46×**。测时 loadavg（JSON 内记）：前 [0.14, 0.41, 1.02] → 后 [2.58, 0.93, 1.18]。

### encode 口径（主干分解）

复跑轮为读数（bench_encode_only_c_dgx_run2.json）：

| 政策 | mean | p95 | min | max |
|---|---|---|---|---|
| 教师（本轮复测） | 27.03 ms | 28.32 | 24.26 | 30.10 |
| 学生 C | 2.52 ms | 2.64 | 2.33 | 2.86 |

比值：**mean 10.73× / p95 10.72×**。loadavg 前 [3.61, 1.37, 1.32] → 后 [4.87, 1.72, 1.43]。

首轮（bench_encode_only_c_dgx.json）教师段受负载污染：教师 mean 38.66／p95 66.97／max 73.70（encode 均值反高于同轮 act 均值，物理上不成立），其比值（13.97×）虚高、不作读数，文件保留备查；学生段首轮 mean 2.77／p95 2.91，与复跑轮同量级。

### B 轮对照（注明：跨轮负载不同，只作参照）

B 轮：act 教师 29.71／学生 B 6.90（4.31×），encode 教师 26.47／学生 B 5.32（4.98×）；B 轮 loadavg 前 [4.02, 19.04, 22.79]。本轮负载显著更轻（1 分钟 0.14 对 4.02、5 分钟 0.41 对 19.04），绝对值跨轮不可直接比；且本轮教师 act 均值（33.58）反高于 B 轮（29.71），座位噪声（大小核调度等，推断）明显——按裁定以比值为判读主据。

## 4. 判读线对照（如实报，资格认定归策略组）

裁定 §2 判读线（DGX 原型口径）：act 均值 ≤5 ms 且同机比值 ≥8×。

- act 均值：3.50 ms ≤ 5 ms —— **达**
- act 比值：9.58× ≥ 8× —— **达**

两项均达。附带说明：裁定事前线性外推预测 C 档 act 约 5.4–5.5 ms，实测 3.50 ms 优于预测；但本轮负载显著轻于 B 轮，绝对值受惠于此，超出部分不能全记为档位之功；比值口径下 9.58× 距 8× 线有余量。本件为 DGX 可达性证据，非正式 §4（正式 §4 仍为 spot-36、同机、固定 manifest、L20 产出）。

## 5. 接入件随档改

本地镜像 `stage0_teacher_small.py` 的 STUDENT_CFG 已改 C 档（第 94 行，唯一改动）：

- 新 md5：**`5c0dc7579126ab7e10e774eef12cea50`**
- 对现行版（c2ac6ff20585dc61589389fc90618b7a）diff 恰一行：`STUDENT_CFG = dict(C=48, NB=4, D=96, NH=4, L=2, E=24, G=4, TA=2, TNH=4)` → `dict(C=32, NB=3, D=64, NH=4, L=1, E=16, G=4, TA=1, TNH=4)`，其余字节不动。

## 6. 落盘清单

- DGX：`small_teacher_scratch/harness/student_C_prototype.npz`（c0fa300a…）、`results/bench_latency_c_dgx.json`、`results/bench_encode_only_c_dgx.json`（首轮，污染备查）、`results/bench_encode_only_c_dgx_run2.json`（encode 读数）
- 本地镜像（harness/）：`bench_latency_c_dgx.json`（md5 9d4155b5…）、`bench_encode_only_c_dgx.json`（d7592287…）、`bench_encode_only_c_dgx_run2.json`（ff94b9cc…）、`stage0_teacher_small.py`（新 md5 5c0dc757…）、本报告
