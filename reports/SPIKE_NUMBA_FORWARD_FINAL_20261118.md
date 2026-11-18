# 编译前向 spike 正式报告（numba njit vs 钉版 numpy 前向）

日期：2026-11-18。执行：工程组（两任执行人，前任完成捕获/转写/首轮 bench，续作完成 bug 定位、修复与复测）。
范围：离线 spike，只量时延与差分两数；不做生产形态、不出规格。

## 0. 结论摘要

- **差分**：修复一处转写 bug 后，2,688 次真实调用全量对打——**argmax/label flips 全类为 0**；priors（logits）max|diff| ≤ 5.15e-05、value ≤ 3.35e-06、u ≤ 1.37e-06，均为 float32 噪声量级，但**不是逐位恰零**（见 §6 事实注记）。
- **时延**：本机限流小核比值 **1.05×**（首轮会话）；修复后同口径复测在更重争用下为 0.99×（中位），同量级、无显著提速。可移植性单列于 §5。
- **覆盖率**：act 整段 100% 编通（encode＋value/u 头＋12 槽自回归解码＋Prefix＋GRU），无子块退守；首次编译 64.4 s（首轮会话）／97.1 s（复测会话，重争用）。

## 1. 输入集与环境

- 输入：钉版 `evaluate_root` 在 selfplay-300002 步 200/215 × 双座共 4 根上捕获的**非采样** act 调用 **2,688 次**（`out/captured.npz`，含每次调用的全部输入与 numpy 参照输出）。分类构成：greedy 2,052／forced 偏离（forced_dev）120／forced 重打分（forced_rescore）4／want_u 512。（注：rescore 只有 4 次——候选臂的批量评估走 CachePolicy 批量路径、不经 `_act_impl` 逐次调用，捕获口径即 `_act_impl` 本人。）
- 环境：本机（L20 装不上 numba——无外网 DNS/pip 镜像、egress 代理约 7.5 KB/s、60 MB llvmlite wheel 传不动，按备选口径回本机）；`PYTHONPATH=pylibs`（numpy 2.2.6＋numba 0.67.0＋llvmlite 0.49.0）＋系统 Python 3.12；taskset 单核口径计时。
- 版本差覆盖：生产 numpy 为 2.5.4（L20）。根级交叉验（本 env numpy 2.2.6 重跑 vs memoprobe 在 L20 numpy 2.5.4 的存档）：4 根 arms 全等、priors max diff ≤ 2.69e-07、q ≤ 2.16e-07——版本差不影响本代码数值。
- 对照组自检：numpy 本 env 重跑与捕获参照逐位一致（每 25 次抽查、0 不合）；仪器化复刻版 `_act_impl`（bisect 用）与捕获参照逐位全等。

## 2. 差分全表（修复后，2,688 次全量，act_nb vs 捕获 numpy 参照）

| 调用类 | 调用数 | priors(logits) max\|diff\| | value max\|diff\| | u max\|diff\| | label flips | 槽位 flips |
|---|---|---|---|---|---|---|
| greedy | 2,052 | 5.15e-05 | 3.35e-06 | — | 0 | 0 |
| forced_dev | 120 | 2.29e-05 | 5.37e-07 | — | 0 | 0 |
| forced_rescore | 4 | 1.91e-05 | 5.37e-07 | — | 0 | 0 |
| want_u | 512 | 4.96e-05 | 1.53e-06 | 1.37e-06 | 0 | 0 |

修复前（同一输入集、同一 bench）：greedy flips 2,051/2,052（槽位 flips 9,731、logits max diff ≈ 1.0e9）、want_u flips 512/512、forced 类 logits max diff 90.49——即 §4 的 bug 签名。

## 3. 时延全表（计时子集 364 次，热身 25 次，交替顺序，perf_counter）

首轮会话（修复前构建；其 numba 侧做的尚不是被证明等价的工作，此表仅作量级参照）：

| 类 | n | numpy 中位 ms | numba 中位 ms | 比值 |
|---|---|---|---|---|
| greedy | 120 | 235.5 | 227.7 | 1.03× |
| forced_dev | 120 | 252.9 | 232.0 | 1.09× |
| forced_rescore | 4 | 256.3 | 230.4 | 1.11×（n=4，无统计意义） |
| want_u | 120 | 244.9 | 246.0 | 1.00× |
| **总体** | 364 | 244.9 | 232.6 | **1.05×** |

p95 两边同在 400 ms 量级、无分离。

修复后复测会话（同口径；本 VM 当窗争用显著更重，绝对值约为首轮会话的 3.6 倍，**绝对值不可移植、不可与首轮会话横比**）：

| 类 | n | numpy 中位 ms | numba 中位 ms | 比值 |
|---|---|---|---|---|
| greedy | 120 | 901.2 | 990.8 | 0.91× |
| forced_dev | 120 | 921.1 | 865.5 | 1.06× |
| forced_rescore | 4 | 698.2 | 878.8 | 0.79×（n=4，无统计意义） |
| want_u | 120 | 786.6 | 895.1 | 0.88× |
| **总体** | 364 | 888.5 | 893.5 | **0.99×**（p95 比值 2469.7/2300.7 ≈ 1.07×） |

两轮会话的总体比值（1.05× / 0.99×）同量级：numba 对此前向无显著提速——numpy 侧耗时主体在稠密内核（matmul/conv）内部，JIT 能省的只有 Python 编排开销，占比不足以形成量级差。

## 4. Bug 根因与修法（bisect 定案，脚本 scripts/bisect2.py）

- **根因（一句话）**：低秩头组合表达式的运算符优先级转写错误——钉版原式 `lg = base + lin(mf,"V") @ lin(h,"U")` 中 `@` 优先级高于 `+`，语义为 `base + (V @ U)`；转写时误按左结合读成 `(base + V) @ U` 并显式写出广播加，logits 自第一个解码槽起系统性错误。
- **定位证据**（一轮 bisect 即定案，未用满两轮）：
  - greedy 调用 #0：第一个被处理槽位即 **槽 0**、即第一分岔点，该槽 logits 与参照差 90.33；
  - 同槽以 numpy 中间量（base／V／U）构造两候选式：act_nb 输出与误读式 `(base+V)@U` 合至 **2.29e-05**、与正解式差 90.33——同时证明该槽上游全部量（encode tok/ctx：max diff 5.01e-06／9.78e-07；value：3.53e-08；uf/mf、GRU 输入与状态、base、V、U）两边一致，分岔只在组合一步；
  - forced 调用 #1（标签被强制、轨迹不分岔）：全部 12 槽逐槽验，act_nb 与误读式逐槽合（≤1.26e-04）、与正解式逐槽差 7.25–61.4——单点根因覆盖整段解码，无第二处分岔。
- **1e9 量级差分的来源**：非第二 bug，而是级联——槽 0 argmax 翻转后 Prefix 状态（cash/slots）轨迹分岔，s≥10 槽的 legal 掩码随之分岔，一侧 NEG（−1e9）对另一侧真实值，差达掩码量级。
- **修法**：`scripts/act_numba.py` 中将显式广播块改回原式 `lg = base + mv @ hu`，并把原误导注释替换为优先级明示。修复后全表见 §2（零 flips）。
- 四类排查结论：布局类——参数表分岔已由前任 V1 偏置前案排除（见 §7）；索引类——槽位/门切片经 12 槽逐槽归因排除；掩码语义类——legal 与 `Prefix.feats` 逐项一致、1e9 差分系级联而非掩码转写；**序（表达式结构）类——即本 bug**。

## 5. 可移植性明示（单列）

**1.05× 是本机限流小核的比值**：测于共享 1.5 vCPU 的本 VM（taskset 单核），绝对时延（中位约 245 ms/次）为本机限流产物，不可移植；修复后复测在更重争用下得 0.99×，与 1.05× 同量级。对 **c8ae 座位核（标注实际运行的硅）：未实测**。结构性预期：此前向的耗时由稠密内核（matmul/conv 等 BLAS 类计算）主导，编排开销占比小，故 numba/numpy 比值对核绝对速度一阶不敏感，**预期在 c8ae 上同量级（约 1.0× 附近、无显著提速）**；但缓存层级、BLAS 内核选择随硅而异，比值仍可能漂移——**预期不等于实测**，立项若需 c8ae 口径，须在座位核上实测后引用，不得以本段预期充读数。

## 6. 事实注记（供立项裁决参考，不替裁决下结论）

编译版差分即便修复后也**非逐位恰零**：priors max|diff| 达 5.15e-05（float32 噪声级、零 flips），来源是手写循环归约顺序与 numpy 向量化归约顺序不同，属转写固有性质而非缺陷。此类差分在「不改变质量」的口径下如何处置（恰零硬闸 vs 噪声级容差），属立项裁决事项，本报告只呈数。

## 7. 改写登记（numpy 原版 → njit 转写）

| 项 | 原版 | 转写 | 验证 |
|---|---|---|---|
| float16 量化（Prefix uf/mf） | `astype(float16).astype(float32)` | numba 无 float16 类型，手工位级 RNE 实现 `_f16r` | 随机 20,000 值逐位比对 numpy 链：0 不合 |
| 参数表布局 | `pol.w` 字典＋`__init__` 派生量 | 冻结为 typed lists（M2/V1/GNW/GNB/LNW/LNB/A3）；**M2 与 V1 布局在下标 28 后分岔**（M2 29–32 = gih/ghh/aemb/teach，V1 29–34 = GRU 双偏置＋value/utility 行），attn/tattn 须用各自基址（M2 33/41、V1 35/43） | 前任曾共用基址致 attn 偏置错位（V1[41] 实测 (256,) vs 应 (384,)，形状检查拦下），已改分岔基址并在 `build_params` docstring 立注记 |
| 参数来源 | 运行时构造 Policy | 自构造的钉版 Policy 对象收割——全部派生量（im2col、weightT 转置、cell-attn tbias、GRU gih/ghh）由钉版 `__init__` 本人产生后冻结 | 与钉版同源，无二次推导 |
| tok_features dtype 提升 | f32/f64 拼接后整体转 f32 | 显式 f64 中间量镜像提升路径 | encode tok max diff 5.01e-06（§4） |
| GN/LN/softmax/conv | numpy 向量化 | 手写循环（GN 按组统计、LN 逐行、softmax 带 max 平移、conv 经 im2col 展开） | 归约顺序不同 → §6 的噪声级差分来源 |
| GRU | `gru(x,h)`：gi/gh 双投影后按 [r;z;n] 切片 | 同式逐元素转写（sigmoid/tanh 逐元素） | forced 全 12 槽归因含 GRU 状态轨迹，一致 |
| 低秩头组合 | `base + lin(mf,"V") @ lin(h,"U")` | 修复后同式 `base + mv @ hu`（曾误写 `(base+V)@U`，见 §4） | §2 全表 |
| 采样/hook | `sample_rng` 采样与 `hook` 覆盖存在于签名 | **排除在 spike 外**：采样调用不捕获、hook 在 evaluate_root 路径从不传（调用点 grep 已验） | 范围明示，非改写 |

## 8. GRU 状态与 forced 上下文的入参处理（明注）

- **GRU 状态**：钉版中 GRU 隐状态在单次 act 调用内由 `h = lin(ctx, "h0")` 创建、随 12 个解码槽在调用内演化，**从不跨调用持续**（记忆化探针已证非采样路径 act 是完整输入的严格函数、13.2 万次零反例）。转写按同一形态处理：状态创建与整个解码循环一并收入纯函数，无外部状态入参、无跨调用缓存。
- **forced 上下文**：forced 标签是显式入参数组＋`has_forced` 标志；函数内在 logits 算出后以 forced 标签替换 argmax 结果（与钉版 `forced is not None` 分支同语义），logits 本身仍按正解式算出并返回——故 forced 类调用同样能验 logits 差分（§2/§4 用法）。
- **采样上下文**：`sample_rng` 不作入参；采样调用（sample=True）不在捕获与对打范围内，其输出依赖不在输入内的 RNG 状态，不构成纯函数——spike 结论不覆盖采样路径。

## 9. 产物与证据

- 脚本：`scripts/act_numba.py`（njit 转写，含 §4 修复）、`scripts/bench.py`（对打/计时）、`scripts/bisect2.py`（本轮 bisect，含仪器化复刻与双候选式归因）、`scripts/capture_driver*.py`、`scripts/debug_bisect.py`、`scripts/quick_probe.py`。
- 数据：`out/captured.npz`（2,688 次调用输入＋参照）、`out/roots.jsonl`。
- 结果：`out/bench_results.json`（修复前）、`out/bench_results_fixed.json`（修复后）、`out/bench.log`／`out/bench_fixed.log`。
- 环境：`pylibs/`（numpy 2.2.6／numba 0.67.0／llvmlite 0.49.0／scipy 1.16.3，pip --target 落盘）。
- 收尾：本轮全部进程已退出、无残留（ps 直查确认）。
