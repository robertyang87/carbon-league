# HANDOVER — iter1 变体包第一批（快速试错）

日期：2026-11-16。构建方：搜索组（本地构建，未推仓、未上考位、未外发）。
依据：所有者令「快速试错快速调整」；策略组微筛注册 REGISTRATION_MICROSCREEN_VARIANTS_20261116.md（验收口径含 TAU=0 等价硬验收）。
结构化记录：[build_record_iter1.json](sandbox://workspace/search-group/iter1/build_record_iter1.json)。

## 0. 一句话

只动了一处源码（比较器提交阈值 CMP_TAU）＋其 CONF 接线；三个变体包彼此只差 CONF/源开关，权重、量化、打包链与 resync 包逐字同源。对照重建先行验明：同链同输入复现 resync 包**逐字节一致**（md5 cba5923c…）。

## 1. 唯一源码改动

**内核（比较器）**：`Comparator` 加构造参数 `tau`（默认 0）。每轮完成结算后，若最优非 A0 候选的叶值 −A0 叶值 ≤ tau，提交 A0（completed_rounds／completed_comparisons 照计、changed=False）。decide 与 decide_multi 共用同一提交点（`_decide_horizons`），规则同行。tau=0 时走原代码路径（阈值判断以 `tau > 0` 为门，非 A0 赢家优势由并列规则保证严格为正，判断永不触发）。

- 变体原型源：`iter1/proto_src_var/comparator.py` md5 **e9eacadb**（冻结原型 53e5c1a1 的副本＋5 处 hunk：契约 docstring、API 行、`__init__`、提交点阈值、自测 case 4）；telemetry／leaf_adapters 为冻结源逐字未动。自测 ALL PASSED（含 case 4：优势 0.30 在 tau=0.5 被压、tau=0.2 放行、默认 0 不变、decide_multi 同规则）。
- 变体生成器：`iter1/make_comparator_tau.py` md5 **f9a82533**（make_comparator.py 的副本：变体 PIN＋变体头＋自证加 1 项）；产物模块 md5 **39000607**（24,574 字符），自证 8/8 PASS，再生逐字节一致（同步门验过）。**此模块不是权威内核 385190bd，不得作权威引用。**

**接线（search 源）**：`iter1/make_search_cmp3.py` md5 **588773e4** 生成 `cnet2_search_cmp3.py` md5 **b8c77aca**（132,494 字符）。照 CMP_H 通路：DEFAULTS 加 `CMP_TAU=0`；段内在 CMP_H 读数旁读 `_tau = float(cfg.get("CMP_TAU", 0))`，构造 `Comparator(..., tau=_tau)`；包 CONF 键经 `cfg = dict(DEFAULTS); cfg.update(CONF)` 覆盖，同 CMP_H 一路。与 cmp2 的全量 diff 恰为：头块标题、change-3 的 CMP_TAU 半句、change-10 头块、DEFAULTS 一行、_tau 读数 4 行、Comparator 构造一行——无其他行。

## 2. 三包身份（＋对照包）

| 包 | 文件 | md5 | 字节 | git blob sha1 |
|---|---|---|---|---|
| V1（TAU=150, H=1） | build_v1/echo4_search_int8_v1.py | 6f349672f264b50f2d4995fe12f729a4 | 9,045,071 | 93e2070c1fdca368ab98af107064c9270afc1cfa |
| V2（H=2） | build_v2/echo4_search_int8_v2.py | 99d4178870354f5c6c73b71f65ac3206 | 9,041,707 | 4ffed7794a612ac20706e472a8966de6b14591d0 |
| V3（TAU=150, H=2） | build_v3/echo4_search_int8_v3.py | fbad1e0e7f08d51ce9e3d1cf777453ac | 9,045,071 | 986fbd255667cdf22c1dcc708974439fa17eea64 |
| CTRL（TAU=0 对照，非变体） | build_ctrl/echo4_search_int8_ctrl_tau0.py | 21197b51ee9d0e1111882ccb73ed2728 | 9,045,055 | 02ae258cf1421b2a5399d6c55a3725f3f227fd23 |

参照 resync 包：cba5923c13fd629208a4991811723980，9,041,707 B，blob c932a0cfdd041d9dbb2cd1a17a55713d7ee29522。三包均 ≤10MB 平台上限。打包器 a1382d0e、量化器 119e9bb9、权重 ckpt20_q8（3faf30a1，双槽）与 resync 构建逐字同。

## 3. 与 cba5923c 的逐项差异

五源（feat 2a45b08b／np d8c15ad4／fastsim aefe6c83 三包全同 resync）：

| 包 | comparator | search | CONF 差异（对成品包 CONF 逐键实测） |
|---|---|---|---|
| CTRL | 385190bd → **39000607**（变体内核） | c3222eea → **b8c77aca**（cmp3） | 无（14 键逐字同；无 CMP_TAU 键 → cfg 缺省 0） |
| V1 | 同 CTRL | 同 CTRL | 仅 +CMP_TAU=150 |
| V2 | 385190bd（原样） | c3222eea（原样） | 仅 CMP_H 1→2 |
| V3 | 同 CTRL | 同 CTRL | CMP_H 1→2 且 +CMP_TAU=150 |

去重未折（按令）。权重/量化零改动。

## 4. verify-pack 记录（全 PASS）

校验器 453f543a（packs 版）`--verify-pack --auth-dir <树>/code`，exit 0、ok=true、mismatches []、missing_embedded []：

- CTRL／V1／V3 对变体树（tree_var：comparator=39000607、search=b8c77aca、余三源权威）→ `out/verify_ctrl.json`／`out/verify_v1.json`／`out/verify_v3.json`
- V2 对权威树（tree_auth）→ `out/verify_v2.json`

## 5. 四套件结果（tau=0 形态，全绿）

方法照批次 3：从成品 CTRL 包抽五源至 `tests_var/`（变体内核模块在 V1/V3 中逐字同），原型名经 shim 绑内嵌模块；冻结 harness 未动（real_env f9bfacb7／pack_bridge 19871b42／smoke 2425f340）。日志 `out/tests_*.log`。

- tests_s1.py **7/7 PASS**；tests_s1_fix.py **15/15 PASS**；tests_s1_fix_r2.py **10/10 PASS**；tests_r2_fix.py **11/11 PASS**（均 exit 0）。

## 6. TAU=0 等价证据（硬验收，PASS）

形态照批次 3 A/B：probe_runner，种子 7000、包坐 seat 0 对基线包（819250bf）、整局 300 步、诊断预算 BUDGET=30（仅诊断副本改 CONF，交付包 BUDGET=0.7 未动）。

- resync 侧：`resync/probe_new_big.json`（批次 3 已归档局，diag/new_diag_big.py）。
- 变体侧：CTRL 包诊断副本 `diag/diag_ctrl30.py` → `out/probe_ctrl30.json`。
- 比对（`compare_equivalence.py` → `out/equivalence_tau0.json`）：**动作流 300/300 步逐字一致（双座）；终局 cash 相同 [7571.62, 8113.83]；比较器遥测除墙钟累加器（cost_accum_s／phase_cost_accum_s／total_cost_accum_s）外全等（calls 263／completed 263／changed 90／leaf 1052／depth {1: 263}／aborts 空）；逐 step trace 299/299 条全等。判定 PASS。**

## 7. 诊断局对照数字（同种子 7000、诊断预算 30、本 VM 口径）

| 臂 | calls | completed_rounds | changed | leaf | depth_hist | aborts | 终局 cash |
|---|---|---|---|---|---|---|---|
| resync（TAU=0, H=1） | 263 | 263 | 90 | 1052 | {1: 263} | 无 | [7571.62, 8113.83] |
| CTRL（变体 TAU=0, H=1） | 263 | 263 | 90 | 1052 | {1: 263} | 无 | [7571.62, 8113.83] |
| V1（TAU=150, H=1） | 263 | 263 | **1** | 1052 | {1: 263} | 无 | [5153.66, 5226.65] |
| V2（H=2） | 259 | 517 | 94 | 2068 | {1: 1, **2: 258**} | 无 | [7935.67, 7887.44] |

- V1 changed **1 vs 对照 90（−98.9%）**，显著低于对照 ✓；leaf 与 completed 不变——阈值只压提交、不减评估。唯一放行改子在 step 192。V1 终局 cash 与对照不同系轨迹在改子提交/压制后分叉所致，单局诊断**不作强度读数**。
- V2 **H=2 完成轮证据**：depth_hist 含 2 且达 **258** 次（259 calls 中 258 次完成第二轮）✓；包面 cmp_stats 与内层快照相等（遥测过包界）。

## 8. 口径与边界（如实声明）

- 全部诊断均为本 VM 口径、单局、诊断预算 30——只证机制与等价，不证计时、更不证强度；强度由微筛注册的考场场次判。
- CTRL 是对照件不是变体件；三变体件未推仓、未上考位、未作任何注册引用。
- 变体内核（39000607）与 cmp3（b8c77aca）只在本批包内成立；权威集仍为 385190bd／c3222eea，后续迭代若转正须另走注册与生成器同步门。
- 无任何一项对不上；无被迫放宽的口径。
