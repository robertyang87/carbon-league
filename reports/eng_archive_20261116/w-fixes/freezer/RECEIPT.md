# W 轮考官冻结器修复回执（W03/W04/W09/W10/W11）

线：工程组 W 轮冻结器修复线。对象：`code/cnet2_train.py` 的
`_selection_snapshot_deps` 冻结器及相邻三处（`_selection_freeze_examiner`、
`_selection_play`、模块级注册表）。评审报告本体未同步到本 VM，以下按
所有者条目描述复现→修复；机制口径有出入处逐项注明。

## 基线与提交

- 基线（动手前 API 读回核对）：`code/cnet2_train.py` md5
  `a3ad86d4a74424134ecc650275a5161e`（118,712 B，blob `0f096cfc…`）。
- 提交 1（枚举器：W04＋W09＋W10）：md5 `6a9b210bf160b2609450726ca1c683c1`
  （138,473 B），commit `657ee7af87b2d0d62f24f6d399f4dd5b2cc0f8aa`。
- 提交 2（W03＋W11）：md5 `80613485c783529984d7adb159e8b5fe`
  （144,176 B），commit `e57e5343228cae1bc14ee53305da96251f84aa39`。
- 两段均为 sha 钉＋落盘后 API 读回逐字节验 md5（见文末落盘记录）。
- 纪律：只改 `code/cnet2_train.py`；选模统计口径与训练热循环未动；
  未触任何活进程。stage1→final 的 diff 经 SequenceMatcher 核对，严格
  限于 W03（注册表/文档段/审计块/守卫＋play 调用）与 W11
  （freeze_examiner）区域。

## 复现（基线 a3ad86d4，`probes/probe_w_repro.py` bug 模式 11/11 命中）

- W03：考官 `import helper`，`sys.modules["helper"]` 预置异字节模块——
  基线冻结照常成功、deps 无任何审计字段；`_selection_play` 无守卫，
  fork 子进程继承被污染的模块表（子进程探针实测读到 poison 值）。
- W04：依赖 dep.py 运行时 `open(join(dirname(__file__), "weights.json"))`
  与经绑定名 FN 的 `open(FN)`（cfg.json）——基线两者均不在冻结集
  （旧版仅考官本体做裸字面量正则扫描，依赖文件的数据打开完全不枚举）。
- W09：依赖以绝对字面量动态加载 leaf.py——基线把 leaf.py 拷进快照，
  但执行快照副本时仍读原路径：冻结后篡改原文件，VALUE 变为
  "tampered"（复制形同虚设，实测坐实）。考官本体的根级绝对加载同理
  只复制不重定向。
- W10：`def _load(SRC)` 形参遮蔽模块全局 `SRC`（绑定 dep.py）——基线
  把 dep.py 误冻（按全局值解析）；局部赋值 `SRC = os.getcwd()` 遮蔽
  同样误冻。正例对照（未遮蔽全局引用、局部绑定优先、`global SRC`
  声明）基线行为正确，修复后保持。
- W11：考官 policy_demo.py（make_agent 型）——基线快照改名
  examiner_snapshot.py，eval_v53 加载约定按 basename 分发
  （`policy_*` → make_agent()，其余 → 模块级 agent），改名后实际取到
  模块级 agent ("module", 0) 而非 make_agent 产物 ("made", 42)。

## 修复

### W10（作用域感知静态求值，提交 1）
- 绑定收集改为作用域感知：函数形参、局部赋值/导入/for/with/except-as
  目标构成遮蔽集（`global`/`nonlocal` 声明除外）；被遮蔽名在作用域内
  记为 `("shadow",)` 哨兵——只有该作用域内单一路径赋值的真绑定可以
  覆盖哨兵，否则该名一律不可归约（动态加载 die 点名、数据文件软跳过）。
- 作用域遍历统一为 `walk_scoped`：函数默认值/装饰器按定义处（外层）
  作用域求值，lambda 形参同样遮蔽。
- 探针：形参遮蔽、局部赋值遮蔽 → 均 die；未遮蔽全局、局部绑定优先、
  `global` 声明三正例 → 均正常冻结（fixed 模式全过）。

### W04（依赖数据文件递归冻结，提交 1）
- 每个被冻 .py（考官本体＋每个依赖）在冻结时做数据打开扫描：
  `open`/`io.open`（排除 os.open 签名）、`np.load`/`loadtxt`/`genfromtxt`、
  pandas `read_csv/read_json/read_pickle`、`Path.read_text/read_bytes`；
  另加旧版裸字面量正则扫描下沉到每个文件。写模式（含 w/a/x/+）跳过。
- 可归约且存在、同目录（考官目录内）的数据文件一并冻结，deps 条目
  新增 `kind` 字段区分 `code`/`data`；`.py` 目标按代码全递归。
  不可归约/不存在/界外者软跳过（不 die）——与代码加载的硬口径刻意
  不同：数据打开在运行期大量存在条件分支，硬 die 会误杀正常考官；
  此点与条目描述「不可归约维持 die」有出入，理由在此，待验收裁定。
- 探针：dep 的 weights.json（直拼字面量）与 cfg.json（绑定名）均入
  冻结集（kind=data、md5 对）、快照内可读回；写模式目标不冻。

### W09（绝对路径加载重定向或 die，提交 1）
- 路径锚点判定结构感知：拼接看左操作数、join 看最后的绝对/文件锚定
  段、dirname/abspath 等包装透传——`dirname(__file__) + "/dep.py"`
  这类 __file__ 锚定式不误判（26 矩阵 add_concat 保住）。
- 依赖内绝对字面量加载：把快照副本中的该字面量按 AST 字节跨度改写
  为 `join(dirname(abspath(__file__)), <相对路径>)`，运行期实际读冻结
  副本；改写逐跨度校验（literal 回读相等＋改写后重解析），dep 条目
  记 `path_rewrites: [{from, to}]` 与 `snapshot_md5`（md5 字段仍记原
  字节）。绝对目标在考官目录外 → die。
- 考官本体的绝对加载：快照字节即身份、不可改写 → die 点名绝对路径。
  非字面量形态（计算式、oneof 候选含绝对项）一律 die 点名，不只复制。
- 验收探针：冻结后篡改原 leaf.py，快照执行结果仍为 "original"。

### W03（sys.modules 替代，提交 2）
- 冻结侧审计：plain import 冻结的依赖在条目记 `import_name`；冻结
  结束时逐名审计 sys.modules——未绑定记 `"absent"`、同字节记
  `"same-bytes"`、异字节当场 SystemExit 点名（模块名＋现存 __file__
  ＋冻结 md5）。全部 import 名登记进模块级注册表
  `_SELECTION_FROZEN_IMPORTS`（name → (md5, 快照路径)）。
- worker 侧守卫：`_selection_play` 先行
  `_selection_enforce_frozen_imports()`——已绑定且 __file__ 为快照
  路径或同字节者放行；异字节 RuntimeError 点名；未绑定者以快照路径
  spec 按原名加载并校验 __file__ 落在快照副本上，失败点名拒绝开局。
- 探针：冻结期污染 → die 点名 helper＋poison 路径；冻结后子进程内
  污染 → 守卫 raise 点名；干净子进程 → 守卫预加载冻结副本，
  `helper.VALUE == "frozen"`。

### W11（快照保留原名，提交 2）
- 仓内约定定位：`code/eval_v53.py` 的 `load_agent_fn`——
  `basename.startswith("policy_")` → `make_agent()`，否则模块级
  `agent`（另有 cnet2_search 系同约定）。
- `_selection_freeze_examiner` 改为先算 deps、快照按考官**原始
  basename** 落盘；仅当某个 dep 的 arc 与该 basename 撞名（dep 反向
  import 考官本名的病态情形）才回退历史名 examiner_snapshot+ext，
  避免互相覆盖。`sel["public"]["examiner_snapshot"]` 记实际文件名。
- 探针：policy_demo.py 快照保名，加载约定实测返回 make_agent 产物。

## 回归（修复版，证据在 `evidence/`）

- 26 项冻结矩阵：26/26（`matrix_fixed.json`）。**一项期望按 W09 新规
  修订**：`absolute_literal`（考官本体绝对字面量加载）由 V06 期望
  「冻结」改为「die 点名绝对路径」——旧行为正是 W09 的「只复制不
  重定向」，修订理由即 W09 本身；矩阵脚本副本中已注明。
- W 探针 fixed 模式：11/11（`w_repro_fixed.json`）；基线 bug 模式
  11/11（`w_repro_baseline.json`）。
- V06 数值不变性：probe_v06 mean_u 两轮均 `0.20390815759383646`
  （与 V06 轮记录逐位同）、probe_v06b 均 `0.5`，snapshot_md5 均
  `84face63ef3c617fe8a3fa682e058a79`。注：examiner_exec 现为
  `.../examiner.py`（W11 保原名），字节不变故 md5 不变。
- probe_v06_deps：deps 文件集与 md5 与 V06 轮证据全等（条目新增
  kind/import_name/sys_modules 字段）；dyn 自加载案例仍 SystemExit，
  理由由旧「不可归约」 refine 为「dynamic-load cycle (dyn.py ->
  dyn.py)」——新解析器可归约 `__file__` 自引用后由环检点名，结论
  （die）不变。
- V03 探针 5 场景与 V03 轮 fixed 证据深相等；V05 两场景（MISSING
  SystemExit 点名 / SWAP tripwire）与 V05 轮证据一致。
- prove_ties：ALL_PASS；run_e2e：ALL_PASS（前任 wave2 日志
  `run_e2e_fixed.log`，rc=0）；test_review_fixes：9/9 OK。
- 真考官证明：见下节（兼作追加交付项）。

## 追加交付项：新枚举器对真考官复导快照集（策略组裁定）

对象：`~/workspace/a4-examiner-proof/` 的真考官 opp_master2r.py＋
master2_src.py（本机持有的是考官本体两件；Studio 那趟 m7bsr1 选模
运行记录里的 deps 清单由策略组持另一侧对照，本节只对本机两件负责）。

- 考官身份：opp_master2r.py md5 `2aebfbc1008d3475ece8e178696e292d`
  ——新版 config 记录值与真身逐字一致（identity_ok=true）；快照按
  W11 保原名 `opp_master2r.py` 落盘为 examiner_exec。
- 新枚举器冻结集（逐项）：仅一项
  `{file: master2_src.py, md5: 404114a0b5e422516322f62e9ddb6870,
  kind: code}`；快照副本字节复核与真身一致。
- 与旧枚举器集合 {master2_src.py: 404114a0…} 比对：**差集为空**
  ——新枚举器在真考官上多冻出的实项为零，冻结集与旧集逐项全等
  （条目结构新增 kind 字段）；未多冻出任何 code/data 实项。
- 短选模（fixture pairs=2、4 workers）：ok=true、4/4 局有效、
  0 错误、mean_u ≈ 5.44e-07（随机候选对 Master2 真身，≈0 合
  预期）、wall 1107.9 s（本 VM 高负载口径；收尾线复跑
  `evidence/real_examiner/rerun_proof.json`）。
- 证明脚本副本 `probes/proof_real_examiner.py`：dep 记录断言按
  file＋md5＋kind 查（W04 后条目带 kind 等字段，不再全字典相等），
  其余与 V06 轮原脚本同。

## 落盘记录（收尾线执行，2026-11-14）

- 提交 1：commit `657ee7af87b2d0d62f24f6d399f4dd5b2cc0f8aa`，
  sha 钉基线 blob `0f096cfc8150eccf618a7a766144e71b120a571f`
  （a3ad86d4），提交后 API 读回 md5
  `6a9b210bf160b2609450726ca1c683c1`，与 stage1 文件逐字节 MATCH
  （blob `b5f439ae4b7706962639cfac16b86f3581a1d6a4`）。
- 提交 2：commit `e57e5343228cae1bc14ee53305da96251f84aa39`，
  sha 钉提交 1 的 blob `b5f439ae…`，提交后 API 读回 md5
  `80613485c783529984d7adb159e8b5fe`（144,176 B），与 fixed 文件
  逐字节 MATCH（blob `316c9299ca22064316ef4c1bca92701bb4aa0bb4`）。
- main 现行 `code/cnet2_train.py` 即提交 2 版（80613485…）。

## 收尾复核（收尾线亲自复跑，不采信前任日志）

- diff 审计：base→stage1 共 11 个 hunk，old 行 1230–1646 严格限于
  `_selection_snapshot_deps` 本体（下一 def 在 1647）；stage1→fixed
  共 5 个 hunk——模块级注册表 `_SELECTION_FROZEN_IMPORTS`、
  `_selection_snapshot_deps` 的 W03 文档段＋尾部审计块、
  `_selection_freeze_examiner`（W11）、新函数
  `_selection_enforce_frozen_imports`＋`_selection_play` 首行守卫
  调用。选模统计函数（_selection_t_crit/_cluster_stats/_paired_se）
  与训练热循环（main 及以下）两段皆未动。
- W 探针：基线 bug 模式 11/11 命中（与前任基线证据逐字段一致，
  `evidence/rerun_w_bug.json`）；fixed 模式 11/11 通过
  （`evidence/rerun_w_fixed.json`）——W03 冻结期 die 点名、子进程
  守卫 raise 点名 poison 路径；W09 篡改后快照仍读 "original"、
  本体绝对加载 die 点名路径；W10 两遮蔽 die、三正例冻结；
  W11 实测返回 make_agent 产物 ("made", 42)。
- 26 项冻结矩阵（fixed）：26/26，逐项 outcome 与前任跑次全等
  （`evidence/rerun_matrix_fixed.json`）。
- V06 数值不变性（fixed）：mean_u 两轮均
  `0.20390815759383646`（换文件后不变）、snapshot_md5
  `84face63ef3c617fe8a3fa682e058a79`
  （`evidence/rerun_probe_v06.log`）；examiner_exec 为
  `.../examiner.py`（W11 保原名），字节不变故 md5 不变。
- prove_ties：ALL_PASS（duel 数值与 A4 基线记录一致，
  `evidence/rerun_prove_ties.log`；回归树内的 cnet2_train 副本
  md5 经核为 80613485…，确系对 fixed 版跑）；test_review_fixes：
  9/9 OK（`evidence/rerun_test_review_fixes.log`）。
- 真考官复导：见上节（收尾线复跑，差集为空）。

## 口径旗标（原样保留，待策略组验收裁定）

- **(a) W04 数据文件不可归约采「软跳过」而非 die**：与原简报
  「不可归约维持 die」口径有出入——数据打开在运行期大量存在
  条件分支，硬 die 会误杀正常考官；不可归约/不存在/界外的数据
  文件软跳过（不 die），代码加载仍维持硬 die。此点待验收裁定。
- **(b) W09 使矩阵 absolute_literal 期望由「冻结」改为「die
  点名绝对路径」**：考官本体的绝对字面量加载在旧版只复制不重
  定向（旧行为即 W09 本身），新版快照字节即身份、不可改写，
  故 die 点名；矩阵脚本副本中已注明修订理由即 W09 本身。
