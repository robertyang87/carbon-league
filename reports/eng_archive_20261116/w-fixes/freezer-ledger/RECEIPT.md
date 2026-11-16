# W04 补账线回执（策略组验收绑定条件：skip 账本）

线：工程组 W04 补账线。对象：`code/cnet2_train.py` 冻结器 W04 数据打开扫描。
基线（动手前 API 读回核对）：md5 `80613485c783529984d7adb159e8b5fe`
（144,176 B，blob `316c9299…`）。

## 落仓

- 提交：`06b5ae81c99761ee6775dc60fdab16a40309cbcb`（单文件、sha 钉基线
  blob 316c9299，落仓前复读 main 确认未漂移）。
- 新 md5：**`5310439c845b388da670131569a29453`**（148,544 B），落仓后
  API 读回逐字节一致；落仓字节复跑账本探针仍 ALL_PASS。
- 纪律：只改 `code/cnet2_train.py`；冻结决策与选模统计/训练热循环未动
  （soft_target 的冻结判定逐分支与旧版等价，仅附加命名与记账）；
  未触任何活进程。

## 第 1 步盘点：现行软跳过分支逐个核对（基线 80613485）

数据扫描 `scan_data_files`/`soft_target`（本体＋每个依赖递归调用，
两层同路）。**基线全部跳过均不落账**——`soft_target` 只返回路径或
None，调用点只收路径，无任何记录结构：

| # | 分支 | 旧行为 | 本版处置 |
|---|------|--------|----------|
| 1 | 求值结果 None（未绑定/运行时计算） | 静默 None | 记账 `irreducible` |
| 2 | SHADOW（W10 函数内遮蔽名） | 静默 None | 记账 `irreducible` |
| 3 | 裸相对字面量 `("lit", 非绝对)` | 静默 None | 记账 `irreducible` |
| 4 | `("dir",)/("pdir",)` 目录形态作打开目标 | 静默 None | 记账 `irreducible`（不归约到文件路径） |
| 5 | oneof 候选含不可归约候选 | 静默 None | 记账 `irreducible` |
| 6 | 归约路径在考官目录内、无常规文件（不存在或被非文件占据） | 静默 None | 记账 `missing` |
| 7 | 归约路径在目录外、无常规文件 | 静默 None | 记账 `missing`（见口径注记） |
| 8 | 归约路径在目录外、常规文件存在 | 静默 None | 记账 `out-of-dir-existing` |
| 9 | oneof 全部候选不存在 | 静默 None | 记账 `missing` |
| 10 | oneof 前序候选不存在、后序候选存在 | 由存在候选决定（运行时同序） | 非跳过事件，仅终局结果记账/冻结 |
| 11 | 打开目标即被扫描文件自身（open(__file__)、Path(__file__).read_text） | 静默 None | **不记账**：该文件字节已冻结，无输入逃逸（显式分类为非跳过） |
| 12 | 写模式 open（on_call 求值前返回） | 静默跳过 | **不记账**：输出非输入，求值前分类 |
| 13 | 未被识别为读打开的调用形态（pnode 取不到） | 静默返回 | **不记账**：从未求值任何路径表达式，无原文可记 |
| 14 | `np.load(io.BytesIO(...))` 等缓冲对象作「路径」 | 静默 None | **不记账**（本版新增求值前分类，见「机制细化」） |
| 15 | freeze_data 旧式裸字面量正则扫描未命中 | 静默不冻 | **不记账**：无表达式，仅字符串提及，非打开 |
| 16 | scan_data_files 遇 SyntaxError 整文件返回 [] | 静默 | **不记账**：非表达式级；文件本身另有处理路径 |

## 实现

- `_selection_snapshot_deps(..., skips_out=None)` 新增尾参（缺省 None
  时返回契约逐字不变，既有探针/调用零改动）；冻结器内部
  `soft_target` 改为三态返回 (路径, None)／(None, 原因类)／
  (None, None=自身)，`on_call` 在求值后未冻结且原因类非空时记一条。
- 账本条目恰三键：`{"file", "expr", "reason"}`——file 为被扫描文件
  相对考官目录路径（与 deps 条目 "file" 同口径，本体即考官文件名、
  依赖层即依赖相对路径，两层同账）；expr 为路径表达式的源码原文
  （`ast.get_source_segment`，回退 `ast.unparse`）；reason ∈
  irreducible|missing|out-of-dir-existing。每调用点一条，函数返回时
  按 (file, expr, reason) 排序后交出，记录确定性。
- 落点（不新造平行文件，挂既有选模记录写出点）：
  `_selection_freeze_examiner` 恒传 skips_out 并写
  `sel["examiner_data_skips"]` 与 `sel["public"]["examiner_data_skips"]`
  （与 examiner_deps 同型）；`_selection_finish` 的胜者导出 meta 增
  `selection_examiner_data_skips`（与 `selection_examiner_deps` 并列——
  预注册核对的就是这份记录）。**空账显式**：无跳过时为 `[]`、键恒在
  （非 .py 考官同样写空账）。
- 口径注记（简报要求注明）：基线代码把目录外路径与不可归约混在同一
  个 None 里，本无分类。本版分类按条件原文钉死：`out-of-dir-existing`
  ⟺ 路径可归约 ∧ 常规文件存在 ∧ 在考官目录外；目录外但无文件者归
  `missing`（missing 定义＝归约路径处无常规文件，不论路径内外、不论
  是全然不存在还是被目录等非文件占据），保住预注册条目的纯度。

## 机制细化一处（请验收留意，与简报字面有一处出入）

首版实现按字面分类时，真考官复导账本非空：唯一一条是
`master2_src.py` 的 `np.load(io.BytesIO(base64.b64decode(b64)))`
（expr 原文 `io.BytesIO(base64.b64decode(b64))`，初判 irreducible）。
实查：这是把内嵌 base64 权重解到内存缓冲再 np.load——其「路径」
根本不是路径，所读字节就在被冻结的 master2_src.py 源码里随文件
同行，没有任何输入逃出冻结，与「打开自身」同类。若按 irreducible
入账，真考官账本恒非空、且条目误导（会让人以为有未冻输入）。
处置：在 on_call 求值前加一个窄分类——路径参数**字面上是**
`BytesIO(...)` 调用（io.BytesIO 或裸 BytesIO）时判为非路径打开、
不记账，理由写入代码注释与 `_selection_snapshot_deps` docstring
（三类求值前边界之一）。收窄边界如实声明：经名字绑定的缓冲对象
（`buf = io.BytesIO(...); np.load(buf)`）静态不可辨，仍按
irreducible 入账（保守方向）。此处置使真考官账本为空账，与任务书
预期及策略组 V06 期「W04 零增项」记录一致；如验收认为应改回字面
入账，是一处小改。

## 探针（probes/probe_ledger.py，证据 evidence/probe_ledger_new.json）

构造考官（exam.py 本体＋dep.py 依赖层）三类各一例＋双控制，结果
账本四条齐、类与表达式原文逐字对：

```json
{"file": "dep.py",
 "expr": "os.path.join(os.path.dirname(__file__), \"dep_absent.json\")",
 "reason": "missing"}
{"file": "exam.py",
 "expr": "\"/tmp/ledger_…/outside/secret.json\"",
 "reason": "out-of-dir-existing"}
{"file": "exam.py",
 "expr": "os.path.join(os.path.dirname(__file__), \"absent.json\")",
 "reason": "missing"}
{"file": "exam.py", "expr": "p", "reason": "irreducible"}
```

控制项：既存目录内 weights.json 正常冻为 kind="data"（不入账）；
写模式 open 的 out.txt 不记账、不冻结；sel 级与 public 级账本同值；
skips_out 直调与记录路径同账；缺省调用返回仍为纯 deps 列表。

真考官复导（同探针 B 部＋proof_real_examiner.py）：身份
`2aebfbc1…` 不变；冻结集不变，恰
`{master2_src.py, 404114a0b5e422516322f62e9ddb6870, kind="code"}`；
账本为**显式空账**（public 键在位、值 []）；证明脚本短选模 4/4 局
有效、errors 0。

## 回归（证据在 evidence/，均以新版字节跑）

- 26 项冻结矩阵：**26/26**（matrix_new.json，failures 空）。
- W 探针 fixed 模式：**11/11**（w_fixed_new.json）。
- V06 不变性：mean_u 两轮均 `0.20390815759383646`、snapshot_md5
  `84face63ef3c617fe8a3fa682e058a79`（probe_v06_new.log），与要求值
  逐字同。
- prove_ties：**ALL_PASS**（prove_ties_new.log；regress 两棵树的
  cnet2_train 副本已同步新版字节）。
- test_review_fixes：**9/9 OK**（test_review_fixes_new.log）。
- 落仓字节复跑账本探针：ALL_PASS（probe_ledger_landed.json）。
