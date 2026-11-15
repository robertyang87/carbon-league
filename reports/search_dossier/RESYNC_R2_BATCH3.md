# R2 第三批构建记录：内核再同步＋遥测穿包＋验包闭环准备

依据：REVIEW_SEARCH_IMPLEMENTATION_R2_20261116 §7（R2-09/R2-10）＋
RULING_R2_20261116 §5 第三批＋RULING_R2_ADDENDUM_20261116（R2-10 交付
条件预处置）。执行：搜索组构建线。日期：2026-11-16。

## 1. 单一权威内核（R2-09 内核层）

生产比较器模块改由**机械生成**：`make_comparator.py` 以三件冻结原型源
（仓内 code/search_proto/ 现行版，md5 钉版断言）拼装——

| 输入 | md5 |
|---|---|
| comparator.py（内核本体） | 53e5c1a1dd6ccc1a1f3343a47fc9b589 |
| telemetry.py（StatsCollector） | 91a7121f02bc848045a5dcae9554ec0e |
| leaf_adapters.py（叶适配器） | a5852c73f7187bb3e179428a81a6fba9 |

产物 `cnet2_comparator.py`：23,572 B，md5 **385190bd2f59d90f4e68a0d2dc6091a1**。
同步闸：同输入重生与产物**逐字节相等**（已验）。段 0 手写分叉
（ad3ec3e4）自此废弃，不再有第二份手写实现。

自证清单（生成器内对产物实测，全 PASS）：

- decide / decide_multi 双入口在位，单轮选优正确；
- 有限性闸：叶值 +Inf / NaN → 整轮作废、回 A0、abort=nonfinite_leaf；
- anytime：H1 完成、H2 轮内抛错 → H1 结果保留、abort=exception；
- C_hi 准入：est 过大 → budget_gate 且零 rollout 调用；est=None 冷启动
  豁免且轮次照常完成（豁免≠测得零）；
- deadline_abs：过期绝对 deadline → A0、无补时（abort=deadline）；
- 分阶段遥测：phase_cost_accum_s 按阶段累计、total_cost_accum_s
  累计整调用总时（原型字段名/记录键 total_cost_s、phase_costs；
  段 0 分叉的 phase_accum_s / "phases" 键随分叉一并废弃）；
- 适配器有限性：VNet 后端返回 +Inf → 原样返回 +Inf 并计数
  （非 tanh 压成 1.0）；启发式适配器 NaN 透传并计数。

生产扩展落位：decide/decide_multi 的绝对 deadline 入参（deadline_abs，
原型本体自带，段 0 分叉曾以另一约定替换——本次恢复原型口径）；
分阶段遥测（StatsCollector 本体自带）。

## 2. search 后继源重生（R2-09 对象层）

`make_search_cmp2.py` 以变换法自钉版基源（code md5 94f7fb64…）重生：
`cnet2_search_cmp2.py`，131,590 B，md5 **c3222eea76ea9d42c160626c2e239ca0**。
与基源 diff 共 16 个 hunk，逐项即源头注释所列 9 条变更，无其他改动。

R2-09 两处对象变化的修法（口径＝冻结 real_env f9bfacb7…）：

- **候选截断改先验 lg0 序**：候选 >4 时，非 A0 候选按其偏离槽的根
  先验 logits 降序（同分按生成序）取前三，keep=[0]+排序后索引
  （real_env._gen_candidates_raw 同式）。Codex 对照根：生成序
  [DOWN,NONE,UP,RIGHT] 与先验序 [DOWN,UP,RIGHT,LEFT] 在本口径下
  合到先验版（NONE 的先验最低被截，LEFT 入选）。
- **H≥2 推进补续弈 guard**：续弈每步双席各过本席 guard——我方席＝
  生产 guard 的 copy（承其真实历史）；对手席＝新建 guard，prevw 以
  对手**当前工人单位**（own_uid[:10]，不含 10/11 招募中心槽）播种
  （real_env 分源语义，R2-01 修复口径）。根步不重过 guard（根动作
  已定：A0 已在 _inner 过 guard、候选已过一致性筛）。改写计入
  stats["cmp_guard_rewrites"]。GUARD 关时续弈不过 guard（同 harness）。

叶值分发按终局种类（real_env._raw_leaf 同构）：灭绝→F01 嵌套叶
（prev_cash＝进入终局步的现金）；纯时间终局→F04 真实终局现金差
（嵌套启发叶不得在时间终局态上运行——段 0 的 tcash=None 迂回
口径废弃）；其余→嵌套启发叶。

C_hi 准入：样本按 (候选数, horizon) 分桶（每桶留最近 64 个，
样本＝decide 调用墙时/叶调用数），桶 max 为 est；冷形状回退全局
max；无样本→None（内核冷启动豁免）。

H 合同（源头注释与本节同文）：CMP_H=1 恰一轮 H=1（出厂形态；
decide_multi([1]) 即内核单轮）；CMP_H≥2 为 anytime 多轮
[1..CMP_H]，完成轮替换已提交结果、中止轮保留前轮。

## 3. 遥测穿包（R2-10）

- 公开 agent 暴露 cmp_stats：打包器模板新增一行转发
  `agent.cmp_stats = getattr(_AG, "cmp_stats", None)`（模板在
  cnet2_search_pack.py 内，非钉版量化器 cnet2_pack.py 119e9bb9…）；
  对照重建证明：旧模块＋新打包器重打包与现行包仅差此一行。
- stats["cmp"] 镜像：段内每次 record_decision 后把
  cmp_stats.snapshot() 写入包 stats 字典——该字典经现行模板既有
  `agent.last_stats = _AG.stats` 通道穿包，eval 侧读
  `agent.last_stats["cmp"]` 即得完成轮/中止/叶调用/分阶段全量；
  持构造后 agent 的消费者另可读 `agent.cmp_stats.snapshot()`
  （活通道）。eval 白名单接法：优先 last_stats["cmp"]（任何本包
  构建都有），或白名单放行 cmp_stats 属性读取。
- 错误计数合 *_errors 汇总约定：段 0 的 cmp_guard_err 改名
  **cmp_guard_errors**（凡按 *_errors 汇总的消费者自此计入）；
  续弈改写另计 cmp_guard_rewrites（非错误，不入 *_errors）。

## 4. 派生包重建

去重回归落地状态核查（switch/dedup/）：check_src_weights PASS、
verify 三件在档，但 A/B 链产出（cmp_h1/cmp_h2/cmp_ship）缺失、
regression_evidence.json 不存在、build_record_dedup 的 regression
字段仍为待填——**未全绿落地**。按任务口径先建**未去重版**（路线 B
在生产注册重建时并入），不阻塞本批。

构建链：打包器＝仓内 code 路线 cnet2_search_pack.py 谱系
（852483f2…＝仓内 5b00e76a…＋MODS 五件增量）＋模板转发一行，
md5 a1382d0ede6e7fac7e605231f214f13b；量化器 cnet2_pack.py 钉版
119e9bb9…未动；权重 ckpt20_q8_from_echo4.npz（3faf30a1…）双槽同源。

新包 `echo4_search_int8_resync.py`（本地交付件，注册归策略组）：

- 字节 9,041,707（≤10,000,000 ✓）；md5 **cba5923c13fd629208a4991811723980**；
  git blob sha1 **c932a0cfdd041d9dbb2cd1a17a55713d7ee29522**。
- 五源身份（包内 _SRC 逐件复算）：feat 2a45b08b… / np d8c15ad4… /
  fastsim aefe6c83… / comparator 385190bd… / search c3222eea…，全合。
- CONF 与现行包**逐字相同**（同一字面量构建）——SEARCH/CMP 相关
  差异为零项；包体差异仅：内嵌 comparator/search 两源替换＋
  模板转发一行（+9,571 B 净增，主因内核恢复 decide_multi/有限性
  等原型全文）。

## 5. 验包闭环（交付成立条件，本地暂存）

后继权威树暂存于本地 `resync/auth_tree/`（**未推仓内 code/ 权威位**，
注册动作归策略组在交付时同步立）：

| 暂存件 | md5 |
|---|---|
| code/cnet2_search.py（后继源入权威位） | c3222eea76ea9d42c160626c2e239ca0 |
| code/cnet2_comparator.py（第五件权威模块） | 385190bd2f59d90f4e68a0d2dc6091a1 |
| code/cnet2_feat.py / cnet2_np.py / fastsim.py | 2a45b08b… / d8c15ad4… / aefe6c83…（仓内现行） |
| code/cnet2_search_pack.py（构建打包器，MODS 五件＋模板转发） | a1382d0ede6e7fac7e605231f214f13b |
| code/packs/cnet2_search_pack.py（验包器，MODS 扩五件＋模板转发） | 453f543a609328bd387171800b4cac9c |
| code/packs/cnet2_pack.py（仓内现行，未改） | 18bf47b76a642aea2fd565ed9ed971f7 |

`--verify-pack` 对新包（packs 路线验包器，--auth-dir 指向暂存树）：
**PASS**（ok=true，五件 embedded==authoritative，mismatches 空）。
对照：现行包对暂存树 verify 为 FAIL，mismatch 恰为
cnet2_comparator＋cnet2_search 两件（继承关系落档）。

certify_pack 全闸复跑（本机口径，种子 986500/986501 各 1 局 vs
Master2 2aebfbc1…，certify_pack fc33be6c…＋eval_v53 45792dfb…）：

| 闸 | 新包 | 现行包（同口径对照） |
|---|---|---|
| init ≤20s | PASS（2.04 / 1.53 s） | PASS（1.40 / 1.29 s） |
| 整局 ≤1500s | PASS（831 / 822 s） | PASS（766 / 832 s） |
| 内部错误 =0 | PASS（0，两局 299 步完赛） | PASS（0） |
| 身份三方一致 | PASS | PASS |
| 子进程干净退出 | PASS | PASS |
| 步硬限 3s / 慢步 ≤3 / 超额池 ≤5s | FAIL（步均 2.21/2.17 s、p95≈2.97 s、max 4.27 s） | FAIL（步均 2.04/2.20 s、max 5.17 s） |

计时三闸两包同挂——本机步均约 2 s 是 VM 口径（部署口径基线步约
0.15 s、C1 证书步均 51 ms 级），非包间差异；新包在任何一项上不差于
对照。生产口径计时认证按注册在主体冻结时于考位（C1）以 S2 口径
复跑，不以本机数充数。峰值 RSS 两包均约 128 MiB。

## 6. 套件与行为自证（对内嵌模块本体）

从成品包解出内嵌五源，以导入路径适配（shim 重导出内嵌
cnet2_comparator，不改测试内容）＋冻结 harness（real_env f9bfacb7、
battery2 fc9d3557、pack_bridge 19871b42、smoke 2425f340、
verify3 17310cc1）运行：

- tests_s1.py **7/7 PASS**
- tests_s1_fix.py **15/15 PASS**
- tests_s1_fix_r2.py **10/10 PASS**
- tests_r2_fix.py **11/11 PASS**

行为探针（diag 变体与交付包同一装配链，CONF 仅注明覆盖项）：

**(a) CMP_H=1 A/B（种子 7000、包坐 0 席、基线对手）**
- BUDGET=2.5：两包动作流 0–13 步逐字一致；首分歧在 step 14——旧包
  该步 cmp_deadline 中止（completed 0 / leaf 0），新包完成并改判，
  属信封边界抖动类（本机单步成本约 2.2 s 与 2.5 s 信封同量级），
  非语义差。全场：旧 calls 271 / completed 232 / changed 113 /
  leaf 928，中止 {deadline 38, single_call_overrun 1}；新 calls 256 /
  completed 210 / changed 82 / leaf 840，中止 {deadline 32,
  budget_gate 14}——budget_gate 是分叉版所无的 C_hi 准入（注册口径
  新增，非行为回退）。
- BUDGET=30（两包全场零中止：旧 255/255 completed、新 263/263）：
  动作流仍 0–13 步一致；step 14 分歧经根重放实证为 **R2-09 注册
  修正类**：该根去重后候选 5 个（A0＋4 替代、同偏 worker-0 槽），
  分叉版生成序 keep=[0,1,2,3]（cand[3] 先验 −70.02 最低且被 guard
  否决，实评估 3＝旧 trace leaf_calls 3）；resync 先验序
  keep=[0,1,2,4]（弃最低先验、留最高先验 cand[4] DOWN −58.44，无
  否决，实评估 4＝新 trace leaf_calls 4），胜出即 cand[4]。其后
  分歧皆为该轨迹分叉的下游。结论：CMP_H=1 形态除两处注册修正
  （先验截断、C_hi 准入）外行为一致，**无未归类分歧**。

**(b) H≥2 续弈 guard（CMP_H=2 diag）**
- 整局（种子 7100，BUDGET=6.0）：帧计数 continuation=2（恰一个
  H=2 rollout 的 2 席×1 续弈步）、screen=729（243 调用×3 替代）、
  production=300（每步 1 次）；cmp_guard_rewrites=0、errors=0——
  续弈 guard 接线在位、双席分源（各自 guard 与历史）已实跑。
- 收获根（种子 7000 step-14 根，CMP_H=2 / BUDGET=30）：根调用
  guard Δ continuation=8（4 候选×2 席×1 步）、screen=3，depth 达 2
  完成，动作与同根 H1 结果一致（DOWN 候选）。
- **C_hi 本机尺度自锁（如实记录）**：BUDGET=6.0 整局首调用的 H2 轮
  被 deadline 杀死后，成本样本被污染（样本＝整调用时长/叶数，含
  被杀轮耗时），其后 242 次调用全 budget_gate（全场 completed 1）；
  BUDGET=3.5 整局同型（224 budget_gate、8 deadline）。该记样本口径
  与冻结 harness 同构（real_env 亦在每次调用后无条件记样本），属
  本机成本尺度（pre-cmp 约 2.2 s/步）产物；生产形态 CMP_H=1 在
  BUDGET=30 下 263 调用零闸零中止。部署口径下 est 低于信封数个
  量级，由 S2 与生产注册计时认证在考位复测（见 §7）。

**(c) anytime 保留（收获根，CMP_H 翻转法）**：agent 以 CMP_H=1 沿
记录轨迹热身（(K,1) 桶样本干净），仅根调用翻 CMP_H=2 并缩 BUDGET：
- BUDGET=12 / 16 / 20 三梯级根调用均为：completed Δ1、depth {1:1}、
  changed 1、leaf Δ5 / 6 / 7（轮 1 的 4＋轮 2 被杀前 1 / 2 / 3 个
  rollout）、abort {single_call_overrun 1}、continuation Δ4 / 6 / 8
  （轮 2 确已启动），返回动作恒为同根 H1 结果（DOWN 候选）——
  轮 2 中止后，轮 1 已完成结果被保留并生效。
- BUDGET=30 的诊断调用在本机抖动下耗时 31.0 s 超预算，同型保留
  （leaf 7、depth {1:1}、overrun、动作 DOWN）。
- 内核级 anytime 另由生成器自证 7/7 与内嵌模块套件用例覆盖；
  全 H2 完成形态见 §6b 收获根（leaf 8、depth {2:1}、零中止）。

**(d) 遥测穿包（BUDGET=2.5 diag 局，种子 7000）**：新包
pack.agent.cmp_stats 快照与内层 _AG.cmp_stats **逐键相等（True）**；
last_stats["cmp"].calls=256 与活通道 cmp_stats.calls=256 一致——
旧包同项为 None（R2-10 缺口已闭）。新包 stats 的 cmp 前缀键为
[cmp, cmp_deadline]（cmp_guard_errors / cmp_guard_rewrites 按包内
计数器惯例于首增时才出现，该局两项为 0）；快照键为原型口径
abort_reasons＋phase_cost_accum_s＋total_cost_accum_s（分叉版的
phase_accum_s 命名已随内核再同步消失）。

## 7. 未决项

- 去重路线 B 回归补齐后，生产注册重建时并入别名形态（本包未去重，
  9.04 MB；并入后约 4.64 MB 量级，另留 L-vnet33 的 Acc4 处置）。
- S2（11-17，C1）仍按作战室恢复令以现行包 fd45f35b… 诊断口径执行；
  本包待策略组注册后另行安排生产口径计时认证（主体冻结口径，
  种子 986500–986519），不在本批。
- 权威树暂存件的仓内注册（code/ 权威位＋验包 MODS）归策略组在
  交付验收时同步立。
- C_hi 样本在本机成本尺度下可被 deadline 杀死的轮污染而自锁
  （§6b）：口径与冻结 harness 同构、本批不改；部署口径影响由
  S2（11-17）与生产注册计时认证在考位实测确认，若考位复现再回
  策略组议口径。
- CMP_H=2 / BUDGET=20 整局探针两次被运行时重启吞掉（探针只在局
  末落盘）；H≥2 证据以 BUDGET=6.0 整局帧计数＋收获根确定性探针
  为准，不再补跑整局。
