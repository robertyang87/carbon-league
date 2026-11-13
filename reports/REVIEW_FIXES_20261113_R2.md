# Carbon League 第四轮复核：原问题关闭与新调用链检查

> 后续完整复核与本轮修复验收见 [第五轮完整 Review（2026-11-13）](REVIEW_FULL_20261113.md)。本文件保留第四轮当时的源码基线与历史结论。

时间：2026-11-12 UTC / 2026-11-13 北京时间。按用户指定的 [REVIEW_FIXES_20261113.md](REVIEW_FIXES_20261113.md) 逐项验收。

审查源码：**5463f504048bcca28868ebce0a063ac8d6c51b8e**；上一轮报告基线为 `14d53320b20d58bdb82676f7fa866b8e11a55062`。本轮另检查了新入仓的 `code/packs`、`code/cloud` 和打包运行手册。所有代码结论均有明确触发范围；没有启动训练、执行云端队列、部署或提交天梯策略。

## 1. 结论

**原 C01、C02、C03 的复现均已通过，可以关闭；种植员死亡分类、ECS 缺 InstanceId、bigeval 小样本诊断也已修正。** 不能继续把这些已修项列为未修。PPO 等 12 个核心文件与上轮通过版本的 Git blob 完全相同，本轮合并没有在这些实现中撤回旧修复。

但修好源文件，还不等于所有入口、产物与验收流程已经同步。本轮确认了三个代码/调用链问题：

1. 新指定的 packs 打包入口读取旧 search 副本；搜索/TTA 配置下，旧终局和缓存问题重新出现。**固定 13 键裸网配置短测正常，不据此认定当前 n1r/Echo3 已坏。**
2. 三个新增历史考试队列仍传 `EVO_SEEDS + N=0`，会被当前 bigeval 立即拒绝，却继续写完成标记。
3. 独立 builder 仍静默跳过 index 中缺失的 replay，发布 `partial=false`；**新的 replica 自动链会拦住这个例子**。

另有一项验收流程缺口：新 runbook 要求完整平台资源认证，但指定工具只覆盖进程内动作耗时和内部错误，不能单独证明全部资源条件。

训练设计有实质修订：D02 联合 CE、D04 截断标签、D06 主要预算算术已修。仍需改清 **Huber 与期望的区别、平台效用目标、最终留出对手的边界**。这些是提案的数学/契约问题，不能写成已测得的新模型训练失败。

完整证据和复跑说明：[第四轮证据入口](review_evidence/20261113_round4/README.md)。

## 2. 已关闭或保留有效的修复

| 原问题 | 本轮核对结果 | 结论边界 |
|---|---|---|
| C01 全终局候选被 MIN_DEPTH 丢弃 | 原例在 MIN_DEPTH=3 时选择 NONE，gain=440、shallow=0；新增根步/较深全终局、混合分支、预算中断、满足深度、自然到时共 7 例通过 | 权威 `code/cnet2_search.py`；packs 副本另见 N01 |
| F01/F04 搜索与评估终局 reward | 原版环境对照 8 个 eval 例、6 个 leaf 例通过 | 构造边界，不测自然发生率 |
| C02 坏回放仍可通过开训链 | 坏 gzip、合法 JSON 内处理异常默认 exit1；旧 meta/基础数组清理；显式 partial 有原 ID/原因且 gate 拒绝；修复输入后重试成功 | 实际运行 builder/set_uw；仅抽取启动 gate，未启动训练 |
| C02 零行/失败混淆 | 合法零窗口与全部坏局分开记账；缓存重试不复活旧 shard；缺失/旧 schema metadata 被拒绝 | 21 项预期检查包括已修例及新边界的观测断言，不代表 21 项都是健康输入 |
| C03 分类错误漏验 | 健康包 exit0；root_errors=2 与 tg_errors=2 分别 exit1，统一 a_errors=2 | 历史 b128、当前 packer、两步真实推理；不是性能认证 |
| bigeval 0/1 有效局诊断 | 均 FAIL=true、exit2，保留健康字段与适用的 first_error | 内部错误用单独计数/seed 字段，未声称具备完整内部 traceback |
| bigeval 请求局数冲突 | N=3、显式 2 seed 在开跑前 exit2，明确错误，不再写 FAIL=false 摘要 | 入口修复正确；旧调用方未迁移见 N02 |
| bigeval 外层首因顺序 | 完成顺序打乱后仍选最低 seed 的外层错误 | 不扩大为所有内部错误字段均按 seed 排序 |
| 出生即死种植员细分账 | pla_dead=2，clash_stay=1、unknown=1，分类和等于总数 | unknown 是明确兜底，不声称已精确识别所有死因 |
| ECS 畸形 Instance 条目 | 缺 ID、null ID 均非零且不报告 RELEASED_VERIFIED；原四项 mock 继续通过 | 6 个纯 mock，无云 API 调用 |
| PPO F01/F06、推理效率、resume | 12 个核心文件在上轮、设计提交、当前基线三点 blob 相同 | 代码恒等核对，未重跑整套长训练/上轮全部 PPO 探针 |

此外，**真实 n1r reference 已经入仓**。本轮直接 exec（无 `__file__`）两步，己方动作合法、错误为零；大小 **4,579,266 字节**、MD5 **ba7f84c3652ea2cd14a2243f2b1eace2** 和 13 键 CONF 与台账一致。此前“真实 n1r 包不在仓库”的限制不再适用；这仍不是完整平台资源或强度认证，Echo3 实际产物未在本轮复测。

## 3. 当前需要处理的问题

### N01 — P2，配置条件下：新发布入口嵌入旧搜索实现，未包含已验收修复

位置：[runbook:28](https://github.com/robertyang87/carbon-league/blob/5463f504048bcca28868ebce0a063ac8d6c51b8e/reports/live/pack_submit_runbook.md#L28)、[packs/cnet2_search_pack.py:135](https://github.com/robertyang87/carbon-league/blob/5463f504048bcca28868ebce0a063ac8d6c51b8e/code/packs/cnet2_search_pack.py#L135)、`code/packs/cnet2_search.py:49、504、1176、1447、1479、1488`。

runbook 指定 packs 下的 packer 为唯一发布入口；它按自己的 HERE 读取源代码，实际嵌入 packs 副本。六份核心副本中五份与 `code/` 相同，唯独 search 是旧版本：没有完整缓存签名/enc_for，没有终局冻结与 previous reward 处理，仍有两处 `left+1`，也没有 C01 的全终局深度例外。

两条独立动态证据：

- 相同 b128 权重分别由两个入口打包，生成包内源 hash 与各自来源一致。真实 TTA 后，consumer 取到的编码与原坐标编码误差：权威源 **0**，旧 `last_enc` 路径 **0.587379**。
- 相同真实 FastCarbon/Feat、固定网络输出的搜索边界例：主源码在全终局 step101 停止；packs 副本预算足时继续到105。自然结束从 step298 开始，主源码停299并正确选择，packs 副本走到300且错选 UP。

**范围：** 以上需要相应搜索/TTA 路径。当前标准 13 键 `SEARCH=False`、未开 TTA，两入口同权重两步动作相同且零错误。保留 n1r reference 的历史字节是合理的，不能改写参考包后继续引用旧 hash/考试成绩。本项不等于当前标准裸网已出现上述错误。

**建议：** 将历史复现工具明确冻结；新的通用生产入口依赖已验收源，或同步修复后重新验收。构建记录每个内嵌模块 hash，并检查源一致性/明确允许的差异，避免只更新 `code/` 就宣告所有 pack 同步修复。

证据：[产物来源与缓存](review_evidence/20261113_round4/runtime/pack-source-results.json)、[搜索边界](review_evidence/20261113_round4/simulator/search-boundaries.json)。

### N02 — P2：旧显式种子调用与新版 bigeval 不兼容，任务失败仍能写完成标记

位置：[q_h14d.sh:7](https://github.com/robertyang87/carbon-league/blob/5463f504048bcca28868ebce0a063ac8d6c51b8e/code/cloud/q_h14d.sh#L7)、`q_h14.sh:11、14、15`、`q_h15.sh:15–20`；入口为 `code/bigeval.py:22–24`。

新增历史脚本继续用 `EVO_SEEDS=<列表>`，命令行 N 却传 0。当前 bigeval 正确要求显式 seed 数与 N 一致，因此这些调用在 worker 创建前失败：

| 调用 | 实际 seed 数 | N | 本地真实入口结果 |
|---|---:|---:|---|
| q_h14 动态补局，构造全部缺局 | 100 | 0 | exit2 |
| q_h14d 两种营地场景 | 57 / 30 | 0 | 两者均 exit2 |
| q_h15 两种营地场景 | 57 / 30 | 0 | 两者均 exit2 |

两个营地队列各循环 3 个版本，原定 `3×(57+30)=261` 局会全部在参数阶段拒绝。q_h15 的独立 S50 常规考试不受这个问题影响。

同时，q_h14/q_h14d 的无参数 `wait` 不汇总子进程失败；独立 Bash 微例中子进程 exit2，随后 wait 返回0并可输出 CB_DONE。q_h15 只等进程消失，也不能区分成功和失败。故脚本能写完成标记，未完成预定考试。

**范围：** 已验证 HEAD 中入口与脚本契约不兼容，未执行任何 cloud 脚本，也没有证据证明当前云机正在运行这些历史队列。

**建议：** 保留严格局数门；固定列表传 57/30，动态补局冻结同一列表再传真实长度，空列表显式跳过。按 PID 检查每个子任务退出状态及结果健康信息，全成功后再写完成标记。

证据：[训练入口结果](review_evidence/20261113_round4/training/results.json)，其中包含完整受影响文件清单和真实拒绝消息。

### N03 — P2，独立 builder：缺失 replay 仍被静默排除，数据声明为非 partial

位置：[cnet2_build.py:253](https://github.com/robertyang87/carbon-league/blob/5463f504048bcca28868ebce0a063ac8d6c51b8e/code/cnet2_build.py#L253)，对应失败声明在 421、467–470 行。

对 index 中两局均符合 teacher/步数条件的 fixture，删除一局 gzip、保留 index，调用真实 build_top 与 set_uw：两者均 exit0，产物仅一局三行，但 `partial=false、failed_games=[]、valid_games=1`，缺失的原 ID 完全没有记录。

原因是 scanner 在构建任务前直接 continue，缺失文件根本不进入新 errs 契约。独立 builder 或只看其声明的消费者会把缺输入与主动过滤混淆。

**这不是原 C02 自动开训漏洞复现。** 当前 replica 链的 replay/index 前检和 valid_games/index 后检能阻止此例；抽取的后一道 gate 实测 exit1。原坏 gzip/处理异常的失败门仍然有效。

**建议：** 仅对原本应参与构建的 indexed replay，将缺失 ID 纳入失败；默认非零，显式 allow-partial 才发布部分数据。teacher/min_steps/重复源等合法过滤另行记账，不能机械地要求所有 index 行都必须成为训练行。

证据：[C02 与扩展边界](review_evidence/20261113_round4/data/c02-results.json)，键 `missing_replay_default / missing_replay_set_uw`。

### N04 — P2，验收流程：指定 timing 工具尚不能完成 runbook 的全部资源认证

位置：[runbook:35](https://github.com/robertyang87/carbon-league/blob/5463f504048bcca28868ebce0a063ac8d6c51b8e/reports/live/pack_submit_runbook.md#L35)、`code/cnet2_search_timing.py:11–17、37–39、60–63`。

手册要求 init、单步硬限、慢步次数/超额池、整局、峰值内存和 p95。指定工具只测 `agent()` 内部调用，未计顶层 exec/init；输出 p50/p90/p99/max，未完整输出 init/p95/slow_count/overage/game_wall/RSS。非零退出只由内部错误计数触发。

一个实际 sleep(3.05) 后返回合法空动作的桩，在仅将循环缩为一步、其他逻辑不改的工具中测到 max=3050ms、a_errors=0，仍 exit0。该探针只验证判定路径，不测模型性能。

**C03 修复有效。exit0 当前只表示内部错误为零，人工查看 max 可以拦下这个例子。** 没有证据证明执行者已经误判通过，或某个已上榜包超时；真正缺口是手册要求的完整测量与判定没有对应实现/补充命令。

**建议：** 明确当前工具是健康与函数耗时子测试；提供外层认证 runner，记录初始化、IPC、逐步额度、慢步池、整场任务和 cgroup peak，并汇总失败。1500 秒按平台整场比赛任务口径衡量，不能只测己方 agent 累计耗时。

证据：[资源门探针](review_evidence/20261113_round4/runtime/timing_resource_gate_probe.json)。

## 4. 训练设计复核：已修内容与剩余数学问题

审查对象仍是 `reports/katago_architecture_design.md` 与 `reports/live/plan_to_first.md`。它们与 CarbonFormer 是独立提案；骨干、是否上线搜索、预算和自博比例不同，本身不构成缺陷。共同需要一致的是计分、合法信息、标签事实和实验声明。

| 原条目 | 当前状态 | 复核结论 |
|---|---|---|
| D01 完整性/信息边界 | 部分修复 | 第8节已明确重构；根同时动作、共同样本、合法历史已补。深层搜索仍需区分物理状态与信息集 key，不能按不同隐藏粒子作先知选择 |
| D02 候选内归一 CE | 核心已修 | 269–279 行改为完整联合 logp；原反例两策略的 CE 现在为 0.798508 / 5.298317，能够惩罚候选总质量下降 |
| D03 价值尾部/期望 | 部分修复 | 承认溢出桶不能恢复均值，独立期望头与禁止重复灭绝罚分已补；新 Huber/胜率语义仍错，见下文 |
| D04 预测当终局真值 | 原问题已修 | terminal/truncated 分开，冻结 target、stop-gradient、真实终局至少70%、截断权重/辅助 mask 均写清；不能再指控其把 bootstrap 称真值 |
| D05 基线与最终验证 | 部分修复 | 旧数字降为历史，要求同制品配对重测；新种子冻结后一次使用已补。但最终留出对手又进入开发门控，见下文 |
| D06 预算冲突 | 原主要问题已修 | 20/200/400 配置的当前加权均值约70（精算69.8495），吞吐已标待测；不再是旧350对60–80混用 |
| D07 BC天花板论断 | 基本修复 | 区分观察、自述、推断，撤回原理性天花板结论；361行仍将行为剖面距离下限放进验收，应降回诊断 |

### M01 — P2，设计：Huber 不保证期望，均值符号也不代表胜率

位置：[架构设计:90](https://github.com/robertyang87/carbon-league/blob/5463f504048bcca28868ebce0a063ac8d6c51b8e/reports/katago_architecture_design.md#L90)、104–106 行；行动计划84行继承其定义。

新头用 Huber 回归，却称搜索叶 E[margin] 的唯一来源。取 δ=1000，分差以90%概率为0、10%为10000：真实均值为1000，Huber 最优点为111.111。MAE 也更偏爱111.111，不能靠“留出 MAE 门控”纠正均值语义。δ 未在稿中固定；该合法参数反例足以否定无条件的期望宣称，其他有限 δ 可同比缩放。

105–106 行又规定胜率估计冲突时以期望头符号为准。即便均值预测完美，90%输100、10%赢10000的均值为+910，胜率仍只有10%；均值符号不是胜率。

**建议：** 真正需要条件均值时使用适当缩放的未裁标签平方损失并检查均值校准；Huber 可保留为更名后的稳健辅助。胜率用 WDL 概率头或明确零点分箱，不以均值符号覆盖概率。336行还残留“超过3000不可分”的旧说法，应与前文 tanh 软压缩订正统一。

### M02 — P2，设计：最大化 E[margin] 仍未对齐公开天梯单局效用

位置：架构设计108、151–158、319–333行；行动计划84行。

公开正常局目标为 `U=0.5×WDL+0.5×sigmoid(M/1000)`，单方执行异常仅按 WDL。来源与读取 hash 见 [公开规则记录](designs/carbonformer_design_sources_20261113.json)。这套规则独立于选择哪种网络架构。

| 正常局结果分布 | E[M] | 胜率 | E[U] |
|---|---:|---:|---:|
| 始终赢100 | 100 | 100% | 0.762490 |
| 90%输100、10%赢10000 | 910 | 10% | 0.313757 |

完美 E[M] 也会给出与 U 相反的动作排序。所以仅把 Huber 改成 MSE、修对灭绝 reward，仍不能宣称冲榜目标完全对齐。

**建议：** 冲榜主价值直接估 E[U]，搜索/主要晋升指标共用该效用；原始分差作为辅助。若有意研究最大化分差，明确它是可能偏离天梯目标的实验。不能使用 `sigmoid(E[M]/1000)` 冒充非线性效用的期望。以上是数学反例，不是当前候选实测结果，更不保证提高单局 U 就一定提高未来整个梯队名次。

### M03 — P2，设计：同一“最终留出对手”又被放入反复选模

位置：[架构设计:258](https://github.com/robertyang87/carbon-league/blob/5463f504048bcca28868ebce0a063ac8d6c51b8e/reports/katago_architecture_design.md#L258)、299、301–303、313行。

258–259 行要求两个对手永久退出对手池与门控、只用于最终确认；299 行又明确把“留出对手”加入每代反复使用的验证集。示例 M6 r1 还与313行整体迁入的 m6r1 池冲突。新种子一次使用已修好，但不能因此把已参与选模的对手称未见对手。

最终确认200局也未给出通过的数值判据，不能在结果出现后再从均分、胜率、U、+1SE 中选有利口径。

**建议：** 固定训练/开发/最终确认的具体对手 ID、权重 hash、种子块和阈值。熟悉对手的新地图确认与未见对手泛化分别命名；不要让同一“留出”身份承担两种职责。

D01 的深层信息集问题是**实现前契约缺口**：隐藏 L/R 的观测完全相同，合法后续动作期望最多5，按隐藏粒子分别选动作却能虚算10。应规定信息历史相同的节点共享决策统计，或固定后续合法策略；没有实测新搜索代码泄漏。详细反例及其他设计边界见 [方法复核说明](review_evidence/20261113_round4/method/README.md) 和 [公式证据](review_evidence/20261113_round4/method/evidence.json)。

## 5. 较小边界与未作扩大解释的事项

- **数据事务性，P3：** 扫描异常默认非零但可留下旧输出；缓存 shard 缺字段时，合并先覆盖数组再失败，旧 meta 可仍在。当前 chain 先清空 DS 并检查 rc，未绕过；建议临时版本目录验证完再原子发布。
- **默认打包路径，P3：** packs 版 packer:106–108 沿用 `HERE/../data/weights`，移动后指向 `code/data/weights`。显式传权重可绕过，默认省略参数路径需修正。
- **n1r 来源链，P3：** 最终包 hash 已核；台账仍缺原始权重文件名/MD5，不能据量化后的包反推原文件身份。保留 reference，补证据或明确源身份未核。
- **AWR 数值边界：** 新 anchor 对合法小正权重 `[1e-6,1]` 生成有限 float32 单位权重；cached trainer 分别转 half 后出现 inf，真实 loss 子函数返回 NaN。本轮未核实际数据权重范围，不称当前训练事故。常规 anchor/m3s 权重转换通过；复用前检查范围并保持损失权重精度。
- **历史脚本使用边界：** `cloud/dgx_full.sh:5` 会用未版本化 `.d0dd809` 副本覆盖 trainer。仓库核心 blob 未回退的结论不覆盖执行该历史脚本之后的远端目录；本轮未执行它。
- **设计次要口径：** step200冲刺的“约1/5成本”不能由现有预算直接推出，按同一 sims 口径后缀约37.75%；`≥100` 是效果门槛，不等于统计显著性。大模型档还应补单文件10,000,000字节实际打包门，不能只比较权重内存与512MiB。

## 6. 下一步与复现边界

建议先统一生产打包源与模块 hash，再迁移显式 seed 调用和完成标记；补齐独立 builder 的缺文件声明与资源认证 runner。训练方案优先统一 U、期望损失与最终留出名单，再执行有限预算原型。**这些结论不否定网络优先、KataGo 式策略迭代路线；它们决定训练标签与验收结论是否可信。**

本轮使用 Python3.12、NumPy1.26.4、PyTorch2.5.1 CPU；72 个变更 Python/Shell 文件通过静态语法检查。动态测试以小型构造局面、真实 builder、真实包短推理、纯 API mock 和数学反例为主。相关探针会成功记录不正确行为，因此“脚本exit0”不自动表示被测对象通过。

没有做完整训练、全局胜率回归、1.5CPU/512MiB整场认证或远端部署核验。原始历史证据保持不变，新结果在独立目录。证据脚本运行应指定新的 `CARBON_REVIEW_OUT`，避免覆盖本轮快照。
