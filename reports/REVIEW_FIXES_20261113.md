# Carbon League 第三轮复核：修复验收与新训练方案

> 后续复核：[第四轮报告](REVIEW_FIXES_20261113_R2.md)。原 C01–C03 已复测通过；新调用链与设计残留以该报告为准。下文保留原审查时点。

审查时间：2026-11-12 UTC / 2026-11-13 北京时间。源码基线：**14d53320b20d58bdb82676f7fa866b8e11a55062**。对比上一轮复核提交 a8eaff21；前两轮报告为 [REVIEW_20261112.md](REVIEW_20261112.md) 和 [REVIEW_FIXES_20261112.md](REVIEW_FIXES_20261112.md)。

## 1. 结论

**上一轮最关键的终局奖励、验证划分、缓存、采样和角色温度问题，原始复现现在均已通过。** 这轮确实有实质进展，不能把已经修好的问题继续列作未修。

但还不能写成“全部关闭、可以直接放量”：扩展测试发现了三个值得先处理的代码问题——**搜索会丢弃已算清的浅层终局结果、构建失败仍能让部分数据通过开训前检查、计时验收漏掉分类规则异常**。另有几项较小边界问题。

同时新增的 KataGo 架构与冲榜方案存在目标函数、价值分箱、实验基线和预算上的不一致，架构文件还被截断。**路线值得做受控原型，但现稿不宜直接作为大规模训练执行基线。** 本轮没有部署、开训、提交天梯策略或操作远端资源；文档中其他执行者的“已派工/待核”状态不等于本轮验证过实机。

## 2. 已通过的修复

| 上轮问题 | 本轮复测结果 | 判定边界 |
|---|---|---|
| F01 PPO 灭绝奖励 | 原 −310 被记成 +10 的例子已对齐为 −310；12 个双席位/record 开关/灭绝组合通过；dense、terminal、worth 与 SELF 镜像一致 | 构造状态与合法动作小例，未跑完整训练或统计自然发生率 |
| F01 搜索终局值 | 6 个 leaf 边界例通过；4 处搜索推进均保留相应终止前现金 | 数值语义已修好；新发现是终局结果被深度门丢弃，见 C01 |
| F04 eval 幸存者 reward | 8 个原版 env.run 对拍通过；原两步例现为 [−299,50]、分差 −349 | 包含 step299 灭绝、自然到时现金变化、存活方当步招募、双方灭绝；首次转移和树收入另由 leaf 微场景覆盖 |
| F02 split 继承 | 新 orig_id 仍留 VAL；缺失/空 orig_id 的旧 lb_cp 均非零退出且不创建输出；自博新局正常合并 | 强制把派生样本改 TRAIN 的执行步骤已删 |
| F03 OPEN_T 回归 | 同一真实 b128 探针现 errors=0，恢复两个基地 RECCOLLECTOR | 不代表最新线上权重的全配置认证 |
| F05 缓存 | 空窗口连跑两次均为空；index score/teacher/order 改动重建；坏 shard 删除且不认证，修复输入后可重试 | 失败输出是否允许进入训练是另一问题，见 C02 |
| F06 actor 角色 | 同路径 learner/SNAP 对象已隔离；快照 .5/1.0 温度给不同 logprob | 同路径、不同路径、混合早停/满局均检查 |
| F07 招募/死亡计数 | 原种植员例总账及 pla_rec/pla_dead/pla_dead0/rec60_p/e_prec25 均为 2 | 更细死因分类仍漏出生即死，见第 4 节 |
| F08 接手说明 | 根 README/HANDOFF 指向 CURRENT；仓外制品及 hash 待核如实声明；取消固定旧 eval hash 闸 | 外部制品仍待实机核验，是已声明待办 |
| F09 常规失败门 | bigeval 内部错误 exit3、外层错误 exit2，均存 FAIL=true；0/1 有效局也存失败；外层 first_error 按最低 seed 稳定选择 | 特殊种子长度和诊断字段尚有边界问题 |
| F09 ECS 原例 | 503、缺 Instance 列表结构、实例持续存在均非零；正常确认不存在才成功 | 仅 API mock，未操作云资源 |
| 第 5 节效率 | 原 600 条有效决策/1196 次 decode，现在恰好 600/600；SNAP 混合批次也正确 | 变长 PackedTraj、buffer、GAE 同时通过 |
| 第 5 节缓存签名 | 旧货量碰撞已失效重编码，误差 0；TTA 恢复误差仍为 0 | 当前编码输入契约未发现新增遗漏 |

修改过的 8 个 Python 文件语法检查通过，清单见 [syntax-and-commits.json](review_evidence/20261113_fixes/syntax-and-commits.json)。resume 四组用例也继续通过。

## 3. 应先处理的代码问题

### C01 — P2：全部候选已终局，搜索仍因深度不足退回必败默认动作

位置：[code/cnet2_search.py:1573](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/cnet2_search.py#L1573)，结合 sweep 中 all(term) 的提前终止与 MIN_DEPTH 门。

F01 的终局值已经正确，但 sweep 的 _search 在全部候选终局后停止推进，随后仍按 depth < MIN_DEPTH 将这次搜索判作 shallow，直接返回原动作，未比较已经确定的终局值。默认 MIN_DEPTH=3 会触发这个冲突。

定向探针使用实际 FastCarbon、Feat 和当前原样 _search，只把网络输出固定为确定动作以隔离控制流：

- 默认 UP 进入敌基地，使我方灭绝，终局 leaf=−220。
- 候选 NONE 截住敌最后一个 collector，使敌方灭绝，终局 leaf=+220。
- MIN_DEPTH=3 返回 UP、shallow=1；仅将 MIN_DEPTH=1 作诊断，则返回 NONE、gain=440。

这是构造局面的可执行反例，不是对正常对局出现频率的测量；本次未发现 mcts 路径存在同一个门。它也不否定已通过的 F01 数值修复。

**建议：** 全部候选已精确终局时，绕过最小启发式展开深度要求，直接比较终局值；对于因预算中断且仍有未解分支的情况继续保留深度保护。至少覆盖根步全终局、较深全终局、终局与非终局混合、预算中断四类。

证据：[simulator/sweep-terminal-depth.json](review_evidence/20261113_fixes/simulator/sweep-terminal-depth.json)、[simulator/sweep_terminal_depth.py](review_evidence/20261113_fixes/simulator/sweep_terminal_depth.py)。

### C02 — P2：坏回放导致部分数据集发布成功，自动链的开训前检查仍会放行

位置：[code/cnet2_build.py:375](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/cnet2_build.py#L375)、[code/cnet2_build.py:396](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/cnet2_build.py#L396)、[code/replica_chain.sh:82](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/replica_chain.sh#L82)。

builder 收集 errs 后只打印，仍生成 npy/meta 并正常退出；失败局以 n=0 保留在 meta.games，games/teacher_sides 汇总仍计入它。replica_chain 只检查构建返回码、set_uw 返回码以及 meta.games 数量是否等于 index 数量，随后启动两臂训练。

实测两局合成回放中破坏一局，实际调用 build_top.py 与 set_uw.py：

- 两者退出码均为 0；
- metadata_games=2、index_games=2，计数检查通过；
- game_rows=[0,3]，原 6 行数据被覆盖为 3 行部分数据。

没有执行 replica_chain 或启动训练；后续会放行开训的结论来自第 82–97 行调用链。坏 shard 已删、manifest 也未认证坏项，**这不是旧缓存复活问题重现**，是扩展失败测试找到的既存独立缺陷。

**建议：** 默认 errs 非空即非零退出，避免发布为完整正式数据集；若业务确实允许部分构建，需显式 allow-partial，保存失败原始局 ID、原因、有效局数，并让启动链按声明判定。合法零行与读取/处理失败必须区分。

证据：[data/data-round3-results.json](review_evidence/20261113_fixes/data/data-round3-results.json) 中 replica_prelaunch_gates；[data/recheck_data_round3.py](review_evidence/20261113_fixes/data/recheck_data_round3.py)。

### C03 — P2：计时验收漏掉 tg_errors 等分类规则错误

位置：[code/cnet2_search_timing.py:25](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/cnet2_search_timing.py#L25)、[code/cnet2_search_timing.py:37](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/cnet2_search_timing.py#L37)；对比 [code/eval_v53.py:522](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/eval_v53.py#L522) 的统一错误汇总。

timing 已能拦截 errors/root_errors，但规则层还会分别累计 tg_errors、pa_errors、px_errors 等。工具只保存和检查前两个计数，因此两个验收入口的“无错误”标准不一致。

当前 packer＋真实 b128，以 TG_MAX='invalid' 作受控故障注入：两步推理产生 tg_errors=2，timing 却输出 errors=0/root_errors=0、退出 0。同一包用 bigeval 则累计 a_errors_total=4、退出 3。

本次只在内存 AST 中把 timing 唯一的 range(299) 缩为 2，其余逻辑及真实包推理不变；用于验证错误门，**不作性能结论**。错误配置为测试注入，不表示线上包存在该配置。

**建议：** 与 eval 共用错误分类和汇总规则，覆盖 errors 及全部数值型 *_errors，并输出分类明细。合法包、根异常包、至少一个规则异常包都应成为门禁回归例。

证据：[runtime/observed/timing_short.jsonl](review_evidence/20261113_fixes/runtime/observed/timing_short.jsonl)、[runtime/observed/rule_invalid_bigeval.json](review_evidence/20261113_fixes/runtime/observed/rule_invalid_bigeval.json)。

## 4. 较小残留与诊断边界

这些问题不应掩盖第 2 节已经通过的修复：

- **P2，特殊配置：** [code/bigeval.py:62](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/code/bigeval.py#L62) 用 len(tasks) 算缺局，第 86 行却用参数 n 判失败。n=3、EVO_SEEDS 只有两个 seed 时，两局成功会存 FAIL=false、n=2，却退出 2。应统一期望局数，或开工前拒绝不一致参数。证据：[runtime/observed/extra_gates.jsonl](review_evidence/20261113_fixes/runtime/observed/extra_gates.jsonl)。
- **P3，诊断：** bigeval 在 0/1 有效局时先退出，精简摘要缺 first_error 和内部错误计数。失败门有效，但定位信息不完整。建议先统一生成健康信息，再决定是否计算均值/标准误。
- **P3，分类账本：** 原种植员同一步出生死亡例现 pla_dead=2，但 pd_clash_stay 仍为 1、死因分类合计也仅 1。出生即死已进入死亡总数，尚未进入细分死因。需补分类事件或明确单列 unknown，避免细分被误当成穷尽分类。证据：[simulator/results.json](review_evidence/20261113_fixes/simulator/results.json)。
- **P3，云工具畸形响应边界：** [tools/ecs_release.py:70](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/tools/ecs_release.py#L70) 检查 Instance 为 dict 列表，但未要求每项有有效 InstanceId。HTTP200 且 Instance=[{}] 或 [{InstanceId:null}] 仍立即报 RELEASED_VERIFIED/exit0。应把缺身份条目视为查询无效。仅纯 mock 证明 parser 边界，未声称真实云 API 出现此响应。证据：[ecs-extra-results.json](review_evidence/20261113_fixes/ecs-extra-results.json)。

## 5. 新训练方案：实现前必须厘清的设计问题

以下审查对象是新增设计文档和执行计划，**不是声称这些尚未实现的模块已经造成线上错误**。修复旧实现，也不能直接证明一个新算法设计成立。

### D01 — 文档不完整，搜索的信息边界与同时行动语义未定义

[reports/katago_architecture_design.md:321](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/katago_architecture_design.md#L321) 的真实仓库内容停在半句，下一行是字面量“...[truncated 4477 chars]”；不是本次读取工具截断。文内引用的 §8 不存在。应先从权威源恢复全文，不能由执行者猜测缺失决策。

第 129–138 行只写节点为 FastCarbon 状态、对手“同理生成臂”，尚未定义两方同时行动如何求值，以及部分可见状态如何进入搜索。实现前至少需规定：

1. 对手动作基于行动前自身信息生成，不能看见本步我方候选后再作为有先见信息的回应。
2. 自博搜索和部署使用一致的信息边界；全图可作监督真值，但不能无说明地成为部署不可用的策略/搜索输入。
3. 自己的候选如何共享对手动作样本、隐状态/belief 如何采样、备份的玩家视角是什么。

如果选择 minimax 或混合策略求解，应明确目标和信息集，不能把保守的最佳回应假设当成环境真实观察。

同时行动、部分可见游戏可以用价值叶搜索，但不能省略这些约束后直接照搬围棋交替节点。此处是缺少实现契约，未证明当前生产代码有新增信息泄漏。

### D02 — 候选内 softmax 的 CE 只约束比例，不保证候选动作整体概率增加

位置：[reports/katago_architecture_design.md:200](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/katago_architecture_design.md#L200)。

设计将联合动作 logp 在候选集合 A 内再次归一化，随后声称高访问动作的实际概率整体上升。反例：A={a,b}、目标各 50%，两个完整策略：

| 完整策略 (a,b,c) | P(A) | 候选内 CE |
|---|---:|---:|
| (0.45,0.45,0.10) | 0.90 | 0.693147 |
| (0.005,0.005,0.99) | 0.01 | 0.693147 |

目标完全相同，但候选总概率下降了 90 倍。条件 CE 可以用于排序；它并不提供文档宣称的质量迁移保证。

**建议：** 若目标是策略改进蒸馏，使用完整自回归策略的联合 log 概率，即 −Σ π̂(a)log πθ(a|s)；若坚持条件排序目标，应另行定义候选总质量约束或采样校正，并先通过上述最小反例。

### D03 — 83 分箱不足以唯一恢复真实 E[margin]

位置：[reports/katago_architecture_design.md:80](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/katago_architecture_design.md#L80)。

双侧溢出箱保留“落入尾部”的概率，但 −5000 与 −13320 都会落入左溢出箱。只有该箱概率、不知道尾部条件均值，无法唯一计算真实期望分差。若用一个固定中心近似，则实际优化的是截断或量化后的效用，应明确说明误差与目标。

**建议：** 增加尾部幅度回归、覆盖实际范围的非均匀分箱，或另设分差期望头。零附近若一个箱跨过零，也不能仅靠该箱质量准确恢复胜率。灭绝已经进入终局分差标签，第 134 行“灭绝头修正”还需定义公式，避免重复计入罚分。

原 tanh 描述也应订正：tanh(−4000/3000)=−0.87006，而 tanh(−13320/3000)=−0.99972，属于软压缩，不是完全不可分；PPO target clamp 是另一个硬裁剪问题，不能合并当作同一证据。

### D04 — 价值网提前“结算”的样本不能冒充终局真值

位置：[reports/katago_architecture_design.md:150](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/katago_architecture_design.md#L150)，对照第 204、235 行。

方案在 step250/280 用当前价值头提前结算、加入价值训练，同时要求价值头只学真实终局分差。Bootstrap 可以使用，但预测值与真实终局标签不是同一事实来源。

**建议：** 明确 truncated 与 terminal，指定目标网络、stop-gradient、bootstrap 损失和真实终局样本比例；未实际走到终局的样本不能直接充当终局归属图真值。搜索访问分布也不是“每步全信息真值”，应保留其估计性质。

### D05 — 旧基线、候选身份与“留出终验”口径仍不自洽

位置：[reports/live/plan_to_first.md:98](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/live/plan_to_first.md#L98)、[reports/katago_architecture_design.md:218](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/katago_architecture_design.md#L218)。

- Stage 0 说在 m3s2r1/snipe 中选初始化，却拿 K4 复刻的 −824 作裸网起点，再沿用 −330 门槛；第 54 行已声明旧规则锚点作废，后文仍用旧 +412 宣称“超过 m3s2r1”。数值可以作为事先目标，但不能未经重测就充当同候选、新规则的已知基线。
- 每代使用固定留出对手的结果决定晋升；文档未规定是否启用未读取的新种子块。这属于持续参与选模的验证协议，不能直接视为独立最终检验。
- 一次 Stage 0 原型失败，只能否定该实现、初始化和预算组合，不能证明整个 KataGo 式路线不成立。

**建议：** 新规则下冻结同一权重、对手、种子、计算预算，比较裸网/rollout 搜索/价值叶搜索的配对增量；记录重新测得的参考制品基线。反复使用的集明确称验证集，最终候选冻结后再用未读取的新数据块确认，或采用预注册的连续检验。+1 SE 可作运营门槛，不宜被描述成强统计证据。

### D06 — 搜索预算算术与吞吐预估不一致

位置：[reports/katago_architecture_design.md:159](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/katago_architecture_design.md#L159) 与第 291 行。

普通步 200 sims、25% 步 800 sims，均值至少为 0.75×200+0.25×800=**350 sims/步**；终局额外提高预算还会增加。吞吐节却按“有效约 60–80 sims/步”估算，二者不兼容。

**建议：** 先给出一致的 low/full/terminal 配置和实际平均访问数，再测局/分钟。不能按当前这两组不同预算直接锁 15–40 局/分或 GPU 日计划；批量化可能改变每次访问成本，但不会把访问次数本身从 350 变成 60–80。

### D07 — “BC 到天花板”“只有搜索能清算”仍缺因果证据

位置：[reports/live/plan_to_first.md:38](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/live/plan_to_first.md#L38)、[reports/katago_architecture_design.md:23](https://github.com/robertyang87/carbon-league/blob/14d53320b20d58bdb82676f7fa866b8e11a55062/reports/katago_architecture_design.md#L23)。

当前证据支持“现有复刻配方仍有明显强度缺口”，不支持从终局残树直接推出“模仿学不出来”。本游戏没有主动卖树动作；树龄、种植时点、对手和局势都会改变残树。新增文档又把这项推断写成已验证事实，与先前将残树降为诊断的修正不一致。

**建议：** 把网络容量、教师捕获、数据覆盖、价值估计与决策计算分别列成待验证解释；对手作者自述与行为推断分别标注证据来源。转向搜索可作为有预算上限的研究选择，不必依赖“BC 原理上不可能”的论断。

数学和文档截断证据统一见 [method/math-evidence.json](review_evidence/20261113_fixes/method/math-evidence.json)。这些是反例与契约检查，不是新架构训练后的效果测量。

## 6. 对训练路线的更新判断与下一步

本轮修复让现有 BC→PPO 管线的核心实现可信度明显提高。现在更合适的做法是：补 C01–C03 与统一失败契约，完成目标机器冒烟与同制品重新定标，然后进行有限预算、可归因的短训。原报告提出的 gamma/worth 目标一致性、价值裁剪分布、采样与部署差异、独立最终检验，仍需实验回答。

新方案可以保留：现有混合骨干、热启动优先、有限预算冷启动探针、学习真实终局分差、候选联合动作搜索、历史对手池、按原局 ID 划分、部署包同测、吞吐先实测再扩规模。**切换到搜索蒸馏不会自动消除目标函数与评测设计问题。**

建议按顺序验收：

1. 代码：终局搜索选择、坏数据非零退出、所有内部错误统一拦截。
2. 文档：补全架构原文，固定信息边界、目标函数、价值尾部、截断标签与预算。
3. 评测：重测同候选新规则基线，冻结开发验证与最终检验的使用规则。
4. 执行：记录源码 SHA、harness/权重/包 hash，再决定原型与训练预算。

本报告提出的是审查建议，没有替用户启动新的研究线或改变正在运行的任务。

## 7. 证据与复现限制

入口：[README.md](review_evidence/20261113_fixes/README.md)。包含 PPO、原版引擎对拍、搜索控制流、实际 builder/merge、当前 packer 与 b128 探针，以及纯 API mock。

脚本均应在独立环境与临时输出目录运行；不能把历史证据 JSON 改写成“新版通过”。Python 3.12、NumPy 1.26.4、PyTorch 2.5.1 CPU；路径整理后只做静态校验，未重复昂贵计时。部分探针以构造输入或内存 AST 缩短循环，报告已逐项标注。

未对最新线上完整制品做资源约束认证，未跑完整训练/胜率回归，未核验远端部署状态。上述已通过结论仅覆盖列出的复现与边界检查。

