# S1 教师接线与收益闸独立复核（2026-11-13）

本轮结论：**新 U 叶、guard 抽取、双席位快照修正已有独立验证支持；S1 整体尚不能关闭，也不能把当前驱动的 PASS 当成整份软 q 已获真实收益验证。** 发现 3 项 P1、6 项 P2。应先修复教师终局路由、认证对象和统计/输入门，再运行正式收益实验及规模标注。

审查区间为 `bc1a84d4..a5c63648`，实际执行源码冻结于 **`a5c63648e037e342b154949c1b4eea6d11eef139`**（13 个提交）。这是 S1 专项，不替代上一轮全库报告，也不追认其他修复线全部关闭。本文源码行号均指该 SHA。提交前另核对 main `5aa7b544`，所审 19 个源码/规范/验收文件与冻结版逐字相同；新增归档及其他数据提交未纳入本轮全面审查。证据与便携脚本见 [S1 证据目录](review_evidence/20261113_stage1/README.md)。

复核采用静态检查、真实 NumPy/Torch 小模型接口对照、真实模拟器及原版规则的确定性控制实验、真实 gate 函数的合成输入反例。**合成反例证明实现或判定存在缺口，不是线上模型退化或自然发生频率的测量。** 没有训练模型、部署、提交比赛包或开展长对战。作者报告的 Studio 正典权重/回放验收与本轮独立结果分别记账。

## 1. 问题总表

P1 表示会直接产生错误教师标签或使核心收益认证失真，应在正式放行前修复；P2 表示实验配置、候选语义或协议执行缺口；P3 为验证及接口边界。

| ID | 级别 | 问题 | 主要位置 |
|---|---|---|---|
| S01 | P1 | 最后一次 continuation 灭绝漏判，真实终局被网络叶值覆盖 | `code/stage0_teacher.py:269` |
| S02 | P1 | 整份 q 的期望收益为负，q 闸与硬动作续弈仍可 PASS | `code/stage1_teacher_gate.py:442` |
| S03 | P1 | 同回放多个根按独立样本算置信区间，可能错误放行 | `code/stage1_teacher_gate.py:455` |
| S04 | P2 | guided 的 guard 历史与 greedy 不一致，生产/续弈 style 也未统一 | `code/stage0_teacher.py:198` |
| S05 | P2 | guard 改动作后仍沿用旧前缀 logits，先验和候选合法性失配 | `code/stage0_teacher.py:119` |
| S06 | P2 | 未完成训练集隔离核验也能正式 PASS | `code/stage1_teacher_gate.py:388` |
| S07 | P2 | 重复 manifest 局 ID 膨胀正式样本量 | `code/stage1_teacher_gate.py:381` |
| S08 | P2 | Stage A/B 未绑定，正式 N 门可调到 2 | `code/stage1_teacher_gate.py:372` |
| S09 | P2 | 声明 10% 开局根，实际采样约 3.2% | `code/stage1_teacher_gate.py:125` |

## 2. P1 复现与修复验收要求

### S01：最后一次展开灭绝仍走网络叶值

`evaluate_root` 在 continuation 循环的**开头**更新 `done`。最后一次 `clone.step` 若造成灭绝且步数仍小于 299，循环退出时 `done=False`；第 269 行只额外检查自然到时，随后进入 `_leaf_u`。正确的 `_terminal_u` 已实现，但这一条路径未调用它。

独立控制实验使用真实 `evaluate_root`、FastCarbon、Feat、BankruptcyGuard，并与原版 `compute_next_board` 对照；仅网络输出固定为可识别的合法哨兵值。默认 **H=2、guard=on**，根在 step100，最后展开到 step103 时最后一个工人进入敌基地而灭绝。原版终局分数为 `[-198,20]`，真实 **U=0.2228574087**，teacher 却输出 **0.876**，记录到叶值函数在 `alive=0` 上被调用。me=0/1 均复现。

两候选实验进一步说明影响：DOWN 臂最后展开灭绝，NONE 臂存活。当前 `Q=[.876,.876]`，`q=[.622459,.377541]`，选择 DOWN；仅将终局臂换成真实结果，`q≈[.00000350,.99999650]`，应选择 NONE。这不是日志误差，能改变蒸馏标签和教师选择；不表示真实权重恰好输出该哨兵值。

修复要求：每次推进后统一检查终止，或者循环退出后重新检查最终灭绝状态；使用对应最后一次推进前的 `pre`。验收须穿过实际 `evaluate_root`，覆盖根步、中间步、最后一步、H=0/1/2、双方席位和自然到时。现有 A2 直接测试 helper，不能覆盖该路由。

证据：[脚本](review_evidence/20261113_stage1/simulator/teacher_terminal_probe.py)、[13 个边界与两臂结果](review_evidence/20261113_stage1/simulator/results.json)。

### S02：报告软收益不等于用软收益放行

`_q_scene` 确实计算 `Σq·Q−Q(greedy)`，也计算行为温度后的分布收益；但 `main:442–447` 的 PASS 只使用 **argmax 动作收益、严格改进率、argmax 真实续弈锚点的均值**，完全不读取两个 soft 字段。独立 du 模式同样每步执行 argmax(q)，不能补上软分布认证。

遵守实际公式 `q∝sqrt(P)·exp(Q/.05)` 的 16 候选反例：

- `Q=[.80,.83,14×.76]`，`P=[.10,.09,14×(.81/14)]`。
- greedy 先验最大；argmax(q) 选 .83，硬动作收益 **+.03**。
- 整份 q 的期望收益 **−.01857149**。
- 原始 `_q_scene` 和 `main` 在 150 行、30 个准确的硬动作锚点 `U_q=.83,U_g=.80` 下仍给 **PASS**。

这里即便 Q 完全准确也能失败；不是只靠提高价值头精度就能解决。`--behavior-temp` 只改变所报软指标，不能把某个行为分布的收益变成硬动作锚点的结论。文档 §2 一方面明确“实际用于训练的 q”，另一方面仍给硬动作阈值，协议自身也需要订正。

另一个必须保留的边界：主 Q 来自同一次教师评估，既用于选动作又用于评价，存在选优偏差；**真实终局锚点确实独立于叶值，不能指控整套门都是网络自证**。问题在于它评的是硬动作，而且 q 门仅要求锚点点估计大于 0。

修复要求：区分硬教师闸与软目标闸；冻结实际训练 q 和实际执行的行为分布，在独立真实续弈子集上估计该分布的期望 U（按 q 抽候选，或按权重汇总候选真实续弈），使对应指标进入放行条件。教师选择所用 `Q_select` 与收益评估 `Q_eval` 的来源分别留痕。若只认证硬教师，必须缩窄结论，不能给整份 soft q 宣布收益已验证。

证据：[原始函数反例脚本](review_evidence/20261113_stage1/method/probe_method.py)、[结果](review_evidence/20261113_stage1/method/evidence.json)。

### S03：两臂已配对，但不同根之间并非自动独立

`sample_scene_specs` 通常从一局抽三个层位根；`t_ci` 与 `bootstrap_ci` 对扁平 rows 计算标准误和重采样，正式 du 判定不使用已有 `game` 分组。同局根共享地图、历史、对手与局势，协议没有提供独立性保证。

可行反例：50 个独立来源对局，每局 3 根；其中 25 局的各根 ΔU 均为 +.24，另 25 局均为 −.16。原始 main 得到 150 行、均值 **+.04**，区间 **[+.007887,+.072113]** 并 **PASS**；按来源局计算，区间约 **[−.0174,+.0974]**，组 bootstrap 也跨 0。

反例不证明真实回放一定完全相关，而是说明当前 95% 解释依赖未成立的假设。每个根的 guided/greedy 确实从同一状态出发、真实终局后相减，这部分配对通过；问题是配对差之间的相关性。

修复要求：若结论面向新对局/地图，按来源 game（需要时按更高层 seed/map）聚类估计区间和功效，或每个独立 game 只取一个确认根。同步报告 `rows / unique_roots / unique_games`。若只讨论固定根集合，应明确更窄的估计对象及根内随机重复，不能沿用跨新对局的泛化表述。证据同 S02。

## 3. P2：接线、候选与实验协议

### S04：相同 guard 配置没有带来相同状态化策略

`stage0_guided_match.py:102,122–142` 的 greedy 保留同一个 `guard_me`；guided 每步调用 `evaluate_root`，后者第 198–199 行新建两席 guard，无历史快照输入。`GUARD_NARROW` 依赖 `lostw>0`，fresh guard 不能知道根之前已发生的失员。gate 的 guided 续弈也走这条 fresh 路径。

真实 guard/teacher 控制例：此前失去一个工人，剩余 collector 在危险格；同一原始政策偏好 NONE。持久 guard 的 `lostw=1`，改成 `c1:UP`；fresh teacher 根以 `q=[.924142,.075858]` 仍选 NONE，`guard_moves=0`。分支快照隔离正确，不能补回缺失的根历史。

还有完整续弈配置差异：producer `selfplay_gen.py:160–163` 为 teacher/late teacher **2**；guided 与 gate 显式/默认 **1**，无对应 CLI 配置。style 是真实网络输入：受控小模型 40 个位置中 2 个动作不同，最大 U 差 .053363。这仅证明配置有语义，不是正典权重强度测量。

修复要求：维护两席实际 guard 状态，将不可变 root 快照交给教师，每候选/场景独立复制；真实状态只随实际动作更新。冻结权重、teacher/late schedule、guard、Feat 历史、探索和续弈配置，生产与评估一致，或明示新配方并重新定标。孤立 root 无历史时单列适用范围。不能只凭 GUARD 四个数值相同就声称增量完全归于搜索。

证据：[guard 脚本](review_evidence/20261113_stage1/runtime/guard_probe.py)、[guard 结果](review_evidence/20261113_stage1/runtime/guard-probe-results.json)、[style 脚本](review_evidence/20261113_stage1/runtime/teacher_config_probe.py)、[style 结果](review_evidence/20261113_stage1/runtime/teacher-config-results.json)。

### S05：guard 后动作与旧自回归前缀 logits 不再对应

`gen_candidates:119–125,148–158` 调用 guard 改写 labels，但仍把改写前 logits 放进 `arm_logits` 并计算 joint prior；可选动作也从旧前缀 logits 枚举。guard 强制早期基地招募后，后续 cash、合法掩码和 GRU 前缀均可能变化。

确定性小模型、真实 CachePolicy/guard/teacher，seed0 初始局面先合法 RECPLANTER 一步：step1 cash=50、无 collector，guard 强制第一基地招 collector。7 个返回候选中，首臂报告先验 **.2387751945**，按最终完整动作重新做 forced-prefix 评分为 **.2701377515**，最大差 **.0313625570**；末两臂仍请求两个各 30 的招募，在 cash50 下第二基地动作已被真实 Prefix 合法掩码禁止。

这里不把“受 guard 后处理策略的概率”偷换成原始策略概率。当前函数承诺返回完整候选的归一化 joint policy probability：就此契约必须按最终前缀重解码、评分并校验合法性；若改定义为 guard 推送后的策略概率，则应汇总所有映射到同一动作的质量，不能继续用旧 logits。最后对实际可执行动作去重。仅对旧先验重新归一化不能修复。

证据：[脚本](review_evidence/20261113_stage1/candidates/check_candidate_priors.py)、[逐臂合法性与先验](review_evidence/20261113_stage1/candidates/results.json)。

### S06：训练集隔离未核完仍能正式 PASS

`--ds-games-file` 可省略；header 写 `NOT RUN`，但正式判定不降级。真实 CLI/采样器加确定性 worker 输入，50 个不同游戏共 150 根时：省略清单或传空清单均正式 PASS。若输入为 `q0.json.gz` 而内部使用 stem `q0`，也会报不交；文件虽要求 stem，程序未拒绝错误命名空间。精确 stem 重叠和 q/du 同名重叠则正确拒绝。

修复要求：正式门必须消费非空且绑定训练数据版本的 ID manifest（或可核验的空训练集声明），记录内容 hash、命名规范、来源 ID 与局数；未核完只允许诊断。清单身份不能只靠调用者提供一组任意字符串。

### S07：重复局 ID 可以膨胀样本量

overlap 用 set，实际采样仍遍历未经唯一性校验的 list。同一 `q0` 写 50 次，真实采样器产生 **150 rows、122 unique roots、1 unique game**，正向 worker 输入下正式 PASS。这不同于 S03 正常每局多个根的统计问题：这里连输入局和部分根都重复。

修复要求：读清单时拒绝重复 origin game ID，生成唯一 root identity，采样后再验唯一性；镜像/改名数据需保留原始 lineage，不把改名当独立来源。

### S08：Stage A/B 锁定没有进入执行条件

`--n-required 2` 无范围门，du **仅 3 根**即可正式 PASS。`--pilot` 不自动限制规范要求的 30 根；50 局 fixture 默认跑 155 根。一次 Stage A 推荐 **600** 后，Stage B 不读取/验证该结果，默认 N=150，使用**同样的 155 根**仍可 PASS。header 未记录锁定 pilot 身份及用户设置的 `n_required`。

修复要求：Stage A 发布不可变 prereg 工件，绑定教师/配置/数据与 pilot roots、估计单位、锁定 N；Stage B 必须消费并验证，正式 N 不得低于规范下限。诊断减样明确降级。确认集与 pilot 的重用规则预先约定；q 闸的一次扩样也应绑定首轮及累计清单，不能只依赖手工约定。

S06–S08 证据：[实际 CLI/采样器探针](review_evidence/20261113_stage1/data/recheck_gate_inputs.py)、[15 项精简结果](review_evidence/20261113_stage1/data/results.compact.json)。昂贵模型/续弈 worker 被确定性记录替代，只验证控制流，绝非真实教师收益 PASS。

### S09：开局层占比实现错误

`DU_OPENING_SHARE=.10` 注释为 scenes 中开局比例；实现是在每局已有三个中后期根之外，以 .1 概率再加一个开局根。期望开局占比是 `.1/(3+.1)=3.2258%`。真实 sampler 对 10,000 个合成完整长度回放生成 31,018 根，其中开局 1,018，比例 **3.28197%**。

修复要求：按冻结的根层配额采样，或将规范明确改为“10% 对局额外加一个开局根”，承认实际分布。不能把现有均值解释为 10% 开局覆盖。证据见 method。

## 4. 已独立通过的部分与适用边界

| 复核点 | 本轮证据 | 可作出的结论 |
|---|---|---|
| A1 新 U 叶 | 小模型 20 个双席位位置，NP/Torch U 最大差 `5.6931e-8`；5 种权重入口；缺 head 明确 KeyError | 新头确实接入，无旧 value 静默回退 |
| 默认 NP 三元组 | 与 bc1 原实现逐值及采样后 RNG 比较均完全相同 | 默认接口在所测路径保持兼容，不仅是 tuple 长度不变 |
| CachePolicy | 直接新鲜编码对照；9 个编码输入变化、teacher 切换、TTA 恢复、权重文件更换、guard re-decode | 所测缓存隔离与 cfg/G=4 包装通过；不宣称任意原地换权重 API 受支持 |
| A2 终局 helper | 原版规则对照 pre-cash、灭绝、自然 299 边界 | helper 算术/pre 捕获正确；S01 调用路由仍失败 |
| A3 guard 抽取 | 6 步实际触发序列，新旧完整 agent 动作、全部 stats/log、lostw/prevw、alias 一致 | 抽取本体有独立等价证据；S04 是新接线/历史问题 |
| 双席位快照 | me=0/1 各 2 臂×2 场景，真实失员分支 | seat 映射、集合深拷贝、跨臂隔离与 root sim/Feat 只读通过 |
| Actor→NP greedy | 同 style1、greedy、相同特征的 12 个位置 | 默认条件等价；不覆盖旧 Actor 环境变量指定的 style/schedule |
| 真实续弈锚点 | 禁止 `_leaf_u` 调用，step298 真实推进到299，U=`.75249991667` | 锚点按真实终局评分，不是叶值自证 |
| packs 两副本 | NP 与 search 分别和权威源码逐字 hash 相同 | 源码副本同步；不代表历史包重建或已部署 |

详细结果：[network](review_evidence/20261113_stage1/network/results.json)、[runtime](review_evidence/20261113_stage1/runtime/guard-probe-results.json)、[simulator](review_evidence/20261113_stage1/simulator/results.json)、[docs](review_evidence/20261113_stage1/docs/results.json)。

## 5. 文档口径、统计细节与 P3

用户补充的五项已核入定版文档：π/IPW 与有意加权分开；报告软 q/行为温度收益；价值监督只取有效标签、approx 单列/unusable 排除；预算另计闸/生成/训练；mean U≈.5 不作为和局证据。这些区分应保留。**它们的文档落地不等于全链实现或收益已验收。** 其中软收益作为正式放行条件仍有 S02。

后续稀疏标注需记录最终 inclusion probability：例如基础 .1，未被基础选中的关键根再以 .5 纳入，则 π=.1+.9×.5=.55。基础率和倍率之外，还需冻结组合算法、去重、IPW 归一化/裁剪及损失尺度。IPW 校正标注抽样，不校正 q 估值误差，也不自动校正生成策略与部署策略的状态分布。这是尚未实现的后续生产契约，不算本轮新代码故障。

统计补正：`ceil((140σ)^2)` 可解释为 iid、近似正态条件下检测真效应 .02 相对 0 的约 80% 功效，但实际 PASS 还要求样本均值≥.02；真效应刚好 .02 且按公式达到 N、未被样本上限压低功效时，整个合取条件的放行率约 50%；截断后可能更低。上限 600 后代码也未报告规范要求的实际功效。修复 S03 后应按独立单位重估 N，并明确目标效应与通过条件的功效。t 函数还使用近似临界值（N>60 用 1.96、小 N 向更高 df 取表），应称近似区间或使用准确分位数；这是次于聚类问题的差异。

其他较低优先级边界：

- 已终局 root299 仍推进到300；正常调用应限0–298，建议入口拒绝或定义终态返回。H=2 实为根步加2次 continuation，共3步，需统一深度/成本口径。
- `BankruptcyGuard.restore` 重绑定 `guard_base`，会切断外部 alias；reset 的 in-place clear 正确。当前生产 agent 不调用 restore，teacher 新建分支后才 restore，未证明当前事故。
- 统计 FAIL 的进程退出码是0；worker 异常非零但旧 `--out` PASS 文件仍留存。未找到已依赖它自动开训的调用点，不声称已误启动训练。下游应读具 run ID 的新结果及 verdict，明确退出码约定与原子发布。
- 空 scene 输出诊断结果但含非标准 JSON NaN；`--max-scenes` 只有实际截断才降为诊断，与 help“forces”不一致。无减样的 cap 本身不削弱证据，按 CLI 边界处理。
- 根与分支的 Feat 是新建对象，已有历史简化并非完整历史 clone；原型使用完整模拟状态推演，尚未接入部署可见信息的 belief 机制。离线教师可使用特权状态，但此实验不能自动证明合法信息搜索或完整生产历史策略的收益。

正典验收留档不足及断言缺口也列 P3：

- `automation/acceptance_s1` 在审查 SHA 只有 README 和4个脚本。canonical `heads_v2.npz`（MD5 `06a03d9cfeec239af3065a49fefec382`）、20局 fixture 的 ID/hash/可取路径、原始 stdout、A3 旧实现来源等不足以从归档直接复证。README 已明说资料在机器上；**不能据此说作者没运行过**。本轮没有把小模型测试冒充 Studio 的 63 局面/50 根验收。
- A3 compare 接受空数组，或两边相同非零 errors；A3_teacher 的 `max(0,NaN,NaN)` 能保持0，合成 NaN 结果仍打印 PASS。应强制数量/身份、finite、q归一化、零错误及分支实际触发。这证明断言漏洞，不是实际结果含 NaN。
- A3_teacher 的 NP/Cache 两臂都被 `evaluate_root` 转成 CachePolicy，不能当独立 uncached/cache 对照；本轮另外做了直接独立对照并通过。其所有根 me=0，不能独自验 me=1；本轮双席位探针补齐。
- 定版文档的 `a0d67414` 是真实历史设计基线，但早于 S1，不能替代本轮冻结源码身份。

## 6. 下一轮关闭条件

1. 修 S01/S04/S05，在同一最终 SHA 复跑终局路由、两席历史状态、候选合法性/先验与原接口回归；不要拼多个阶段基线的自测结论。
2. 修 S02，冻结“训练 q、行为分布、评价续弈策略、独立 Q/真实终局”的明确契约与预注册阈值。
3. 修 S03/S06–S09，把来源局聚类、清单/根唯一性、训练隔离、层配额、Stage A/B 身份及 N 锁定真正接入正式 verdict；输入缺失只能诊断。
4. 固定制品与完整配置，补齐可取 fixture/命令/原始输出，按修复后的 Stage A/B 协议运行小规模真实 q 收益闸与独立 ΔU 闸。这里只能在实际结果出来后谈收益通过。
5. 两道闸均通过后按预算稀疏标注，接联合动作蒸馏，最终以冻结裸网的独立真实 ΔU 检验训练改进。教师收益、蒸馏成功、裸网增强是不同结论。

平台完整资源认证和冻结裸网收益验证仍为开放项。本报告只提交审查文档与复现证据；没有修改训练/搜索业务源码。
