# Carbon League 修复复核（main@c64bbeda）

日期：2026-11-12。审查对象：[robertyang87/carbon-league](https://github.com/robertyang87/carbon-league)。固定基线：**c64bbeda8a00b2f4d79ace589f201bab3cca5163**。本轮检查原报告入库提交 5f338eb 至该基线的修复，与[初次完整审查](REVIEW_20261112.md)配套阅读。

入库前再次核对远端 main：1a0d8ee73dcccb95c1c987fde46cc55a83531aa5。相比被测基线只新增 5 笔回放/列表数据提交，被审查的代码与操作文档未变化；本报告的动态测试仍严格归属 c64bbeda。

## 1. 验收结论

**22/22 笔修复提交确实已进入 main；不能据此认定 R01–R15 全部关闭。** 代码修复、复测通过、部署生效是三个不同状态。本轮确认若干原始回归已修好，也发现仍影响训练目标和验收可信度的缺口，以及一个采样开局的新回归。

最优先的两个问题：

1. **R02：训练和搜索仍可能奖励已经灭绝判负的一方。** 新代码会停止模拟，但仍用现金差作为输赢及终局价值；评分语义没有贯通。
2. **R06：重标执行备忘仍强制把所有派生样本改回 TRAIN。** 按现有步骤执行，会覆盖刚修好的原始对局划分；旧重标 shard 也缺迁移保护。

修复这两项、同步严格的终局计分并完成下述数据/采样回归检查后，再做新版 harness 定标和部署验收。不要把旧版 +785 等锚点直接沿用到新规则。用户说明各机器拉取和 harness 替换尚未执行；本轮没有登录考场/训练机器、部署、训练、天梯提交或操作云资源。

## 2. 本轮证据范围

- 核对 22 笔修复的提交记录，检查 11 个修改过的 Python 文件，语法均通过。清单见[commit-syntax-summary.json](review_evidence/20261112_fixes/commit-syntax-summary.json)。
- 运行仓内原版 env.run 与本地 eval 的完整合法两步灭绝场景及正常 299 步边界场景。
- 执行真实 rollout、PackedTraj、Cnet2Buffer、GAE、decoder 和 resume 代码块；用轻量策略/构造边界状态控制输入，未运行完整 PPO 训练。
- 使用当前打包器和仓内 b128 权重做真实包故障注入、299 步纯网计时、采样开局改前/改后对照。
- 使用真实 builder/merge 对合成小数据集运行缓存与 split 回归；云资源检查只用 API mock。
- 环境：Python 3.12、NumPy 1.26.4、PyTorch 2.5.1 CPU。计时不是目标考场资源限制下的资格证明。
- 这不是完整胜率回归或全功能覆盖；未取得并复测当前 Echo2/n1r 的完整发布制品，也未证明现网包装层与仓内原版的所有评分细节一致。

## 3. R01–R15 状态表

“原例通过”只表示初次报告中的复现得到修复，不代表相关模块已无其他问题。

| 原条目 | 本轮判定 | 证据与残留 |
|---|---|---|
| R01 多走一步 | 原例通过 | eval/rollout 均为 299 次转移、末次观测 298；search 两路径去掉 left+1（搜索 horizon 为静态核对） |
| R02 灭绝与最终奖励 | **未关闭，P1** | 提前停止、变长轨迹/GAE 已通过；训练/search 仍按现金判终局；eval 存活方评分也未严格复刻原版，见 F01/F04 |
| R03 resume 参数覆盖 | 原例通过 | 显式全覆盖、全继承、显式 KL=0、只覆盖 LR 四例均通过；有效参数 manifest 已实现 |
| R04 对手温度污染 | 部分通过 | 新建独立 SNAP 正确；同权重缓存对象跨 primary/SNAP 角色复用仍污染，见 F06 |
| R05 异常考试漏闸 | 原例通过，接口待补 | 当前坏 teacher 包退出 3、磁盘 FAIL=true；外层异常/缺局仍能存 FAIL=false，见 F09 |
| R06 重标验证泄漏 | **未关闭，P1 条件性** | 新 orig_id 路径通过；执行文档覆盖 split、旧 shard 无 orig_id 仍泄漏，见 F02 |
| R07 招募死亡账本 | 总账通过，分类未闭合 | 原负入库消除；类型/时段/种植员死亡诊断仍漏当步出生死亡，见 F07 |
| R08 构建缓存 | 部分通过 | step 窗口和教师表变化能失效；空构建复活旧 shard、index 分数变化沿用旧标签，见 F05 |
| R09 残树硬闸 | 文档修正通过 | 残树/种树量降为诊断，胜负分差为主指标；不代表相关新实验已通过 |
| R10 终局增长口径 | 算术修正通过 | 2093 改为双方差值 702；需用新版规则重算基线，执行备忘仍有旧 eval hash，见 F08 |
| R11 纯网计时空数组 | 原例通过 | 299 步正常完成，searched=0、搜索分位数 null；错误退出码仍非验收门，见 F09 |
| R12 TTA 编码缓存 | 原例通过，**有新回归** | 原坐标偏差降至 0；OPEN_T 分支缺 T_now 导致 guard 异常；签名仍不完整，见 F03/第 5 节 |
| R13 退休分支 workflow | 静态修正通过 | checkout/pull/push 和 live README 已指向 main；本轮未实际触发采集工作流 |
| R14 接手入口/制品 | 未关闭 | 新增 CURRENT，但根入口无指针、声称在仓 wrapper 缺失、旧 hash 冲突，见 F08 |
| R15 ECS 验证误报 | 原 503 复现通过 | 查询持续失败非零退出；HTTP200 畸形结构及仍存在的退出状态待补，见 F09 |

## 4. 需要继续修复的发现

### F01 — P1：灭绝已停局，但 PPO 与搜索仍把现金优势当胜利（R02）

位置：[rl2_ppo_rollout.py:245](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/rl2_ppo_rollout.py#L245)、[同文件:275](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/rl2_ppo_rollout.py#L275)、[cnet2_search.py:542](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/cnet2_search.py#L542)。

rollout 的 dmargin、最终 margin 和终局 worth 仍来自 cash。search 新增终止分支，却明确返回 cash[me]−cash[opp]。灭绝判负并不等于现金归零，因此仅停止推进不够。

构造边界状态：A 无工人无树、现金 20；B 存活、现金 10。相同边界在原版引擎终止后 rewards=[−300,10]，分差 −310；真实 rollout 报告 margin=+10。测试以 reward scale=1 隔离符号，dense/terminal/worth 三种模式均得到约 +10.1，包含错误的胜利奖励。默认训练尺度会改变幅度，不会修正错判方向。此例是构造状态的引擎对照，未据此声称已测量正常训练中的发生频率。

search 的原样 leaf 函数测试也返回 +10：单方灭绝时应为 −301（不同终止步）；双方同时灭绝时应为 0。错误终局值可让搜索偏好已判负分支。

**建议验收条件：** 共用原版状态转移与终局 reward 规则，保留必要的前一步 reward；eval、rollout、search 统一消费。最后一条奖励补终局计分修正，SELF 镜像同步取负。至少覆盖单方灭绝、双方灭绝、存活方当步现金变化、正常到时四类，检查三种 reward 模式与搜索终局值。

证据：[ppo-fixes-results.json](review_evidence/20261112_fixes/ppo-fixes-results.json)、[simulator-fix-results.json](review_evidence/20261112_fixes/simulator-fix-results.json)。

### F02 — P1，条件触发：操作文档和旧 shard 会重新制造验证泄漏（R06）

位置：[lb_expert_iteration_prep.md:77](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/reports/live/lb_expert_iteration_prep.md#L77) 及第 119 行；[merge_dagger.py](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/merge_dagger.py)。

新 lb_relabel 写 orig_id、新 merge 继承原局 split 的路径正确。但开工备忘仍要求“lb 局一律改回 train”，把 src=="dagger" 的 val 全置 false。实测原验证局 0efcfa…-1.json.gz 的派生样本正确进入 VAL 后，按文档处理就进入 TRAIN，原局仍留在 VAL。

旧 lb_cp_*.npz 缺 orig_id 时仍按派生名重新哈希，原例也会进入 TRAIN。不是所有无 orig_id 的新对局都应拒绝，但已知重标来源不能当作独立新局处理。

**建议验收条件：** 删除文档的 val 强制覆盖；旧重标制品补原局 ID 或明确拒绝导入；以 original_game_id 检查训练/验证交集为零，而非只比验证集行数。该问题是按计划开工或复用旧制品时触发，本轮不声称已经发生新一轮训练泄漏。

证据：[data-results.json](review_evidence/20261112_fixes/data-results.json)。

### F03 — P2，新回归：采样开局触发 guard 时 KeyError('T_now')（R12）

位置：[cnet2_search.py:699](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/cnet2_search.py#L699)、[同文件:974](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/cnet2_search.py#L974)。

OPEN_T 采样分支使用 teacher=T，但不设置 T_now；只有非采样分支设置。新 enc_for 调用却直接读取 st["T_now"]。

用真实 b128 权重、OPEN_T=3、OPEN_SEED=8、TEACHER=1、GUARD=True、SEARCH=False，第 0 步即复现：改前 errors=0、两个基地 RECCOLLECTOR；改后 errors=1、last_error=KeyError('T_now')，动作仅 RECPLANTER。保护层异常回退，改变了实际动作。该结论限定为受支持的采样配置，不代表线上贪心包也触发。

**建议验收条件：** 每个分支明确本次实际 teacher，再调用 enc_for；验证采样、普通贪心、TTA、阶段教师切换与替代网络路径。复跑本例须 errors/root_errors=0。

证据：[sampling_before_after.out](review_evidence/20261112_fixes/runtime/observed/sampling_before_after.out)。

### F04 — P2：eval 存活方终止 reward 仍与仓内原版不同（R02）

位置：[eval_v53.py:433](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/eval_v53.py#L433)。原版依据：[carbon.py:190](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.py#L190)。

原版先在少于两名 ACTIVE 时把存活方也设为 DONE，再只给 ACTIVE 方更新 reward；因此灭绝终止当步，存活方保留上一步 reward。修复后 eval 却使用终止后的现金。

从正常初始地图用合法动作完整运行两步：A 连续在同一基地招募且不移动、第二步自撞灭绝；B 两步分别在两个基地招募。原版 rewards=[−299,50]、分差 −349；新版 eval scores=[−299,20]、分差 −319。胜负相同，均分偏 30。

**建议验收条件：** 按原版状态更新顺序保留 reward，不能单用最终现金推回；若现网确有不同规则，应提供现网结果证据并明确选用哪套规范。与 F01 一起统一实现。

证据：[simulator-fix-results.json](review_evidence/20261112_fixes/simulator-fix-results.json)。

### F05 — P2，条件触发：缓存仍能悄悄复用被排除的行及过期标签（R08）

位置：[cnet2_build.py:183](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/cnet2_build.py#L183)、[同文件:284](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/cnet2_build.py#L284)、[同文件:305](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/cnet2_build.py#L305)。

两个真实 builder 复现：

- **空重建复活旧 shard。** 先生成 step=[2]；把 STEP_MIN 改为 100，第一次构建输出 []，同配置第二次却输出 [2]。n==0 直接返回，旧 npz 未删除，新 manifest 又把旧文件认证为当前配置。切终局窗口时，提前结束的回放尤其可能零行。
- **index 修正不使标签失效。** gzip 不变，仅将 index scores 从 [100,0] 改为 [3000,0]；新 metadata 已更新，value 仍是 0.033325，应为 0.761719。缓存未包含会影响内容的 margins/tids/sides 等任务参数。

**建议验收条件：** 零行和失败任务不保留有效旧 shard；manifest 记录每项成功状态/行数，只认证本次成功产物。把每项生成内容的输入和元数据纳入指纹。连续运行两次均输出相同正确结果，并单独测试 index 更新。

证据：[data-results.json](review_evidence/20261112_fixes/data-results.json)。

### F06 — P2，条件触发：actor 缓存跨角色复用仍污染 SNAP 温度（R04）

位置：[rl2_ppo_rollout.py:58](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/rl2_ppo_rollout.py#L58)、[同文件:167](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/rl2_ppo_rollout.py#L167)。

缓存 key 只有 adapter 和 wkey；primary 将 actor.temp_kind=True。若同 key 后来或同场又作为 SNAP，取得的是同一对象，角色属性没有隔离。动态确认 cached_snapshot_is_primary=True，传给快照的温度 0.5/1.0 得到相同 logprob。不同 key 的新快照已正确，不应把此问题扩大成所有对手仍错误。

**建议验收条件：** 角色进入缓存身份，或每次调用显式传温度策略而不修改共享对象。覆盖 primary→SNAP、同路径同场、普通独立 SNAP 三种情况。

证据：[ppo-fixes-results.json](review_evidence/20261112_fixes/ppo-fixes-results.json)。

### F07 — P2：总账修好了，细分行为账本仍漏同一步出生死亡（R07）

位置：[eval_v53.py:224](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/eval_v53.py#L224)、[同文件:312](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/eval_v53.py#L312)。

总 recruit_count/spend/deaths 改用 worker counter，原例正确为 2/60/2、入库为 0。但招募类型、时段和种植员死亡仍靠存活 ID 差推断。两步连续招募种植员并自撞，pla_rec、pla_dead、pla_dead0、rec60_p、e_prec25 都是 1，实际各应为 2。会影响开局招募结构、零收益死亡等研究归因。

**建议验收条件：** 总账和细分账本消费同一份含发生步/单位类型的事件数据；细分计数相加应与总计一致。

证据：[planter-ledger-residual.json](review_evidence/20261112_fixes/planter-ledger-residual.json)、[simulator-fix-results.json](review_evidence/20261112_fixes/simulator-fix-results.json)。

### F08 — P2：接手入口和部署核验说明仍不自洽（R14，关联 R10）

位置：[CURRENT.md:17](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/reports/live/CURRENT.md#L17)、[lb_expert_iteration_prep.md:106](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/reports/live/lb_expert_iteration_prep.md#L106)。

- CURRENT 声称 code/exp/echo2.py “仓内”，但该路径不在审查基线 Git tree，无法执行其 wrapper hash 核验。权重和 n1r 完整包仍缺从仓库可恢复的来源。
- 根 README/HANDOFF 未指向 CURRENT，仍会把接手者引向旧说明。
- 备忘要求同步最新 eval 后核 MD5 6d24da0e；当前 eval 的完整 MD5 是 40f3875d5967274cd549b73b82cb6efa。同步修复后必然与旧前缀不符。

**建议验收条件：** 根入口指向 CURRENT；制品明确仓内/仓外真实位置、源码 SHA、构建参数和校验值；将新版 harness hash 与重新定标的基线绑定。不能为了满足旧 hash 回退正确修复。

证据：[docs-results.json](review_evidence/20261112_fixes/docs-results.json)。

### F09 — P2：部分失败状态仍与机器可读成功标记矛盾（R05/R11/R15）

这些是各工具在原始问题已修复后的剩余边界；不能把它们说成原修复完全无效。

- [bigeval.py:50](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/bigeval.py#L50)：三局中两局成功、一局外层异常，正确退出 2，但保存 FAIL=false、err=1、n=2。只读 FAIL 的下游可能误判。FAIL 应覆盖内部错误、外层错误和缺局；根 fallback 也应保留首个堆栈。
- [cnet2_search_timing.py:37](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/code/cnet2_search_timing.py#L37)：坏 teacher 包 299 步全根失败，输出 root_errors=299，却退出 0。它可以作为统计工具，但不能只用退出码当自动验收门；若承担门禁，应错误非零退出。
- [ecs_release.py:119](https://github.com/robertyang87/carbon-league/blob/c64bbeda8a00b2f4d79ace589f201bab3cca5163/tools/ecs_release.py#L119)：HTTP503 持续到期限现在正确退出 1；但 HTTP200 且 Instances={} 的缺字段结构仍被当成空列表并报 RELEASED_VERIFIED。实例持续存在到期限则输出 STILL_PRESENT 却退出 0。应校验 Instance 列表结构，完成状态与退出码保持一致，或明确区分“删除已接受”和“释放已验证”的契约。

证据：[gates.out](review_evidence/20261112_fixes/runtime/observed/gates.out)、[timing_checks.out](review_evidence/20261112_fixes/runtime/observed/timing_checks.out)、[ecs-results.json](review_evidence/20261112_fixes/ecs-results.json)。

## 5. 补充风险与效率问题

**终止后的无效推理。** rollout 先为全部 games/selfg/snaps prepare/finish，再判断 done。两场 SELF 中一场 1 步结束、一场 299 步，记录 600 条有效决策，却 decode 1196 次，多出 596 次；轨迹长度和 GAE 本身已通过。这主要浪费算力并影响吞吐口径，应只为未结束局推理，全批结束就退出。

**缓存签名未覆盖所有编码输入。** CachePolicy._sig 缺 tok_kind/tok_mask/tok_cargo/hist_cargo/hist_move。两个正常 Feat.observe 特征在总货量相同、地图通道饱和、两采集员货量由 (1000,800) 改为 (1100,700) 时签名相同，正确编码与缓存结果最大差 0.061544。原标准 TTA 问题确实由 0.587379 降至 0；本轮尚未证明上述签名碰撞在现有线上调用序列中自然触发，因此列为实现契约风险，不宣称已造成线上强度下降。建议完整覆盖编码输入，或明确持有当前根编码，避免用不完整签名推断身份。

证据：[ppo-fixes-results.json](review_evidence/20261112_fixes/ppo-fixes-results.json)、[tta_restore.json](review_evidence/20261112_fixes/runtime/observed/tta_restore.json)、[cache_observations.out](review_evidence/20261112_fixes/runtime/observed/cache_observations.out)。

## 6. 对训练合理性的更新判断

**BC → 带历史对手的 PPO 主线仍合理；本轮修复尚不足以证明现有训练闭环可靠。** resume 参数覆盖和 rollout 边界确有改善，但 F01 会直接把错误胜负送入奖励，优先级高于扩模型、加搜索或增算力。F02/F05 会影响新训练数据和验证可信度。

原报告的以下方法问题也没有因为 22 笔实现修复而自动得到实验回答：

- gamma=0.995 与 worth 差分塑形不保证保留不折扣终局目标，需要目标明确的对照实验。
- critic 输出/回报截断需实际分布与裁剪比例支持；不能仅凭结构断言已造成损害。
- 训练采样与部署贪心的差异，需用同一制品、双策略口径评估。
- 反复用于选型的固定考试集，应与最终未用于选择的检验集分开。
- 动作一致率约 0.87 不是“BC 已到上限”的证明；残树不是未清算资产的直接证据。

推荐顺序：统一游戏终止和奖励 → 修复 split/cache/采样回归 → 小规模确定性冒烟 → 记录各机器源码/harness/制品 hash → 新旧版本同种子对照并重算锚点 → 再运行可归因的短训与独立检验。以上是复核建议，不代表本轮已经部署或启动任务。

## 7. 复查证据

证据目录：[review_evidence/20261112_fixes/](review_evidence/20261112_fixes/)。入口见其中 [README.md](review_evidence/20261112_fixes/README.md)，含复现命令、环境与局限。

这些脚本是审查探针：有的断言检查修复通过，有的输出仍存在的反例；**脚本退出 0 不代表本报告全部问题关闭。** 结果文件记录 c64bbeda 当时输出。未来源码修改后应生成新结果并注明新 SHA，不覆盖本次历史证据。

