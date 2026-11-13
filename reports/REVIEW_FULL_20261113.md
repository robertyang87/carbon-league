# Carbon League 第五轮完整复核

日期：2026-11-13，北京时间。针对第四轮 [REVIEW_FIXES_20261113_R2.md](REVIEW_FIXES_20261113_R2.md) 的修复声明，并重新检查游戏规则、数据、网络、训练、搜索、打包、认证和考试自动化。

主审基线：`98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c`，已包含 Muse 所列 `4b44362d` 修复；新增实现先补核至 `62e0dee16419725ece6f18a22f095c83bb6fcd0f`，最终代码审查截止 **`57557a83a73cd4d6b26b56ebebb4c0df8e13a155`**。62e 比主审基线增加/修改六个实现文件：`cnet2_model.py`、`cnet2_train.py`、`stage0_heads_patch.py`、`stage0_smoke_heads.py`、`producer/selfplay_gen.py`、`producer/sp_probe_s29.sh`；575 再增 `stage0_train_heads.py` 与 `producer/sp_probe_studio.sh`。下文区分三个证据基线，不把旧测试覆盖冒充新结构已通过。

## 1. 结论

**Muse 的主要修复确实已进入源码，多项原复现可以关闭；“所有代码与执行契约均已关闭”仍不成立。** 本轮重新跑了原版环境对照、真实 builder/loader、网络与 PPO 接口、真实打包器和认证器、考试 helper 及提交客户端的隔离反例。既没有把读回成功当作正确性证明，也没有把已修旧例重复判成未修。

最先处理的事项：

1. **新增网络头破坏旧权重的 PPO 加载兼容性**，须先完成模型/恢复状态迁移，再从旧 checkpoint 重启。
2. **多个旧造数入口仍越过真实终局；DAgger 的特征前缀与训练前缀不一致。** 新 self-play producer 的终局逻辑有改善，但不能替代这些独立入口的修复。
3. **资源认证器存在假 PASS：首次 agent 调用被排除在单步预算外，最后一次推进未计入整局时间；指定对手加载失败还会静默替换对手。** 当前 PASS 不能证明平台资格。
4. **数据发布并非整版事务；考试续跑也未验证累计健康全集。** 逐文件 replace、旧 DONE、旧失败 seed 都留下跨轮状态问题。
5. **AWR 的半精度风险已有实际入口的可达反例；新设计的 MAE 停止门仍与条件均值学习冲突。** 不应再只靠人工一次范围检查或训练 loss 下降判定关闭。

没有实测证明当前已发布模型的强度退化，也没有独立核验 DGX/spot 的部署、生产缓存重建、线上提交状态或新方案的强度收益。本轮只写审查文档与证据，不修改业务实现，不运行真实训练、云队列、部署或天梯提交。

完整可复现证据：[本轮证据入口](review_evidence/20261113_full/README.md)。优先级含义：P1 为阻塞现有支持入口的回归；P2 为确定的数据/目标/验收错误，应在相关路径复用前处理；P3 为限定路径或恢复性、交接问题。它们均不自动等于生产事故。

## 2. 上轮修复验收

| 条目 | 本轮验证 | 裁定 |
|---|---|---|
| N01 packs 旧 search | 权威源与 packs 逐字一致；原终局/TTA 边界通过；注入漂移后 check 非零、build 拒绝产物 | 本地生产链原问题关闭；standalone 包身份的后验检查见 F08 |
| N02 `EVO_SEEDS + N=0` / 裸 wait | h14 动态列表冻结后计数；h14d/h15 为 57/30 个且唯一；逐 PID 失败会传播 | 原调用错误关闭；累计健康结果、旧 DONE 见 F07 |
| N03 缺 replay 静默漏局 | eligible 缺文件默认 exit1；显式 partial 记录原 ID/原因；补回后恢复完整构建 | 原漏局例关闭；原子发布范围见 F05，重复来源误拒见 P3 |
| 原 P3 合并中途覆盖旧数组 | 坏 shard 缺 maps 时 exit1，旧 meta/数组逐字节保留 | 原 mid-merge 例关闭；发布阶段是不同边界 |
| N04 资源工具 | 每次 sleep3.05s 的原例现 FAIL；真实包两步短测与错误分类门通过 | 原反例关闭，但新工具尚不满足完整认证契约，见 F04 |
| M01 期望头、胜率 | 设计改为 U/M 的 MSE 条件均值，WDL 单独建模；Huber 标为诊断 | 原数学错误在文档层关闭；残留 MAE kill 见 F09，新 smoke 的 loss 另核 |
| M02 主目标 | 设计以逐局 U、E[U] 和最终裸网 ΔU 为主，区分异常局 | 提案层关闭；现有 BC/PPO 并未因此自动切换目标 |
| M03 留出与判据 | M6 r1 归入开发；熟悉对手新种子与未见身份分开；每类 200 局、ΔU≥.02 且配对 95% CI 下界>0 | 原名单/缺阈值问题关闭；实际 manifest、一次性身份消费仍待落实 |
| 默认权重、DGX 护栏、台账 | packs 默认路径指向 repo/data；无确认的 dgx_full 在副作用前拒绝；n1r 原始权重身份明确记“未核” | 这些具体改进有效；确认后的 DGX 失败语义另见 P3 |

n1r **最终包身份**可复算：4,579,266 字节、MD5 `ba7f84c3652ea2cd14a2243f2b1eace2`；Echo3 为 4,576,790 字节、MD5 `7b7620499411a088cf4dc17c0e929bd7`，均与台账一致。两者无 `__file__` 执行的两步动作检查通过。**原始训练权重身份仍不可由解量化包补回**，不能混为一项。

## 3. 覆盖范围与已通过证据

| 链路 | 实际覆盖 | 边界 |
|---|---|---|
| 游戏/模拟器 | 原版 Board 36 个对拍例；8 个 eval、6 个 leaf 终局例；主源/pack sweep 及双方 MCTS 根/深层/自然到时；隐藏碳隔离、分支特征隔离 | 确定边界与短合法轨迹，不是对所有可达状态形式证明 |
| 数据 | 真实 builder/merger 与当前 loader；缺文件、partial、缓存、split、发布失败；一局真实新增 M9XJ-it460 corpus schema | 未全量验收所有新增 corpus；未重建生产数据 |
| 网络/PPO | 28 个局面含双方、真实 UID 换槽、晚期/最大单位边界；联合 logp、loop/packed 梯度、server/NumPy、预算、roles、GAE、resume 参数 | 98ff 的 joint logp 最大误差 4.77e−7、梯度最大差 1.12e−8；62e 的新增头单列追加验证 |
| 打包/运行 | 当前生产 packer、hash 闸、TTA cache、真实 n1r/Echo3、分类错误包、四种资源反例 | 重建使用 n1r 解量化 q8 权重；不是原始 fp32；仅短测 |
| 考试/工具 | 实际 bigeval 主函数配假 executor、shell helper 配假任务、ladder 配假 opener；134 份 cloud shell 的 bash -n | 没有执行原始云脚本或发送 POST；语法通过不是业务正确 |
| 方法/设计 | 修订稿逐条核对；均值/MAE/效用/q 分布数学例；当前 reward helper | 不把设计修改、smoke 或 loss 下降等同于强度提升 |

规则来源于本轮北京时间 **11:23** 重新读取的 [平台指南](https://117.72.210.134/static/agent-guide.md)、[首页](https://117.72.210.134/) 与 [回放 API](https://117.72.210.134/static/replay-api.txt)，HTTP 200，证书链及目标主机名验证通过；内容 hash 和时间保存在 [来源记录](review_evidence/20261113_full/sources/verified-public-sources.json)。重要契约：初始化 20s 只含编译/导入/顶层加载；agent 内延迟加载计入首步；单步 1s 基准、3s 硬界、每局 3 次慢步/5s 超额池；文件≤10,000,000 字节、1.5CPU/512MiB、整局任务 1500s。正常终局为 step299，不存在 obs299 后再下第300次动作。

## 4. 需要处理的确定问题

### F01 — P1：新增头的旧权重加载兼容回归（62e 追加）

`cnet2_model.py` 无条件新增 utility/wdl/margin 共 6 个参数键；`cnet2_train.py` 只在 BC_INIT 路径放宽 missing-key 白名单，PPO adapter 的严格加载入口没有相应迁移。旧 checkpoint 启动该入口会报缺 6 键，而不是把新头初始化后继续。恢复训练还必须同时迁移参数组/optimizer 状态，不能只将一个 `strict=True` 改掉就宣布兼容。

这阻塞的是**用新版源码从旧权重启动/恢复**，不推断已运行的旧进程受影响。追加测试及其他新文件结论见第 5 节。

### F02 — P2：终局契约未覆盖所有造数入口

位置：[`dagger_gen.py:51`](../code/dagger_gen.py#L51)、[`lb_relabel.py:123`](../code/lb_relabel.py#L123)、[`awr_gen.py:65`](../code/awr_gen.py#L65)、[`awr_weights.py:43`](../code/awr_weights.py#L43)。这里分别指 DAgger play、LB selfplay 和 AWR；不包含已修的主评估/PPO rollout，也不包含 LB 读取已有 corpus 分数的分支。

同一条从合法初始局面开始的两步招募序列，原版环境在 step2 结束，rewards=`[-299,50]`、margin=`−349`。不改模拟器，只将网络调用换为确定动作后：

| 实际生产入口 | 输出 | 错误标签 |
|---|---|---|
| DAgger | 300 行，298 行在真实终局后 | margin/value=0 |
| LB selfplay 默认 LB_FROM=270 | 30 行，全部在真实终局后 | margin/value=0 |
| AWR_STEPS=299 + builder | 299 行，297 行在真实终局后，回放全 ACTIVE | reward/value=0；rescore 又误当 mirror |

DAgger 本例没有 planter，但 value loss 不受 policy uw 屏蔽，不能以 uw=0 排除影响。新 producer 即使正确，也不会让旧入口自动修复。应共用带终止原因、原始 reward、最终 margin/U 的 GameResult；终局后零样本，所有重评分器沿用相同结果。自然触发率和已训练模型受影响程度尚未测。

证据：[label-results](review_evidence/20261113_full/data/label-results.json)、[awr-results](review_evidence/20261113_full/training/awr-results.json)。

### F03 — P2：DAgger 的行为前缀与训练前缀不一致

[`dagger_gen.py:64–76`](../code/dagger_gen.py#L64) 按实际动作 `y` 推进 Prefix、生成 mf/uf，却保存重标动作 `ym`；模型的 teacher forcing 以保存的 `ym` 推进 GRU。同一样本因此表达两种不同的前缀。

两个 planter 的真实动作 `[NONE,NONE]`、重标动作 `[UP,LEFT]` 时，第二单位特征与按训练前缀重算结果相差 **0.5**。这不是网络容量或 loss 权重问题。若训练的是实际前缀下的局部纠错，须分存 `prefix_actions` 与 `target_actions`；若训练改进联合动作，须顺序推进改进前缀并重算对应特征。普通 BC builder 的前缀一致性本轮通过。

证据：label-results 的 `dagger_prefix`。

### F04 — P2：资源认证存在假通过，也有跨局误拒绝

位置：[`certify_pack.py:100`](../code/certify_pack.py#L100)、`:128–143`、`:274`、`:331–352`。

| 确定反例 | 实测结果 | 与平台契约的差异 |
|---|---|---|
| 仅首次 agent sleep3.05s | first_call=3050.2ms，init≈3.184s；后续 max≈0.2ms，整体 PASS | 首调用被预热且动作丢弃，错误计入初始化，漏掉首步硬界 |
| 最后一次对手推进 sleep3.05s；显式 game-limit=2s | 外部≈3.253s，报告 game≈0.131s，PASS | game_wall 在最后 advance/summary 完成前截断；以短阈值验证控制流，未等待1500s |
| 两局，每局两次1.02s慢步 | 每局均在次数/超额池内，却合计 slow=4而FAIL | 单局额度错误跨局相加 |
| 显式给不存在的对手路径 | harness=true、a_errors=0、PASS，证书仍列该路径 | 宽泛 except 静默换成空动作对手 |

首调用必须作为真实 step0 计时、推进一次；整局计时覆盖实际结束；每局独立判定 slow/overage；显式对手/依赖加载失败必须退出。证书应绑定候选包、对手、harness hash 和完整参数。

本机无法创建独立 cgroup，**没有完成 1.5CPU/512MiB 平台配额认证**。两步真实包 PASS 是短冒烟。工具对未知单位键的宽容还表明资源 PASS 不能自动当作完整动作协议合规证书。

证据：[certifier-probes](review_evidence/20261113_full/runtime/certifier-probes.json)、[独立计时复现](review_evidence/20261113_full/independent/certifier-completion.json)。

### F05 — P2：数据发布不是整版原子切换，旧可选权重会残留

位置：[`cnet2_build.py:571`](../code/cnet2_build.py#L571)、[`cnet2_train.py:32`](../code/cnet2_train.py#L32)。临时目录确实保护了**合并阶段**；最后逐字段 `os.replace` 仍允许消费者看到新 maps=2 行、旧 y=6 行、旧 meta.total=6。实际 loader 正常返回混合版本；第二次 replace 注入 OSError 后，混合版本仍留在正式目录。

另一个正常退出反例：同目录先 weighted 构建，再关闭 weighted 重建。旧 `w.npy` 没被移除，默认 `BC_USE_W=1` 仍按文件存在加载旧 2/1 权重；所有数组行数相同，也无法靠 shape 检查发现。

建议发布完整不可变 generation，原子更新指针，**消费者在打开第一份数组前固定该 generation 路径**；manifest 声明本代字段、样本数、hash、配置，禁止未声明派生文件混入。只把目录换成可随时切换的 symlink 而逐文件重新解析，仍可能跨版本。当前 replica_chain 的删目录、串行非零停止避开上述复用路径；没有证明它越门训练。

证据：data-results 的 `publication_replace_failure`、`unweighted_rebuild`。

### F06 — P2：AWR 半精度风险可以由默认入口产生

位置：[`awr_weights.py:71`](../code/awr_weights.py#L71)、[`awr_collector_anchor.py:18`](../code/awr_collector_anchor.py#L18)、[`cnet2_train.py:200`](../code/cnet2_train.py#L200)。ESS 和 w_max 约束不提供正权重下界；anchor 的 `mean(w)/w` 会放大小权重倒数，cached trainer 再分别 half 化 w/uw。

本轮不直接手塞极小 w：构造 200 个非平局分差标签，调用真实默认 AWR weights → anchor → cached loss，得到 ESS=.211（通过）、min_w=`3.2104e−6`、max_uw=`428293.7`；原 float32 权重均有限，half 后 uw=inf，真实 batch_loss=NaN。

这是**支持入口下的可达输入反例**，还不是实际训练语料的范围调查或已发生事故。权重分布fixture使用合成分差标签与占位回放，未验证为完整合法对局轨迹。复用前扫实际范围有价值，但不能作为永久关闭依据；宜保持 loss 权重 float32，或对权重变换和加载设有限性/乘积稳定性硬门，并记录被拒数据身份。

证据：awr-results 的 `weight_precision`；没有执行 optimizer step。

### F07 — P2：考试续跑只验证增量，DONE 没有本轮身份

位置：[`q_h14.sh:6`](../code/cloud/q_h14.sh#L6)、`:20–47`、`:68–97`，h14d/h15 同类 helper；[`bigeval.py:12`](../code/bigeval.py#L12)。

- live 中旧 seed 有错误时，`miss()` 仍将其算作已完成。100 局中一个旧失败 seed 被跳过，其余99局新摘要健康即可通过 helper；原100个健康全集未核。
- helper 只核 tag/n/FAIL，受控摘要的 actor/对手错误、seed重复也能通过；无摘要且无输出仅 warn 后 exit0。后者是 helper 的 fail-open 边界，未证明当前正常 bigeval 会以 exit0 漏写摘要。
- 本轮失败写 FAIL，却不撤销旧 DONE；真实失败分支的测试留下 `sx.fail` 和旧 `sx.done` 共存。下游仍只看文件存在，会把上轮完成当本轮完成。
- bigeval 输入 `EVO_SEEDS=17,17,N=2` 会 exit0、n=2、se=0，但只有一个独立 seed。当前57/30固定列表均唯一，不认定它们已污染。

**对 q_h14 空列表的裁定：** 同一 run manifest 的全部预期 seed 都存在、唯一、健康，且 actor/对手/权重/配置/规则身份匹配，才返回 `ALREADY_COMPLETE` 成功；否则重试或失败。现状“一律空列表失败”是保守阻塞，不能只翻成成功。仅凭live中的seed不足；现格式未绑定完整配置身份。

应以 run_id/config hash 驱动任务、累计结果核全集，再原子发布可核验状态；所有消费者检查同一个身份，失效旧标记。证据：[automation/results](review_evidence/20261113_full/automation/results.json)。

### F08 — P2：远端制品核验与提交重试还有两个独立缺口

**包身份：** [`packs/cnet2_search_pack.py:49`](../code/packs/cnet2_search_pack.py#L49) 允许 standalone 无权威树构建，只警告；手册要求随后在干净仓库 `--check-source`，但这个命令不读取待发布包。实际以冻结旧 search 独立构建成功，内嵌 hash=`a15308190f4a527ab029b6cf783071a6`；再对干净源码检查仍 exit0，其权威 hash=`14230fa124e0a9b90d59d6df03c6765a`。本地 N01 修复有效，允许的远端交付步骤却没有验到制品。应核候选包内嵌源与权威提交并绑定最终包 hash，或禁止未经权威源检查的构建用于发布。

**POST 重放：** [`ladder.py:66–81`](../code/ladder.py#L66) 的内层 `_open_retry` 对所有方法最多重试6次；外层限制非GET一次并不能阻止它。假 opener 模拟“服务端接受但回包前断连”，观察两次 POST。若无幂等协议，应先核原请求状态再显式恢复；否则可能重复尝试/消耗额度或第一次成功、重试报错。本轮没有发真实请求，未认定 Echo3 发生重复提交；服务端去重效果未知。

证据：pack-export-results 的 standalone/clean recheck；automation/results 的 `ladder_post_ambiguous_retry`。

### F09 — P2：价值 MAE kill 仍会否掉更正确的均值学习

位置：[`katago_architecture_design.md:397`](katago_architecture_design.md#L397)、[`live/plan_to_first.md:110`](live/plan_to_first.md#L110)。其他段落已正确写明 MSE 学条件均值、MAE仅诊断，但 Stage1 仍要求“价值 MAE 两次不降即 kill”。

合法正常局 U 分布：90%为 .1、10%为 .9，条件均值=.18。预测从 .10→.14→.18，MSE 从 .0640→.0592→.0576 连降、均值偏差归零，MAE 却 .080→.112→.144 连升。若指 margin 均值，亦有同类反例。因此这不是统计噪声，而是判据错位。

删除 MAE 的独立停止身份；以主 U 损失/校准、真实教师质量、裸网开发 ΔU 和预算门决定。证据：[method/evidence](review_evidence/20261113_full/method/evidence.json)。

## 5. 审查期间新增实现：62e0dee → 57557a8

### F01 的追加实证与网络兼容范围

从98模型真实导出的旧小权重调用62版 `Cnet2Adapter.make_model`，报缺 utility/wdl/margin 六键；旧 checkpoint.net 严格加载同样失败。即使受控迁移模型缺键，旧 Adam 的59参数与新模型65参数也使 optimizer.load_state_dict 报参数组大小不符。应迁移白名单缺键及 optimizer 状态，并对两种恢复入口回归测试。

正向对照：按白名单迁移后，旧 policy logits/value 误差为0；新 npz 经 NumPy policy、真实 packer 及首步动作正常，任意改变三新头不改变旧 policy/value。**新增头没有破坏所有推理制品**，问题是旧权重的严格加载和恢复迁移。

真实普通 BC loss 反向传播时，六个新参数 grad 均为 None，旧 trunk/value 有梯度；独立 head_losses 则能给六参数梯度。随后575确实加入了会保存 merged npz 的完整头训练脚本；因此现状是“主 BC/PPO 旧目标 + 独立新头训练入口”，既不能说完全没实现，也不能说原有训练已切到 E[U]。

证据：[training/latest-heads-results](review_evidence/20261113_full/training/latest-heads-results.json)。

### F10 — P2：新 margin 期望头再次使用 Huber，smoke 也未形成失败门

`cnet2_model.py:164,209–212` 宣称 margin 输出 E[M]/1000；`stage0_smoke_heads.py:49` 实际 smooth_l1_loss，575的 `stage0_train_heads.py` 直接复用。9个M=0与1个M=10000的小张量例，经现有存储/反推后的标签均值为1006.066：头输出m_k=.111111（原始单位111.111）时dL/dm_k约0，输出1.006066（原始单位1006.066）时梯度为+.2，梯度下降会离开均值。这是新实现偏离已修文档，不是原文档未改。正式期望头需 MSE；若保留 Huber 应只当更名后的稳健诊断头。

旧 float16 tanh 反推不能恢复 raw margin：10000→10060.66，13320及20000均被截到约11400.62。smoke已披露尾部限制，不另冒称隐瞒；这些例子的正常U误差仅约1e−6至6e−6，不能把大分差误差夸大成同量级U错误。但正式 margin 均值、异常 outcome 类型和 bare/mixed continuation 仍须真实标签/版本绑定。

smoke真实控制流的故障注入还证实：训练loss为NaN只打印 finite=false并正常退出；验证loss为NaN时仍可能打印 finite=true并正常退出；缺旧stem.weight也因 strict=False 继续。575完整训练入口增加了缺键断言，能拒绝缺主干的旧权重；但对其真实main注入NaN训练/验证结果并使用不更新参数的假optimizer，仍保存npz、打印TRAIN-DONE、正常返回。没有执行真实优化，也没有声称保存的参数已被NaN污染；已证明的是非有限指标没有阻止产物发布。应将训练/验证/保存前参数有限性纳入硬门。

证据：[method/latest-heads-evidence](review_evidence/20261113_full/method/latest-heads-evidence.json)、training/latest-heads-results、[完整头训练出口](review_evidence/20261113_full/training/latest-full-heads-results.json)。

### F11 — P2：新 self-play producer 的正确终局没有完整进入数据契约

**通过：** 真正调用新 `_Gen.play`，两步灭绝得到 `[-299,50]`、margin=−349、U≈`[.206812,.793188]`；正常对局299次推进，动作移位及DONE padding正确；直接 build_game 生成双方共4行，没有终局后样本。双席位使用独立 agent 状态、两份 probe 的生产者 hash 绑定方向也正确。

仍有四个确定缺口：

1. **早终局被默认过滤。** index.steps写实际nsteps；builder默认min_steps=200，实际主构建将上述灭绝局过滤后exit0/total0。padding至300帧不会改变这个过滤。新自博数据需显式接受真实短终局，区分完整终局与截断/损坏；不能为了通过过滤伪写局长。
2. **辅助碳图标签错误。** producer仅存部分可见的sim.obs，没有full_carbon；builder:205回退可见carbon，训练端却对全图做MSE。seed0初始真实全图103格非零，回放全0，实际build_game的carbon目标也全0，103格错误。应另存仅供监督的完整碳图，或显式可见mask；完整隐藏真值不得因此进入actor输入。该问题可发生于正常完整局，独立于短局过滤。
3. **恢复依赖文件存在而非完整提交。** gzip直接写最终路径、index在整批末尾写；有gzip无index时反复报exists而无法恢复，有index无gzip时按seed跳过、games=0/errors=[]/exit0。应以每局原子replay+校验元数据和可恢复索引事务提交，重跑核文件完整性/身份，修复orphan而非只skip。
4. **吞掉的策略异常未进入结果。** 调用真实search最外层fallback，root_errors=2时生成器仍err=None并写正常回放。外层try只能捕获抛出的异常，不能捕获包内已吞异常；需前后统计错误增量，分类标记/拒绝受污染对局。

raw margin/U已写入corpus是有效进展；builder dataset仍只保留旧value列，这一点新脚本已明确披露，不能说它已完成正式U数据链。

P3边界：直接从repo的producer路径运行会因只加入当前目录而无法import上层eval_v53；fleet脚本复制到code根目录，避开此问题。workers=3/max-games=1的受控执行实际完成3局，说明预算是批边界软限制，不能当精确硬上限。未运行两份远端脚本或测真实生成吞吐。

证据：[data/latest-selfplay-results](review_evidence/20261113_full/data/latest-selfplay-results.json)。

### F12 — P2：新增完整头训练的校准统计须按预测值分箱

575的 `stage0_train_heads.py:evaluate` 按真实目标 `ut` 分箱再计算桶内预测误差，并称 `val_u_calib_binned`；预测校准应检验 E[U|预测u]≈u，按预测u分箱。直接调用原evaluate/targets并替换loader/net的小例：真实U约.1/.9各半、预测全.5，正确的常量均值却被报约.40003；真实U全.5、预测.2/.8各半，当前统计报0，按预测分箱却有.3误差。既会误报也会漏报。建议保留target分层残差但更名，并另报按预测分箱的校准、样本数与不确定性。

证据：[method/latest575-calibration-evidence](review_evidence/20261113_full/method/latest575-calibration-evidence.json)。

## 6. 训练方式到底是否合理

**以神经网络为主要能力载体、把受限搜索用于制造更好的监督、最终验裸网，是合理且值得做小预算原型的路线。** 本库已有多教师 BC、结构化单位编码、自回归联合动作、历史对手池和 PPO；本轮接口检查没有支持“联合动作 PPO 天生写错”这种判断。网络规模更大、换 Transformer 或沿用 KataGo 名称，均不能替代正确的数据/目标和独立收益验证。

当前主 BC 仍读 `tanh(M/3000)`；旧 PPO 仍按分差/身家与 win bonus 构造 reward，默认 gamma=.995，critic target 还有[-1,1]裁剪。新提案已改成 E[U]，不代表这些入口已切换。正常局应先算 `U=.5W+.5sigmoid(M/1000)` 再学其条件均值；异常单边执行故障用 W，非额定双故障按规则排除，不能靠 sigmoid(E[M]) 代替 E[U]。

实际提取旧 `game_rewards`，使用支持的 terminal 模式及默认 scale/bonus：稳赢100时 reward=.12、U≈.76249；80%输100/20%赢10000时期望 reward=.324、期望U≈.39000。旧目标会反向偏好后者。该例说明目标不等价，不声称远端当前就是该模式，也不把已明确隔离的旧线实验称为新提案的文档修复失败。

网络优先还必须解决三件事：

1. **监督的物理含义一致。** 所有生成器同终局、同奖励，前缀/联合动作一致；raw margin/U 保留，bare 与 mixed continuation 分清。先解决 F02/F03/F06，扩大数据才有价值。
2. **教师整体目标值得学。** 不只检查 argmax Q；若训练目标是软分布 q，应独立续弈验证实际 q/温度的期望收益。数学反例中 argmax 提升+.01而 q 期望反降约.00366，证明“选中最好动作更好”不足以认证完整蒸馏分布。我方 CarbonFormer 提案也须执行同一标准。
3. **收益证据来自冻结裸网。** 同初始权重/对手/预算比较 BC、terminal-U PPO、根教师蒸馏；绑定包/权重/hash、完整种子与配对统计。未见对手身份一经用于决定下一代，就不再是未见身份；换 seed 不能恢复其未见资格。

Top3 采用 KataGo 思路是用户提供的重要方向线索，本轮未取得它们的训练源码、受控消融和预算对照，不能独立证明领先全由网络或搜索贡献。当前证据支持先把数据、U目标和蒸馏闭环做可信，再按同预算测裸网提升；不支持提前承诺某骨干必胜。

## 7. 限定路径与交接问题（P3）

| 项目 | 位置/已证事实 | 范围与处理 |
|---|---|---|
| 快照树 ID 恢复 | `fastsim.py:316` 按当前主人重建计数；合法 seed7、7步抢树后恢复，再种树覆盖存活敌树，P1现金应31.67却30.85 | 主 cnet search 有+500绕过；旧 `nn_search.py:136/145` 无保护。应保存真实计数/按创建者ID恢复，不宣称默认搜索本轮回归 |
| 重复来源误拒 | builder 的 seen.add 在文件存在检查前；A缺失副本/B完好副本按A,B失败、B,A成功 | 保守失败；聚合来源、校验一致性后选择可用副本 |
| 推理服务首因不回传 | `rl2_ppo_server.py:87–94` load在try外；缺权重实际FileNotFoundError，worker无错误消息 | 默认客户端可能等600s；未实际等待。初始化也应纳入错误/退出协议 |
| DGX 确认后仍假DONE | `dgx_full.sh:25–33` 尾段复制失败、假训练命令返回7，仍exit0/FULL_DONE | opt-in新护栏确实有效；本例只抽取尾段、没有训练。确认不能替代失败即停 |
| cloud command 提前展开 | `.github/workflows/cloud-assistant.yml:38–41` 直接插值进shell，远端变量在runner先展开 | 旧workflow、b64路径可绕开；通过env值传参，未调用云API或读取凭证 |
| 当前入口过时 | `reports/live/CURRENT.md:41–43` 仍说n1r仓内无完整副本 | 已有最终包且hash可算；CURRENT应引用ledger。不要把原始权重未核误改为已核；本轮未独立核线上榜单 |
| 历史 marker 生产者 | q_h14b/c 仍有裸wait/按进程消失写同名DONE | 无证据当前正在运行；清退/冻结入口，避免与新manifest混用 |
| Studio probe依赖版本 | 新脚本仅在依赖文件缺失时从DGX同步code目录，已有文件仅打印hash、不核固定版本 | producer/weights固定hash匹配，两机seed区间不重叠；依赖代码仍须锁定。脚本还会写code/weights/tmp，不能按头注释理解为仅写probe数据目录；外部SSH超时未核 |

细节与精确反例见各域证据 notes；不把历史脚本、未上线提案和当前主入口混成同等生产影响。

## 8. 关闭顺序与交付边界

建议按依赖而非提交数量关账：先完成新增头兼容迁移与共同 GameResult/前缀契约；再封住权重精度、数据 generation 发布与 producer 恢复；随后修正认证计时和制品身份；最后让考试按 run manifest 核健康全集，再做有限预算的 U/蒸馏原型。

每项关闭至少包含：修复源码 SHA、原复现重跑、对应调用方/消费者检查、实际产物 hash。涉及远端还需逐机部署版本、缓存重建 manifest 与真实运行证书；本报告没有代替这些证据。

对 Muse 三层说法的最终口径：**提交和读回已核实；多项原缺陷行为已核实修复；跨入口/新增代码/整链验收仍未全部关闭。** 历史 n1r 原始权重缺口继续记账；远端部署状态保留“未独立核验”，不把未知状态写成确定未部署，也不因审查擅自部署。
