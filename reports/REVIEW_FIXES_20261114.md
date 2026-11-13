# 2026-11-14 review fixes

初始同步基线：ace6db9dd69f70a1cf284c93307a87a765e0858f；提交前再次同步到acd8f37，保留并发入仓的0500f8c双臂新鲜度修复。本次只修改代码与回归；没有部署、重启训练、重建数据、开新现场实验或上榜提交。

## 修复范围

|审查项|改动与验收|
|---|---|
|F01 教师根历史|extract_scene已经观察根，continue_game首轮直接复用根x/aux，不再重复observe；实际8步前缀的首次续弈特征及历史逐位相同|
|F02 数值保护|恢复后检查完整actor/critic/reference及optimizer；训练检查V/ADV/RET、loss、梯度/范数、更新后的状态；非有限值非零退出并记录numerical_failure.json，禁止发布坏状态|
|F03 checkpoint|完整state写同目录临时文件、fsync、atomic replace；保留state_previous.pt；actor/state哈希写入checkpoint_complete.json后才发布完成日志|
|F04 margin精度|index与shard按同一RAW_FIELDS dtype量化后比较，不用float64对float32的固定1e-4误差；合法±7000.12及65536.12通过，真实篡改/NaN仍拒绝|
|F05 恢复预热|resume-vwarm是正式迭代前独立阶段，写warmup.jsonl；不推进正式it，不触发正式snap/EVAL，不改actor optimizer、KL系数或league更新；预热进度和共享头optimizer可恢复|
|F06 根前guard|按前缀公共worker身份重建lostw/prevw，实际续弈继承两席状态；v3的q/anchor/guided teacher只接收我方真实guard snapshot，真实对手guard只用于真实续弈/oracle诊断|
|F08 AWR|最新main已增加generation-owned w/uw的写前拒绝；保留原修复，新增完整有效weighted generation反例，确认拒绝后所有文件不变、loader仍可加载|
|F09 双臂健康回执|最新main的0500f8c已增加每臂证明窗口内日志与最新权重mtime必须推进，并保留本次启动PID检查；本轮保留该并发修复，并增加真实shell判断片段回归。GPU信息仍是平台可用时打印的人工证据，不声称已自动认证GPU健康|

附带补齐已排期的残局窗口：`--step-min 200 --step-max 299`为[200,299)；显式窗口替代默认跨阶段层，不会把max-scenes冒充步数过滤。窗口及scene_contract绑定header和pilot artifact；不同窗口/旧契约不能复用。prefix simulator使用get_state/set_state，保留实体ID计数器。代码身份新增Feat/FastCarbon hash。

## 使用语义变化

- `--resume-vwarm N --iters 0`：只预热N批，正式it不动。
- `--resume-vwarm N --iters M`：先预热，再执行M个正式迭代；M仍须由调度方按实际剩余预算指定，不自动重计30轮。
- 用部分预热checkpoint续跑时，保持同一个`--resume-vwarm N`，以auto/checkpoint保留critic来源；已完成的批次不重复执行。显式`--utility-source init`表示再次重置critic，因此重新执行预热。
- 全actor逐位不变对应`--sep-value 1`；兼容的共享模式仅更新utility/value头，策略参数与actor optimizer不变。旧的fresh-run `--vwarm`计号语义保留。
- 调度监视器以`checkpoint_complete.json`的phase/it与文件SHA256为完成依据；`train.jsonl`现在只在完整提交后追加。warmup单独记账，不移动it85/95/105。
- checkpoint完整state是恢复权威；actor NPZ不能替代critic/optimizer。多文件提交期间若中断，旧marker与新文件hash不匹配时应拒绝该完成回执；state_last/state_previous仍可供核验恢复。
- 新teacher gate会使旧代码hash/pilot契约失效，须按新版重新做正典验收；12局残局冒烟仍不作收益认证。现有formal门槛没有放宽，12根pilot仍为diagnostic-grade。
- replica沿用最新main的120秒证明窗口，每臂日志和最新权重mtime均须推进；存活但冻结会拒绝。失败不擅自停止训练。GPU利用率由调度方另行核验，自动回执仅证明已实现的PID与新鲜度条件。

## 验证

运行环境：CPU tiny CNet2/合成回放/临时generation；不是现场部署或正典heads棋力认证。

- PPO U mode：18/18。
- PPO resume：33/33，包括真实rollout/GAE/backward、只预热、预热后正式更新、部分预热续跑、NaN backbone/optimizer拒绝与健康恢复点保留。
- 新审查回归：9/9（含多个精度子例、实际prefix续弈/guard、q/anchor隔离、断写保存、双臂存活与新鲜度回执、AWR只读拒绝）。
- gate/pilot身份：4/4，含显式窗口与跨窗口pilot拒绝。
- mirror identity：15/15。
- 教师信息集合：20/20；双世界：37/37。
- FastCarbon对拍：36/36。
- pack source check：ok=true，嵌入源码无不一致。
- bash -n、Python语法检查、git diff --check通过。

默认教师测试使用仓内b128及seeded utility头验证信息流，不是正典heads收益/校准认证。双臂回执以本地无副作用shell stub覆盖；本轮没有在DGX/L20上执行真实GPU任务。运行日志保存在本轮工作区/workspace/carbon-fix-validation/。

## 部署边界

这些是入仓修复，不等于现场进程已使用新代码。部署须使用完整提交（新增依赖rl2_checkpoint.py），不要只替换旧trainer或launcher单文件。仍由统一调度方在已批准边界核对actor/critic/ref身份、剩余U轮预算和留出轨迹校准，再决定部署/恢复；本次不执行这些现场动作。
