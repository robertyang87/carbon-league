# Round 5 — pack/export 与资源认证复核

基线 `98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c`；该基线的 `code/certify_pack.py` 与修复声明中的 `4b44362d` 文件 SHA-256 相同（`6774ac96b8fb1e3a165b80325f1361def5e5c01013144739f5bb3af1411bf7a2`）。只读业务源码，本地短探针；无部署/提交/长评测。不能用哈希一致代替行为正确，以下同时给实际行为证据。

## 已确认的改进

- **N01 本地生产链已修：** `code/packs/cnet2_search.py` 与权威源码逐字一致，历史版本明确移入 `historical/`。生产 packer `--check-source` exit0；临时拷贝里注入源漂移后 check exit1、build exit2、无产物。正常重建 `_BUILD_HASHES` 四项均与实际内嵌 `_SRC` MD5 一致。使用从 n1r 包提取的解量化 q8 权重重建，TTA consumer 编码误差0、标准13键前两步动作与冻结 n1r 相同。不能据历史 reference 的旧源继续列原 N01；远端验收边界另见下面。
- **N04 原慢步反例已拦截：** 每次 `agent()` 实际 sleep3.05s 的探针，测量步 max3051.0ms，认证 exit1/FAIL，step_hard 检查失败。新工具确实输出初始化、p95、slow/overage、game wall、RSS及分类错误；不是“什么都没修”。
- **错误门继续有效：** 新生产packer以n1r解量化权重构建 TG_MAX='invalid' 与 teacher999 两个故障包，certifier两步均 exit1/FAIL，a_errors=2。合法真实 n1r/Echo3 短证书 exit0、a_errors=0。
- **P3已修/标明：** packs默认权重位置修为 `../../data/weights`（packer:160–165）；n1r来源缺项已在ledger明确标“源身份未核”。

证据：`pack-export-results.json`、`real-pack-certifier-results.json`、`certifier-probes.json`。

## F1 — P2：首次 agent 调用被移到初始化，真实首步超硬界仍获 PASS

位置：`code/certify_pack.py:128–143`，以及 `151–162`；runbook:35也沿用了“exec＋首次调用的init”口径。

平台指南规定初始化20秒只覆盖编译、导入、顶层加载，**agent内延迟加载仍计入首步**（`https://117.72.210.134/static/agent-guide.md`；本地已取快照 `/workspace/carbon-design-20261113/live-agent-guide.md:38`）。新工具在READY前真实调用一次agent、丢弃动作，重置模拟器后才测step0。这样首次冷加载或一次性全局计算只受20秒门，而非首步3秒/慢步池。

实际未改源码探针 `cold_3050`：只在第一次调用sleep3.05s，之后返回合法空动作。证书 first_call_ms=3050.2、init=3.184s，两个“测量步”max0.2ms、slow=0、overage=0，**overall PASS/exit0**。平台首次动作已超过3秒硬界。作为对照，每次都慢的原例确实FAIL。

建议：READY在编译/顶层初始化完成后发送；首个真实agent调用直接作为step0计时、保留动作推进一次，不额外预热或重放step0。首次调用若需要分项统计，可同时标注cold-call，但不能从逐步预算中剔除。

证据：`certifier_probe.py`、`certifier-probes.json` 的 cold_3050/warm_3050/legal。

## F2 — P2：慢步次数与超额池按所有 games 合并，合规多局被误拒绝

位置：`code/certify_pack.py:331–334,349–352`。

平台限制按每方**每局**重新计算。当前将全部games的steps摊平，再对总slow数与总overage做一次门。`--games 2` 时每局各2次慢步均合规，却被合计4次判不合规；超额池同样跨局相加。

实际探针 `two_games_two_slow_each`：2局各2个测量步，每次sleep1.02s，每局只有2次慢步、每局超额约0.04s。证书汇总slow=4、overage=0.083s，**exit1/FAIL，仅slow_count失败**。首调用预热的另一个问题与此独立；本例只分析工具自己的测量步。

建议：在每个game内计算并存slow_count、overage与门，overall要求所有game通过；总体聚合仅用于报告，不使用平台单局阈值拒绝跨局总数。

证据：同上，two_games_two_slow_each。

## F3 — P2：指定对手无法加载时静默改为空动作对手，证书仍报原对手并 PASS

位置：`code/certify_pack.py:100–116,166` 与结果字段 `367`。

`eval_v53`导入、指定对手加载、模拟器创建共用一个宽泛except。对手路径拼错或导入失败进入fallback后，`opp_fn`仍为None；测试继续用 `{}` 作为对手，结果却保留命令里原opp路径。资源负载与实际局面不再是声明的考试条件。

实际 `invalid_opponent` 传一个不存在的 `--opp`：harness=true、a_errors=0、fatals空、**PASS/exit0**，证书仍列不存在的对手文件。没有在任何远端执行该条件；这是本地受控装载失败例。

建议：显式指定的pack/opponent和harness加载失败一律非零，记录首因；只有未指定opp时才明确选择idle模式。不要用可选导入fallback吞掉真实考试条件错误。

证据：`certifier-probes.json` 的 invalid_opponent。

## F4 — P2，远端条件：源检查没有绑定到待发布包

位置：`code/packs/cnet2_search_pack.py:49–65,192–201,209–217`；runbook:28的远端无权威树回退步骤。

本地build与check确实能拦漂移。但在standalone机器上，build允许无权威树、仅警告和记录hash；手册规定随后“在仓库环境跑 `... --check-source`”。这个CLI只比较**当前本地两份源**，没有待检pack参数，也未读取远端包的 `_SRC/_BUILD_HASHES`。因此它不能证明已经交付的远端包使用了这些源码。

实际临时目录复现：拷贝当前packer及依赖，唯独search换为明确冻结的历史实现，且没有上层权威目录。standalone build **exit0**（warning、source_check.checked=false），生成包search hash=`a15308190f4a527ab029b6cf783071a6`；随后在干净真实仓库跑文档规定的check仍 **exit0**，权威search hash=`14230fa124e0a9b90d59d6df03c6765a`。这不是认定现存n1r/Echo3历史包应改写，也不是证明真实生产方已走错分支；是文档允许路径无法闭合待发布制品检查。

建议：提供 `--check-pack FILE`，核验候选包实际内嵌源与记录hash，再对比冻结权威提交/明确allowlist，并记录最终pack hash；或禁止无权威源构建作为可发布产物。仅在另一机器对干净目录check不够。

证据：`pack_export_probe.py`、`pack-export-results.json` 的 standalone_drift_build/clean_repo_recheck_after_standalone_build/hash字段。

## 其他边界与根agent交叉证据

- 根agent独立验证 `code/certify_pack.py:274` 在最后sim advance/summary完成前结束game_wall计时。`--steps-per-game 1 --game-limit-s 2`，对手sleep3.05s，外部总3.253s却report game0.131s且PASS；idle对照0.168s/report0.118s。该记录见 `/workspace/carbon-review-round5/independent/check_certifier_completion.py` 与 `certifier-completion.json`，应与F1一起描述N04的新实现未完全对齐平台。未运行1500秒等待，采用显式较短阈值验证同一控制流。
- `wrong_action_key` 返回不存在的单位ID也获PASS；本工具目前只做资源/内部异常门，未做平台动作合法性门。本次真实制品烟测另检查了自身unit ID；不要把资源PASS自动推广为动作协议完整认证。
- 当前工具不记录最终pack内容hash；证书仅有路径，复现性还依赖外部台账。建议将pack/opponent/hash、harness SHA、完整args和cgroup配置写入正式证书；本轮不把这一元信息改善单独与核心行为缺陷混排。

## 真实产物与实测限制

| 产物 | MD5 /字节 | 本轮结果 |
|---|---|---|
| n1r_int8_reference.py | ba7f84c3652ea2cd14a2243f2b1eace2 /4,579,266 | 与ledger一致，exec无__file__两步动作合法、errors0；短certifier PASS |
| echo3_int8.py | 7b7620499411a088cf4dc17c0e929bd7 /4,576,790 | 与ledger一致，exec无__file__两步动作合法、errors0；短certifier PASS |

两者为明确冻结的历史产物，未改写、未重新上线。本轮第一次直接检查已入仓Echo3；不能沿用旧轮“Echo3产物未在仓内”的限制。ledger对历史外部比赛/上榜的描述未在本地重新认证。

新生产重建和故障注入使用**从真实n1r包解量化得到的q8权重**，不是原始训练fp32文件，也不是上一轮旧b128fixture；不能据此补齐原权重MD5。重建后只检查短程动作/缓存与错误门，不声称全局策略等价。

所有certifier真实包记录只跑2个测量步，且其额外warmup如F1所述。环境 `/sys/fs/cgroup` 虽rw挂载，当前用户无写权限；没有独立创建1.5CPU/512MiB cgroup。`--mem-limit-mib512` 只是本地RSS门，不构成同平台CPU配额认证。未做全局强度或300步资源认证。
