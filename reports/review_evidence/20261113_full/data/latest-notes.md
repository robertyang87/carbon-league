# 新增 producer 追加审查（62e0dee）

独立基线 `62e0dee16419725ece6f18a22f095c83bb6fcd0f`，工作树 `/workspace/carbon-review-round5/latest`。本补充不改写原 98ffadfa 的复核结论。审查文件 `code/producer/selfplay_gen.py` 与 `code/producer/sp_probe_s29.sh`；后者只读，未运行。Python 脚本 MD5 实测 `b275e952da14d644e6fab41e4ce941c8`，与 shell 固定校验值一致。

证据：`latest-selfplay-check.py` / `latest-selfplay-results.json`。11 个预期观测断言均通过，其中有“确认边界存在”的断言，不能把它当作生成器无问题。所有生成数据在临时目录；真实网络推理替为确定动作；CLI 只用代码公开的 `SPGEN_STUB=1` 本地验证路径。没有训练、远端 shell、部署或真实权重运行。

## 已通过 / 与旧入口区分

- **新入口终局规则正确。** `selfplay_gen.py:173–194` 最多推进 299 步；合法两步招募灭绝例得到 rewards `[-299,50]`、margin `-349`，与先前原版环境 authority 一致。正常空动作局是 299 次推进、300 帧，最后 DONE。
- **raw margin / U 记录正确。** 例子 U 为 `[0.2068124697,0.7931875303]`，和为 1；值写进 index 和回放 result。`_u_of_margin:126–129` 的实现正确（文档 :44 的 W 三分支文字排列有笔误，代码与 :127 注释正确）。没有 bootstrap 或预测终局冒充真实值。
- **动作移位与 DONE padding 不产生终局后训练行。** step 0 action 空，obs(t) 对 action(t+1)；直接调用真实 build_game 对两侧仅生成 `[0,1,0,1]` 四行，padding 通过 status 截断。新 producer 本身没有旧 dagger/lb 的固定走满 300 继续训练问题；旧入口未改，不能因此宣称全链都已修。
- **前缀语义与 BC 相容。** 新回放存的是 agent 最终执行动作（包括 guard/探索结果），builder 按同一动作生成标签并推进 Prefix；没有旧 dagger 的实际 y / 目标 ym 分离。此格式没有行为 logp，适合当前 BC，不应当作可直接进入 PPO 的 on-policy probability 记录。
- **dataset 目标尚未迁移是已声明限制。** 文档 `:48–59` 明确当前 builder 仍只产 `value=tanh(M/3000)`。这不是声称 U 已接入 trainer 的修复；raw U/margin 的持久化与正式 E[U] 学习闭环分开判断。

## L01 — P2：新终局数据被默认 builder 的长度门丢掉

位置：`selfplay_gen.py:225` 写 `steps=nsteps`；`code/cnet2_build.py:255` / `:281–283` 默认 min_steps=200，main `:386` 未为 selfplay 放宽。

上述正确的两步终局回放（已经 pad 为 300 帧）通过正式 `B.main` 构建，结果 **exit 0，total=0，valid_games=0，filtered_games={min_steps:1}**。直接 build_game 有四条合法决策，说明问题在 scanner 的历史 BC 筛选规则，不在 padding/动作格式。

后果：所有 200 步前的灭绝局都会被默认自博→BC 接口排除，特别是本应学习的早期灾难/胜负边界。新生成器“能记录正确终局”不等于训练数据已保留这些终局。

建议为 source=selfplay / 完整 terminal 记录提供显式准入策略或 min_steps 参数，保留真实 nsteps；不要把 index.steps 伪造为 299/300 来规避筛选。证据键 `builder_default_early_terminal` / `direct_builder`。

## L02 — P2：只存局部 carbon，却被 builder 当全图辅助标签

位置：`selfplay_gen.py:167–168`、`:186–189` 仅存 `sim.obs(p)`；`fastsim.py:361–375` 明确隐藏格为零；`cnet2_build.py:205` / `:214` 在没有 full_carbon 时把 visible carbon 当全图 target；`cnet2_train.py:157` 对全图计算 MSE。

实测 seed 0 初始地图实际有 **103 个非零格**。当时没有 worker，agent 可见 carbon 全零；新回放没有 `full_carbon` 字段。真实 build_game 产出的第一行 carbon 目标全零，与真值 **103 格不一致**。证据 `auxiliary_carbon`。这是对普通完整局同样适用的共同初始帧问题，不依赖早终局过滤，也不是指 actor 应读取全图。

建议把 `sim.carbon` 作为仅用于监督的 full_carbon 保存，继续只把 sim.obs 交给 agent；或保存 target visibility mask 并明确屏蔽未知格损失。不能将隐藏零当作“该格真实无碳”。

## L03 — P2：回放与索引没有可恢复的提交协议

位置：`selfplay_gen.py:163–164`、`:208–215`、`:283–290`、`:312–318`、`:330`。

回放直接写最终 gzip 路径，index 只在整个任务结束后覆盖写。没有临时文件、逐局 journal 或重建索引逻辑。

- **回放已落盘、index 未写就中断：** 用真实 `_Gen.play` 生成一局但不写 index，随后按同 seed 运行实际 CLI；已有文件返回 `err='exists'`，main 视为 GAME-ERROR，exit 1，index 写成空列表。重复调用仍然如此，不能自行补回索引。
- **index 有局、gzip 丢失：** 保留 entry、删除 gzip，运行同 seed CLI；按 have 集合完全跳过，**games=0、errors=[]、exit 0**，文件仍不存在。下游新 N03 会拦，但生成器的完成/恢复结果本身不可信。
- index 损坏时 `:285–288` 静默置空，再与已有文件冲突；回放写中断可能留下半 gzip，同样被 exists 阻塞。这两项为源码可见的同一机制，本轮未额外模拟真实断电。

建议按局临时写入+校验+原子 rename，提交小 manifest/journal；恢复时验证 index↔文件双向一致、校验现有回放并恢复或重做孤儿。已有索引但缺文件应重做或明确失败；损坏 index 不应当成空历史并覆盖。需把模型/配置 hash 与生成身份一起持久化，以免恢复时把不同策略的同 seed 混作已完成。

## L04 — P2：内层吞掉的推理错误仍被接受为正常训练回放

位置：`selfplay_gen.py:175`、`:195–197` 只判断抛到外层的异常；真实 `cnet2_search.py:1905–1910` 会累计 root_errors 并返回空动作；生成器从未检查 agent.stats。

探针执行原样抽取的当前 search 最终 fallback wrapper，只把 `_inner` 改为抛出确定错误。两步内累计 **root_errors=2**，新生成器却返回 **error=None 并写出正常回放**。证据 `hidden_errors`。这验证真实 wrapper 与新调用端的控制流接缝，不是声称已有线上权重在出错。

建议记录每局前后 `errors` 和数值型 `*_errors` 的增量，任何策略/规则异常使该 teacher 样本失败或隔离并明确标注；不能仅依赖外层 try/except。缓存 agent 跨局使用，必须取增量，不能把历史累计误归到下一局。

## L05 — P3：probe 的局数/时间预算不是所写的硬上限

位置：`selfplay_gen.py:305–309` 一次提交 workers 个任务；`:309–322` 把整个批次全部收完再检查预算。`sp_probe_s29.sh:53–57` 最多 workers=56，却写 max-games=20 / max-seconds=180。

实际本地 STUB CLI 设置 workers=3、max-games=1，生成 **3 局**后才报 budget_hit=games；相同机制让 56 workers 的首批超过“最多 20 局”。max-seconds 也只在批次外检查，不中断当前批次；外层 timeout 420 是另一个总进程保护，不等于 180 秒生成窗口。证据 `max_games_overshoot`，不作实机吞吐结论。

另外 `:238` 的 CPU 起点在 worker 初始化，`:248` 每次返回累计 CPU，`:311` 再逐任务相加，会重复累计同 worker 先前任务和 warmup CPU；而生成 wall 从 warmup 后计时（`:303`）。该静态指标口径需要修正后才用 eff_cores 做资源效率结论。本轮未跑实机吞吐 benchmark。

## L06 — P3：仓库路径直接执行的导入失败；部署复制路径可避免

位置：`selfplay_gen.py:93–97` 只加入自身 producer 目录，依赖 eval_v53 等位于上一级 code。

清空 PYTHONPATH，直接 `python /repo/code/producer/selfplay_gen.py --help`，实测 ModuleNotFoundError: eval_v53。这是仓库直接使用入口问题。**不能据此声称 spot 脚本必然失败：** `sp_probe_s29.sh:42–46` 先把文件复制到 code 根目录后执行，恰好解决该布局；同时 MD5 断言与本基线文件匹配。

建议在模块中按真实布局加入 code 父目录，或固定包入口；保留复制路径兼容性，说明两种调用方式。

## 复跑

```bash
CARBON_REPO=/path/to/checkout-at-62e0dee \
CARBON_REVIEW_OUT=/path/to/results \
/path/to/python latest-selfplay-check.py
```

默认只 stdout，可用环境变量外置结果。policy stub / fallback 故障注入 / STUB CLI 范围都在脚本中明示。没有执行 `sp_probe_s29.sh`、调用远端资源或启动训练。单侧异常比赛只计 WDL 的平台规则不由这次正常两侧终局 fixture 覆盖；正常局 U 数值已验证。
