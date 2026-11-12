# Simulator / search：第四轮复核

复核 commit：`5463f504048bcca28868ebce0a063ac8d6c51b8e`。业务源码只读；没有训练、修改旧证据或远端操作。

**权威 `code/` 下 C01 与种植员死亡细分类可关闭。新增 `code/packs/cnet2_search.py` 没有同步此前终局、步数与 C01 修复；开启搜索时能复现错误。当前固定 `SEARCH=False` 的裸网配置不进入这条路径。**

## 已通过

1. 原 C01 复现原样重跑：`MIN_DEPTH=3` 现在与诊断用 `MIN_DEPTH=1` 一致，选择 NONE，默认 UP 的真实终局值为 −220，NONE 为 +220；`shallow=0`，`changed=1`，`gain=440`。规则与价值没有换成测试替身。修复位置：`code/cnet2_search.py:1573`；默认配置 `MIN_DEPTH=3` 在第 36 行；全分支终局提前退出在第 1529 行。
2. 扩展 7 项搜索边界检查全部通过（包含上面的两个深度配置）。详细结果见下表。
3. 上轮 8 个真实原版 `env.run` 与 `eval_v53.play_game` 对照例全部通过：自然 299 到时、不可到达的第 299 次动作、最后真实动作花费、单方灭绝且存活方招募、双方同灭绝、最后一步灭绝、同一步招募并死亡的 planter/collector。
4. 上轮 6 个 `compute_next_board` 与搜索 `leaf` 数值对照全部通过：存活方招募、存活方树收入、双灭绝且现金不同、首次 transition 灭绝、自然终局花费、最后一步灭绝。存活方使用终止前 reward 对应现金、灭绝方使用 `step − episodeSteps − 1` 的语义仍正确。
5. 原种植员出生即死账本现在 `pla_dead=2`、`pd_clash_stay=1`、`pd_unknown=1`，其他 `pd_*` 为 0，分类合计等于死亡总数。招募、死亡、零产出死亡、早期相关类型计数均仍为 2。新增 bucket 定义在 `code/eval_v53.py:190`，出生即死归类在第 278–287 行。

| 搜索场景 | 实际深度 / 分支停步 | 结果 |
|---|---|---|
| 根步全部终局，默认 MIN_DEPTH=3 | depth 1；101 / 101 | 选 NONE；shallow=0；值 −220 / +220 |
| 同例 MIN_DEPTH=1 诊断 | depth 1；101 / 101 | 与默认配置一致 |
| 一支根步终局，另一支第 2 步终局 | depth 2；102 / 101 | 选 NONE；shallow=0；值 −219 / +220 |
| 根步 mixed，预算阻止首次 rollout | depth 1；101 / 101 | 保留 UP；shallow=1；未豁免未解分支 |
| mixed，展开一次后预算阻止下次 rollout | depth 2；102 / 101 | 保留 UP；shallow=1；终局支保持冻结 |
| mixed，非终局支达到 MIN_DEPTH=3 | depth 3；103 / 101 | 正常比较并选 NONE；shallow=0 |
| 从 step 298 开始，一支自然到时、一支灭绝 | depth 1；299 / 299 | 选 NONE；值 0 / +22；没有模拟第 300 步 |

注意：当前 `_terminal` helper 识别的是灭绝（第 549 行），自然到时由 `H=min(cfg["H"],left)`（第 1526 行）限制；测试结果中的 `terminal` 字段表示完整游戏终止条件，包含自然到时。没有把 helper 已扩展到自然到时作为通过依据。

原版依据：`code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.py:180–198` 先判灭绝并设置 DONE，再仅对 ACTIVE 更新 reward；`zerosum_env/core.py:216` 在 `episodeSteps−1` 终止。当前搜索奖励实现位于 `code/cnet2_search.py:552–563`。没有使用最大现金差作为原版灭绝 reward。

## 剩余确定问题：pack 搜索副本遗漏修复

建议与 runtime 发现的“打包来源过旧”合并成一个问题，不重复计数。搜索影响级别为 **P2，开启搜索时触发**；当前固定裸网配置 `SEARCH=False` 不执行 `_search/_mcts`，本项不声称当前裸网对局触发了终局搜索错误。

`code/packs/cnet2_search_pack.py:135` 从自身 `HERE` 读取模块，即会嵌入 `code/packs/cnet2_search.py`。这份搜索源码：

- 第 504 行 `leaf` 仍直接从现金与资产启发式计算，缺少 `_extinct/_terminal` 与终止前现金；
- 第 1176 行（`_mcts`）与第 1447 行（`_search`）仍是 `left+1`；
- 第 1479–1482 行无条件继续推进分支，缺少灭绝冻结；
- 第 1488 行仍无条件执行最小深度保护，缺少 C01 全终局例外。

对两份实际 `_search` 源码施加相同固定网络输出，得到以下确定差异：

| 构造例 | 权威 `code/cnet2_search.py` | `code/packs/cnet2_search.py` |
|---|---|---|
| 根步两支均已灭绝终局；预算不允许首次 rollout | 直接比较精确 −220 / +220，选 NONE | shallow=1，退回必败 UP；自己的 leaf 还误算为 −120 / +120 |
| 同样两支根步已终局，预算充足 | 冻结在 step 101 | 继续推进到 step 105，并用资产启发式计分 |
| step 298，默认支自然到时、NONE 支使对手灭绝 | 停在 299，分值 0 / +22，选 NONE | 推到 300，分值 0 / 0，错误退回 UP |

`code/fastsim.py` 与 `code/packs/fastsim.py` SHA-256 完全相同，均为 `500859dc534c666982ba92e223b1e3c89bce8c3fe18c8efc060b6077bb5e0843`。`cnet2_feat.py` 两份也完全相同。这里的遗漏在搜索层，不是模拟器副本本身不同。`FastCarbon` 本来就实现棋盘转移；完整 episode 终止与 reward 仍须由其调用者处理。

建议唯一化打包模块来源，或在构建时校验与权威源码的内容哈希，并让真实打包来源也执行这些终局测试。不能只把 C01 的一行改动复制过去，因为此副本还缺失此前 R01/F01/R02 的配套实现。

## 证据与复现

- [原版 8 个评估 + 6 个 leaf 对照结果](results.json)
- [7 个主源码边界 + 3 个 pack 探针结果](search-boundaries.json)
- [原 C01 脚本原样重跑结果](sweep-terminal-depth.json)
- 脚本：[review_simulator_r4.py](review_simulator_r4.py)、[search_boundaries.py](search_boundaries.py)、[original_sweep_repro.py](original_sweep_repro.py)。

```bash
export CARBON_REPO=/path/to/carbon-league
export CARBON_REVIEW_OUT=/tmp/carbon-r4-evidence
/path/to/python review_simulator_r4.py
/path/to/python search_boundaries.py
/path/to/python original_sweep_repro.py
```

环境本次使用 `/workspace/carbon-review-venv/bin/python`。若不设置 `CARBON_REPO`，脚本从自身目录与 cwd 向上查找 `code/fastsim.py`。输出每次进入全新的临时子目录，不覆盖本轮或历史结果。

`review_simulator_r4.py` 从上一轮脚本复制，只新增 `pd_unknown` 与分类合计断言、更新当前源码位置；实际对照仍运行仓内原版环境。`original_sweep_repro.py` 原样复制前轮探针；它的退出码不是通过判据，结论来自输出与新增脚本的明确断言。

`search_boundaries.py` 从当前文件提取原样 `_search` 及其 leaf/helper 函数；使用真实 `FastCarbon`、`Feat`、`key_units`、`feat_copy`。只固定网络输出和预算时钟，以可重复地隔离控制流。pack 源码调用的同类基础实现经过哈希一致性校验。`pack` 项是保留错误行为的探针，脚本目前对其错误表现作断言以确保复现，不代表该路径通过正确性验收。构造局面不用于估计自然对局发生频率、策略胜率或真实机器性能。

本轮未对 `_mcts` 做新的动态全路径测试；该副本同样缺少修复的结论基于实际源码。新增动态证据覆盖默认 `MODE="sweep"` 的 `_search`，不声称穷尽全部搜索配置。
