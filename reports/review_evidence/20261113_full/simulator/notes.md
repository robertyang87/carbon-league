# 第 5 轮：游戏规则、模拟器、信息边界和搜索

固定复核版本：`98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c`。源码和历史证据保持只读，没有训练或远端操作。

**N01 的搜索源码同步可以关闭。** 当前 `code/` 与 `code/packs/` 中 `cnet2_search.py`、`fastsim.py`、`cnet2_feat.py` 分别逐字节一致；对两份实际搜索函数的动态验证也通过。旧 C01 和终局奖励修复没有回退。

本轮补记一个 **P3：`FastCarbon.load_observation` 对被抢树的 ID 恢复错误**。它是现有 API 缺陷，不是本次同步引入的回归；主 cnet2 搜索已有局部绕过，旧 `nn_search.py` 直接调用时仍受影响。下文给出正常种子地图上 7 步合法前缀的确定复现。

## 覆盖范围与通过证据

没有仅检查 diff。重新阅读 `FastCarbon` 的初始化、观测恢复、ID、招募/移动/碰撞/存款/种植与抢树/吸收/枯萎/再生顺序；检查 `Feat` 的可见碳过滤、历史、镜像估计和分支复制；检查搜索入口、对手根动作生成、sweep 与 MCTS 的根步及后续推进、预算停止、终止前现金传递、终局分支冻结、自然步数边界和动作后处理。

| 验证 | 当前结果 |
|---|---|
| 原版 Board 对拍：地图、两个 300-transition 棋盘链、微型规则场景、双席位可见性、旧 agent 兼容 | 36 项全部通过 |
| 原版完整 episode 与当前 eval | 8 项全部通过，包含自然 299 结束、单/双灭绝及同一步出生死亡 |
| 原版 `compute_next_board` 与搜索 leaf | 6 项全部通过，包含存活方本步招募/树收入、初步及最后一步灭绝 |
| 权威 sweep 搜索边界 | 7 项通过，含根终局、深度 2 全终局、混合终局/非终局、两处预算中断、自然 299 边界 |
| 上轮 pack 的 3 个失败场景 | 全部关闭：不再 shallow 丢弃已解终局，不再走到 105 或 300 |
| 主/pack 各自实际 MCTS | 各 3 项通过，根终局、较深终局、自然到时/灭绝混合 |
| 双席位隐藏碳扰动 | 每席位改变所有不可见格子的碳，全部网络输入数组保持相同；模拟器过滤也通过 |
| 分支 Feat 恢复隔离 | 分支观察下一状态后，root 的数组、历史、前态字典与 token 映射均未变化 |

300-transition 对拍只验证 **Board 转移**，不把第 300 次转移当成真实比赛合法步。完整 episode 的停止与 reward 由独立 8+6 项原版对照检查。

原版事实依据仍是 `zerosum_env/envs/carbon/carbon.py:180–198`：灭绝方得到 `step−episodeSteps−1`；存活方在终止 transition 被置为 DONE，保留上一 reward。自然到时由 `zerosum_env/core.py:216` 在 step 299 终止。`FastCarbon` 自身是棋盘转移器，不代替 episode adjudicator。

### N01、C01 搜索检查的具体结果

`code/cnet2_search.py` 与 `code/packs/cnet2_search.py` 的 SHA-256 均为 `72719a75bab3db627b49ffa60a82e80d5d7ef99feb7f81d714c96248ea99c4d7`。

- 根步全终局：主/pack 的 sweep 和 MCTS 都冻结在 step 101，默认 UP 为 −220，NONE 为 +220，返回 NONE。
- 较深全终局：MCTS 两份实现都保留候选在 101 的终局；默认支在 102 才终局，值分别 −219 / +220，选择 NONE。
- 从 step 298 起一支自然到时、一支灭绝：两份 MCTS 和修复后的 pack sweep 都停在 299，值 0 / +22，选择 NONE。
- mixed 且预算不足：权威 sweep 仍保留 MIN_DEPTH 保护，根步截断或只再展开一步时均 `shallow=1`；没有把“某支已终局”误当作“所有分支已解”。

代码定位：最小深度条件第 1573 行；sweep `left` 上界第 1526 行；MCTS 上界第 1240 行；灭绝 leaf 第 552–563 行；四处推进分别在第 1299、1361、1512、1561 行附近保存该分支的终止前现金。当前 `_terminal` helper 只判断灭绝，自然到时靠 `H=min(H,left)` 限制，这是实际实现而非遗漏测试的假设。

这里关闭的是当前源码和构建输入的搜索差异。已发布/保留的历史单文件 artifact 不会随源码更新自动改变，其来源与认证由 runtime 复核负责。

## P3：抢树后恢复快照，再种树会覆盖现存敌树

**位置：** `code/fastsim.py:316`（pack 副本同位置），结合第 185–188 行的 `_new_tree_id` 和第 575–579 行的种树插入。

**原因：** 树被抢走后保留创建者 ID，例如 `player-0-tree-0` 可以属于 player 1。恢复函数遍历当前主人 `p`，却把 ID 后缀写入 `tc[p]`。因此 player 0 没有当前自有树时，自己的计数器恢复为 0；随后新树获得仍在敌方名下存活的同一个 ID。字典插入覆盖旧树对象，两方 `player_trees` 列表还可能指向同一个被改写的 ID。

**确定复现：** 正常 `reset(seed=7)`，默认初始现金与基地。每次前缀转移同时运行原版 Board 与连续 FastCarbon，逐步完全一致：

1. 双方在第一个基地招募 planter。
2. P0 从 64 右移到 65，P1 从 70 左移。
3. P0 在 65 种下 `player-0-tree-0`；P1 继续左移。
4. P0 连续下移离开；P1 到达 65 后停留，实际抢走该树。
5. step 7 保存 `full_observation()` 并 `load_observation()`；下一步 P0 在 125 停留种树。

原版与连续 FastCarbon：P1 保留 `(id=player-0-tree-0, pos=65, age=6)`，P0 新增 `(id=player-0-tree-1, pos=125, age=1)`。

恢复后的 FastCarbon：只剩字典中的 `player-0-tree-0`，被改成 `(owner=0, pos=125, age=1)`；原本 P1 的树消失。该步 P1 现金原版为 **31.67**，恢复分支为 **30.85**，碳格也已不同。因此不是纯字符串命名差异。

**配置/调用范围：** 任意 `load_observation` 后继续种树的调用者，在上述抢树状态可触发。旧 `code/nn_search.py:136、145` 直接使用此恢复 API，未补计数。当前 `cnet2_search.py:1298、1511` 额外给 tree counter 加 500，规避了本例；`cnet2_search_recruit_rank.py:78` 也有同类 workaround。因此不将其表述为当前默认 cnet2 搜索、裸网包或这次 N01 修复的线上事故。

**建议：** 至少根据树 ID 的原始创建者前缀恢复各自 high-water mark，而不是当前主人；若需要恢复原版精确 ID 序列，完整快照/clone 还须携带历史单调计数器。统一恢复契约后，调用者才能去掉各处固定偏移 workaround。应以“被抢树仍存活 + 创建者再次种树”的原版对拍验证，不能只用从 reset 连续推进的 parity 证明 snapshot 正确。

## 部分观测、同时行动与未扩大解释的边界

- 当前根候选不会把自身本步动作作为对手网络输入。对手 root action 从当时 `xo/fo` 产生，再复用于不同候选；未来分支则从推进后的各自观测取动作。没有在本轮发现当前根动作泄漏。
- `Feat.observe` 从己方工人位置计算 visibility（第 186–199 行），没有把普通 `full_carbon` 自动送入 actor。`reveal` 与 `oracle_carbon` 是明确的诊断入口；本轮没有把打开 oracle 的实验称作合法部分观测对局。
- 地图估计、镜像外推和对手私有历史属于近似。测试证明当前 actor 输入不随不可见碳直接变化，不证明这套 belief 校准良好或战力最优。
- `SHARE_OPP`、`LOCAL_R` 等是显式近似搜索选项。本轮静态检查这些分支，没有把默认 sweep/MCTS 的固定网络微型验证扩张成全部组合的正确性或强度保证。
- 已排除 worker ID 复用继承死人历史的初步疑点：`Feat.observe:355–356` 会删除不再存活工人的历史；现存 ID 的 max+1 不与 root 活工人冲突。本轮不将该疑点列为缺陷。
- 旧死亡分类回归仍为 `pla_dead=2=pd_clash_stay(1)+pd_unknown(1)`。

## 可迁移复现与结果

```bash
export CARBON_REPO=/path/to/carbon-league
export CARBON_REVIEW_OUT=/tmp/carbon-round5-evidence
/path/to/python eval_terminal_probe.py
/path/to/python search_sync_probe.py
/path/to/python simulator_contract_probe.py
```

本次 Python 为 `/workspace/carbon-review-venv/bin/python`。未指定 `CARBON_REPO` 时从脚本和 cwd 向上找 `code/fastsim.py`。每次输出新建临时目录，不覆盖旧证据。

- [eval_terminal_probe.py](eval_terminal_probe.py) → [8 eval + 6 leaf 原版对照 JSON](carbon-simulator-r5-2h4axdzz/results.json)
- [search_sync_probe.py](search_sync_probe.py) → [主/pack sweep 与 MCTS JSON](carbon-search-r5-lsx048db/search-boundaries.json)
- [simulator_contract_probe.py](simulator_contract_probe.py) → [36 项 parity、seed=7 快照缺陷、双席位 visibility、clone JSON](carbon-contract-r5-7jgbwqfx/simulator-contracts.json)

搜索探针从当前文件 AST 提取原样 `_search/_mcts/leaf/helper`，实际使用 FastCarbon、Feat、key_units 与 feat_copy，只固定网络输出和时钟；没有训练或伪造胜率。最后一份探针的 `seized_tree_snapshot.pass_` 明确为 false，脚本退出 0 只表示证据成功生成，不能当成该 API 通过。

目录中先前的 `carbon-search-r5-thjg2yhb`、`carbon-contract-r5-izqwfxqc`、`carbon-contract-r5-8a65toow` 是本轮探针扩充过程的保留快照；最终结论以以上三份链接为准。源码 `git status --short` 为空。
