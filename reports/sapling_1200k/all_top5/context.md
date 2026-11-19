# 天梯前五逐决策背景查询

冻结截点 `4a4677c9` 覆盖 1,542 场双席官方分数齐全的对局，累计 459,160 个联合行动、918,320 个行动前的本席可见决策视角。另 1 场执行故障留有 257 步正常前缀。此次追加的 14 场均走满 299 步，新增 8,372 个逐席视角。经济转折窗口保留了 534 个去重后的局内联合行动，可逐步查看交战双方当时的工人、载货、位置、命令和到账。[上下文索引收据](context_appendix_4a4677c9_receipt.json) · [转折前后逐步动作](appendix_4a4677c9/turn_actions.json.gz) · [初盘隔离核验](appendix_4a4677c9/exposure.json)

此前冻结截点 `aca9b8a3` 含 1,528 场双席官方分数完整的对局，累计 454,974 步、909,948 个赛前可见的逐席决策视角。另有 1 场执行故障的 257 步正常前缀。新增七场的经济转折前后共 280 个联合步，保留了双方整队、实际命令及现金流水。[七场回放上下文收据](context_appendix_aca9b8a3_receipt.json) · [逐步发令与结算](appendix_aca9b8a3/turn_actions.json.gz) · [新旧实验地图隔离](appendix_aca9b8a3/exposure.json)

截至仓库快照 `b5e6a7f8`，有完整双方分数的 **1,521 局**共 452,881 次联合推进，按两个物理席索引 **905,762 个行动前视角**。另有 1 场执行故障，双方正常的前 257 步及各自视野保留在单列回放，故障方的官方分数为空。共 1,522 个不同比赛 ID、地图种子和初始碳图。最新 19 局另可逐步读取经济转折前 10 步至窗口结束的双方工人、真实动作和实际结算。
[最新 19 局上下文](context_appendix_b5e6a7f8_receipt.json) · [转折前后联合动作与收支](appendix_b5e6a7f8/turn_actions.json.gz) · [上批 15 局上下文](context_appendix_2cc7c358_receipt.json) · [故障局的有效前缀](appendix_97de1ebc/unscored/README.md) · [新旧实验地图隔离](appendix_b5e6a7f8/exposure.json)


截至 `cbd6d545`，冻结前五累计 **1,465 局、436,331 个联合回合、872,662 个行动前物理席位视角**。新批 25 局的 14,676 个决策均核对两席公开碳图、现金与命令。24 局走到步限，1 局正式灭绝提前结束。[本批逐局实收和转折](appendix_cbd6d545/README.md) · [决策上下文收据](context_appendix_cbd6d545_receipt.json)

截至 `39a745e0`，冻结前五累计 1,440 张独立地图、428,993 次联合推进、857,986 个物理席决策。本批 8 场均为 Master6 对外实战，共 4,784 个行动前上下文。[双方现金、行动与转折前局面](appendix_39a745e0/README.md) · [上下文收据](context_appendix_39a745e0_receipt.json)

截至 `c5c8cf8d`，已核对 1,432 张独立地图、426,601 次联合推进、853,202 个物理席决策。新增 11 场均为前五对外实战，其中 Master5s 6 场、Master6 4 场、ClaudeFable5.2-v7 1 场，共 6,578 个行动前上下文。[逐局现金、攻防与回溯](appendix_c5c8cf8d/README.md) · [上下文收据](context_appendix_c5c8cf8d_receipt.json)

截至 `32b555ee`，前五回放共 1,421 张独立地图、423,312 次联合推进、846,624 个物理席决策。这批新增 Master6 的 14 场对外胜局，共 8,372 个行动前上下文。[逐局回放与收入](appendix_32b555ee/README.md) · [上下文收据](context_appendix_32b555ee_receipt.json)

截至 `1fdacb1b` 再补五场，累计 1,407 张地图、419,126 次联合推进、838,252 个席决策。[四场上下文](context_appendix_f7109df6_receipt.json) · [单场上下文](context_appendix_1fdacb1b_receipt.json)

在一个真实决策点，先读该席出手前能知道的盘面，再按需打开实际动作和结算标签。截至 4bfdabf5 的三个批次有 1,347 局、401,385 个联合推进、802,770 个物理席决策。555c0c56 补编增加 10 局、2,791 个联合推进、5,582 个席决策，累计 1,357 局、404,176 个联合推进、808,352 个物理席决策。 截至 04:20 UTC 冻结的 bf049043，再增四局；五批累计 1,361 局、405,372 个联合推进、810,744 个物理席决策。 收盘后的 Master6 索引另补 bd3c0021 十二局，六批合计 1,373 局、408,960 个联合推进、817,920 个物理席决策。 5484590b 更新再增五场对外实战，七批共 1,378 局、410,455 个联合推进、820,910 个物理席决策。99919cbf5 补编四场，八批共 1,382 局、411,651 个联合推进、823,302 个物理席决策。0137949d4 补入四场 Master5s 对外局，九批累计 1,386 局、412,847 次联合推进、825,694 个物理席决策。41e4177e 新增十六场对外，累计 1,402 局、417,631 次联合推进、835,262 个物理席决策。仓内已有无损原始回放，查询器按索引直接取单局，读后核验压缩包 SHA-256，支持无网络的本地训练与分析。[读取接口](context.py)

| 批次 | 局数 | 联合推进 | 可查席决策 | 索引与验收 |
| --- | ---: | ---: | ---: | --- |
| 核心前五 | 1,324 | 394,508 | 789,016 | [地址索引](context_index.tsv.gz) · [全批收据和实例](context_receipt.json) |
| 后续新增 499886f1 | 8 | 2,392 | 4,784 | [地址索引](context_appendix_499886f1_index.tsv.gz) · [全批收据](context_appendix_499886f1_receipt.json) |
| 后续新增 4bfdabf5 | 15 | 4,485 | 8,970 | [地址索引](context_appendix_4bfdabf5_index.tsv.gz) · [全批收据](context_appendix_4bfdabf5_receipt.json) |
| 后续新增 555c0c56 | 10 | 2,791 | 5,582 | [地址索引](context_appendix_555c0c56_index.tsv.gz) · [全批收据](context_appendix_555c0c56_receipt.json) |
| 11:56 批末（bf049043） | 4 | 1,196 | 2,392 | [地址索引](context_appendix_bf049043_index.tsv.gz) · [全批收据](context_appendix_bf049043_receipt.json) |
| 04:20 UTC 收盘后 Master6 单列补编（bd3c0021） | 12 | 3,588 | 7,176 | [地址索引](context_appendix_bd3c0021_index.tsv.gz) · [全批收据](context_appendix_bd3c0021_receipt.json) |
| 5484590b 新增对外场 | 5 | 1,495 | 2,990 | [地址索引](context_appendix_5484590b_index.tsv.gz) · [全批核验](context_appendix_5484590b_receipt.json) |
| 99919cbf5 新增对外场 | 4 | 1,196 | 2,392 | [地址索引](context_appendix_99919cbf_index.tsv.gz) · [全批核验](context_appendix_99919cbf_receipt.json) |
| 0137949d4 新增对外场 | 4 | 1,196 | 2,392 | [地址索引](context_appendix_0137949d_index.tsv.gz) · [全批核验](context_appendix_0137949d_receipt.json) |
| 41e4177e 新增对外场 | 16 | 4,784 | 9,568 | [地址索引](context_appendix_41e4177e_index.tsv.gz) · [全批核验](context_appendix_41e4177e_receipt.json) |

## 找到一个决策

从仓库根目录执行以下命令。第一条打开 Master5s 在第 142 步运货时的决策背景，也取这个采集工此前的可见轨迹及每格最后一次被看到的碳量。第二条显式取行为和终局标签。第三条从补编分卷取一个席决策。

```bash
python reports/sapling_1200k/all_top5/context.py 01b56d406361dbb513d6d7417dc0184f-1 142 0 --memory --entity player-0-worker-36
python reports/sapling_1200k/all_top5/context.py 011008e4dee32c2fdbfa37044c3c2608-1 165 1 --labels actions outcome
python reports/sapling_1200k/all_top5/context.py --manifest reports/sapling_1200k/all_top5/appendix_499886f1/manifest.json 0
```

每个地址由批次、比赛 ID、动作前步号 `t`、物理席 `0/1` 组成。也可使用批内连续整数 `ordinal_begin + 2*t + seat`。索引为每局保存半开区间，`Corpus.locate()` 与 `Corpus.ordinal()` 可以互转。第 0 帧是开局观察。若该局有 T 次推进，最后一个决策为 T−1，归档第 T 帧只供终局标签读取。补编有独立序号空间。 Python 调用方也可用 `Corpus(index=索引路径)` 读取成绩卡身份。程序按索引文件的 SHA-256 匹配对应批次的核验收据，原样复制或改名后仍可定位。

```python
from reports.sapling_1200k.all_top5.context import Corpus

corpus = Corpus()
game = corpus.open("01b56d406361dbb513d6d7417dc0184f-1")
row = game.decision(142, 0, lookbacks=(3, 5, 10), memory=True)
public_state = row["at"]
commands = game.labels(142, 0, ["actions"])["joint_commands_from_t"]
trail = game.entity(142, 0, "player-0-worker-36")
for observation in game.training_rows(0):
    features = observation["at"]
```

## 出手时能读到什么

| 区域 | 具体字段与时间 |
| --- | --- |
| 精确观察 | `at.observation` 按回放的 `specification.observation` 从物理席0补齐 `shared` 字段，并保留本席私有字段。包括步号、席号、本席剩余加时、完整双方资产、所有公开树记录、本席225格碳输入。物理席0存储的隐藏全碳图位于诊断标签。 |
| 双方资产 | `at.assets_by_physical_seat[s]` 含现金、每个基地 ID 和位置、每个工人 ID／位置／载货／类型、每棵树 ID／位置／年龄／当前所有者／来源工人字段以及刚结束那一步的吸收收入。实体 ID 可用于串起同局历史。 |
| 可见碳和记忆 | `visibility_mask` 逐格给出可见性。采集工看到自己的格；种树工看到环形棋盘上的四邻格。所有己方工人的视野取并集。`carbon_known_or_null` 保留可见的真零，未知填 null。开启 `--memory` 可获得每格最后一次实际看到的值及步号，未曾看到为 null。 |
| 已发生的行动 | `at.own_previous_command` 是从 t−1 走到 t 的本席原始命令，开局为 null。通过 `entity` 可查询双方已公开实体的位置、载货和树龄轨迹，以及我方在此前各步发送的命令。敌方可见位置的变化可直接观察，同一位置上的不同意图有时产生相同结果。 |
| 布局历史 | 默认给 t−3、t−5、t−10 三个真实观察。早于开局的回溯为 null。开启 `--history` 可取从第0帧到 t−1 的全部本席公开历史；当前观察另放在 `at`。 |
| 规则和时间 | `game_rules` 给出本局尺寸、资源增长、招工、移动和冶炼、工人和树限制、种树与夺树成本、吸收率、树寿命、步限、agent／act／run timeout 等实际规则。`declared_platform_time_limits` 给出归档确有记录的步时、硬顶、慢步次数与初始加时等。当前可用加时读本席观察。原始配置、spec 和 info 可用 `game.metadata()` 读取。天梯来源、对局日期、物理席玩家名、策略 ID 和快照排名可用 `corpus.provenance(gid)` 读取；`--metadata` 同时给出两组资料。可按 t=0 和 t>0 区分开局和后续调用。 |

模型默认输入基于观察和批准的局规。地图种子、环境 UUID、排名、对手名字只在查询地址或单独的归档来源信息中使用。对手的搜索树、私有网络表示、意图和未来路线在这些观察中没有实测字段，研究者预测的目标或路线应另设预测列。[官方视野规则](../../../code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.py) · [官方共享／隐藏字段投影](../../../code/env_src/carbon_challenge-master/zerosum_env/core.py) · [规则定义](../../../code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json)

## 监督与诊断标签

默认 CLI 输出只有 `decision`。若请求其他内容，`labels`、`metadata` 和 `entity_history` 分别位于独立键。标签名 `actions` 读取归档第 t+1 帧的双席联合命令；`next` 读取推进后本席的下一观察；`outcome` 读取两席官方终局奖励、状态与本席官方分差；`full_carbon` 取 t 帧真实全图；`opponent_private` 取 t 帧对方独有碳视野与加时；`runtime` 原样返回 t+1 两席的归档 info，空对象表示没有记录。现金终局、灭绝奖励和当前盘面现金可以独立查询。网络得分、搜索 Q、预期回报以及决策者的意图均没有随动作记录。

这 1,347 局都记录了平台限时配置。2,694 个开局席调用和 800,076 个后续席调用的逐席运行时 info 全为空，本批 0 局在平台计时字段中单列初始化时限。可以恢复每席当时所余加时和注册限时，无法从这些回放恢复真实冷启动耗时、逐步 CPU／墙钟耗时或缓存冷热。

## 三个真实决策点

| 用例 | 当时可见的背景 | 此后实际发生 | 回放与经济证据 |
| --- | --- | --- | --- |
| Master5s 运货，t142，席0 | 查询采集工 `player-0-worker-36` 的位置、货量、两席队形和基地附近的威胁，可回看此前 3／5／10 步。 | 选择 RIGHT，下一步入账 454.83。 | [逐局记录](economy_games/01b56d406361dbb513d6d7417dc0184f-1.md) |
| Master6 截获，t165，席1 | 可查看双方采集工 `player-1-worker-42` 和 `player-0-worker-24` 的位置、载货与此前轨迹。 | 己方选择 RIGHT，碰撞获得 142.8573 货物，之后全部存入基地。 | [逐局记录](economy_games/011008e4dee32c2fdbfa37044c3c2608-1.md) |
| 种树，t19，席1 | 己方现金 32.17，种树工 `player-1-worker-3` 的位置、周邻可见碳与敌我树分布均可取。 | 该工人留在原地，下一帧在第215格出现己方树 `player-1-tree-0`，树龄1。 | [逐局记录](economy_games/003aed5cd8b63c5c4d82dc67e7dd78f2-1.md) |

## 完整性核验

逐局从仓内卷或历史 Git blob 读回并复算压缩字节 SHA-256，遍历全部决策地址。核验包括双席身份、共享字段、双方动作必须在第 t+1 帧且命令只能指向动作前已经存在的工人或基地、末帧请求拒绝、225格掩码逐格与隐藏真值和所录私有碳向量对齐。独立引擎核验的逐局证书覆盖每一局和每次推进。本文的新增扫描不再模拟这些比赛。核心批有 770,700 个席决策同时包含“已看见的零碳”和“未看见的格”，两次补编分别有 4,579 和 8,707 个。

泄漏反例保持动作前可见历史相同，替换地图 seed、UUID、对手标签、对手私有观察和历史命令、隐藏全碳图、t+1 联合命令和资产、终局奖励。默认输入、完整历史、碳记忆和自己单位的历史均逐字段不变；单独请求动作及终局标签时能读到对应替换值。检查还覆盖双席开局、两席最后一次有效出手和正式止步。生成与核验入口分别为 [context_catalog.py](context_catalog.py) 与 [context_audit.py](context_audit.py)。
