# 前五回放追加核验

截止 `559ba262`，本次有更新的成绩索引从 6739 条增加到 6814 条，本次核验其中 4 张新图。此前已归档的重复引用 2 条，双方均未进入冻结前五的新增引用 69 条。累计 1,624 张图，互打 121 张。每张图的物理席、官方比分、种子与真实棋盘均已核对。[逐条去重清单](score_delta.json) · [归档及回放身份](manifest.json)

|前五策略 / 席|对手|官方终局分（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [RyanX-AIPowercore-it720-preview / 1](6a0a524f9c52f3509316dbe00bb26e36-1.md) | RyanX-AIPowercore-it920-evo31 | 4,716.98 / 6,437.91 | t78–88 | -684.99 | 0.00 / 738.48 | 100.03 / 71.45 |
| [claude-AlphaCarbon-Master6 / 0](90bf3ab81d9c6b07133627b2f0d50cd9-1.md) | RyanX-AIPowercore-it920-evo31 | 6,094.23 / 7,136.91 | t268–278 | -919.46 | 0.00 / 946.31 | 117.86 / 83.95 |
| [claude-AlphaCarbon-Master6 / 1](bb517d5c62548d43044ec9335c537096-1.md) | RyanX-AIPowercore-it920-evo31 | 6,626.45 / 7,097.42 | t246–256 | -1,083.10 | 51.00 / 1,093.06 | 206.95 / 67.31 |
| [RyanX-AIPowercore-it720-preview / 0](bfc2127bb6741f405cd0f3a6cb88f6b2-1.md) | RyanX-AIPowercore-it920-evo31 | 6,193.98 / 7,317.53 | t115–125 | +695.76 | 849.27 / 136.48 | 107.74 / 124.76 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 1,196 次联合推进，4 场全部吻合。每席行动前上下文复查包括 2,392 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 1,013,760 字节，SHA-256 `a96de0e24d3c391c8badebee7b9f376f160f099f2bff1ccde437901d1cb12424`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_559ba262_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_559ba262.tar)

已见初始棋盘登记组织为独立考试的排除项。每个回放的两个席位与本批身份关系、指令和真实结算可按步复算。[地图登记](exposure.json) · [转折前后实际动作](turn_actions.json.gz)

本批 Master6 另有逐工时机复核。[两场决策对照](Master6.md)
