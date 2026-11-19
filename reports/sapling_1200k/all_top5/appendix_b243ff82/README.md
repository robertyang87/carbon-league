# 前五回放追加核验

截止 `b243ff82`，本次有更新的成绩索引 从 672 条增加到 685 条，本次核验其中 2 张新图。此前已归档的重复引用 1 条，双方均未进入冻结前五的新增引用 10 条。累计 1,629 张图，互打 121 张。每张图的物理席、官方比分、种子与真实棋盘均已核对。[逐条去重清单](score_delta.json) · [归档及回放身份](manifest.json)

|前五策略 / 席|对手|官方终局分（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [ClaudeFable5.2-v7 / 0](67a8bebd8c22430a7fc0905220bd7671-1.md) | RyanX-AIPowercore-it920-evo31 | 4,367.46 / 5,033.20 | t183–193 | -800.33 | 0.00 / 714.68 | 145.92 / 34.74 |
| [claude-AlphaCarbon-Master5s / 0](bea5046421b817a05fc98012bb1e403e-1.md) | RyanX-AIPowercore-it920-evo31 | 6,569.70 / 6,242.00 | t105–115 | -985.53 | 0.00 / 1,013.75 | 109.20 / 70.97 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 598 次联合推进，2 场全部吻合。每席行动前上下文复查包括 1,196 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 501,760 字节，SHA-256 `7897783ebb7cabca3cdb5da6a5e0482a48de885c18b81352f861ca58e8b80376`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_b243ff82_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_b243ff82.tar)

Fable v7 的对手从第 64 步起持续领先，终场领先 665.74。Master5s 在第 105–115 步同样遇到对手集中回库，随后在第 251 步取得持续到终局的领先，最终赢 327.70。两局来自不同地图。[Fable v7 逐局账目](67a8bebd8c22430a7fc0905220bd7671-1.md) · [Master5s 逐局账目](bea5046421b817a05fc98012bb1e403e-1.md)

两场均由冻结前五策略执物理席 0，对手均为 RyanX-AIPowercore-it920-evo31。另有 1 局官方异常，合计 1,630 个已见初盘。[初盘排除核对](exposure.json) · [逐步实发命令与结算](turn_actions.json.gz) · [十步与三十步收入口径](future_economy_check.json)

在包含本批、截至本轮的归档检出中运行 `python reports/sapling_1200k/all_top5/appendix_b243ff82/prepare.py` 可重建证据。程序优先读取已发布的原始回放。
