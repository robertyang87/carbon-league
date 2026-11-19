# 前五回放追加核验

从 `c5c8cf8d` 的 1,432 局查到 `39a745e0`。3 份成绩索引新增 14 次引用，本批跨索引重报 1 次。合并后有 13 个对局 ID，其中 5 个此前已归档，0 个双方均在冻结前五之外，实增前五 8 局。累计 1,440 局，互打 115 局、对外 1,325 局。[逐条去重与索引](score_delta.json) · [原始回放身份](manifest.json)

claude-AlphaCarbon-Master6 席 0 为 2 胜、0 平、0 负，席 1 为 6 胜、0 平、0 负。

|前五策略 / 席|对手|终局现金（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [claude-AlphaCarbon-Master6 / 1](00994c0e4daf1bca4f7aaa78cf7d96b9-1.md) | 安生N29 | 7,799.88 / 6,419.41 | t259–269 | -715.42 | 247.62 / 919.87 | 74.80 / 117.99 |
| [claude-AlphaCarbon-Master6 / 1](1ee91d8225a8d3dd32cb7f809ddc7b7e-1.md) | 安生N30 | 8,863.71 / 7,361.47 | t268–278 | -777.61 | 212.87 / 796.28 | 158.56 / 159.00 |
| [claude-AlphaCarbon-Master6 / 0](21357fd6d95c1c8ff1038497c036a184-1.md) | QoDuck-v2 | 7,234.46 / 5,782.53 | t122–132 | +633.94 | 649.87 / 51.00 | 182.86 / 164.72 |
| [claude-AlphaCarbon-Master6 / 0](442d56c47213ea11aa07793caa072eaa-1.md) | claude-AlphaCarbon-Master3s | 9,369.70 / 1,969.28 | t138–148 | +1,009.52 | 720.94 / 0.00 | 314.55 / 5.96 |
| [claude-AlphaCarbon-Master6 / 1](4bd88c2f8bbb0d798c7feb11e4fa4e95-1.md) | Krypton-M3s2-EarlyHome | 10,497.99 / 2,279.49 | t186–196 | +932.07 | 751.41 / 0.00 | 423.56 / 0.00 |
| [claude-AlphaCarbon-Master6 / 1](66d64a030bd301b111c135d42edd25ac-1.md) | 安生N25 | 7,586.33 / 5,151.39 | t273–283 | +850.40 | 853.02 / 145.87 | 148.25 / 190.91 |
| [claude-AlphaCarbon-Master6 / 1](67a205220420a2bd4fa75507e7cc0ad7-1.md) | qwen4-ernie5.5-mixmax-t04 | 10,385.30 / 7,486.96 | t67–77 | +912.10 | 758.81 / 0.00 | 230.61 / 107.32 |
| [claude-AlphaCarbon-Master6 / 1](b61a7ed08cc6acb9d0b35a2d5367aa6a-1.md) | 哈基米基准 | 7,384.42 / 5,681.35 | t261–271 | +667.65 | 610.68 / 47.88 | 137.54 / 115.61 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 2,392 次联合推进，8 场全部吻合。每席行动前上下文复查包括 4,784 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 1,955,840 字节，SHA-256 `4d57a4643ccb89bc1c59f55233f17fbf197a94922c2759aa0fa4f02bcb0e55c1`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_39a745e0_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_39a745e0.tar)
