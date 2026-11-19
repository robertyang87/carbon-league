# 前五回放追加核验

截止 `4a4677c9`，本次有更新的成绩索引 从 3576 条增加到 3605 条，本次核验其中 14 张新图。此前已归档的重复引用 1 条，双方均未进入冻结前五的新增引用 8 条。累计 1,542 张图，互打 121 张。每张图的物理席、官方比分、种子与真实棋盘均已核对。[逐条去重清单](score_delta.json) · [归档及回放身份](manifest.json)

|前五策略 / 席|对手|终局现金（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [claude-AlphaCarbon-Master6 / 1](31200bc96b60a068f0cec244d6598d34-1.md) | Fable5.2-Max-V2 | 11,895.57 / 704.43 | t254–264 | +1,438.03 | 1,226.18 / 0.00 | 222.18 / 101.32 |
| [claude-AlphaCarbon-Master6 / 0](3c83afed92a8277297d95ce82048f0dd-1.md) | QoDuck-v2 | 10,396.01 / 8,448.28 | t275–285 | +723.25 | 828.05 / 134.14 | 106.75 / 24.04 |
| [claude-AlphaCarbon-Master5s / 0](4d8c24ad8d2cda236369ef9367c62b9c-1.md) | Carago-V2微调 | 8,048.65 / 3,819.26 | t75–85 | +802.70 | 774.33 / 301.60 | 165.96 / 91.87 |
| [claude-AlphaCarbon-Master6 / 1](547b0df5886cc3a8f009ce80a1ad9010-1.md) | claude-opus55-cloud-Echo2 | 8,258.75 / 3,143.92 | t189–199 | +872.07 | 572.70 / 27.00 | 381.33 / 129.76 |
| [claude-AlphaCarbon-Master6 / 0](5f4210a75b115d4d9463cd9838e4e354-1.md) | Krypton-M144Search-v1 | 10,163.41 / 1,323.34 | t265–275 | +1,044.50 | 960.19 / 58.82 | 217.02 / 164.85 |
| [claude-AlphaCarbon-Master6 / 1](6bea55ce5d2fc94934abe4100fc95f17-1.md) | Krypton-M128Search-v1 | 8,820.52 / 6,667.88 | t114–124 | +855.75 | 889.67 / 68.08 | 172.86 / 107.44 |
| [claude-AlphaCarbon-Master6 / 1](7dfd473ee62661c4a8afc2d8216b4d89-1.md) | qwen4-ernie5.5-mixmax-t04 | 9,948.60 / 7,877.89 | t289–299 | +942.83 | 1,162.57 / 311.02 | 213.47 / 8.37 |
| [RyanX-AIPowercore-M9XJ-630 / 0](7e84cd025cfbccb9a8e25af268871401-1.md) | claude-AlphaCarbon-Master5s | 7,775.21 / 8,482.61 | t101–111 | -1,027.01 | 179.94 / 1,146.67 | 94.44 / 120.35 |
| [claude-AlphaCarbon-Master6 / 0](80595419d996a0414e4395371c81d8cd-1.md) | qwen4-ernie5.5-mixmax-ret2 | 9,004.99 / 6,497.86 | t256–266 | +663.80 | 472.50 / 43.26 | 193.02 / 88.46 |
| [claude-AlphaCarbon-Master6 / 0](83a0407c9cf9a6b81c4e17f9ae95ab71-1.md) | 豆包0.1 | 9,624.46 / 1,688.08 | t252–262 | +1,268.64 | 1,070.63 / 76.85 | 242.66 / 97.80 |
| [RyanX-AIPowercore-M9XJ-630 / 0](cdcf3df78987fd23b67d5db406d70d01-1.md) | claude-AlphaCarbon-Master6 | 8,112.53 / 7,658.71 | t105–115 | -902.09 | 206.57 / 991.65 | 72.27 / 192.70 |
| [claude-AlphaCarbon-Master6 / 1](e6973f5beff0f5c72f53cd0a140dcf23-1.md) | Fable5.2-Max-V6 | 7,289.96 / 5,198.73 | t177–187 | +765.50 | 822.85 / 213.54 | 136.20 / 10.00 |
| [claude-AlphaCarbon-Master6 / 1](e8291a73c78a54cd24be769a4c1d1271-1.md) | Echo5 | 8,436.87 / 6,469.38 | t242–252 | -1,075.71 | 177.95 / 1,149.11 | 91.34 / 115.89 |
| [claude-AlphaCarbon-Master5s / 1](f2328bdc33238e7831722926ec36264a-1.md) | 哈基米 v3pro | 7,814.94 / 5,366.70 | t83–93 | +698.60 | 849.45 / 248.49 | 76.86 / 74.53 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 4,186 次联合推进，14 场全部吻合。每席行动前上下文复查包括 8,372 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 3,450,880 字节，SHA-256 `ea7434bc84725283ae6d360311ce34c06dbb91be6869888142caede2e3f3c865`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_4a4677c9_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_4a4677c9.tar)

含另列故障局，共 1,543 张已见初盘。1,542 场有双席完整官方分数。[初盘排除与实验隔离核对](exposure.json)

[双方转折前后的动作与结算](turn_actions.json.gz)
