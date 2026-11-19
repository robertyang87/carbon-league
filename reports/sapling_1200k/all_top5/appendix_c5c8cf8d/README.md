# 前五回放追加核验

从 `32b555ee` 的 1,421 局查到 `c5c8cf8d`。扫过全部冻结索引，10 份成绩索引产生 15 条新增引用，其中 1 条在同一增量内重复。按比赛 ID 与局号去重后，2 个已归档，1 个双方均在冻结前五之外，新增前五 11 局。累计归档 1,432 局，前五互打 115 局、对外 1,317 局。[完整逐条去重与来源](score_delta.json) · [物理席与原件身份](manifest.json)

本批 ClaudeFable5.2-v7 席 0 为 0 胜、0 平、0 负。本批 ClaudeFable5.2-v7 席 1 为 1 胜、0 平、0 负。本批 claude-AlphaCarbon-Master5s 席 0 为 4 胜、0 平、0 负。本批 claude-AlphaCarbon-Master5s 席 1 为 2 胜、0 平、0 负。本批 claude-AlphaCarbon-Master6 席 0 为 2 胜、0 平、0 负。本批 claude-AlphaCarbon-Master6 席 1 为 2 胜、0 平、0 负。

|前五策略 / 席|对手|终局现金（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [claude-AlphaCarbon-Master5s / 0](15ac6798c31ce374cdd4a8ded47a5401-1.md) | Krypton-M144Search-v1 | 6,800.30 / 3,805.69 | t236–246 | +654.17 | 671.68 / 140.36 | 175.61 / 15.70 |
| [ClaudeFable5.2-v7 / 1](32a3f9648b647c91b88bfd9adff310c0-1.md) | 安生N33 | 8,325.64 / 7,077.46 | t288–298 | +863.07 | 867.57 / 296.57 | 434.47 / 0.00 |
| [claude-AlphaCarbon-Master5s / 1](72d5846586ec5b1190973ef92582114a-1.md) | Fable5.2-Max-V7 | 8,105.14 / 5,488.64 | t252–262 | +912.26 | 869.81 / 162.24 | 215.22 / 10.55 |
| [claude-AlphaCarbon-Master6 / 1](7b4eb1f28c73fa493b5a976ad02256d4-1.md) | 小肥鱼v5 | 11,273.55 / 1,430.33 | t215–225 | +953.70 | 658.60 / 102.00 | 350.60 / 0.00 |
| [claude-AlphaCarbon-Master5s / 0](83f447003b198774cc7ccca7e411d834-1.md) | 安生N27 | 8,917.65 / 6,719.81 | t189–199 | +1,024.29 | 1,100.83 / 114.09 | 131.51 / 56.88 |
| [claude-AlphaCarbon-Master6 / 1](8c4c3262ca5e88ad03f74da02bfdcaf7-1.md) | 安生N23 | 9,914.43 / 17.90 | t289–299 | +1,439.89 | 1,323.70 / 0.00 | 116.19 / 0.00 |
| [claude-AlphaCarbon-Master5s / 0](8fdca31919d813dc0383c30f4590d4c3-1.md) | claude-opus55-cloud-Echo2 | 9,092.00 / 6,531.60 | t274–284 | +718.99 | 871.49 / 117.70 | 60.36 / 227.43 |
| [claude-AlphaCarbon-Master6 / 0](9633973b1fb8ba2bde2aa27f2e300235-1.md) | Krypton-CloudSearch-v1 | 9,433.46 / 319.00 | t221–231 | +1,076.77 | 818.19 / 0.00 | 390.81 / 0.00 |
| [claude-AlphaCarbon-Master5s / 0](a2ee1e332429e89145226d87ed393d97-1.md) | 安生N30 | 9,259.87 / 4,928.16 | t120–130 | +1,221.52 | 1,265.23 / 135.99 | 70.56 / 95.33 |
| [claude-AlphaCarbon-Master5s / 1](e08ce71a43a9795b7ab4d39158f43909-1.md) | Echo4 | 10,014.96 / 5,529.78 | t207–217 | +757.97 | 782.23 / 256.86 | 120.96 / 222.96 |
| [claude-AlphaCarbon-Master6 / 0](e6e334a725f9616d9343722afe27bea0-1.md) | 哈基米基准 | 6,168.93 / 5,069.47 | t119–129 | +1,096.80 | 1,123.85 / 179.40 | 224.25 / 127.48 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 3,289 次联合推进，11 场全部吻合。每席行动前上下文复查包括 6,578 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 2,621,440 字节，SHA-256 `8df31a09af53f97cf50478d817c5b3cc5b5ee62e973db203977a9c1e6cd8a8e7`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_c5c8cf8d_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_c5c8cf8d.tar)
