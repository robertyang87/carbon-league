# 前五回放追加核验

截止 `aca9b8a3`，本次有更新的成绩索引 从 1424 条增加到 1434 条，本次核验其中 7 张新图。此前已归档的重复引用 1 条，双方均未进入冻结前五的新增引用 2 条。累计 1,528 张图，互打 119 张。每张图的物理席、官方比分、种子与真实棋盘均已核对。[逐条去重清单](score_delta.json) · [归档及回放身份](manifest.json)

|前五策略 / 席|对手|终局现金（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [claude-AlphaCarbon-Master6 / 1](1a1fb2a9bbcbc85ac932649b15cf2d4c-1.md) | lin-CarbonMaster-v1 | 9,418.30 / 2,367.59 | t238–248 | +1,372.25 | 1,025.18 / 0.00 | 347.09 / 0.00 |
| [claude-AlphaCarbon-Master6 / 0](639ca12035541df4268615cdaede25fd-1.md) | 111.1 | 7,996.16 / 5,523.00 | t162–172 | +920.48 | 848.89 / 82.93 | 161.00 / 124.23 |
| [claude-AlphaCarbon-Master6 / 1](6a8fafbd549ae7c188e18ccdf307a430-1.md) | claude-opus55-cloud-Grove1 | 10,982.86 / 0.46 | t246–256 | +1,208.31 | 1,030.08 / 0.00 | 178.21 / 0.00 |
| [claude-AlphaCarbon-Master6 / 0](a4dcd21bd0bf2ebe78838ffc97fc6221-1.md) | carsonsafe | 9,557.90 / 7,308.36 | t172–182 | +997.21 | 936.74 / 0.00 | 165.30 / 51.40 |
| [claude-AlphaCarbon-Master6 / 0](ac0470cdb4f4882221bc1a0f439f5a10-1.md) | Fable5.2-Max-V4 | 7,370.84 / 5,835.38 | t94–104 | +856.85 | 966.93 / 215.25 | 90.51 / 123.11 |
| [claude-AlphaCarbon-Master5s / 1](c6324652b9f049fdaebad126276f182d-1.md) | Krypton-CS-Plant-v1 | 10,712.06 / 3,553.89 | t106–116 | +1,195.31 | 1,014.55 / 25.51 | 354.12 / 5.92 |
| [claude-AlphaCarbon-Master6 / 0](f510998f38085bb3f8ca29ac2cb25184-1.md) | Fable5.2-Max-V4 | 6,599.28 / 5,452.06 | t282–292 | -687.41 | 0.00 / 608.23 | 100.75 / 106.29 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 2,093 次联合推进，7 场全部吻合。每席行动前上下文复查包括 4,186 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 1,658,880 字节，SHA-256 `d3886500b16d97bc60f02cd6b1a51fe66595c1f2aeb190ce33f8e91c6a75917a`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_aca9b8a3_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_aca9b8a3.tar)

含另列故障局，共 1,529 张已见初盘。1,528 场有双席完整官方分数。[初盘排除与实验隔离核对](exposure.json)

[双方转折前后的动作与结算](turn_actions.json.gz)
