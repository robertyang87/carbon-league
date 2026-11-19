# 前五回放追加核验

从 `cbd6d545` 的 1,465 局查到 `336e8b8c`。19 份成绩索引新增 31 次引用，本批跨索引重报 1 次。合并后有 30 个对局 ID，其中 4 个此前已归档，16 个双方均在冻结前五之外，实增前五 10 局。累计 1,475 局，互打 117 局、对外 1,358 局。[逐条去重与索引](score_delta.json) · [原始回放身份](manifest.json)

RyanX-AIPowercore-it720-preview 席 0 为 0 胜、0 平、1 负，席 1 为 0 胜、0 平、0 负；claude-AlphaCarbon-Master5s 席 0 为 2 胜、0 平、0 负，席 1 为 3 胜、0 平、0 负；claude-AlphaCarbon-Master6 席 0 为 3 胜、0 平、1 负，席 1 为 1 胜、0 平、0 负。

|前五策略 / 席|对手|终局现金（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [RyanX-AIPowercore-it720-preview / 0](16bc7ee748afb9192047bab0ceaf7286-1.md) | claude-AlphaCarbon-Master5s | 6,972.15 / 8,026.00 | t133–143 | -810.50 | 92.09 / 930.44 | 123.55 / 57.86 |
| [claude-AlphaCarbon-Master6 / 0](1783e381db10df78e87bf12117d23ecf-1.md) | master3 | 7,406.00 / 6,427.55 | t185–195 | +1,006.57 | 1,277.23 / 357.71 | 120.90 / 85.12 |
| [claude-AlphaCarbon-Master5s / 0](2e1b4d8f42d17f9153496364fba9010b-1.md) | 安生N27 | 8,448.40 / 6,654.57 | t288–298 | +985.85 | 1,121.63 / 397.34 | 292.77 / 1.21 |
| [claude-AlphaCarbon-Master6 / 0](3ab7424e471a6b51563317ea9f18fa5a-1.md) | Makabaka-v3 | 11,319.76 / 6.80 | t212–222 | +1,299.36 | 1,200.76 / 0.00 | 187.57 / 0.00 |
| [claude-AlphaCarbon-Master5s / 0](4e964046201ad2bd02d80942c63ea4cd-1.md) | 安生N23 | 10,023.98 / 4,240.31 | t117–127 | +1,156.65 | 1,001.26 / 102.60 | 269.89 / 0.00 |
| [claude-AlphaCarbon-Master6 / 0](54cc67fd830e4be31c48927b96b9d900-1.md) | ClaudeFable5.2-v8 | 5,978.91 / 6,217.43 | t83–93 | -773.64 | 196.58 / 765.93 | 48.04 / 181.34 |
| [claude-AlphaCarbon-Master5s / 1](79c54825a0a6d384454413cbded7c470-1.md) | 测001 | 7,755.56 / 6,234.03 | t189–199 | +1,192.12 | 1,046.08 / 40.10 | 88.29 / 26.29 |
| [claude-AlphaCarbon-Master6 / 1](8bec7413cfddb5c932e130f8ed97aeb5-1.md) | 哈基米基准 | 10,686.73 / 2,190.31 | t191–201 | +1,393.58 | 1,247.97 / 0.00 | 299.53 / 0.00 |
| [claude-AlphaCarbon-Master5s / 1](c93515fede77d9434fe22f4e82fd018c-1.md) | First111 | 8,493.65 / 5,872.90 | t250–260 | +633.02 | 740.06 / 300.15 | 176.20 / 65.30 |
| [claude-AlphaCarbon-Master6 / 0](ddb7d7b02fd6bf7ff172a95f0106b6eb-1.md) | 哈基米 pro2 | 6,985.30 / 3,156.97 | t149–159 | +967.40 | 973.48 / 111.86 | 20.95 / 195.92 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 2,990 次联合推进，10 场全部吻合。每席行动前上下文复查包括 5,980 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。每场真实 225 格开局均已单独登记，供考场清单下一次更新核入。归档文件 2,396,160 字节，SHA-256 `c9b739f9423359326a05d7c3f777ee95f166eecc06d9e4667e32ea3c54625a10`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_336e8b8c_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_336e8b8c.tar)

新败局 Master6 在 t296 仍领先 238.05，ClaudeFable5.2-v8 的两名采集工下一步分别交回 188.90 和 198.56，从 t297 起领先至终局。两支策略上一次同席对局是 Master6 净胜 8,590.76。八组有历史交手的同名对照、两组逐工回溯与现金账见附页。[同名对手与失利回溯](contrasts.md) · [双工回款的库外拦截](terminal_intercept.md) · [地图排除候选清单](map_registry_candidate.json) · [未来 10/30 步监督覆盖](future_economy_check.json)
