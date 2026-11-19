# 前五回放追加核验

从 `39a745e0` 的 1,440 局查到 `cbd6d545`。19 份成绩索引新增 44 次引用，本批跨索引重报 12 次。合并后有 32 个对局 ID，其中 3 个此前已归档，4 个双方均在冻结前五之外，实增前五 25 局。累计 1,465 局，互打 116 局、对外 1,349 局。[逐条去重与索引](score_delta.json) · [原始回放身份](manifest.json)

RyanX-AIPowercore-M9XJ-630 席 0 为 1 胜、0 平、0 负，席 1 为 0 胜、0 平、0 负；RyanX-AIPowercore-it720-preview 席 0 为 1 胜、0 平、0 负，席 1 为 0 胜、0 平、0 负；claude-AlphaCarbon-Master5s 席 0 为 3 胜、0 平、0 负，席 1 为 6 胜、0 平、0 负；claude-AlphaCarbon-Master6 席 0 为 3 胜、0 平、0 负，席 1 为 11 胜、0 平、1 负。

|前五策略 / 席|对手|终局现金（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|
|---|---|---:|---|---:|---:|---:|
| [claude-AlphaCarbon-Master5s / 0](036aa16163420d20ad7d70bb92f48fdc-1.md) | 哈基米基准 | 8,770.41 / 7,470.35 | t277–287 | +1,445.65 | 2,329.89 / 889.36 | 55.65 / 50.54 |
| [claude-AlphaCarbon-Master6 / 0](0f1196d2b1d3292cc9d4192e62f2d889-1.md) | Krypton-Pro | 8,663.94 / 6,087.49 | t219–229 | +726.51 | 561.45 / 0.00 | 216.50 / 55.62 |
| [claude-AlphaCarbon-Master6 / 1](0f685e7891c4040ca35e0bf23fea9581-1.md) | RyanX-AIPowercore-Lite-V8 | 10,386.38 / 14.67 | t289–299 | +1,615.50 | 1,518.11 / 0.00 | 97.39 / 0.00 |
| [claude-AlphaCarbon-Master6 / 0](18be79a0eab5ec97400502e5ce148471-1.md) | Fable5.2-Max-V6 | 8,662.00 / 6,663.72 | t147–157 | -701.95 | 49.36 / 714.60 | 109.42 / 62.72 |
| [claude-AlphaCarbon-Master6 / 1](1e41e65ef797b70e54643cb43fee6412-1.md) | 哈基米 v3 | 2,302.62 / -139.00 | t114–124 | +482.67 | 299.70 / 0.00 | 283.46 / 0.00 |
| [claude-AlphaCarbon-Master6 / 1](39afd54331a072dc9142d8e09e64b29b-1.md) | Krypton-M144Search-v1 | 9,604.80 / 653.68 | t212–222 | +1,199.07 | 1,011.62 / 30.00 | 297.18 / 62.67 |
| [claude-AlphaCarbon-Master6 / 1](4f70b78d13e848d1f4a2dd1c4b4ed19e-1.md) | Fable5.2-Max-V7 | 11,929.36 / 73.56 | t270–280 | +1,423.71 | 1,180.48 / 0.00 | 304.50 / 0.00 |
| [claude-AlphaCarbon-Master5s / 1](5673ce137d0a2ea79addd2cbc7ed37ea-1.md) | Fable5.2-Max-V8 | 7,591.99 / 5,277.14 | t213–223 | +804.39 | 649.04 / 0.00 | 188.31 / 52.98 |
| [claude-AlphaCarbon-Master6 / 1](65364a63078232db415aadc49ae5b18a-1.md) | stephen-K4TacSearch-v2 | 11,376.10 / 304.25 | t289–299 | +1,715.76 | 1,507.45 / 0.00 | 208.32 / 0.00 |
| [claude-AlphaCarbon-Master6 / 1](6ad6ab37a8caa9130639a298ee11d02a-1.md) | Echo4 | 10,955.67 / 1,814.31 | t271–281 | +965.78 | 700.52 / 0.00 | 359.74 / 66.74 |
| [claude-AlphaCarbon-Master5s / 1](7cc903c193162e6265cbd0d962790b76-1.md) | stephen-K4TacSearch-v6 | 6,043.13 / 5,052.61 | t217–227 | +688.97 | 668.89 / 202.83 | 167.78 / 130.54 |
| [RyanX-AIPowercore-it720-preview / 0](8d06052179a60bcc71ab86269cea85b6-1.md) | claude-AlphaCarbon-Master6 | 5,819.14 / 5,580.43 | t92–102 | +711.04 | 773.62 / 115.53 | 115.76 / 72.82 |
| [claude-AlphaCarbon-Master6 / 1](8f7e0aa71adcc9d6acf7350ded77c558-1.md) | 安生N23 | 9,642.45 / 8,604.81 | t289–299 | +1,138.78 | 1,082.10 / 271.21 | 265.16 / 48.23 |
| [claude-AlphaCarbon-Master5s / 0](9ae087ee9e6c57354a3ff58b7e7eac26-1.md) | Fable5.2-Max-V9 | 14,055.79 / 236.14 | t192–202 | +1,020.61 | 871.93 / 0.00 | 363.85 / 0.00 |
| [claude-AlphaCarbon-Master6 / 1](9c386ed1f990fd40416a59c17c50de6a-1.md) | Krypton-CS-Plant-v1 | 6,213.44 / 4,815.35 | t197–207 | +1,017.99 | 922.47 / 75.84 | 71.21 / 148.44 |
| [claude-AlphaCarbon-Master6 / 1](aea1509742046a614fce5294446cdb3a-1.md) | 安生N27 | 10,188.59 / 1,847.50 | t90–100 | +1,016.19 | 720.87 / 0.00 | 466.04 / 30.04 |
| [RyanX-AIPowercore-M9XJ-630 / 0](b5e17029acf92f5e6fab63f848d054ac-1.md) | 安生N33 | 8,241.56 / 6,662.15 | t155–165 | +731.55 | 582.97 / 66.25 | 165.18 / 71.63 |
| [claude-AlphaCarbon-Master6 / 1](c587b390755d39ba4b41b356ee7b7c17-1.md) | 小肥鱼v5-late1 | 6,933.44 / 6,767.54 | t91–101 | +901.95 | 1,011.84 / 271.73 | 2.78 / 351.19 |
| [claude-AlphaCarbon-Master5s / 0](d3f70abc463e8fe234a66b0152a6b47d-1.md) | Fable5.2-Max-V7 | 7,866.72 / 5,696.48 | t196–206 | +714.81 | 561.29 / 0.00 | 372.02 / 100.76 |
| [claude-AlphaCarbon-Master6 / 0](d52ec5edd86c0e2be03394fc814941c4-1.md) | claude-AlphaCarbon-Master3s | 9,759.63 / 1,515.99 | t270–280 | +1,186.45 | 1,045.05 / 38.25 | 32.25 / 217.59 |
| [claude-AlphaCarbon-Master5s / 1](dd3ab1629f52192b6f2d332d99cc367e-1.md) | 安生N30 | 9,194.89 / 7,502.99 | t82–92 | +1,121.59 | 1,273.93 / 176.34 | 43.88 / 140.54 |
| [claude-AlphaCarbon-Master6 / 1](e12b395124fb7d8f976adbce3a496ad5-1.md) | Krypton-Pro | 8,800.30 / 5.50 | t289–299 | +1,362.91 | 1,198.40 / 0.00 | 164.51 / 0.00 |
| [claude-AlphaCarbon-Master5s / 1](ecf672c3a0a68c2ccf2357791cc632d2-1.md) | 哈基米 v3 | 7,300.13 / 4,303.51 | t246–256 | +868.85 | 694.43 / 0.00 | 147.11 / 76.10 |
| [claude-AlphaCarbon-Master5s / 1](eedd89a100270eb131fed1074dbcf04a-1.md) | Echo3 | 8,797.81 / 4,967.15 | t207–217 | +1,201.84 | 825.09 / 0.00 | 209.56 / 142.33 |
| [claude-AlphaCarbon-Master5s / 1](f9c3e9c07e27ab31b729ed0b82a5541c-1.md) | ClaudeFable5.2-v8 | 6,320.71 / 5,133.61 | t140–150 | +688.12 | 703.46 / 26.41 | 148.12 / 32.93 |

每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)

官方 Board 和 FastCarbon 各复算 7,338 次联合推进，25 场全部吻合。每席行动前上下文复查包括 14,676 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 5,785,600 字节，SHA-256 `317833060aa2efedc4f67f199bf5aad14b86f18ecbfcb2b8724abe96fc66e0b4`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文](../context_appendix_cbd6d545_receipt.json) · [完整无损回放](../../../../data/sapling_all_top5_20261119/appendix_cbd6d545.tar)

同名 Krypton-Pro 在两张地图分别得 6,087.49 和 5.50，Master6 另以 238.71 分负于 it720。三局并列核对首工、实收、招工和关键时段的双方命令。[逐局现金与动作对照](contrasts.md) · [未来 10/30 步实收标签核验](future_economy_check.json)
