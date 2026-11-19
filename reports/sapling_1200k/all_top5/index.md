# 天梯前五逐局索引

共 1,324 个独立对局 ID，其中前五互打 110 局，对外 1214 局。五个选手视图合计 1,434 条，因为互打局在两位选手名下都可查到。下载表每个 ID 只有一行，保留两物理席。

| 排名 | 选手 | 对局数 | 胜 / 平 / 负 | 页码 |
| ---: | --- | ---: | --- | --- |
| 1 | claude-AlphaCarbon-Master5s | 220 | 206 / 0 / 14 | [1](index/rank1-p01.md) · [2](index/rank1-p02.md) · [3](index/rank1-p03.md) · [4](index/rank1-p04.md) |
| 2 | claude-AlphaCarbon-Master6 | 269 | 247 / 1 / 21 | [1](index/rank2-p01.md) · [2](index/rank2-p02.md) · [3](index/rank2-p03.md) · [4](index/rank2-p04.md) · [5](index/rank2-p05.md) |
| 3 | RyanX-AIPowercore-it720-preview | 270 | 239 / 0 / 31 | [1](index/rank3-p01.md) · [2](index/rank3-p02.md) · [3](index/rank3-p03.md) · [4](index/rank3-p04.md) · [5](index/rank3-p05.md) |
| 4 | RyanX-AIPowercore-M9XJ-630 | 311 | 219 / 0 / 92 | [1](index/rank4-p01.md) · [2](index/rank4-p02.md) · [3](index/rank4-p03.md) · [4](index/rank4-p04.md) · [5](index/rank4-p05.md) · [6](index/rank4-p06.md) |
| 5 | ClaudeFable5.2-v7 | 364 | 296 / 0 / 68 | [1](index/rank5-p01.md) · [2](index/rank5-p02.md) · [3](index/rank5-p03.md) · [4](index/rank5-p04.md) · [5](index/rank5-p05.md) · [6](index/rank5-p06.md) · [7](index/rank5-p07.md) |

## 后续补编

[新增八局](appendix_499886f1/README.md)和[最后十五局](appendix_4bfdabf5/README.md)逐局附相同的原始回放、资金流和出手背景，三批合计 1,347 局。

## 数据与口径

[逐局双席 TSV 压缩表](index_games.tsv.gz) · [来源与哈希](index_sources.json) · [逐帧清单](manifest.json)

打开选手分页，点对局 ID 看天梯回放，点“经济与回溯”看该局收入明细及转折前的布局。“防0/防1”跳到物理席防守统计。缺少证据的局保留在索引中，空值不得当作零。

逐局证据覆盖为 齐全 1324 局。独立验证状态为 consistent 1324 局。

转折是整局绝对值最大的连续十步席0现金领先变化；短于十步的对局取全段。到达转折起点前 3、5、10 步的资产位置均保存在 TSV。若回溯早于开局则取第0步，保留实际步号。终局分差按官方奖励计算；灭绝时它与最终盘面现金差可能不同。

以 seat0_ 和 seat1_ 开头的列始终对应物理席。收入金额按现金账分类，抢到的货物在实际入账前仍属在途；最后三十步与三段区间存为 JSON。布局资产快照的数组顺序为 cash、collectors、planters、trees、remaining_tree_turns_30、cargo_total、cargo_near_bank、cargo_approach_threatened、blocked_bases。布局快照只记录对局实际到达的步数。

[11:56 同步的首批 10 局](appendix_555c0c56/README.md) · [04:20 UTC 截止补齐的 4 局](appendix_bf049043/README.md) · [全量 1,361 张图说明](README.md)

[收盘后 Master6 主索引新增 12 局](appendix_bd3c0021/README.md) · [六批全部 1,373 局](README.md)

[补编五张图及逐局分析](appendix_5484590b/README.md)

[新增四局逐场资金与布局](appendix_99919cbf/README.md) · [累计 1,382 张唯一地图的双引擎核验](appendix_99919cbf/validation.md)

[截至 0137949d 的 Master5s 对外四局及逐步收支](appendix_0137949d/README.md) · [累计 1,386 局](appendix_0137949d/validation.md)

[截至 41e4177e 的 16 场新对外局及行动追溯](appendix_41e4177e/README.md) · [累计 1,402 张唯一地图、两种引擎核分及正式考试隔离](appendix_41e4177e/validation.md)

[Master5s 新四场，对手含 Echo6 和 Echo5](appendix_f7109df6/README.md) · [Master6 对 Fable v8 新局](appendix_1fdacb1b/README.md)

[截至 32b555ee 新增 14 场 Master6 对外比赛，查每场转折和此前行动](appendix_32b555ee/README.md)

[截至 c5c8cf8d 的 11 局新增对外实战，含双方资金、攻防动作和转折前布局](appendix_c5c8cf8d/README.md)

[截至 39a745e0 新增的 Master6 对外八局，含两引擎复算与每局双方的现金转折](appendix_39a745e0/README.md)

[截至 cbd6d545 的 25 场新增回放](appendix_cbd6d545/README.md) · [Master6 的败局与同一对手两种收入结局](appendix_cbd6d545/contrasts.md)
