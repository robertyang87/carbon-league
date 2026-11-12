# ladder  2026-11-12 09:06（Mac 掉线，Muse Spark 接手中转；榜单与对局为公开接口实拉）

```
 1 claude-AlphaCarbon-Master3s2          2869 ± 39 103/106 heavy ladder  d679ed45ab595e8b7bb1ea51c3451154
 2 claude-AlphaCarbon-Master3s           2782 ± 30 139/146 heavy ladder  c9859c8e70b08007f11900837a036b3f
 3 stephen-K4TacSearch-v1                2626 ± 24 129/172 heavy ladder  f78b721a1641c088badfc0acf352a272
 4 RyanX-AIPowercore-M6                  2616 ± 24 119/164 heavy ladder  c06029321163eb0a8867815397ae64a3
 5 Carago-V2微调                           2609 ± 24 103/147 heavy ladder  a2670aef3ae1823aa2532fca9edb1c3e
 6 RyanX-AIPowercore-M7Bs-b010           2602 ± 41  27/38  heavy ladder  67a89340f860f030de13ca23d350d203
 7 claude-opus55-cloud-Echo2             2584 ± 24 110/159 heavy ladder  df8c016180803b5b3a2675650c11a7ed
 8 AgyCarbon-P2C-Godslayer-CV            2571 ± 23 112/166 heavy ladder  e8f770e5117b0ae0acea77f70850a032
 9 claude-opus55-cloud-Echo1             2569 ± 23 109/162 heavy ladder  08c61154810a06ddd2fcff37b25fd023
10 claude-AlphaCarbon-Master2s           2551 ± 18 283/348 heavy ladder  77dd0f049ce4e9264ada7274cce158b5
11 Carago-V2                             2529 ± 22 126/190 heavy ladder  c7663d5361ee186df0f3decd7874f730
12 RyanX-AIPowercore-M2                  2509 ± 19 194/278 heavy ladder  436e1c2f3756656fde90f8a47a91f953
```

变化：Master3s2 总对局 106 已过 100；RyanX 新版 M7Bs-b010 空降第 6（27/38）；Fable5.2-Max-V5 第 13（2504 ± 38，24/43）。

## Master3s2 的输局（全部 3 局，05:45 建议要求记录）

- 输 Carago-V2微调 −972
- 输 RyanX-AIPowercore-M6 −599
- 输 RyanX-AIPowercore-M6 −176
- 对 claude-opus55-cloud 全家 27 局全胜（Echo1/Echo2/Grove1/v16b/v17/v17d2/Scout1）。

## 我们的输局

- claude-opus55-cloud-Echo1（对局 162 局）最近输局：RyanX-AIPowercore-M7Bs-b010 -684；claude-AlphaCarbon-Master2 -352；claude-AlphaCarbon-Master3s2 -3326；claude-AlphaCarbon-Master3s2 -2316；AgyCarbon-P2C-Godslayer-CV -141；stephen-K4TacSearch-v1 -963；claude-AlphaCarbon-Master3s2 -1888；Carago-V2微调 -437；claude-AlphaCarbon-Master3s2 -2812；claude-AlphaCarbon-Master3s2 -2711
- claude-opus55-cloud-Echo2（对局 159 局）最近输局：claude-AlphaCarbon-Master3s2 -2612；RyanX-AIPowercore-M7Bs-b010 -21；claude-AlphaCarbon-Master3s2 -1949；RyanX-AIPowercore-M6 -357；stephen-K4TacSearch-v1 -920；claude-AlphaCarbon-Master3s2 -1752；RyanX-AIPowercore-M6 -612；claude-AlphaCarbon-Master2 -49；claude-AlphaCarbon-Master3s2 -2168；Carago-V2 -525

## 回放盘点（本地 top_corpus，Mac 侧 07:07 数，未变）

| 选手 | 局数 |
|---|---|
| claude-AlphaCarbon-Master3s | 143 |
| RyanX-AIPowercore-M6 | 159 |
| stephen-K4TacSearch-v1 | 164 |
| AgyCarbon-P2C-Godslayer-CV | 156 |
| claude-AlphaCarbon-Master2s | 344 |
| claude-opus55-cloud-Echo1 | 157 |
| claude-opus55-cloud-Echo2 | 153 |

## 09:06 接手侧盘点（Muse Spark）

- 公开快照仓（JUZI1/carbon-league-replays，滚动 400 局）已收：目标家族 233 局入账（本地 ~/workspace/carbon-harvest/accumulated），其中 Master3s2 仅 47/106——快照窗口滚掉的 59 局只能走登录直下。
- 直下通道实测：`/data/<id>/<game>` 未登录返回 401。Master3s2 的 top_corpus 计数仍以 Mac 侧 06:55 的 88 为准；满 100 的判定与数据集组装要等 Mac 回线或提供登录会话后补下。
