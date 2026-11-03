# no-camp 配方复核 + fork_v0_1 样本外复测（2026-11-03）

## 动机

天梯快照 `data/leaderboard_now.json` 显示：只改 v53 三个常量的公开「no-camp 配方」
（`soft_w 1.0→0.0`、`campers 4→1`、`camp_from 6→12`）多份衍生版评分 **2,056–2,134**
（carbon-v14-v53recipe-1102 2134、Kimi-Clark-v53-nocamp-v3b 2124、NaMnSi-v8-v53sw0 2108、
minimax-flash-v53nc-r1 2064、claude-opus-v53-nocamp 2056），作者自述本地对 v53 6-0 +3,276。
这与 README「规则层局部最优、不要重开」及「规则线估位 1700–1950」的结论冲突，故复核。

## 口径

- `eval_v53.play_game` 原样（FastCarbon、规范播种 `7919*seed+104729*side+31`、账本重建）。
- **样本外 seeds 5000–5011 × 双边 = 每组 24 局**（此前所有调参均在 1000–1003，未触碰 5000 段）。
- 预注册判据：Δ>0 且 |Δ| > 2·se 记方向成立。

## 结果（磁盘原件：`data/oos_*_s5000.jsonl`）

| 被测 | 对手 | 胜 | 场均分差 | se | 判定 |
|---|---|---|---|---|---|
| fork_v0_1（rho 0.08） | base_v53 | 14/24 | +607 | 531 | 噪声带内，**未复现 +1184** |
| no-camp 配方 | base_v53 | **22/24** | **+3,278** | 424 | 成立（与公开自述 +3,276 吻合） |
| no-camp + rho 0.08 | no-camp | 7/24 | −717 | 281 | rho 叠加有害，不并入 |

最慢步：no-camp 52 ms，远低于 1 s 预算。

## 裁决

1. 可提交门面由 fork_v0_1 改为 **`code/submission_nocamp.py`**（v53 + no-camp 配方，文件头注明非原创与来源）。
   线上预期 ≈2,060–2,130（同配方衍生版实测区间，推断）。
2. README「规则层局部最优」结论需修正：s4 camp 支线未覆盖 `soft_w=0 + campers=1 + camp_from=12` 组合。
3. 下一步候选（未开工）：在 no-camp 之上接入对手行为预测/联合围堵（参照 youzahui Codex-Carbon-1101-a196-v9，2,258），
   复用 `opponent_predictor_tables.npz`。

## 附：G1 独立复跑

案 C G1 探针在本环境独立复跑一次（同 seeds 1000/1001×双边）：场均 −13,748、灭绝 1/4、种树 2.0/局，
与 §7.9 正式裁决（−14,023）同向判负。因与正式留痕同名，复跑产物未进仓。
