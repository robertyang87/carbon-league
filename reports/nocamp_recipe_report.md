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

## 续：自我进化（evo）启动记录

- 工具：`code/evo.py`（变体＝父文件末尾追加 `PARAMS.update`，Bot 在 step 0 拷贝 PARAMS，故精确；评测复用 `eval_v53.play_game`）、
  `code/evo_loop.py`（v1：只对冠军镜像自博）、`code/evo_loop2.py`（v2：陪练池＋假设队列，协议写在文件头，先写后跑）。
- 本地陪练池普查（nocamp 视角，seed 6000 ×2 边）：对 base +2,836、GE-v53 +3,181、v18 +2,097、canopy +3,466，其余公开策略 +9k~+19k；
  NN 影子 shadow_alpha_v2 / shadow_deepseek −18k 量级（太弱，不能当榜首代理）。本地已无比 nocamp 强的对手。
- **榜首对 nocamp 的败因（回放 `data/summaries/a2_highscore_win_v53sw0.json`，NaMnSi-v8-v53sw0＝同配方）**：
  存碳 6,905（8 采集）vs Alpha 10,024（6 采集）；种树/抢树 17/9 vs 37/15；场上树 1–3 vs 5–10；
  步 240/270 在途 977/1,133、终局滞留 682 vs 229。→ 败在树经济＋存碳慢（在途货被敌树 10%/步抽税）。纯镜像自博看不到这一点。
- 因此 v2 加入两个树重型陪练（均以 nocamp 为底）：`spar_tf`（全程低种树门槛、2 种植员）与 `spar_tw_nc`（switcher_treewar 的步 50–100 加码＋nocamp），
  对 nocamp 分别 +129±185、−34±225（8 局），与 nocamp 同级。
- v1 跑 3 轮无晋升即停（单参小扰动在镜像下平坦），状态存档 `data/evo_state_v1.json`。
