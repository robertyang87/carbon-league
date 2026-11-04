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

## 更正：陪练池退化（evo2 判废）＋ 旧线 switcher_treewar 惰性

- **发现**：evo2 第 1–2 轮中 `planters=2`/`early_planters=3`/`plant_min=6`/`seize_min=4` 对两个陪练的 Δ 恰为 0.0 → 逐局核对（seed 7000 side 0）：
  `nocamp`、`spar_tf`、`spar_tw_nc` 三者对 base 终分**逐位相同**（1,038.68）。中等幅度改种树门槛不生效——种树的约束在种植员编制（`planter_from`：采集 ≥4×种植），不在门槛。
  覆盖确实生效（`rho=0.08`、`plant_min=40/ratio=3`、`early_planters=0` 均改变终分）。
- **后果**：evo2 陪练池实为纯镜像；此前「spar_tf +129、spar_tw_nc −34」是 RNG 噪声。evo2 已停，状态 `data/evo2_state.json` 仅作留档，不采信。
- **旧线连带**：`switcher_treewar.py` 对 nocamp 终分与 `base_v53.py` 逐位相同（−3,565.6，种树同为 17）——旧项目「三型变速陪练」中的 treewar 型在该局是惰性的，相关历史结论需复核。
- **修正版树重型陪练**（以 nocamp 为底，对 nocamp seed 7000 单局实测）：
  `spar_tf2`＝planter_from 2、early_planters 3、retire_from 250、plant_min 5、plant_ratio 1.15、seize_min 4 → 种树 23（nocamp 11）；
  `spar_tf3`＝planter_from 2、early_planters 4、retire_from 250、retire_keep 2、plant_min 3、plant_ratio 1.05、seize_min 2、planters_late 3 → 种树 29。
  纪律补丁：今后任何陪练/变体先做「同 seed 逐位比对」确认非惰性再入池。

## 天梯新形势（JUZI1/carbon-league-replays 快照 2026-11-03 14:26 UTC）

- 新榜首 **claude-AlphaCarbon-Master 2,652**（41/43）；AlphaCarbon-2 跌至 2,479；前 20 几乎全是 NN/RL（NaMnSi-RL-T 系）与 Godslayer。
- 规则线：youzahui v13 2,224（#21）；**no-camp 配方衍生版已跌到 1,952–2,014（#46–88）**。距榜首约 650 分 → 参数微调不可能弥合，必须走 BC＋自博 RL。
- 已起 NN 管线：`code/nnfeat.py`（共享特征：信念图＋14 平面自我中心 15×15＋18 标量）、`code/nn_build_bc.py`（从回放取 rating≥2380 的教师方）、
  `code/nn_train_bc.py`（3 残差块 48 通道循环填充 CNN，二面体增广）、`code/nn_agent.py`（NumPy 推理＋己方防撞＋合法招募）。
  BC 数据：280 局、380 个教师方、1,254,695 样本（4 GB，未进仓，可由脚本＋回放快照再生）。

## NN 第一阶段结果（BC）

- BC 训练 3 轮（`nn_train_bc.py`，W48×3 残差块）：留出局 top-1 —— ep0 72.4%、ep1 75.6%、**ep2 77.4%**（采集 75.4%、种植 68.6%、基地 92.7%）。
- 招募闸门（`recruit_choice`）：BC argmax 几乎从不招募（类别失衡），改为「有空位且现金够 30＋种树储备即招、类型由网络定」。
  ep0 有闸 vs nocamp：−2,286/−3,543/−2,311/−3,072（无闸时约 −12.5k）。
- **最终 BC＋闸门＋终局回家规则 vs nocamp（seeds 6500–6507 ×2 边，16 局）：4 胜，均差 −191 ± 293（噪声内，持平）**；
  账本均值：存碳 6,663 vs 7,542、树收入 **4,222 vs 2,417**、死亡丢碳 **442 vs 98**；154 ms/步（4 进程并行），最慢 257 ms。
- 解读：BC 已追平 nocamp（天梯约 2,000），短板是护货（死亡丢碳 4.5×）与存碳吞吐；下一步 PPO 自博（`nn_rl.py`）。

## NN 第二阶段：PPO 第一轮（判负，已停）

- 配置：BC 终版起步，lr 1e-4，每轮 24 局（nocamp 50%、spar_tf2/tf3 各 15%、自身快照 20%），采样温度 0.5，团队级 GAE。
- 17 轮（约 1.8 小时）：采样胜率 46% → 4–25%，对 nocamp 均差 +400 → −1,400 左右，**单调变差**，已停。日志 `data/nn_rl1_log.jsonl`。
- 推断原因：价值头从未训练（BC 无价值目标），团队级优势噪声大，更新步长偏大，又无 KL 约束拉住 BC 策略。
- 现最优仍为 BC 终版（权重 `data/weights/nn_bc_w48_v1_final.npz`）。

## 护货盾 + RL v2 启动

- `nn_agent.SHIELD`：满载采集员优先选「下一步不会被携货 ≤ 自己的敌方采集员够到」的格子。
  同 seeds 6500–6507×2 对 nocamp：无盾 4 胜 / −191、死亡丢碳 442 → **有盾 7 胜 / −20、死亡丢碳 250**；配对 Δ +171 ± 274（方向正、噪声内），保留。
- RL v2（`nn_rl.py`）修正：价值头预热 3 轮、lr 2e-5、KL(π_BC‖π) 系数 0.5、去掉自身快照（nocamp 60% / tf2 20% / tf3 20%）、每 5 轮贪心对 nocamp 固定 8 局。
- 第 0 轮基线（BC＋招募闸＋终局回家＋护货盾，贪心，seeds 640000–640003×2）：**6/8 胜，均差 +409 ± 285**。

## RL v2 判废（共享主干 bug）→ RL v3

- RL v2 第 2 轮（仍在「仅训价值」预热期）采样胜率 54% → 0%、KL 0.056 → 0.185：价值头与策略共用卷积主干，价值梯度把策略拖走。
  RL v1 的退化很可能同因。日志 `data/nn_rl2_log.jsonl`。
- RL v3：独立价值网络（BC 权重初始化、lr 3e-4、只吃价值损失），策略网只吃 PPO＋熵＋KL；预热期策略网不更新（KL 应为 0，作为自检）。
- 参赛账号改为 `claude-opus55-cloud`（原 muse-spark1-3 已退出）。

## RL v3 结果 → v2 提交件

- RL v3（独立价值网、lr 2e-5、KL 0.5）12 轮、约 290 局；贪心固定 8 局对 nocamp：第 0 轮 +409、第 5 轮 +119、第 10 轮 +222（均噪声内），KL 仅 0.0016。
- **新 seeds 700000–700015 ×2 边、32 局配对**：v1 20 胜 +14 ± 246；**RL v3 终版 22 胜 +427 ± 143**；配对差 +413 ± 249（≈1.7 se）。
- 打包为 `code/submission_nn_v2.py`（权重 `data/weights/nn_rl3_last.npz`，打包前后同局逐位一致）。推荐上传 v2。
- 下一步：新回放快照（2026-11-04，400 局全新）扩充 BC 数据，热启动续训，再接 RL。
- 新快照（2026-11-04 00:2x UTC）只覆盖约 2 小时、以中游规则系为主：教师方仅 27 局 / 99,693 样本（+8%），暂不重训 BC，留待累积。
- RL v4：从 RL v3 终版热启动，lr 5e-5、KL 0.2、每轮 32 局、更新 1 遍、价值预热 2 轮，跑 4 小时（seeds 从 800000 起）。

## 陪练扩充：Master 影子

- 数据：两份快照中 claude-AlphaCarbon-Master 的 45 局、148,269 样本（`BC_TEACHERS_ONLY`）。
- 训练：从 BC v1 热启动、lr 5e-4、4 轮；Master 留出局 top-1 77.7%（采集 77.2%、种植 68.5%）。权重 `data/weights/nn_shadow_master_v1.npz`。
- 对 nocamp（seeds 640000–640003 ×2，贪心＋同一规则层）：4 胜，均差 +445 ± 518（逐局 630/1993/2144/−1201/−779/2145/−547/−827）。
  同 seeds：v1 +409、v2 +683。→ 与 v1/v2 同级、方差大、风格不同；作联赛多样性成员入池，不作「更强对手」。

## RL v5：联赛对手池

- RL v4 只跑完价值预热 2 轮即停（改走联赛），日志 `data/nn_rl4_log.jsonl`。
- RL v5：从 v2（RL v3 终版）热启动，lr 5e-5、KL 0.2、每轮 32 局、更新 1 遍、价值预热 2 轮，计划 5 小时（seeds 从 900000 起）。
  对手池（`data/nn_rl5_pool.json`）：nocamp 40%、spar_tf2/tf3 各 10%、NN v1 10%、NN v2 15%、Master 影子 15%。
- 第 0 轮贪心基线（= v2，seeds 640000–640003×2）：6/8 胜，+683 ± 185。
- 容器重启中断 RL v5（已到第 8 轮；第 5 轮贪心 +491、5/8，噪声内）。日志 `data/nn_rl5_log_part1.jsonl`。
  从第 8 轮 `rl_last` 续跑为 RL v5b（同配置，价值网重新预热 2 轮，seeds 从 950000 起，4 小时）。
  新增 `code/make_spars.py`：一键重建 nocamp.py、exp/spar_tf2/tf3 与 NN 包装文件。
