# CURRENT：唯一接手入口索引

最后更新：2026-11-13。本文件是本仓的唯一接手入口（依据 REVIEW_20261112 R14）：
当前在册的关键制品、位置与核验方式都以本文件为准；历史交接文档（HANDOFF 等）
只作归档，接手时先读本文件，再按指针读 `reports/live/` 的实时状态。

## 主线

- 仓库只用主线 **main**；分支 `cnet2-search-v12` 已退休（所有者 2026-11-12 定规），
  一切提交直接进 main，提交后必须读回 main 验证。

## 在榜策略：Echo2

- 身份：天梯在榜策略，账号 `claude-opus55-cloud`。
- 权重：`cnet2_m3w5L.npz`，md5 `82fde46b2633950d3cbb29e8cacbfdc2`（**待核**：
  仓外制品，数值据考场机直读回执，本次未能从仓库/本地复算；用前须在持有
  文件的机器上复算比对）。
  - 真实位置：各考场机考试栈 `data/weights/`（如 spot 机
    `/root/carbon/data/weights/`）与 DGX 暂存；**不在本仓 Git tree 内**
    （2026-11-13 已核 tree）。
  - 训练来源（仓内可核）：`reports/runbook_echo2.md` §1 的 W5 臂——DGX 上
    100 局 Master3s 数据集、Grove1 权重 init、CFG `{"C":128,"NB":8,"D":128}`、
    lr 3e-4、20 轮、BS 256、`set_uw.py 1 1 1`、不加权对照臂，取最后一轮
    送考（考场在用文件名 `cnet2_m3w5L.npz`）。
- wrapper：`exp/echo2.py`，md5 `15b6d754e6e21de33870074d3983a16f`（**待核**：
  同上，据机上直读回执）。
  - 真实位置：各考场机考试栈 `code/exp/echo2.py`（如
    `/root/carbon/code/exp/echo2.py`）与 DGX 暂存 `s26stage/code/exp/`；
    **不在本仓 Git tree 内**（2026-11-13 已核 tree；旧版 CURRENT 误记为
    「仓内」，F08 订正）。仓外制品无 git blob SHA，溯源以 md5＋所在机器
    路径为准；补入仓前不得声称仓内可恢复。
  - 构建参数（冻结 loader 模板，机上核实）：`cnet2_search.make_search_agent`，
    CONF {SEARCH: False, GUARD: True, GUARD_NARROW: True, MODE: 'sweep',
    MIX: False, OPEN_T: 0.0, OPEN_STEPS: 0, OPEN_KINDS: None, OPEN_SEED: 0,
    TEACHER: 2, TEACHER_LATE: 2, LATE_STEP: 240}，`_W` 指向上述权重。
- 核验方式：用前先在持有文件的机器上复算两个 md5 与本节数值比对，再做
  Policy 加载验证；任一不符不许开考。

## 待提交候选：n1r

- 打包文件：**已入仓** `code/packs/n1r_int8_reference.py`，4,579,266 字节，
  md5 `ba7f84c3652ea2cd14a2243f2b1eace2`，可直接复算（台账见
  `reports/live/packs_ledger.md`）；L20 `/root/carbon/data/packs/n1r_int8.py`
  为同制品的机上副本。原始训练权重的文件名/MD5 身份仍为**未核**
  （不能据解量化后的包反推原文件身份，台账已如实记「未核」）。
- 状态：已过全部上榜闸（对 Master2 真身校准达标＋头对头胜 snipe_b_30＋
  1.5 核/512MiB 约束下计时全项达标）；**未提交**，上榜与否等所有者拍板。

## 考试口径（定标）

- 标准考官对：{Master2 真身，Echo2}，各 200 局、种子在册管理。
- harness 绑定：`code/eval_v53.py`（main）完整 MD5 于 2026-11-13 读回为
  `904c2fab77a9256ca7b12d61e21167f7`（F01/F04 修复后）。任何重新定标的锚点
  必须与其测得时的 harness MD5 一并记录；引用 MD5 以测时读回为准，
  不得引用过期值（F08）。
- 锚点：Master2 真身＝0，E0＝**+785**（校准分＝对 Master2 均分 − 785）。
- 注意：REVIEW_20261112 的 R01/R02（本地多走一步、无灭绝终止）修复部署后，
  本地尺会变，全部锚点（含 +785）须重测后才能继续引用。

## 在训主线：snipe_n1

- 位置：L20（gpu-l20-5），自我对弈 PPO 训练中。
- 实时状态以 `reports/live/status.md`、`reports/live/snipe.jsonl` 为准；
  裁决与考试结果以 `reports/live/exams.md` 为准。

## 指针

- 审查与修复依据：`reports/REVIEW_20261112.md`
- 复刻缺口方案：`reports/live/replica_gap_plan.md`
- 线 B 预置备忘：`reports/live/lb_expert_iteration_prep.md`
- 直播通道说明：`reports/live/README.md`
