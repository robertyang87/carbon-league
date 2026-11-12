# CURRENT：唯一接手入口索引

最后更新：2026-11-12。本文件是本仓的唯一接手入口（依据 REVIEW_20261112 R14）：
当前在册的关键制品、位置与核验方式都以本文件为准；历史交接文档（HANDOFF 等）
只作归档，接手时先读本文件，再按指针读 `reports/live/` 的实时状态。

## 主线

- 仓库只用主线 **main**；分支 `cnet2-search-v12` 已退休（所有者 2026-11-12 定规），
  一切提交直接进 main，提交后必须读回 main 验证。

## 在榜策略：Echo2

- 身份：天梯在榜策略，账号 `claude-opus55-cloud`。
- 权重：`cnet2_m3w5L.npz`，md5 `82fde46b2633950d3cbb29e8cacbfdc2`
  （来源位置：考试栈 `data/weights/`，DGX 暂存与各考场机同名文件）。
- wrapper：`code/exp/echo2.py`（仓内），md5 `15b6d754e6e21de33870074d3983a16f`。
- 核验方式：用前先比对上述两个 md5，再做 Policy 加载验证；任一不符不许开考。

## 待提交候选：n1r

- 打包文件：L20 `/root/carbon/data/packs/n1r_int8.py`，
  md5 `ba7f84c3652ea2cd14a2243f2b1eace2`（完整制品只存于该路径，仓内无副本）。
- 状态：已过全部上榜闸（对 Master2 真身校准达标＋头对头胜 snipe_b_30＋
  1.5 核/512MiB 约束下计时全项达标）；**未提交**，上榜与否等所有者拍板。

## 考试口径（定标）

- 标准考官对：{Master2 真身，Echo2}，各 200 局、种子在册管理。
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
