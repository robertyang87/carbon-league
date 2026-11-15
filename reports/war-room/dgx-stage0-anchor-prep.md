# DGX 线 Stage 0 本地准备：heads_patch 锚点核对（2026-11-13 05:40 CST）

作战室 05:36 派活（所有者点名）：本线＝DGX 侧 Stage 0 唯一执行人。
通道仍故障期间只做本地准备：核对 heads_patch 三处锚点在本机克隆 main
文件里的命中情况。结论如下，通道恢复时直接用。

## 三处锚点（自 hidden_files/stage0_heads_patch.py 逐字提取，ast 解析）

- MODEL_ANCHOR（cnet2_model.py，37 字符）：
  `        self.value = nn.Linear(D, 1)\n`
- CTX_ANCHOR（cnet2_model.py，203 字符）：
  `    def ctx_from(self, pre, teacher):\n        """style enters only here: trunk summary [B, C+D+64] + teach[teacher] -> ctx."""\n        return Fn.relu(self.ctx(torch.cat([pre, self.teach(teacher)], -1)))\n`
- TRAINER_ANCHOR（cnet2_train.py，103 字符）：
  `        bad_ = [k for k in mk_.missing_keys if not k.startswith("tattn.")] + list(mk_.unexpected_keys)\n`

补丁口径：每处锚点计数必须恰为 1，≠1 即非零退出、不留半成品；
幂等标记分别为 `self.utility = nn.Linear(D, 1)`、`def heads(self, ctx):`、
`Stage 0 new heads`。

## 本机核对结果：本机无从判定（与缺口清单一致）

- 本 VM 全盘（~/workspace 无深度限制 find）**没有 cnet2_model.py、
  没有 cnet2_train.py** 任何副本。
- ~/workspace/carbon-league-repo 是快照导出（有 MANIFEST.md、无 .git），
  其 code/ 下无任何 cnet2_* 文件，不是可用克隆。
- foundation_stub/ 只有 cnet2_build/cnet2_feat/cnet2_search/eval_v53/
  fastsim 快照，无此两文件。
- 与 stage0_gap_list.md 的记载一致：「本地无 main tip 的这两文件副本」。

## 通道恢复后第一趟的锚点动作（并入 step-0，不另发车）

step-0 ff 拉 main 最新之后、跑 heads_patch 之前，在 DGX carbon-repo 的
code/ 目录内对 cnet2_model.py / cnet2_train.py 做同口径 grep 计数：
MODEL=1、CTX=1、TRAINER=1 才跑补丁；任一 ≠1 即停手回报，不硬改。
（补丁本体自带同一断言，此为前置人工核对，双保险。）

## 队列续排（作战室 2026-11-13 05:38 补充，所有者定：每台机器 ≥2 小时任务、不许空等）

通道恢复后连续执行、不一趟一停；任何一环 FAIL 先报卡点再往下，不许跳环：

1. step-0（自扫残留＋ff＋四件 md5）→ 2. heads_patch（锚点命中才跑）→
3. smoke_heads（ds_m3s2、骨干冻结、只训新头；报 loss/U 范围/WDL 准确率/
   分差 MAE）——原三趟。
4. **三头全量训练**：smoke 过后用 ds_m3s2 全量训三头（骨干冻结口径先行），
   报每头收敛曲线与校准误差（U 的 held-out 均值校准，照 M01 口径）。
5. **教师质量闸小场**：教师 q 指导 vs 裸网策略，对 Master2、Echo2 各
   200 局（满载配方，种子域 994000 段，与定标错开），出双场数字——
   gen-1 开产前的闸，结论报作战室。（Studio 线有离线半边：ds_m3s2 样本
   局面跑根候选教师的一致率/熵/首选吻合率，与本场互证。）
6. **gen-1 建集预演**：用定版产局器（作战室本地已备，通道恢复后入仓）
   在 DGX 产 200 局小样，走完 katago_chain 草稿链（建集→双臂短训
   3 epoch 只验链路），回报每环读数。

1–6 合计 >2 小时。吞吐与教师成本两个数字不归本线（gpu-l20-5 控制台
云助手实测中，作战室直管）。红线不变：不碰天梯、不碰 snipe、不另起
复刻训练。
