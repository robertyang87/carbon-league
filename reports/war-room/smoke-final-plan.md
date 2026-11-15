# 最终版冒烟执行预案（DGX 线，唯一执行人；作战室 2026-11-13 00:28 裁定）

## 触发与前置
- 触发：第三波修复（F06/F07/第 5 节两项：终止后无效推理、CachePolicy 签名补全）入 main 后，作战室发话才开跑。C 组已关闭（00:25，8 笔入 main）；A 组 F01/F04/F03 已关闭（c43d08b1/313c1298/2e5ffcaf）。
- 开跑前：DGX 仓库 fetch＋ff-only 到当期 main tip（只读核验无在训进程，不动任何在跑的东西），先报回基线 SHA，再跑冒烟。
- 范围：标准趟（自博弈＋bot 局，恒等式全套）＋构造灭绝趟＋R12（TTA=2）顺手复跑。不部署、不动考试、不动 snipe_n1。

## 构造灭绝场景（预研完成，2026-11-13 00:29）
- 注入点已定位：repo 版 rl2_ppo_rollout.play_batch 在函数内直接 `sim = FastCarbon(CFG).reset(seed=seed)`（旧 recon 副本第 172 行；最终版开跑前按当期代码复核行号与 CFG 来源）。
- 路线：冒烟脚本（harness 侧，不改仓库代码）在 import 后替换 rollout 模块命名空间里的 FastCarbon 为薄包装：reset 照常建局，随后把一侧状态改到「距灭绝一步」——清其树与基地、工人留 1 个且置于必死位置（或直接清工人并保留 _extinct 的现金条件边界），使第一个真实 step 后 `_extinct(post, p)` 为真。被测的 play_batch 主循环、term 标记、跳步、worth 记账全部走原码。
- 备选路线（注入点若被第三波改动）：改起始资源（CFG 起始现金/单位）＋脚本化自杀 bot 驱一侧灭绝；两者皆不通才按裁定把卡点写清回作战室，不许跳过。

## 验收清单（作战室口径）
标准趟：
- 全记录 len(recs)＝steps＝len(dworth)＝len(dmargin)；sum(dworth)、sum(dmargin) 对 margin 误差在 float32 容差内；非灭绝局 steps 恰 299。
构造灭绝趟（三项全要）：
1. 记录长度对终止步：term 局 steps ＜299 且 len(recs)＝steps＝len(dworth)。
2. 灭绝局 dworth 和＝**原版终局分差**（F01 修正后语义，不是现金差）：除 sum(dworth)＝margin 的内部恒等式外，再用原版计分对同一终局状态独立复算一遍终局分差与 margin 对照（参照 A 组 F01 验收法：构造例修后 margin 与原版逐字一致）。
3. 终局后无多余推理步：term 局在标记后不再推进（steps 冻结、planted 不再变），同批其他局照常打满。
R12：TTA=2 后 sig_mismatch 应触发、残留视图差应非零、enc_for 消费差应恰 0.0（权重用 cnet2_b128m2t3.npz，与审查同款）。

## 回执
- 三件套：基线 SHA＋进程证据（DGX 直读）、日志认领段（Range 分段取回兜底）、产物目录 /home/arm/junyang/carbon/data/smoke_pb_final/（新目录，与 v2 的 smoke_pb_2345 分开）。
- 发本对话＋同步作战室。
