# 裁定：对照闸口径修订——同机对照为放行硬闸＋回放哈希标注纠正（策略组，2026-11-18）

- 凭据：准备报告 reports/eng_verify/STAGE4D_RETRY_TRUNK_REPORT.md §6（commit 44a76574，策略组读回 md5 87c656e3 合）；harness reports/eng_verify/retry_trunk_harness.py（commit 46cf3202，读回 md5 53df4f2e 合）；草稿脚本 reports/eng_verify/direct_control_roots.py（commit bab63480，读回 md5 09d2705e 合）。

## §1 事实（验合一致）

- 本地 1 根对照冒烟 RC=0：act 调用数 648 恰与预测一致（680−32）；同机确定性复跑 q/Q/Q_std/priors/arms 全部逐位一致——harness 确定、结构忠实。
- 本地对照 vs L20 教师分片：arms 逐位一致，priors 透传逐位一致（max|dp|=0.0），q 最大差 2.20e-07、Q 最大差 7.03e-08、argmax 相同；gen_candidates priors 最大差 1.357e-06（两次运行值完全相同——稳定的机器间差异）。
- 隧道审批 14400c86（通往 spot-64 100.72.73.119）：~03:25 再查仍 pending；本轮未对 spot-64 发起 SSH；座位直读闸、mesh 分发验通、机上身份核验、座位对照闸、全量、比对均未做——停报合规。
- 源材料本地转运全合（18M，SCP_RC=0）；mesh 钥对已本地生成（指纹 SHA256:sLYABsIs9NxzauiUoehOW/sygVzvb3dcP2V2OhpKN0E，私钥仅存本机）。

## §2 裁定：对照闸口径修订

- 原闸「q 与 Q_mean 须逐位一致」的前提是跨机逐位可复现，实测已证伪：残差 ~1e-7 是稳定的跨机浮点噪声（L20 vs 本机），不是 harness 错误；诊断 (a) 已排除。原 (b) 容忍上限 ≤1e-12 作废。
- **批准同机对照方案（首选）**：座位上另跑生产 `evaluate_root` 直接调用（教师 trunk，8 根对照根，草稿脚本 direct_control_roots.py md5 09d2705e 批准使用），与 harness 对照输出逐位比对（q/Q/Q_std/priors/arms 全部 np.array_equal）。**同机对照逐位一致为放行硬闸**；不过即按原诊断 (a)(c) 处置（(a) 同机两次不一致→harness 非确定性→作废；(c) 其余→harness 有误→作废）。
- **跨机比对改为参考性**：harness 输出 vs L20 教师分片，max|dq| ≤1e-6 且 argmax 相同、arms 一致即记「跨机噪声内一致」；不作为放行依据，不触发停表。
- gen_candidates priors 的机器间差异（1.357e-06）不影响重跑：重跑不调 gen_candidates，先验取教师分片透传（max|dp|=0.0）。

## §3 回放哈希标注纠正（收讫）

- 停报／提案 v2／v2 批复中写的回放「sha256 6ed2623e／6607fe5e」实际是文件的 **MD5**；真 sha256（与教师 generation 身份块记录一致）：994400 → 55d713d64c002d6c204988280804da47b96c979464f4ae9740b2676338581436；994401 → 866a8ce7cc7125af8962e51f6db7b1a41686c3c13dbab35b2cec0c653a66baf5。
- 值本身未变、生成身份自洽，只错 paperwork 标签；harness 双重核验（md5 且 sha256）。此前各件（含策略组 v2 批复 530c70e3）中该标签按本裁定纠正，后续一律用真 sha256 标注。

## §4 执行

- 隧道审批 14400c86 到后，执行按提案 v2 §7 照序续跑：① 座位直读闸 → ② mesh 分发验通 → ③ 身份机上核验（回放按 §3 真 sha256 核验）→ ④ 对照闸（同机对照逐位一致为硬闸＋跨机比对参考性）→ 全量 396 根 → v4 比对 → 结果入仓 → Tier 四项实测值报策略组（只报数）。
- Tier 落点不由执行方判，只报数；判读待比对结果件到后由策略组落。
