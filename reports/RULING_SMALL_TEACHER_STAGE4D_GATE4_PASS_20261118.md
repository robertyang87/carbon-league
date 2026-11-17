# 裁定：阶段-4D 闸 4——跨机逐位一致（0.0）替代同机硬闸，放行全量（策略组，2026-11-18）

- 凭据：停报件 reports/eng_verify/STAGE4D_RETRY_TRUNK_REPORT.md §7（commit 64025123，策略组读回 md5 75e3cf94、19,196 B 合）。裁定对象：批复 530c70e3 开工四件套之闸 4、裁定 ef932ea3 对照闸口径。

## §1 事实（验合一致）

- 闸 1–3 全过：spot-64（64 vCPU／121 GB／load 0.00）；mesh 验通（本机↔64 免密、64↔DGX／L20 直连 RC 0；spot-29/63 离线未验，如实记录）；身份 35 件全合、回放双验全合（md5＋真 sha256）、config 重算 9a2b2440 全值合。
- 闸 4(a)：harness 对照跑 8/8 完赛，每根 act 调用 648（min=max=mean）；诊断 gen_candidates arms_match 全 True、priors 差 0.0。
- 闸 4(d) 跨机参考：harness 对照输出 vs L20 教师分片，8 根逐根对位，994400／994401 **max|dq|＝0.0、max|dQ|＝0.0，argmax／arms／priors 全同——逐位一致**（强于参考线 ≤1e-6）。
- 闸 4(b) 停因：已批脚本 direct_control_roots.py（09d2705e）L81 把 cnet2_feat 的实例方法 Feat.observe(self, obs) 当类级工厂调用（TypeError: takes 2 positional arguments but 3 were given），脚本自始不可运行；其 evaluate_root 调用签名亦与生产 _eval_root 不一致。主代理独立复核属实：**与 harness 无关、非数值不等**。

## §2 裁定：选项②——跨机逐位一致替代同机硬闸，放行

1. 同机硬闸的目的，是在存在跨机浮点噪声时隔离机器噪声、单独证明 harness 忠实。本次跨机比对全字段逐位 0.0，是直接对教师生产分片本身的证明：harness 对照输出与生产分片在 q／Q／argmax／arms／priors 上逐位相同——harness 若有任何结构偏差，8 根不可能全字段 0.0。证据强度不低于同机对照（同机证 harness 与同机生产调用一致；本次证 harness 与生产产物一致，更直接）。
2. 跨机降格为参考性的唯一理由是浮点噪声（本机 VM↔L20 曾测 2.2e-07）；spot-64↔L20 实测噪声为 0，该理由在本座不适用。
3. direct_control_roots.py 的缺陷是其自身 API 误用，不构成对 harness 的怀疑；该脚本本次作废不用（缺陷在案，后续如需同机直跑另行修正报批）。

## §3 放行条件

- **放行全量 396 根**（换 trunk 重跑），座位 spot-64 不变。
- 学生重跑的自身确定性复核（抽 2 根重跑、须与自身逐位一致）仍须执行——此为学生侧确定性要求，与同机对照无关，不因本裁定免除。
- 全量产出后：比对件经 shard_compare v4 出 Tier 1/2 四项实测值＋compared_roots 与可比动作条目数，报策略组判读（Tier 落点归策略组，执行方只报数）。
