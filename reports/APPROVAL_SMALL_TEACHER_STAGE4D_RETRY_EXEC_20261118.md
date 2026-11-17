# 批复：换 trunk 重跑执行提案——批准开工（策略组，2026-11-18）

- 凭据：执行提案 reports/eng_verify/STAGE4D_RETRY_TRUNK_PROPOSAL.md（commit 063aed1d，策略组读回 md5 dd0c4b0c、12,992 B，逐项核对与裁定 6388b27e §2 一致）。

## §1 核对要点（均合）

- 口径：臂集/先验取教师分片 gen-3f2c17f8b7beb5b6 的 arms/priors，不调 gen_candidates；信念场景 trunk 无关同码同种复现（seed_base=7000+step，生产调用一致）；对手根动作教师 trunk 采样＋guard 修正后冻结、只取用不重采样；只换 trunk 三处（延续步解码、叶 _leaf_u、guard 内重解码），其余教师原值。
- 对照闸（开工前硬闸）：同一 harness、教师 trunk、8 根，q 与 Q_mean 逐位一致＋arms/priors 透传 assert；不过即停并按 (a)(b)(c) 诊断，其中 (b)（偏差均匀 ≤1e-12 且 argmax 相同）报数字、停、等裁定——不擅自作废、不 proceed。
- 座位：主 spot-36、备选 spot-37/38；排除 spot-34、L20、DGX、Studio——与座位原则一致。
- 用量：396 根 × 680 ≈ 269,280 次调用；机队约 5.4 秒/根（区间 5–8 秒），2 worker 约 18 分钟（区间 15–30 分钟），总墙时预算约 1 小时（含 overhead，估计，未校准）；L20 的 2.7 秒/根不是机队速率，首分片落盘后以实测重钉 ETA。
- 身份三件套齐全（学生 npz b22020ff、教师件 c6ade5ce、config 9a2b2440、回放 v2、k0_annotate 7f567a2e、教师 generation、学生 plug-in 5c0dc757、权重 06a03d9c、代码栈全扫、比对器 v4 1e9ad085）；保底条款按裁定 §2 原文复述。

## §2 批复

- **批准开工**。执行程序：开工三件套（座位直读确认＋身份三件套核验＋对照闸通过）齐才放全量；停表任一触发即停报，不硬跑。
- Tier 落点不由执行方判，只报数；判读待比对结果件到后由策略组落。
