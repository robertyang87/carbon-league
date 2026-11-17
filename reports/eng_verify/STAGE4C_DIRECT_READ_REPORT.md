# 阶段 4C：直读 q 评估——映射判定报告（提案作废）

- 时间：2026-11-18 01:06–01:20 CST
- 批复：`reports/APPROVAL_SMALL_TEACHER_STAGE4B_DIRECT_READ_20261118.md`（commit `dc8b0595`，API 读回 md5 `a87b45fa…` 合）
- 提案：`reports/eng_verify/STAGE4B_DIRECT_READ_PROPOSAL.md`（commit `7b87ba63`，API 读回 md5 `3334c5ae…` 合）
- 结论：**提案作废**。按批复条件 ①，映射读清后即停——未写 harness、未跑任何读数、未动 L20、未提交任何文件。

## 1. 验合（全部 API 读回）

| 件 | 身份 | 读数 |
|---|---|---|
| 批复 | `reports/APPROVAL_SMALL_TEACHER_STAGE4B_DIRECT_READ_20261118.md` | md5 前 8 位 `a87b45fa` 合 |
| 提案 | `reports/eng_verify/STAGE4B_DIRECT_READ_PROPOSAL.md` | md5 前 8 位 `3334c5ae` 合 |
| `stage0_teacher.py` | 仓库 `code/stage0_teacher.py` | md5 `c6ade5ce…` 合（25,186 B） |
| `k0_annotate.py` | 仓库 `code/k0_annotate.py` | md5 `7f567a2e…` 合（49,151 B） |

## 2. 每 arm 的 q 映射（`stage0_teacher.py` c6ade5ce，`evaluate_root` L296 起）

生产调用（`k0_annotate.py` L341–351，`_eval_root`）：`teacher=2`、`K=16`、`M=8`、`H=2`、`seed_base=seed_teacher+step`、`feats=(fm, fo)`、`belief="sample"`（默认参数 L86：`--belief sample --teacher-style 2 --K 16 --M 8 --H 2`）。分片每根存 `q` 与 `Q`（`k0_annotate.py` L365–366：`q[:n]=r["q"]`，`Q[:n]=r["Q_mean"]`）。

对一根、每臂 `i`，`q[i]` 由以下五步构成：

1. **场景世界（诸臂共享，每根）**（`evaluate_root` L397–402）：`M=8` 个世界 `roots[m]`，由 `belief_carbon(feats[0], players, config, rng=default_rng([seed_base+m, 0xC0]), mode="sample")` 采样——从 `me` 座信念克隆、未知格每场景采样，再经 `_legal_root` 合法化。
2. **对手根动作（诸臂共享，每场景）**（L404–427）：每场景一个，由 trunk 采样解码（`act(..., sample_rng=default_rng(seed_base+m), temp=1.0)`）＋ `BankruptcyGuard` 产生；先于候选臂算定，**与所选臂无关**。
3. **臂集**（L429，`gen_candidates` L209–288）：贪心解码＋强制前缀单位偏离，最多 `K=16` 臂；`priors` 为终臂的归一化联合策略概率（逐臂强制解码重打分、合法性复核后 `joint_logp` 的 soft-max）。
4. **逐（臂，场景） rollout**（L434–492）：克隆场景世界；走根步（该臂＋对手根动作）；再走至多 `H=2` 个延续步——**每步双方均经 trunk 贪心解码＋guard**；**每步后都查终局**（L459、L488）。任一步终局则 `Q[ci,m] = _terminal_u`（F01/F04 终局打分，无 trunk 调用，L490）；否则 `Q[ci,m] = _leaf_u`＝叶局面上的 trunk u 头（`pol.act(..., want_u=True)`，L492）。
5. **聚合**（L494–496）：`Q_mean = Q.mean(axis=1)`（8 场景均值）；`q = wq / Σwq`，其中 `wq = exp((Q_mean − max Q_mean)/0.05) × sqrt(priors)`。

即：**每臂的 q ＝ 8 个信念场景 ×（根步＋至多 2 步双方延续）rollout 结果的均值，再经先验加权 soft-max**。分片的 `Q` 字段就是第 5 步前的 `Q_mean`，同样是 rollout 聚合量。

## 3. 为何单次直读无法复现

- 单次直读（每（根，动作）一次 trunk 调用，如「施加该臂、读叶 u」）得到的是**单世界、单步、无延续、无对手采样、无先验加权**的叶值，与教师 q 是**不同的量**：缺场景平均（M=8）、缺延续 rollout（H=2 步、双方 trunk 解码）、缺对手根动作采样、缺先验加权。
- 调用量对账：每根满臂 680 次 trunk 调用（设计公式 `8 + (1+D+A) + 5AM`；其中 `5AM = 5×16×8 = 640` 恰为每（臂，场景）的 rollout 调用：4 次延续解码＋1 次叶 u）。提案 §5 的「6,336 次 trunk 调用」（每（根，动作）1 次）只有按单次直读才成立——按真实映射应为 396 根 × 680 ≈ 269,280 次，且仍不是直读。
- 忠实复现映射的唯一做法是把整个 `evaluate_root` 用学生 trunk 重跑一遍（臂集钉死为教师臂）——那不是「直读」，而是**换 trunk 重跑搜索**；且会引入新的口径不一致：先验取教师的还是学生强制解码重算的？对手根动作取教师采样的还是学生重采样的？无论哪种取法，得到的都不是「学生 trunk 在教师动作上的价值保真度」，而是学生搜索量——与直读口径要区分的东西混为一体。
- 连带：校验关 1（交叉核对）的前提「直读 q 与学生搜索 q 应接近——同一 trunk、同一叶评估」同样不成立：学生搜索的每臂 q 也是 rollout 聚合量，不是叶评估。

## 4. 裁定

- 按批复条件 ① 与提案 §5 风险条款原文（「若第 1 步发现教师每 arm 的 q 含搜索内 rollout 聚合、无法以单次 trunk 直读复现，则直读口径本身不成立——提案作废」），**提案作废，回策略组**。
- 本趟未写 harness、未跑读数、未动 L20、未提交任何结果件。v4 搜索一致性读数（94 根 / 228 条，见 `STAGE4B_COMPARE_V4_REPORT.md`）不受本判定影响，仍按原口径记录。
- 直读四项实测值：**无**（未执行，无读数可报）。可比根数：**无**（未执行）。四关校验：**未执行**（前置条件未满足）。
- Tier 落点不由本趟判，只报数——本趟无数可报。
