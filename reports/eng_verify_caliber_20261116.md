# 提速线身份验证轨 · 逐根比对口径（DGX 线落字，2026-11-16 21:40 CST）

工具：DGX /home/arm/junyang/eng_verify/shard_compare.py（md5 aee0cd6f6b0f1324dac42813b2c08527）。
基线：eng_ref_20261116 封口两局已验分片（selfplay-994400.npz 21439570／selfplay-994401.npz 79214a05）。
候选：任一候选码对同两局的重标输出分片（同 npz schema）。

## 根集与对齐

- 根集＝每局 198 根，键 (step, seat)；候选须同序同集，否则报 STRUCTURAL_MISMATCH 并停。
- 手臂对齐按手臂身份（arms 的 12 元编码）而非槽位；有效手臂＝每根前 n_arms 个。任一根手臂集不同 → 该根计 arms_set_diff、退出 q 统计（单独报数，不静默对齐）。
- 身份前置：候选 _meta 的 config_sha256／games_sha256／该局 replay_sha256 须与基线相等；code_md5s 只报不判（候选码本应不同）。

## 四数定义（逐局报，另 pooled 两局）

1. **q_max_abs_delta**：比对根×有效手臂上 |q_候选 − q_基线| 的最大值。
2. **frac_roots_delta_pos**：根 delta（＝该根有效手臂 |Δq| 最大值）>0 的根占比（精确浮点不等）；分母＝比对根数。容差记法并报 >1e-9／>1e-6／>1e-3 三档占比。
3. **argmax_flips**：argmax_a q（按手臂身份；并列取各自文件内最低槽位）不同的根数。
4. **du 差异**：du_root := max_a q_a − Σ_a p̃_a·q_a（p̃＝priors 在有效手臂上重归一）——分片原生的「决策值」代理。报 du_max_abs_delta 与 du_mean_signed_delta（比对根上）。
   - 明示：此 du 非门内 rollout 口径的 du（u_guided − u_greedy，配对续弈）；若策略组要后者，属另一重型检查，请其另令，本轨不自改。

## 自验（已跑）

- 基线对基线：四数全 0、身份全合、198/198 根比对。
- 合成扰动（10 根 q 加 1e-7 噪声＋1 根交换 top-2）：max|Δq|＝0.012155（＝交换幅度）、Δ>0 占比 11/198、>1e-6 占比 1/198、翻转 1——全合预期。

候选一到，本轨当轮跑全两局、四数报作战室；验证包经本轨收片对账转策略组。
