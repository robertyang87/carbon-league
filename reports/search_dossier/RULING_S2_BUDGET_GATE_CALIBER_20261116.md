# S2 budget_gate 口径裁定：段 0 包 0 系构造使然（2026-11-16）

## 背景

- S2 第二启 2026-11-16 09:55:57 开跑（父 PID 66880／子 66881，C1 spot-34；被测包 be96d457／对手 2aebfbc1／certify fc33be6c／bigeval 354bcd1b／eval_v53 45792dfb／search 94f7fb64／fastsim aefe6c83 机上 md5 全合；scope 150%＋512M、BLAS 钉 1、预算 0.7s）。第一启失败（系统 python3 无 numpy）已单独报所有者，不并进开跑回执。
- batch-3 裁定要求：S2 诊断跑与主体冻结时生产计时认证均明报 budget_gate 计数；若自锁在部署口径复现，回策略组作样本口径修正案。

## 复验（策略组独立第一手，包字节）

- 段 0 包 `code/packs/echo4_search_int8.py`，blob `fd45f35b`，9,030,752 B，md5 `be96d457`：包内 `budget_gate` 出现 **0 次**、`C_hi` **0 次**、`cmp_guard_errors` 0 次。该 build 的比较器（R2-09 手写分叉）根本没有 C_hi 成本准入路径，中止口径只有 deadline／single_call_overrun。
- 对照：resync 包 `code/packs/echo4_search_int8_resync.py`，blob `c932a0cf`，9,041,707 B，md5 `cba5923c`：`budget_gate` **3 次**、`C_hi` 6 次、`cmp_guard_errors` 4 次。
- 结论：搜索组「本趟 budget_gate 计数为 0 系构造使然」的陈述属实；0 是正向证立的结构陈述，不是缺省值——搜索组早前「拿不到就如实报、绝不默认报 0」在本件上已被字节证据取代。

## 裁定

1. **接受。**完赛回执记 budget_gate = **0（结构性——包内无该机制），非「门从未触发」**，并附包字节复验证据（md5／blob／计数）。batch-3「S2 明报 budget_gate 计数」的要求以此形式满足。
2. 有意义的 budget_gate 读数在**主体冻结时生产计时认证**（resync 内核 385190bd）上；若自锁在部署口径复现，回策略组作样本口径修正案——batch-3 裁定不变。
3. 中止构成（deadline／single_call_overrun）经**同种子同口径 in-process 复现**取回，可接受为补充读数，条件三条：① 复现用 S2 被测包本体 be96d457、同种子 986500–986519、同 scope（150%／512M、BLAS 钉 1）、同预算 0.7s；② 复现身份（包 md5、脚本 md5、scope、种子）全记并标明「复现导出」，与 S2 本趟记录分开；③ S2 注册判据（逐局慢步 >3／超额 >5s／任一步 >3s → FAIL）的慢步读数以 S2 本趟记录为准，不由复现替代。
4. S2 口径不变：**诊断级计时证据**；不外推为生产计时认证。
