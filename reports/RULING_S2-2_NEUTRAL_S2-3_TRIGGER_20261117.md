# 确认：S2-2 中性成立，触发 S2-3（2026-11-17，策略组）

承：加速裁定 05204f30；NOTE_S2-2_PENDING_VERIFICATION_20261117（962c07a0）。

## §1 独立复算（策略组直读仓内原件）

- duel 件 `data/k0_calib/duels/lr0.0005_s400_b2_seed7.json`（commit 632c54cc，md5 3da4e7e0，4,046,285 B）：9,900 行＝50 局×198 根（步 200–298）；学生 md5 单值 922f50fa…、父权重 06a03d9c… 单值、roots_sha256 单值。
- 逐局配对 ΔU（局内 du 均值，再局间平均）：**ΔU +0.00002048**、σ_game 0.028542、se 0.004036；**t CI95 [−0.008091, +0.008132]**；自举 CI95 [−0.007358, +0.008189]（与 ledger 所载 [−0.00750717, +0.00818435] 下界差 1.5e-4，系自举蒙特卡洛波动，判读不变）。
- ledger `data/k0_calib/ledger.jsonl`（commit b0fc0bb5，md5 84e95a06）：12 行，第 12 行即 S2-2（β=2.0、种子 7、ΔU 2.0481087085355843e-05、ci95 [−0.00750717, +0.00818435]、neutral True、50/9,900）；前 11 行与旧 ledger 逐字节合（带尾换行 md5 2af0122f），差分恰 +1 行。两 commit 先后（ledger b0fc0bb5 → duel 632c54cc）与 S2-1 单 commit 先例不同，内容无差、不作问题。

## §2 确认

**S2-2 中性成立**：点估计 ≈0，两条 CI 均含 0。DGX 所报读数与复算逐值合（t CI 至 1e-6；自举下界差 1.5e-4 见 §1）。

## §3 触发 S2-3

按 05204f30，S2-2 中性 → **S2-3（β=0.25，lr 5e-4×400、种子 7，与 S2-2 同设置）开跑**，由 DGX 手动显式启动（非自动接档）。S2-3 落盘后策略组同轮复算；**S2-3 若判中性，按 β 非单调异常停报**；若判非中性，其读数并入 β 测绘、回策略组议。

## §4 driver 诊断收讫（DGX 直读，策略组认）

- 根因：driver 31 行 `r2["neutral"]` 读错对象——`do_cell` 返回的原始合并字典无 "neutral" 键，neutral 只在 `rec`（k0_calib_driver.py:191）里；早返路径同样无该键。
- **更正本组先前假设**：入仓步骤没有死在这里——两 driver 全文 grep（gh_api／put_file／git push／warehouse）零命中，入仓根本不在 driver 流程里；S2-2 未入仓是入仓为独立手动步骤、此前没人跑过。
- 修复：仅 k0_stage2b_driver.py（8fccf867→**16d4d83d**，备份 .bak_20261117；钉版 k0_calib_driver.py 未动，仍 8b1ad464）：加 `_neutral(r)` helper（按 :191 同定义由 cluster_bootstrap_ci95 是否含 0 推导；r 为 None 抛错），31／36 行各改一字。只读验 _neutral(S2-2)=True、_neutral(S2-1)=False，均与 ledger 合；py_compile 通过。
