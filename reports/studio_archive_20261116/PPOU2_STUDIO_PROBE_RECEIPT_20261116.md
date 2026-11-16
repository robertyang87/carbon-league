# Studio ppou2 训练探针轮末回执（2026-11-16）

- 派单：作战室 2026-11-16 08:46，所有者 08:45 批准；本线唯一执行人。
- 开跑：2026-11-16 09:15:49 CST，train 主进程 PID 66628 / PGID 66628，14 workers，--device mps，--resume state_frozen_it40.pt，kl 0.05，--games 4000，--lam 0.95。
- 探针目录（Studio 本机）：/Users/arm/junyang/carbon/data/ppou2_probe_studio/
- 看守状态：probe_watch.json 2026-11-16 10:12:44 CST，status = STOPPED_ROUND_DONE（首个 warm:false 行＋state_last.pt 落盘后 SIGTERM 整组），group_empty = true。

## 轮末验收（it=41，首个 warm:false 行直读）

| 指标 | 值 |
|---|---|
| it | 41 |
| 实收 games | 4333（目标 4000，超收 333） |
| roll_s | 2020.4 |
| upd_s | 1375.5 |
| buf_s | 7.4 |
| pre_s | 502.3 |
| games_per_min | 128.7 |
| kl | 0.1913（kl_coef 0.05） |
| pol_s / opp_s | 14935.2 / 10255.9 |
| pl / vl / ent | 0.0106 / 0.0123 / 0.712 |
| clipfrac / gnorm | 0.166 / 9.822 |
| winrate | 0.664 |
| 末行 rss_mb | 23248 |
| workers_rss_mb | 4906 |

## 墙时

- 开跑 09:15:49 CST → state_last.pt mtime 2026-11-16 10:12:27 CST：**56.6 分钟**（3398 秒；看守 elapsed_s 3415.3 秒 ≈ 56.9 分钟，一致）。
- 判读线 45 分钟：**56.6 > 45，超约 11.6 分钟（约 +26%）**。

## 内存

- 峰值 peak_rss_mb = **27670.2 MB（约 27.0 GiB）**，对照 64G 上限：过，有约 2.3 倍余量。
- 末行 rss_mb 23248 MB，workers_rss_mb 4906 MB；停机时 rss_mb_at_stop 9338.7 MB。

## MPS 回退

- 日志 `grep -c "will fall back to run on the CPU"` = **0**。
- 回退算子：无（`operator 'aten::…'` 去重结果为空）。
- 日志含 MPS 的警告行数：**0**。
- 更新步无算子回退 CPU。

## 落盘与进程组

- state_last.pt：82,585,520 B，2026-11-16 10:12，sha256（checkpoint_complete.json 记录）06218eb21db17f3c793fece36a3d0c9c091fede775dd4d4299d5218338e2c1cc，本线直读 shasum 复核逐字合。
- ckpt_41.npz：13,972,532 B，2026-11-16 10:12。
- ckpt_last.npz：13,972,532 B，2026-11-16 10:12，sha256（checkpoint_complete.json 记录）d401b064550a6613e4defd681948167ed2b55a0fa1d6af45151e36c93adcc976，本线直读 shasum 复核逐字合；ckpt_41.npz 与其逐字节相同。
- 进程组：`pgrep -f 'rl2_ppo_trai[n]'` 返回 0（exit 1，无残留），组已空，无需杀组。
- 附注：日志尾有 multiprocessing resource_tracker 警告（关机时清理 60 个 leaked semaphore），属关机清理提示，不影响本轮结果。

## 判读结论

**落地未过。** 内存（峰值约 27 GiB < 64G）与 MPS 回退（0 条、无回退算子）两项均过；但整轮墙时 56.6 分钟超过 45 分钟判读线约 11.6 分钟，按任务书口径（>45 或更新步算子回退 CPU＝落地未过），Studio MPS 不能作慢速第二席继续评估。另注意 kl 0.1913 明显高于 kl 目标 0.05（kl_coef 0.05），供策略组参考。

## 后续

- 本 cron（ppou2-studio-probe-watch）已按任务书自撤。
- 回执文件：~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/ppou2-studio-probe-receipt-20261116.md
