# ppou2 Studio 探针日志摘要（2026-11-16）

- 运行：2026-11-16 09:15:49–10:12:27 CST（Mac Studio，M2 Ultra）；train 主 PID/PGID 66628，18 进程（主＋resource_tracker＋2 server＋14 worker）；device=mps（PYTORCH_ENABLE_MPS_FALLBACK=1）；输入为主臂 it=40 冻结态，只跑一轮（it=41）。
- 日志全文仅两段（原文 4,328 B 随本批入仓 `ppou2_probe.log`）：
  1. **it=41 轮末 metrics 行**（与 `ppou2_probe_train.jsonl` 同行）：games 4,333、games_per_min 128.7、winrate 0.664、dec 1,291,323；分段 roll_s 2020.4／upd_s 1375.5／pre_s 502.3／buf_s 7.4；kl **0.1913**（kl_coef 0.05）、clipfrac 0.166、pl 0.0106、vl 0.0123、ent 0.712、gnorm 9.822；server 2 台 req 11,348／fwd 3,130／t_fwd 60.15／models 11；rss_mb 23,248、buf_mb 13,761.6；checkpoint 封印 SHA-256 与落盘件合（见检查点清单）。
  2. **停机清理警告**：resource_tracker 报 60 个 leaked semaphore 对象待清理（UserWarning，非报错）；看守随后确认进程组全空、残留 0。
- 全程无 traceback；MPS 回退 0 条（无 aten 回退算子、无 MPS 警告行）。看守 probe_watch.json 终态 STOPPED_ROUND_DONE；内存峰值 27,670 MB／64 GB。
- 整轮墙时 3,415.3 秒 ≈ 56.9 分钟（mtime 口径 56.6 分钟）。预注册判读：45 分钟线未过（详见探针回执与裁定 PPOU2_STUDIO_PROBE_RULING_20261116）。
