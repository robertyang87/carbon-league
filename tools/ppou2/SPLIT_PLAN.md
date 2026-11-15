# ppou2 分臂方案（所有者 2026-11-15 14:08 令，作战室转达）

## 落位裁定（本线自定，按 state 完整性与机器稳定性）
- **主臂 m 留 gpu-l20-6**（周包稳定机、不可抢占）：本线主线（Echo4 出处、it=40 检查点在即），不放抢占机。
- **对照臂 c 迁杭州 J 区抢占机 carbon-l20-spot-1**（gn8ia.8xlarge／L20 48G，¥6.4435/时，释放率 0–3%）：被回收的损失上限＝1 轮（ckpt-every 1、state 逐轮落盘），且另设 state 回同步兜底（见下）。
- 串行 driver 就此退役：两臂各机独立连续跑，不再轮换。检查点考试仍全部在 gpu-l20-6 的 probe_m2 跑（c 的 ckpt 先回拷 gpu-l20-6 再打包考试）。

## 现行体积（2026-11-15 14:10 直读 gpu-l20-6）
- .venv 5,615 MB（torch 2.15.0／numpy 2.5.4／scipy 1.18.1）；code 3,758 MB；data/weights 392 MB；rl3/ppou2_c 608 MB；rl3/ppou2_m 583 MB。源盘 197G、余 156G。
- c 现 it=10 在跑（kl 0.0516 @ 0.15）；m 停 it=20（state_match 真）。

## 复制路线（IP 到达后按序试）
- A（首选）：gpu-l20-6 直连新机 tailnet rsync 全量（含 .venv），关键文件 md5 对账（state_last.pt、init npz、trainer、league.json）。
- B：新机以阿里云 PyPI 镜像同版本重建 .venv（同路径 /root/carbon/.venv），其余经本 VM 中继 rsync（≈4.7 GB）。
- C：全量中继（最慢，约 2–3 小时），仅兜底。
- 目标机前置核验：tailscale 在网、本线 ssh 钥可用、根盘余 ≥30G、Ubuntu 同族（.venv 二进制兼容）。
- **裸镜像定案（作战室 2026-11-15 14:14）**：新机 Ubuntu 24.04、无 NVIDIA 驱动，不换镜像——本线装驱动，预算 +约 30 分钟（估计），与文件复制并行（ETA 不叠加）。源机基准（14:15 直读）：driver 580.126.09／CUDA 13.0／Python 3.12.3／Ubuntu 24.04.5、kernel 6.8.0-147——目标同族，.venv 整拷可用。装驱动脚本 install_driver.sh（580-server 优先、nouveau 拉黑、modprobe 验活，失败走一次重启）。钥：本 VM 钥已内置 bootstrap；同步钥 spot_sync 由 C3 在 bootstrap 后追加并读回验证。

## 分臂执行序（IP＋复制就绪后）
1. 先停看门狗 cron＋serial_driver（防它在新布局下误拉 c 回 gpu-l20-6），再动任何臂。
2. c 在 gpu-l20-6 轮边界停（state_match），整组验收（组灭＋GPU 清零）。
3. ppou2_c 运行目录收尾同步到新机（state_last.pt／train.jsonl 停后刷新），md5 对账。
4. 新机 resume c（launch_ppou2_v5.py --kl-value 0.15；state 已 0.15，remote_edit_kl 读回确认），验三件套。
5. gpu-l20-6 resume m（v5 --kl-value 0.15），验三件套。m it=21 为主臂 0.15 首轮校准点，落地当轮报策略组。
6. KL 看护改双机版：本 VM 轻量 watcher 每 60 秒读两机 remote_kl_check（calib_pending／≥0.10 报先／≥0.13 触发即 HOLD 该臂待裁——两臂现均为 0.15，触发后一律 HOLD），由 cron 看护拉活。
7. 抢占兜底＝**2 分钟同步环（所有者 2026-11-15 14:13 补令，覆盖原 ≤30 分钟口径）**：拉式环跑在 gpu-l20-6（稳定机、不随本 VM 替换死），每 120 秒从新机拉 state_last.pt／train.jsonl／eval.jsonl／league.json／ppou2_c.log／新 ckpt_*.npz，落点 **gpu-l20-6:/root/carbon/data/rl3/ppou2_c_mirror/**，环日志 /root/carbon/data/rl3/c_mirror_sync.log（逐趟时间戳＋rc＋镜像 train 行数）。脚本 sync_c_mirror.sh（setsid 启动、PGID 落盘）。鉴权：gpu-l20-6 新生成专用钥 /root/.ssh/spot_sync（公钥已发作战室请 C3 在 staging 加入新机 authorized_keys）。
   **开训前验环（硬序）**：环先起 → 首趟落盘（镜像 state md5 与新机源对上）→ 探针验增（新机侧写 sync_probe.txt，一个环周期内出现在镜像且内容合）→ 两项回执到手才许在新机开训。开训后巡检口径：镜像 train 行数／state mtime 与源端差 >1 个环周期即异常当场报（训练产出逐轮落行，轮内零字节属正常，看的是源端有新行而镜像没跟上、或环日志断趟）。真被回收则在 gpu-l20-6 以镜像 state 恢复 c（退回串行或等新机），损失 ≤ 约 2 分钟＋在飞半轮。

## 进度口径与检查点（13:41 新规）
- 进度：两机各自 train.jsonl 轮次（已完成轮/目标轮）＋state_last.pt mtime；driver 日志退役后看 watcher 日志。
- 检查点位置：gpu-l20-6 /root/carbon/data/rl3/ppou2_m/ckpt_*.npz；新机 /root/carbon/data/rl3/ppou2_c/ckpt_*.npz（回同步镜像在 gpu-l20-6 /root/carbon/data/rl3/ppou2_c_mirror/）。
- ETA（估计，IP 到达后起算）：核验约 5 分钟＋复制 20–60 分钟（看路线）＋分臂开训约 15 分钟，合计约 45–90 分钟（估计）。
