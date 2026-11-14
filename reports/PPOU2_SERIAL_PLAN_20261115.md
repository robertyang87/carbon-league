# ppou2 双臂按轮边界串行实施方案

作战室 2026-11-15 01:34 裁定：(c) 串行采纳；(a) 原样重发否决；(d) 移机否决；(b) 缩 mb 冻结为后备，未经策略组批不得动备案 pin。
本方案报作战室确认后执行；确认前主臂不重发，对照臂不停。

## 0. 现状与约束
- 对照臂 c（λ1.00，PID/PGID 91224）：暖机 it=1..3 已完，锚已捕获验证（md5 573fb3af…，actor==init 180 键差 0.0）；首轮正式（it=4，warm=false）约 01:53（估计）。
- 主臂 m（λ0.95）：00:41:18 OOM 死，无任何 state；重发为 fresh 启动。
- 两臂配置除 out 路径与 lam 外逐字相同；effective_params 逐字相同。不改 pin、不改代码。
- 单臂峰值水位约 32–36 GiB（c 实测 36,468 MiB），卡 44.39 GiB；单臂余量约 6.8 GiB。

## 1. 切换机制（state 落盘为界）
- **边界信号**：运行臂 `train.jsonl` 新行落地，且该行 `checkpoint.files.state_last.pt` 的 sha256 与磁盘文件实测一致 → 该轮已完整落盘，可以切。
- **让出**：SIGTERM 整进程组（PGID，orphan-worker 协议口径）。验收三件：① 组内 PID 全不在册；② 该臂 `spawn_main` 归零（括号口径 `pgrep -f 'spawn_mai[n]'`）；③ nvidia-smi 无该臂进程。server 进程若残留（不假设一定同组），按 PID 补杀，再验 GPU 空。
- **接手**：单臂 launcher（v3 改：只起一臂，其余 argv/env/池 SNAP 编码逐字同 v3）＋ `--resume <对方 state_last.pt>`（m 首次为 fresh，无 state）。resume 只加这一个参数；kl/lr 等显式值与备案一致（R03：显式值胜过 checkpoint）。验收（5 分钟内）：进程在册、GPU 利用率 >0、league.json 11 键、spawn_main 15（14 worker＋1 tracker）。
- **在途损失**：切换时下一轮已在途的 rollout 部分作废，上限约一轮 rollout（约 26 分钟）；按检查点粒度切换，分摊 ≤2%。
- **驱动**：我 VM 上 nohup Python driver（与 anchor/first-round watcher 同模式，复用 remote_round_check.py 的读数方式），日志在 `~/workspace/carbon-live/ppou2/serial_driver.log`；VM 重启后重验在位（已知坑：nohup 不 survive 替换）。

## 2. 切换粒度与次序
- **首次切换**：确认后第一个轮次边界，且不早于 c it=4（首轮正式落地、首轮 KL 入账后再切）。
- **此后按检查点边界**：m 跑满 20 轮切，c 跑满 40 轮切，交替。
- **节奏**：m 每块约 7.6 小时，c 每块约 15.3 小时；距 11-20 12:00 硬停约 106 小时，一个完整循环约 23 小时，合**约 4.6 个循环**，m 约 92 轮、c 约 184 轮，各约 4–5 个检查点（估计）。（2026-11-15 01:45 作战室算术更正：原「约 5 个半循环／m 约 100 轮／c 约 200 轮」作废；时间盒仍成立，结论不变。）

## 3. 主臂锚重捕计划
- m fresh 启动后暖机 it=1..3；现有 anchor watcher 已同时盯两臂，it=3 warm 行一到即拷 `state_last.pt` → `state_anchor_frozen.pt`，记 md5＋sha256。
- 再用 `check_anchor_equal.py` 验 actor==init（对照臂同法已验：180 键，最大差 0.0）。
- 随后与首轮正式 KL 一并补交策略组（锚 md5 备案余项）。
- m 锚未捕获前不做任何检查点考试（无锚不考）。

## 4. 主臂重发开工前预估（标「估计」，按实测轮长算）
实测基数（对照臂）：it=1 含启动 38 分 40 秒；it=2/it=3 各约 23 分钟/轮。
- 以 m 启动为 T0：锚（暖机 it=3 落）约 **T0+85 分钟**；首轮正式约 **T0+108 分钟**；首个检查点 it=20 约 **T0+8 小时**。
- 若确认后首切在 c it=4（约 01:53），m 启动约 01:55（估计），则锚约 03:20、首轮正式约 03:43、it=20 检查点约 10:00（均为估计）。
- 残留风险：单臂峰值 36.5 GiB、余量约 6.8 GiB；更长的对局批仍可能顶穿单臂。监视在位；若单臂也 OOM，则按 (b) 后备提策略组批 mb，不自行改 pin。

## 5. 确认后执行步骤
1. 写 single-arm launcher（v3 改单臂＋`--resume` 透传），核 config diff（与在跑臂只差 out/lam）。
2. 挂 serial driver（nohup，验在位、验首读为真值）。
3. c 不停到首切边界；到界后按 §1 让出→验收→接手→验收。
4. m 暖机期间 anchor watcher 自动捕获；捕获后验 actor==init，报策略组。

## 校正二执行注（2026-11-15 02:26 策略组裁，02:30 执行）

- kl_coef 0.20 → 0.05，双臂同值（launcher v4 已改，dry-run 验过与 v3 留档仅 kl 一项不同）。主臂 0.20 版暖机 it=1 已停（SIGTERM PGID 99683，组 gone、GPU 空）并整目录归档，02:30:43 fresh 重启（PID/PGID 104993，--kl 0.05，resume null，验收四项全过）；对照臂下次 resume --kl 0.05。
- §4 开工前预估作废：主臂重发 02:30:43（实测，launch record）；运行中修正估计：暖机 it=1 约 02:57、锚约 03:45、首轮正式约 04:08、it=20 检查点约 11:30（均估计，按 m 首版 it=1 约 26 分钟＋约 23 分钟/轮推）。
