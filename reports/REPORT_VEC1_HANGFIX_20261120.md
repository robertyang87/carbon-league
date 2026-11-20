<!-- 落仓头（工程组，2026-11-20）：执行人报告原件逐字转录（本地 md5 44f2bd4f91726590331bc96d9917bd56）；主代理独立核验见 reports/VERIFY_VEC1_HANGFIX_20261120.md（T2b/T3/T4 逐项 DGX 直读复核、T5 本人重跑 PASS）。 -->

# vec1 推理服务器静默挂死修复 — 执行人验证报告（策略组裁定 ②，76472417）

执行人：工程组（DGX）。执行人不落仓；主代理独立核验后落仓。
开工三件套：开工 2026-11-20 12:10（DGX 定位原件）；预期完成约 1 小时
（估计，侦察＋实现约 20 分钟、故障注入与回归约 25 分钟、报告约 15 分钟）；
实际定稿 12:35。进度口径：① 原件定位＋读码 → ② 修复＋diff → ③ 复现
（T2b）→ ④ 注入（T3）→ ⑤ 回归（T4）＋超时单元（T5）→ ⑥ 报告。
价值谱正式训（DGX PID 1898309）全程未动，完工时仍在册（elapsed 1:32:25）。

## 0. 结论先行

- 修复完成并验证通过，销项条件齐备：故障注入下主进程 35 秒内以
  rc=1 响亮失败（原为无限挂死），无残留进程；正常跑单 worker 同种子
  同口径分片与修前**逐字节同**。
- 修后件：`vec_selfplay.py` md5 **2dbd1485784c87733f437e911e9c408a**
  （原 e59f07001d26ae1584fd126071ccf4fc；改动见 `vec_selfplay_hangfix.diff`，
  158 行含上下文，仅故障路径）。
- 原件位置纠正：派单所写 `/home/arm/junyang/carbon/code/katago_m3s2/
  vec_selfplay.py` 不存在；DGX 上现行 vec1 是 L20 镜像件
  `/home/arm/l20_katago_mirror/tools/katago_m3s2/vec_selfplay.py`
  （md5 e59f0700，与仓内 `tools/katago_m3s2/vec_selfplay.py` 逐字节同，
  27,313 B）——消融跑亦从该件执行。修复在该件上做，镜像件本身未动；
  测试在自建 staging 树 `/home/arm/junyang/vec1_hangfix/tree/` 跑
  （bridge/layout/model 原样复制、src 符号链接到镜像 src）。

## 1. 故障机理（读码确认，与 D1 报告 §7.4 一致）

- worker 侧 `ServerClient.forward`：`self.resp_q.get()` 无界阻塞。
  服务器若在 `net.eval().to(device)`（D1 的 OOM 点）或运行中死亡，
  worker 永等；主进程随后在 `p.join()` 无界阻塞；`server.join()` 同样
  无界。三处均无超时、无存活检查。
- 另核实一处细节：死掉的服务器在进程表里呈 zombie（`Z`，见 T2b 日志），
  `os.kill(pid, 0)` 对 zombie 仍返回存活——故存活检查不用 kill 信号，
  改读 `/proc/<pid>/stat` 的 state 字段（`Z` 即死）。

## 2. 修复（故障路径のみ，正常路径零改动）

- `ServerClient.forward`：`resp_q.get(timeout=30 s)` 轮询；每 tick 查
  服务器存活——已死即 raise `RuntimeError('inference server died (pid …)
  while waiting for response rid …')`；存活但累计超 300 s 无响应即 raise
  `TimeoutError('inference server unresponsive for 300 s …')`。响应到达
  后的 `assert rid` 与 `res is None` 分支原样保留。请求发送、成批、
  响应解包一字未动。
- `_worker_main` 增加 `server_pid` 参数透传；`main()` 将 `server.pid`
  传入各 worker。
- `main()` 的 join：worker join 改为 5 s tick 轮询；服务器已死时给
  worker 60 s 自行退出宽限（worker 至多一个 30 s tick 即可发现），
  超宽限仍活着的 worker 由 main `terminate()`；随后 `req_q.put('STOP')`，
  `server.join(timeout=60 s)`，超时未退则 `terminate()` 并记入
  `failed`（原 `failed = []` 后置定义前移，避免覆盖该项）。
  明确不设整跑绝对时限（8,000 局生成跑数小时是正常态）——只给
  “等死服务器的 worker”设限。
- 超时值依据（报告正文有述）：DGX 批 2 包 verify 步时 mean 1.669 s／
  p95 3.708 s／max 23.745 s（争用态）；单次前向响应只是单步的一小部分，
  300 s 为最坏步时的 12 倍以上、p95 的 80 倍以上，不会误杀健康慢轮，
  同时把挂死 bound 到分钟级。`VEC1_RESP_POLL_SECS`／
  `VEC1_RESP_TIMEOUT_SECS`／`VEC1_SERVER_JOIN_TIMEOUT_SECS`／
  `VEC1_WORKER_JOIN_GRACE_SECS` 四个环境变量仅供测试时覆盖，
  生产默认即上述值。
- `py_compile` 通过；`diff` 全文见 `vec_selfplay_hangfix.diff`。

## 3. 验证（全 CPU 口径，避开价值谱训练的 GPU；gen4 ckpt 42888eab＋
##    hof gen0/gen1r/gen2、deep-frac 0、value-scale u、dtype bf16、threads 4）

### T1 回归基线（修前，games=2，workers=1）
- 12:14:03→12:16:21，rc=0，`selfplay done: 2 games`；分片 1 件 598 行
  （2 局×299）、11 键齐；md5 **e9deed0a82c58efaa7b5a7a5aad73380**。

### T2 首次复现（作废，自述）
- 以“最大 RSS”认服务器，误杀了一个 worker（三个 spawn 子进程 RSS
  均约 560 MB，无法区分）；主进程 +120 s 仍活是剩余 worker 未跑完
  （4 局约 288 s），不是挂死。结论作废，过程留档（t2 日志未随附，
  DGX 上 `t2_repro.log` 可查）；组杀清理后无残留进程。

### T2b 复现（修前，games=8，workers=2）——挂死证实
- 服务器正确识别：最低 PID 的 spawn_main 子进程（pipe_handle=12，
  首个 spawn），12:22:31.6 `kill -9`，随后呈 zombie（`Z`，RSS 0）。
- 两 worker 的累计 CPU 时间（TIME 列）在杀后 6 次快照中冻结于
  00:00:48、RSS 冻结；主进程 12:24:31（杀后 +120 s）仍活。
  结论：修前确为无限挂死，与 D1 观察一致。组杀清理，无残留。

### T3 故障注入（修后，games=8，workers=2）——响亮失败
- 同口径，服务器 3949146 于 12:26:36.06 `kill -9`；主进程
  **35 秒后退出，rc=1**（12:27:11.11）。
- stderr 原文（`inject_fixed.stderr.log`）：
  `RuntimeError: inference server died (pid 3949146) while waiting for
  response rid 320`（worker 0）／`… rid 319`（worker 1）；
  `SELFPLAY FAILED: worker 0 (exit 1, 4 games, 0 shards); worker 1
  (exit 1, 4 games, 0 shards); inference server (exit -9)`。
- 出盘为空（无分片、无 selfplay_config.json——失败跑不写完成标志，
  与原有失败语义一致）；ps 复核无残留进程。

### T5 超时分支单元测试（修后模块直接测）
- `test_timeout_path.py`（VEC1_RESP_POLL_SECS=2、
  VEC1_RESP_TIMEOUT_SECS=10）：A. 服务器 pid 为活着但永不响应的
  `sleep` → **10.0 s 准时 `TimeoutError: inference server
  unresponsive for 10 s (rid 1, …)`**；B. 等待中杀掉 `sleep` →
  **4.0 s `RuntimeError: inference server died (pid …) while waiting
  for response rid 1`**（3 s 时杀，下一个 2 s tick 发现）。
  `TIMEOUT PATH TESTS PASS`，rc=0。

### T4 正常跑回归（修后，games=2，workers=1，与 T1 同种子同口径）
- 12:27:25→12:30:09，rc=0，`selfplay done: 2 games`；分片 598 行、
  11 键齐；md5 **e9deed0a82c58efaa7b5a7a5aad73380**——与修前 T1
  **逐字节同**。正常路径输出未变（单 worker 下 vec1 本即逐字节确定，
  D1 已证）。

## 4. 交付物

- 本地 `~/workspace/genexp/hangfix/`：`vec_selfplay.py`（修后件，
  md5 2dbd1485…）、`vec_selfplay_orig.py`（修前件，e59f0700…）、
  `vec_selfplay_hangfix.diff`（158 行）、本报告、
  `test_timeout_path.py`、`t2_repro.sh`、`t2b_repro.sh`、
  `t3_inject.sh`、DGX 日志副本 `t2b_repro.log`、`t3_inject.log`、
  `inject_fixed.stderr.log`、`inject_fixed.stdout.log`、
  `repro_orig.stderr.log`。
- DGX 原件：`/home/arm/junyang/vec1_hangfix/`（staging 树
  `tree/tools/katago_m3s2/vec_selfplay.py` 即修后件，md5 已验；
  `out/` 下 T1/T2b/T3/T4 产物与日志；`tree/src` 为到镜像 src 的
  符号链接）。镜像原件未动（md5 e59f0700 复核如初）。
- 待主代理：独立核验后落仓（修后件＋diff＋本报告），新 md5 报
  策略组登记；是否以此件替换生产 vec1、由策略组／作战室裁定——
  Gen-6 生成已定用 vec2，本修复不堵任何在跑线。

## 5. 未做与如实记

- 未在 GPU 上跑注入／回归（价值谱正式训占 GPU，DGX 内存 104/121 GB；
  故障路径与设备无关，CPU 口径已覆盖三条分支；D1 的 GPU OOM 触发点
  本身是服务器启动失败，T3 的 kill -9 与之等价——服务器进程死亡）。
- 未测 `server.join` 超时分支的真实触发（STOP 后服务器正常退出是
  即时返回；该分支为防御性代码，与 worker 超时分支同机理）。
- T2 作废经过如实记于 §3，不计入证据。
