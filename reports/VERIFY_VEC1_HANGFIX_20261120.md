# vec1 推理服务器静默挂死修复 独立核验备忘（工程组主代理，2026-11-20）

核验对象：执行人报告 `reports/REPORT_VEC1_HANGFIX_20261120.md`
（本地 `~/workspace/genexp/hangfix/REPORT_VEC1_HANGFIX_20261120.md`）与
修后件／diff／测试。本备忘只记我本人独立复核的结果。

## 1. 文件与 md5

修后件 vec_selfplay.py md5 **2dbd1485784c87733f437e911e9c408a**、
修前件 e59f07001d26ae1584fd126071ccf4fc、报告 44f2bd4f91726590331bc96d9917bd56、
diff 158 行——与执行人终报逐字合 ✓。DGX 镜像原件
`/home/arm/l20_katago_mirror/tools/katago_m3s2/vec_selfplay.py` 直读
md5 e59f0700（27,313 B），未动 ✓；DGX staging 树修后件直读 md5 2dbd1485 ✓。

## 2. diff 全文通读（仅故障路径）

改动四处，正常路径零改动：① 新增 `queue`／`time` 导入；
② 四个超时常量＋`_pid_alive()`（读 /proc state，`Z` 即死——
os.kill(pid, 0) 对 zombie 误判存活，此处绕开属实）；
③ `ServerClient.forward` 的无界 `resp_q.get()` 改为 30 s tick 轮询，
每 tick 查存活（死即 RuntimeError）＋累计超 300 s 即 TimeoutError，
响应到达后的 assert／res is None 分支原样保留；
④ `main()` 的 worker join 改 5 s tick 轮询（服务器死时给 worker 60 s
自行退出宽限、超宽限 terminate），`server.join(timeout=60)`、
超时 terminate 并记入 failed；`failed = []` 仅前移，无语义变。
超时值依据：300 s 为批 2 包 verify 最坏步时（23.745 s）的 12 倍以上、
p95（3.708 s）的 80 倍以上，不误杀健康慢轮；不设整跑绝对时限
（8,000 局跑数小时是正常态）——口径合理 ✓。

## 3. 证据复核（DGX 直读／重跑）

- **T2b 复现（修前挂死证实）**：t2b_repro.log——服务器 kill 后主进程
  +20／+40／+60／+80／+100 s 快照均 STILL ALIVE ✓。
- **T3 故障注入（修后）**：t3_inject.log——kill 12:26:36.056，
  主进程 35 s 后退出（12:27:11.114），rc=1 ✓；stderr 原文逐字合
  （两 worker `RuntimeError: inference server died (pid 3949146) … rid 320/319`；
  主进程 `SELFPLAY FAILED: worker 0 (exit 1, 4 games, 0 shards);
  worker 1 (exit 1, 4 games, 0 shards); inference server (exit -9)`）；
  日志记 `no vec_selfplay processes remain`（无残留）✓；
  出盘为空（无分片、无 config，失败跑不写完成标志，与原有失败语义一致）。
- **T4 正常跑回归**：DGX 直读 base_orig 与 base_fixed 分片 md5
  同为 **e9deed0a82c58efaa7b5a7a5aad73380**——修前 T1 与修后 T4
  逐字节同，正常路径输出未变 ✓（单 worker 下 vec1 本即逐字节确定）。
- **T5 超时分支**：我本人在 DGX staging 树上重跑
  test_timeout_path.py——A 活着但永不响应 10.0 s 准时 TimeoutError；
  B 等待中杀服务器 4.0 s RuntimeError；`TIMEOUT PATH TESTS PASS`，
  rc=0 ✓（与执行人报告逐字合；本地本 VM 因无 bridge 模块无法导入，
  故在 DGX 重跑）。

## 4. 如实记（执行人已披露，核验不推翻）

- T2 首次复现作废（误认服务器杀错 worker），不计入证据；T2b 为有效复现。
- 未在 GPU 上跑注入／回归（价值谱训占 GPU；故障路径与设备无关；
  D1 的 GPU OOM 触发点本身是服务器启动失败，与 T3 的 kill -9 等价）。
- `server.join` 超时分支未真实触发（防御性代码；STOP 后正常退出即时返回）。

## 判读与落仓口径

**修复核验通过，销项条件齐备。** 落仓口径：修后件以**新件**
`tools/katago_m3s2/vec_selfplay_hangfix.py`（md5 2dbd1485）落仓，
现行生产 vec1（e59f0700）原地不动；是否以此件替换生产 vec1、
由策略组／作战室裁定（Gen-6 生成已定用 vec2，本修复不堵任何在跑线）。
