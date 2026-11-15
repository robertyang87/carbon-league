# H28 K4 逐局线拉取 — 阻塞状态（2026-11-13 00:51 CST）

## 已完成（分支 sync/h28 上已有）
- 汇总 bigeval.jsonl：smoke + r1 + g1 三行齐全，g1 行与机上直读逐字一致（200 triples，46 胜 / 均 -544）。
- 分支头 be6fa25c @00:07:22（H24 机上回填）。live 文件 k4v2r1_e0.jsonl / k4v2g1_e0.jsonl 仍 0 行。

## 阻塞原因（已实证 4 次）
- DGX -> spot-28 (47.116.17.244:22) ssh 全部 Connection timed out：
  00:17（chunk 1-5，run 43256840041）、00:24（reach 探针 43258104076）、00:27（reach2 43258782816）、00:51（retest S28K4RT0050，run 43262984890）。
- DGX 自身出网正常（同趟 curl api.github.com 200）；本 VM 直连 :22 曾 TCP-OPEN（00:23），但本 VM 经 egress 代理 ssh 被 reset（198.19.0.1:3128），直连拉取此路不通。
- 机器本身未证实回收：H24 00:23 机上直读成功（g1 终值＋CHAIN_ALL_DONE）；:22 仍有应答。
- GitHub API 额度 00:30 耗尽（5000/5000），00:48:54 重置，现已恢复（00:49 查 remaining 4966）。所有者 00:36 定规：额度共用、节制使用。

## 驱动状态
- s28k4pull-r1.py / s28k4pull-g1.py 均已修好两处：① fetch_log 改 curl --http1.1 取日志（urllib blob 读必断）；② summary_games 改 JSONL 按 tag 选行（be6fa25c 合并后文件为 3 行）；③ claim 改 60s 轮询＋只看 dispatch 后创建的 run（省额度）。
- r1 驱动已停（pid 5815 已杀，00:40）；本地累积文件 0 行；未落盘任何 chunk。
- fetch_log_curl.py（curl 取日志）在 hidden_files/，两驱动共用。

## 恢复后操作（按序）
1. 先单趟探针复测 DGX->spot-28 ssh 通否（同 S28K4RT0050 配方）；通了再往下。
2. `nohup python3 s28k4pull-r1.py`（自动从 0 行续拉 40 chunk，每批 4 个，落盘后提交）。
3. r1 200/200 落盘验讫后，再跑 `s28k4pull-g1.py` 同法拉 g1。
4. 缺种子用 grep -E 按种子定向补；逐行对分支汇总 games 表验 (seed, side, margin)。
