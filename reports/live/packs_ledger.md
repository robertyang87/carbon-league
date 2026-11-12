# Pack 台账（packs ledger）

打包 → 上天梯的唯一登记处，流程见 `reports/live/pack_submit_runbook.md`。每个 pack 一行；作废包标 VOID，各线见 VOID 的 md5 一律拒收（不进考场、不提交）。

| 名称 | 源权重（md5） | 打包器 | CONF | pack md5 | 字节 | 等价性 | 计时 | 状态 |
|---|---|---|---|---|---|---|---|---|
| n1r | snipe m3n1 线权重（int8 打包） | cnet2_search_pack.py（原在所有者 Mac 上执行，2026-11-13 固化入 main） | 原版 13 键（PACK_CONF_TEMPLATE.txt） | ba7f84c3652ea2cd14a2243f2b1eace2 | 4,579,266 | 已过（对 Master2 真身合并 600 对原始 +969，按新锚折算 +184 ≥ 150 闸；对 snipe_b_30 头对头合并 600 对 +110.3） | 已过（2026-11-12 cgroup 1.5 核/512MiB：init 138.9ms、单步 max 75.6ms、p50/p95 38.7/63.8ms、慢步 0、整局 24.64s、内存峰约 91MiB；在进程内计时，天梯 IPC 未测） | 全部闸已过，**未提交**（所有者未决，与 Echo3 是两回事） |
| Echo3 | cnet2_m3s2r1L.npz（b6db08c9b076257c54817377bb93130d） | cnet2_search_pack.py（v12 件套，NO_ROLL=1） | 原版 13 键逐字 | 7b7620499411a088cf4dc17c0e929bd7 | 4,576,790 | **已过**（C2／spot-30：对 Master2 真身 200 局 seed 996000，+1192±58 对基线 +1197±63，187 胜、err 0；Echo2 场 +410±56 带内、144 胜、err 0） | **已过**（C03 修复版工具、cgroup 1.5 核/512MiB：init 145.0ms、单步 max 97.6ms、慢步 0、整局 20.05s、内存峰值约 90MiB） | 验证全过（2026-11-13）；所有者 03:14 已令推送，作战室提交天梯；pack 文件入仓 `code/packs/echo3_int8.py` 待 git 通道（原件同值在位于 L20／DGX s26stage／spot-30，md5 已多方验同） |
| （无名，VOID） | cnet2_m3s2r1L.npz | cnet2_pack.py（裸 agent 打包器，**用错**） | 装不下 13 键 CONF（仅 teacher=2 余全关） | 28d09892f6a2e9173b8127ed8db4bae6 | 4,443,283 | — | — | **VOID**：2026-11-13 作废，永不进考场、永不提交；C2 见此 md5 直接拒收 |
