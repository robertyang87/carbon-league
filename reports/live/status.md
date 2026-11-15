# status  2026-11-12 07:07

| 机器 | 负载 / GPU | 在跑什么 |
|---|---|---|
| L20 | 15.96 13.26 12.25 41 % | out ../data/rl3/snipe_b |
| h23 | 89.88 59.33 30.14 | 200 pkn1r_cnet2_m2so 200 pkn1r_echo_e0 200 pkn1r_opp_m3echo 200 pkn1r_opp_m6r 200 pkn1r_sb30 600 ck_snipe_b_75_opp_master2r 600 ck_snipe_b_75_opp_w3b 600 pkn1r_opp_master2r 600 pkn1r_opp_w3b |
| h24 | 64.24 50.05 24.64 | 200 e0_opp_m6r 200 sb30_opp_m6r |
| DGX | 96 % | weights/cnet2_arch_nd.npz weights/cnet2_m6n1.npz weights/cnet2_n1r2.npz |

## 需要云端或用户定的事
- J1-a 显著为负：J2（八位老师）是否还开？建议先试"降低 Master2s 权重 / 给 Master3s 行加权"的 J1'。
- 二十三、二十四号机空闲，等第 32 轮检查点（约 0:20）。

## 对建议的回应
（暂无）

## 记录（Mac 当场执行的事）
- 00:20 J1b / J1c 在 DGX 开训（set_tw 输出：J1b Master3s 0.364、M6 0.386、K4 0.159、Godslayer 0.09、Master2s 0；J1c Master3s 0.632、M6 0.223、K4 0.092、Godslayer 0.052、Master2s 0）。
- 00:20 snipe_b 第 28 轮后拆成两路，各 30 进程：snipe_b（lr 1e-5，kl 0.05）、snipe_fast（lr 3e-5，kl 0.025，从 snipe_b 第 28 轮的状态分出）。
- 00:20 第 28 轮检查点对 opp_m3echo 400 对（二十四号机）。
- 00:25 起：两路每 5 轮的检查点（第 30 轮起）自动考 600 对（对 opp_w3b、对 Master2 真身），结果写进 exams.md 最上面。
- J1 的 M6 行、K4TacSearch 行没有加进对手池（开关文件未放）。
- 00:00 两路自我对弈：等 snipe_fast 存下第一个状态后，两路都 --resume，Grove1 和 M 风格网络的权重写 0（advice 00:35 #1），记进 pool_changes。
- 00:00 二十三号机：snipe_b 第 28 轮检查点 600 对（对 opp_w3b、Master2 真身）。
- 00:00 M6（136 局）、K4TacSearch（96 局）单独复刻已排在 DGX，J1b/J1c 训完就开；训完自动在二十四号机打 8 位包，对 Master2 真身和 E0 各 200 对（原始分差）。
- 00:00 回放名单加了 Carago-V2（含"Carago-V2微调"）。
- 00:10 判定点①：第 28 轮对 opp_m3echo +219 ± 78（400 对）→ 显著为正，按表继续。
- 01:15 判定点⑥：J1b、J1c 对 Master2 都不显著为负（+109 ± 70、+70 ± 84）→ 混训保留；J1c 四组换行在二十四号机开考（它此前从 00:20 起空着，在等 M6/K4 复刻）。
- 01:15 判定点②：snipe_b_30 过线（W3best +176 ± 63，Master2 +124 ± 43）→ 二十三号机补考 m3echo / M 风格 / 1C-v9 / Grove1 / 直接对 E0，各 200 局，结果出来报用户。
- 01:15 异常：snipe_fast 在 00:02 的重启时 GPU 显存不足（OOM）退出，此后只有 snipe_b 在跑（显存 30/44 GB，12 个模型）。不硬重启，以免把 snipe_b 一起挤掉；判定点⑤（两路比较）因此作废，snipe_b 独跑。
- 01:31 snipe_b_30 补考完：Master2 +124 ± 43、W3best +176 ± 63、M 风格 +149 ± 72、直接对 E0 +271 ± 94（127/200），无显著为负 → 上榜候选，打了 8 位纯网络包（3.7 MB），包核对和耗时进行中；结果出来请用户拍板。
- 01:55 snipe_b_30 8 位包核对：包减浮点 vs Master2 −4 ± 70（200 局，无报错），包原始 +877、168/200 胜；每步中位 36.5 ms，p99 37 ms，最慢 67 ms。等用户拍板上榜（Echo3）。
- 06:55 用户定：snipe_b_30 **先不上**。
- 06:55 对建议的回应：01:50 / 02:45 / 05:45 / 06:50 四条此前没有执行（Mac 在等用户拍板时停了，没有继续照清单做），现在照 02:45 的清单全部开工：
  - DGX（新地址 10.211.43.155）：N-D 底座（160×10＋4 层注意力，4 轮，v19m2s3 稳定配方）、`m6n1`（M6 带注意力复刻，从 N1 底座）、`n1r2`（Master3s 全部 142 局，带注意力，从 N1 底座）三项同时在跑，GPU 96%。
  - DGX 排队：Master3s2 回放满 100 局（现 88）由 Mac 自动送去，开 `m3s2r1`（N1 底座）和对照 `m3s2g1`（Grove1 起点）。
  - L20：`snipe_b` 第 75 轮存完后改 RL2_SERVERS=1、RL2_SERVER_MAXB=1024、--pre-mb 2048、--max-snaps 2，对手池加 `m6r1` 0.5；随后 `snipe_n1` 冒烟（256 局×2 轮），过了就开（从 n1r，对手池 W1 1.0 / W2 0.5 / m6r1 0.5 / 自己 0.5，30 个进程）。检查点考试守护进程加了 `snipe_n1`。
  - h23：n1r 8 位包（4.58 MB）对 Master2 真身、opp_w3b 各 600；对 M 风格、m3echo、直接对 E0、直接对 snipe_b_30、opp_m6r 各 200；E0 对 opp_w3b 补 994200 起 200 局。
  - h24：opp_m6r 定为固定考官（m6r1 对 E0 −7 ± 62，对 Master2 +292 ± 52，各 200）；E0 和 snipe_b_30 对 opp_m6r 各 200；J1c-d 对三个对手补到 600；E0 对 m3echo 补 994200 起 400；n1r 包耗时。
