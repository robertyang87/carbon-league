# status  2026-11-12 00:00

| 机器 | 负载 / GPU | 在跑什么 |
|---|---|---|
| L20 | 10.07 14.04 16.45 100 % | out ../data/rl3/snipe_b
out ../data/rl3/snipe_fast |
| h23 | 42.43 16.63 10.95 | 600 ck_snipe_b_28_opp_master2r 600 ck_snipe_b_28_opp_w3b |
| h24 | 0.04 7.90 11.45 | 空闲 |
| DGX | 96 % | weights/cnet2_top_j1b.npz weights/cnet2_top_j1c.npz |

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
