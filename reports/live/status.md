# status  2026-11-11 23:50

| 机器 | 负载 / GPU | 在跑什么 |
|---|---|---|
| L20 | 25.03 23.22 18.65 99 % | out ../data/rl3/snipe_b
out ../data/rl3/snipe_fast |
| h23 | 0.00 0.49 9.56 | 空闲 |
| h24 | 31.75 8.67 8.19 | 400 snp28_m3echo |
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
