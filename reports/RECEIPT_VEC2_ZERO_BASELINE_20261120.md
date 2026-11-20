# vec2 闭环重开 · 第一步收据：零基线补测（vec1 对 vec1）

## 头号数字（只报数，不判分叉）
- **margin KS（零基线）= 0.06272401433691754**，线 0.05（暂行、待校准）。
- 与线的距离：**高出线值 0.0127**（约为线值的 25%）。
- 与 vec2 复测值 0.06989247311827958 的关系：**落在同一量级**——基线比复测值低 0.0072（约低 10%），两者同在 0.06–0.07 区间、同在线值之上。

## 分布腿 (b) 全数（gate 1a9a7a26，n=120/臂）
| 指标 | 实测 | 线 | 判 |
|---|---|---|---|
| margin KS | 0.062724 | 0.05 | over |
| mean margin shift / pooled SD | 0.014906 | 0.05 | 内 |
| win-rate shift | （通过） | 0.02 | 内 |
| searched-frac shift | （通过） | 0.01 | 内 |
| rows-per-game 相对差 | 0.032258 | 0.01 | over |
| aux_delta KS h1 / h2 / h3（对照） | 0.00630 / 0.00656 / 0.00855 | 0.05（正式线，不动） | 内 |

- margin 均值：旧臂 −27.75，新臂 −42.13；SD：旧 710.8，新 1164.7；分位（25%/中位/75%）：旧 [−946.06, −21.61, 862.42]，新 [−955.36, −80.83, 840.66]。
- 行数：旧臂 55,614 行（463.45 行/局），新臂 53,820 行（448.50 行/局）；schema 两臂 problems=[]。
- gate_pass=False：其打印的「不过→弃置」是闸对 vec_selfplay2 的内置结论文案，**不适用于本次基线测量**——本次交付物是实测 KS 数，不是 vec2 判定。分叉（线未校准 vs vec2 实差）由策略组裁。

## 方法（与 L20 vec2 复测逐项同口径）
- 种子块：968400–968519（策略组登记 d9218d1d）；A 臂基数 968400，B 臂基数 968460，逐局种子 = 基数 + 1000003×gid（gid 0..119），240 个逐局种子互不重合（差恒为 60）。
- 双臂直接调 vec_selfplay.py（gate 的 --run-games 把同一 --seed 传双臂，会使 vec1 对 vec1 逐字相同、KS 恒为 0，故不用）；闸以 --old-dir/--new-dir --games 120 只跑 (b)＋schema，前向腿与速度腿未跑。
- 参数：--src master3s2.py（57f9a594）、--ckpt work0/gen2/ckpt.npz（镜像 md5 b944a2b9）、--deep-frac 0.0、--device cuda、--dtype bf16、--value-scale u、--threads 4、--workers 24、无 --hof；每臂 120 局。
- 文件身份：gate 1a9a7a26（仓内合）、vec1 e59f0700（仓内合）、layout 234ea89e / model f9e6fe7d（DGX 镜像件，与仓内现头仅注释差）、bridge f31b07dc。
- 座位：DGX（GB10，torch 2.14.1+cu130）。A 臂约 5 分钟、B 臂约 6 分钟（墙钟，非测速腿）。L20 未碰；Gen-4/NETEXP 未碰；vec_selfplay2.py 本趟未运行。
- 运行件：RUN_ARTIFACT.md（种子块＋派生＋命令原文，跑前已钉）；闸报告全文 gate_report_baseline.json；双臂 config：armA_config.json / armB_config.json。

## 未做
- 第二步的任何候选（bf16＋TF32 全关变体、改判据、吞吐重测）一概未碰；aux_delta 的线与判据未动。
