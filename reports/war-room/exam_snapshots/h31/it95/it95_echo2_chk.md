# it95 检查点 Echo2 场（C2/spot-31）

- 被测方：snipe_n1 it95 冻结网（PPO-U 第 95 轮，2026-11-14 14:51:03 冻结，kl 0.1116）
  vs 对手：Echo2（claude-opus55-cloud-Echo2 真身）
- 对照基线：m3s2r1（U 前冻结态，Master3s2 替身、注意力版）
- 种子：999000–999199，同种子配对，各 200 局；判据 ΔU ≥0.02 且种子聚类 CI 下界 >0
- 包：code/exp/it95.py，md5 9e16085acd3549df210c6209ab3fd3cb，4,598,510 B
  （DGX 打包，--verify-pack rc=0；源权重 ckpt_95.npz md5 9ece5a33cdc6bc6730c5136d48976a07）
- 开考：2026-11-14 15:28:57 CST，链 rc=0 双场，完赛 15:33:03 CST

## 开考前检查（spot-31 直读 15:17:59 CST，全过）
- bigeval.py 354bcd1bc2dc8bc188c813f53d28c893／eval_v53.py 45792dfbea3101812836fe77e6752706（成对）
- certify_pack.py 1f4811b450d0c93417f6465c6835f215（按策略组裁定：此版签发证书标 TEMPORARY、不消费）
- fastsim 双处 aefe6c83826efae4de83bef8c643c80b（code/＋packs/）
- exp/echo2.py 15b6d754／exp/m3s2r1.py 73542cdd；机器 load 0.00、空闲

## 启动命令 env 段转录
- ENV_SEGMENT: EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
- CMD1: /root/carbon/.venv/bin/python bigeval.py exp/it95.py exp/echo2.py 999000 200 it95_echo2
- CMD2: /root/carbon/.venv/bin/python bigeval.py exp/m3s2r1.py exp/echo2.py 999000 200 m3s2r1_echo2_base95
- 档案首行 env 块：EVAL_* 均显式 null（CNET2_TEACHER/RL2_SNAP_TEMP/CNET2_TEMP_KIND）
- 快照环：smoke（2 局）开考前验活落盘，2 分钟全量快照逐趟核行数增长；完赛本地快照双场各 200 行

## 场一：it95 vs Echo2（启动器原文）
- wins 124，mean +255，se 50，lo2se 154，worst −1619，worst_ms 120.0，err 0，FAIL false
- a_md5 9e16085a／b_md5 15b6d754／harness_md5 354bcd1b
- 独立复算（本地快照 200 行）：wins 124，mean 254.9，se 50.2，lo2se 154.4，worst −1619.34，U mean 0.5893
- 种子完整唯一 999000–999199；err 全 null；hh 全 354bcd1b

## 场二：m3s2r1 vs Echo2（启动器原文）
- wins 145，mean +466，se 61，lo2se 345，worst −2290，worst_ms 154.0，err 0，FAIL false
- a_md5 73542cdd／b_md5 15b6d754／harness_md5 354bcd1b
- 独立复算（本地快照 200 行）：wins 145，mean 466.0，se 60.7，lo2se 344.5，worst −2289.74，U mean 0.6632
- 种子完整唯一 999000–999199；err 全 null；hh 全 354bcd1b

## 配对 ΔU（it95 − m3s2r1，同种子；裁决归策略组）
- mean −0.0739，sd 0.4333，se 0.0306
- 种子聚类 bootstrap 95% CI（20,000 重抽）：[−0.1336, −0.0132]
- 正态近似 95% CI：[−0.1340, −0.0138]
- Δmargin mean −211.1
- 判读注记：CI 上界 <0，未达 ΔU ≥0.02 且 CI 下界 >0 的筛查线；与 it85 Echo2 场 ΔU −0.0913 同向（对 Echo2 低于基线）

## 证书口径
- C2 本线未签发任何证书。策略组裁定已落记：修复版 certify 验收前签发的证书标 TEMPORARY、不得消费；重签不一致时临时证书自始无效、差异按 W06–W08 实弹实例上报；修复版验收后直接用修复版签发。
