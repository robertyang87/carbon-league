# 从当前局面预测 30 步后的现金分差变化

同一套 105 权重的小模型，在旧 21 张留出图和新增 26 张诊断图上，逐图均优于零变化、最近 10 步趋势和最近 30 步趋势的外推。其输出为双方沿当时策略再打 30 步的现金分差变化，以碳计。例：从席一看现在落后 500，预测 +200，预计 30 步后约落后 300。

| 评估集 | 图数／座位根 | 头 RMSE | 零变化 | 10 步趋势 | 30 步趋势 | 头 MAE | 非零变化方向命中 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 Top5 留出 | 21／3,780 | 293.92 | 600.31 | 908.40 | 556.96 | 220.89 | 76.0% |
| 新增诊断 | 26／4,680 | 334.50 | 843.85 | 913.20 | 622.78 | 252.86 | 80.4% |

每图等权。先在图内平均两席和各步的平方误差，再跨图平均、开平方。对照使用同一时间戳前的最近现金趋势；开局不足回看长度时用已有前史，第一帧输出零。

旧 Top5 留出相对 30 步趋势的 RMSE 差为 -263.05，以地图为单位配对重抽的 95% 区间为 [-300.52, -225.54]。

新增诊断相对 30 步趋势的 RMSE 差为 -288.28，以地图为单位配对重抽的 95% 区间为 [-322.97, -252.33]。

## 输入、训练与使用

对双方分别计算相同定义的 35 个量，涵盖现金、招工能力、工人结构、运力、货量与到基地的距离、敌方轻工对返程格的威胁、载货工附近的敌树、树的余寿和上一帧树收入、周围可见碳，以及过去 3／10／30 步现金趋势。地面碳仅从自己工人当步视野更新，实见记录的权重在 12 步内线性降为零。

以己方减对方的 35 维差量，加上早期与晚期的两组时间交互，共 105 个权重。按训练图等权 RMS 缩放，固定岭系数 0.01，无截距。训练只消费原清单 79 图、14,196 个座位根；每三步采一根，只纳入此后确有 30 个联合推进的根。权重包 18,220 字节，SHA-256 为 fff096dd217767db0c81f2679fa231fdd5e8e2f5b76e43f51fba1e847a84696b。

旧 21 图和新 26 图的 ID、种子和内容均与训练图互斥。新图此前已做战术研究，作为迁移诊断。模型输入使用该行动席的公开观察，地图身份、选手名、未来动作和全图碳真值未进入模型。在训练图上逐步污染不可见格、加入虚假全图和身份，180 次预测最大变化为零。同步交换双方玩家列表和观察席编号、保持同一棋手视角时，预测最大变化也为零。

接口要求逐步给同一席观察。比较某个后继前先复制该根的观察历史，分别更新各分支的合法观测。这个头估计后继出发的 30 步现金变化；从根到后继的现金变化须另计。两臂应在同一时间点比较。策略更换会改变兑现方式，替代动作的收益与 Sapling 自身续局质量仍需对照检验。

```python
from value import CashMargin30, History

head = CashMargin30()
history = History()
# 每一帧，从第零步开始
history.update(observation)
expected_change = head.predict(history, observation, configuration)
```

## 两席和失败局面

| 集合／席 | 图数 | RMSE | MAE | 平均有符号误差 |
|---|---:|---:|---:|---:|
| 旧 Top5 留出／0 | 21 | 295.71 | 222.13 | +40.69 |
| 旧 Top5 留出／1 | 21 | 292.11 | 219.64 | -37.72 |
| 新增诊断／0 | 26 | 332.81 | 251.55 | +0.06 |
| 新增诊断／1 | 26 | 336.18 | 254.18 | +6.45 |

旧样本席零平均高估、席一平均低估，合并两席会掩盖这一点。以下从各集合中按绝对误差筛出方向也判错的不同地图。随后 30 步各项现金差取自实际游戏入账。

- 旧 Top5 留出 c0261123，席 0，ClaudeFable5.2-v7 对 RyanX-AIPowercore-M9XJ-630。第 219 步，预测 -563.06，实际 +324.64。主要现金项为双方正常存入差 +222.83、种树支出差 +191.19、夺树支出差 -100.00。
- 旧 Top5 留出 b455a0c8，席 0，claude-AlphaCarbon-Master6 对 ClaudeFable5.2-v7。第 264 步，预测 +572.93，实际 -268.30。主要现金项为双方正常存入差 -474.90、双方树收入差 +100.71、夺树支出差 +100.00。
- 新增诊断 6e93f17d，席 1，RyanX-AIPowercore-M9XJ-630 对 小肥鱼v5-late2d。第 210 步，预测 +386.02，实际 -765.69。主要现金项为双方正常存入差 -583.43、双方树收入差 -477.90、种树支出差 +315.66。
- 新增诊断 62a40fb9，席 1，Echo4 对 claude-AlphaCarbon-Master6。第 138 步，预测 +64.30，实际 -812.23。主要现金项为双方正常存入差 -557.49、双方树收入差 -224.72、招种植工支出差 -90.00。

## 按对手

| 集合 | 对手 | 图数 | 座位根 | 头 RMSE | 30 步趋势 RMSE |
|---|---|---:|---:|---:|---:|
| 旧 Top5 留出 | ClaudeFable5.2-v7 | 9 | 810 | 306.48 | 586.58 |
| 旧 Top5 留出 | RyanX-AIPowercore-M9XJ-630 | 7 | 630 | 280.07 | 538.14 |
| 旧 Top5 留出 | RyanX-AIPowercore-it720-preview | 10 | 900 | 267.94 | 526.71 |
| 旧 Top5 留出 | claude-AlphaCarbon-Master5s | 8 | 720 | 267.06 | 501.02 |
| 旧 Top5 留出 | claude-AlphaCarbon-Master6 | 8 | 720 | 343.74 | 624.75 |
| 新增诊断 | AgyCarbon-P2C-Godslayer-CV | 1 | 90 | 336.49 | 756.22 |
| 新增诊断 | ClaudeFable5.2-v7 | 2 | 180 | 260.93 | 601.69 |
| 新增诊断 | Echo4 | 1 | 90 | 400.71 | 737.44 |
| 新增诊断 | Fable5.2-Max-V6 | 1 | 90 | 339.06 | 515.52 |
| 新增诊断 | Fable5.2-Max-V9 | 1 | 90 | 446.03 | 703.00 |
| 新增诊断 | First111 | 1 | 90 | 297.54 | 610.16 |
| 新增诊断 | First111-1 | 1 | 90 | 326.63 | 494.08 |
| 新增诊断 | Krypton-CS-Plant-v1 | 1 | 90 | 437.54 | 808.28 |
| 新增诊断 | Krypton-Cloud500 | 1 | 90 | 274.97 | 529.80 |
| 新增诊断 | Krypton-M2BC3s2-v2 | 1 | 90 | 301.20 | 617.13 |
| 新增诊断 | RyanX-AIPowercore-Lite-V8 | 1 | 90 | 514.51 | 842.22 |
| 新增诊断 | RyanX-AIPowercore-M9XJ-630 | 2 | 180 | 253.05 | 438.07 |
| 新增诊断 | RyanX-AIPowercore-it720-preview | 1 | 90 | 346.69 | 710.38 |
| 新增诊断 | claude-AlphaCarbon-Master3s2 | 1 | 90 | 264.51 | 424.78 |
| 新增诊断 | claude-AlphaCarbon-Master5s | 6 | 540 | 328.19 | 618.53 |
| 新增诊断 | claude-AlphaCarbon-Master6 | 16 | 1440 | 358.05 | 646.18 |
| 新增诊断 | stephen-K4TacSearch-v1 | 1 | 90 | 289.19 | 503.27 |
| 新增诊断 | stephen-K4TacSearch-v6 | 1 | 90 | 372.61 | 589.60 |
| 新增诊断 | 哈基米  v3 | 1 | 90 | 253.02 | 392.29 |
| 新增诊断 | 哈基米 v2 | 1 | 90 | 299.25 | 785.35 |
| 新增诊断 | 哈基米 v2 pro | 1 | 90 | 312.50 | 571.07 |
| 新增诊断 | 哈基米 v3 | 1 | 90 | 368.48 | 782.45 |
| 新增诊断 | 安生N29 | 1 | 90 | 389.47 | 614.78 |
| 新增诊断 | 小肥鱼v5-late2d | 6 | 540 | 268.72 | 572.33 |
| 新增诊断 | 测001 | 1 | 90 | 258.36 | 479.19 |

## 逐图误差

| 集合 | 图 ID | 席零／席一 | 头 RMSE | 席零／席一 RMSE | 零变化 | 10 步趋势 | 30 步趋势 |
|---|---|---|---:|---:|---:|---:|---:|
| 旧 Top5 留出 | 56169fed | RyanX-AIPowercore-M9XJ-630／ClaudeFable5.2-v7 | 277.4 | 280.8／274.0 | 357.5 | 961.6 | 594.8 |
| 旧 Top5 留出 | c0261123 | ClaudeFable5.2-v7／RyanX-AIPowercore-M9XJ-630 | 292.3 | 293.3／291.3 | 320.7 | 923.4 | 471.4 |
| 旧 Top5 留出 | e73135fa | ClaudeFable5.2-v7／RyanX-AIPowercore-it720-preview | 290.3 | 289.1／291.6 | 486.0 | 1057.3 | 648.7 |
| 旧 Top5 留出 | 2e09893b | ClaudeFable5.2-v7／RyanX-AIPowercore-it720-preview | 270.9 | 267.6／274.0 | 320.6 | 768.2 | 485.3 |
| 旧 Top5 留出 | 3377ed87 | RyanX-AIPowercore-it720-preview／ClaudeFable5.2-v7 | 219.3 | 219.7／218.9 | 295.9 | 1037.1 | 469.9 |
| 旧 Top5 留出 | f306679c | ClaudeFable5.2-v7／claude-AlphaCarbon-Master5s | 266.2 | 267.5／264.8 | 397.3 | 902.9 | 518.6 |
| 旧 Top5 留出 | ef8c2fbd | ClaudeFable5.2-v7／claude-AlphaCarbon-Master5s | 274.3 | 275.8／272.8 | 344.9 | 847.8 | 515.0 |
| 旧 Top5 留出 | 0e4aac28 | claude-AlphaCarbon-Master6／ClaudeFable5.2-v7 | 258.0 | 262.4／253.4 | 367.5 | 944.6 | 625.8 |
| 旧 Top5 留出 | b455a0c8 | claude-AlphaCarbon-Master6／ClaudeFable5.2-v7 | 502.4 | 509.8／494.9 | 1055.1 | 1194.4 | 847.1 |
| 旧 Top5 留出 | 9aae789e | RyanX-AIPowercore-M9XJ-630／RyanX-AIPowercore-it720-preview | 255.6 | 255.2／256.1 | 309.8 | 950.3 | 527.9 |
| 旧 Top5 留出 | eca9e0ff | claude-AlphaCarbon-Master5s／RyanX-AIPowercore-M9XJ-630 | 277.4 | 280.4／274.3 | 320.8 | 716.2 | 445.8 |
| 旧 Top5 留出 | ff09b130 | RyanX-AIPowercore-M9XJ-630／claude-AlphaCarbon-Master5s | 269.3 | 269.0／269.5 | 323.1 | 874.8 | 505.9 |
| 旧 Top5 留出 | 09a12e7f | claude-AlphaCarbon-Master6／RyanX-AIPowercore-M9XJ-630 | 350.4 | 344.7／355.9 | 1191.5 | 931.8 | 761.9 |
| 旧 Top5 留出 | 5a5f72bc | RyanX-AIPowercore-M9XJ-630／claude-AlphaCarbon-Master6 | 229.1 | 229.7／228.6 | 914.7 | 565.9 | 370.5 |
| 旧 Top5 留出 | 20ab1325 | RyanX-AIPowercore-it720-preview／claude-AlphaCarbon-Master5s | 250.0 | 250.1／249.9 | 357.7 | 1059.1 | 516.5 |
| 旧 Top5 留出 | 0a4b378d | RyanX-AIPowercore-it720-preview／claude-AlphaCarbon-Master5s | 212.2 | 214.0／210.4 | 315.4 | 840.2 | 461.5 |
| 旧 Top5 留出 | abb23c77 | claude-AlphaCarbon-Master5s／RyanX-AIPowercore-it720-preview | 194.1 | 191.2／196.9 | 259.4 | 805.3 | 437.4 |
| 旧 Top5 留出 | 5b4f5cea | RyanX-AIPowercore-it720-preview／claude-AlphaCarbon-Master6 | 392.3 | 402.9／381.3 | 1115.6 | 669.0 | 471.4 |
| 旧 Top5 留出 | 5fb0f13d | claude-AlphaCarbon-Master6／RyanX-AIPowercore-it720-preview | 343.0 | 344.8／341.2 | 465.5 | 1151.8 | 744.5 |
| 旧 Top5 留出 | d840e4de | RyanX-AIPowercore-it720-preview／claude-AlphaCarbon-Master6 | 205.4 | 205.5／205.3 | 289.2 | 808.1 | 413.2 |
| 旧 Top5 留出 | e71d5897 | claude-AlphaCarbon-Master5s／claude-AlphaCarbon-Master6 | 363.2 | 368.3／358.1 | 993.2 | 801.0 | 590.1 |
| 新增诊断 | 33a38179 | 小肥鱼v5-late2d／ClaudeFable5.2-v7 | 289.3 | 289.5／289.1 | 438.6 | 983.7 | 707.6 |
| 新增诊断 | 3e9f194c | claude-AlphaCarbon-Master6／小肥鱼v5-late2d | 290.9 | 286.7／295.0 | 404.3 | 934.8 | 648.0 |
| 新增诊断 | 62a40fb9 | claude-AlphaCarbon-Master6／Echo4 | 402.3 | 400.7／403.9 | 481.2 | 1017.6 | 737.4 |
| 新增诊断 | 881edae9 | Krypton-Cloud500／claude-AlphaCarbon-Master6 | 280.4 | 285.7／275.0 | 1242.8 | 904.0 | 529.8 |
| 新增诊断 | 8dd16c37 | claude-AlphaCarbon-Master6／哈基米  v3 | 253.1 | 253.0／253.2 | 820.1 | 610.3 | 392.3 |
| 新增诊断 | a81d56fe | First111-1／claude-AlphaCarbon-Master6 | 330.6 | 334.4／326.6 | 412.7 | 889.1 | 494.1 |
| 新增诊断 | e5dbdca2 | 小肥鱼v5-late2d／RyanX-AIPowercore-M9XJ-630 | 230.5 | 231.4／229.6 | 283.2 | 852.5 | 406.5 |
| 新增诊断 | 5ce27313 | claude-AlphaCarbon-Master5s／测001 | 257.8 | 258.4／257.2 | 400.8 | 761.0 | 479.2 |
| 新增诊断 | 6e93f17d | 小肥鱼v5-late2d／RyanX-AIPowercore-M9XJ-630 | 272.5 | 273.0／272.0 | 286.9 | 1107.0 | 467.5 |
| 新增诊断 | 840dabcc | claude-AlphaCarbon-Master5s／First111 | 302.8 | 297.5／308.0 | 431.2 | 831.7 | 610.2 |
| 新增诊断 | 86c15608 | claude-AlphaCarbon-Master5s／小肥鱼v5-late2d | 291.9 | 292.4／291.4 | 434.9 | 1016.7 | 661.5 |
| 新增诊断 | 8c51712e | 小肥鱼v5-late2d／ClaudeFable5.2-v7 | 231.9 | 228.8／235.0 | 352.4 | 703.4 | 472.6 |
| 新增诊断 | 9d97c112 | 安生N29／claude-AlphaCarbon-Master5s | 390.6 | 391.8／389.5 | 566.1 | 1054.0 | 614.8 |
| 新增诊断 | c8b53081 | Fable5.2-Max-V6／claude-AlphaCarbon-Master5s | 338.1 | 337.2／339.1 | 438.1 | 791.2 | 515.5 |
| 新增诊断 | ca4a03a7 | 哈基米 v3／claude-AlphaCarbon-Master5s | 366.7 | 364.9／368.5 | 514.2 | 1104.3 | 782.5 |
| 新增诊断 | 0dd8c601 | claude-AlphaCarbon-Master6／RyanX-AIPowercore-Lite-V8 | 528.1 | 514.5／541.3 | 1466.1 | 995.5 | 842.2 |
| 新增诊断 | 0ec137b7 | 哈基米 v2 pro／claude-AlphaCarbon-Master6 | 320.4 | 328.1／312.5 | 1086.2 | 695.9 | 571.1 |
| 新增诊断 | 17b212c8 | AgyCarbon-P2C-Godslayer-CV／claude-AlphaCarbon-Master6 | 337.1 | 337.7／336.5 | 449.0 | 1160.5 | 756.2 |
| 新增诊断 | 2063cf3e | claude-AlphaCarbon-Master6／Krypton-M2BC3s2-v2 | 305.2 | 301.2／309.2 | 1202.0 | 667.8 | 617.1 |
| 新增诊断 | 3d0d21d6 | stephen-K4TacSearch-v6／claude-AlphaCarbon-Master6 | 375.1 | 377.6／372.6 | 1117.3 | 819.9 | 589.6 |
| 新增诊断 | 529b2d23 | claude-AlphaCarbon-Master6／Krypton-CS-Plant-v1 | 444.2 | 437.5／450.8 | 1372.4 | 1086.4 | 808.3 |
| 新增诊断 | 5bd3791f | RyanX-AIPowercore-it720-preview／claude-AlphaCarbon-Master6 | 350.4 | 354.0／346.7 | 1104.7 | 933.1 | 710.4 |
| 新增诊断 | be51be0d | stephen-K4TacSearch-v1／claude-AlphaCarbon-Master6 | 285.5 | 281.7／289.2 | 330.9 | 836.7 | 503.3 |
| 新增诊断 | dd08a437 | claude-AlphaCarbon-Master6／claude-AlphaCarbon-Master3s2 | 267.9 | 264.5／271.3 | 795.0 | 621.5 | 424.8 |
| 新增诊断 | f2bf32ab | claude-AlphaCarbon-Master6／Fable5.2-Max-V9 | 454.7 | 446.0／463.2 | 1616.7 | 1050.4 | 703.0 |
| 新增诊断 | fd9f6d09 | claude-AlphaCarbon-Master6／哈基米 v2 | 303.3 | 299.2／307.3 | 1121.1 | 967.9 | 785.3 |

[冻结权重](cash_margin_30.npz) · [训练清单](frozen.json) · [预测接口](value.py) · [完整误差](scores.json) · [旧图逐根预测](old_val_predictions.tsv.gz) · [新图逐根预测](new_diag_predictions.tsv.gz) · [反例现金明细](failure_ledgers.json)

从本目录运行。OLD_REPLAYS 为原始 Top5 回放目录，NEW_REPLAYS 为新清单的缓存根目录。

```bash
OPENBLAS_NUM_THREADS=1 python train.py --old "$OLD_REPLAYS"
OPENBLAS_NUM_THREADS=1 python evaluate.py --old "$OLD_REPLAYS" --new "$NEW_REPLAYS"
OPENBLAS_NUM_THREADS=1 python verify.py --old "$OLD_REPLAYS"
OPENBLAS_NUM_THREADS=1 python failures.py --old "$OLD_REPLAYS" --new "$NEW_REPLAYS"
python report.py
```
