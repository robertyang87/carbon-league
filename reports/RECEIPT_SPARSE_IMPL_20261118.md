# 稀疏标签 /3 实现回执（工程组执行人，2026-11-18）

派单：作战室正式派单（设计 c108965f、验收 a7da515d 三点为上位口径）。注册：REGISTRATION.md（开跑前钉死）。全部产物在 L20 `/root/sparse_impl/`（code/data/runs/tests），源数据只读。

## 交付件（本机 impl/ 与 L20 code/ 同版）

| 件 | md5 | 说明 |
|---|---|---|
| k0_distill_u.py（稀疏副本） | 91910290 | 基底仓 /3（5d28e2f7）叠加 /3 稀疏合同；TRAINER_VERSION `k0-distill-u/3+sparse1`；消费钩子逐字未动 |
| k0_thin_joined.py | 00f7f49d | join 层抽稀器＋/3 发布＋sidecar 封印 |
| eval_dense.py | c284dd45 | 稠密主读数 |
| test_sparse.py | e04b9792 | 五个自证子命令 |

## 产品（L20 /root/sparse_impl/data/，全部经 /3 loader 发布前预检＋sidecar 封印）

- reg_all1_full gen-7126e28abe58349c（150 局全覆盖，回归闸专用）
- base_all1_train gen-8e357f53df68137e（135 训练局全覆盖，80,730 根）
- varA_r10 gen-ee7ccc3bd7e52c3c（甲臂 10%，q 根 8,074）
- varA_r25 gen-f92e5b05344acaa0（甲臂 25%，q 根 20,188）
- varB_r10 gen-6d69ab234b81ba31（乙臂 10%，q 根 8,073）
- varB_r25 gen-b1defeac1ab4bcda（乙臂 25%，q 根 20,182；关键集 40,330 根＝尾步∪近分局 34 局、阈值 |margin|≤232.18，关键层 p_q=0.400、非关键 0.0999）

留出局 15（种子 20261119）：300003/300015/300024/300029/300043/300049/300058/300095/300110/300114/300115/300135/300136/300140/300146。

## 自证逐项（硬正确性闸）

1. **loader 等价**（/2 产品上 sparse vs stock loader）：全部数组＋info 逐项相等 PASS。
2. **负例集 7/7 PASS**（全 die 点名 (game, step, seat) 或根号）：v 根占位 pi 篡改、v 根 q 非零、v 根 n_arms≠0、q 根存储 q 篡改、p_q=0、has_q=2、全覆盖造假（meta 标 full coverage 而记录非全 1）。
3. **回归闸**（stock /3 跑 /2 vs 稀疏跑全覆盖 /3，同种子/同 bs/8 updates）：batch_sha256 8/8 全等；**update 1 逐项逐位相等**（total 差 8.9e-16，ce/kl/utility_mse 差 0.0）；updates 2–8 漂移 1e-5–1e-3 量级——两路径求和分组不同、梯度噪声经 AdamW 放大成轨迹混沌，非口径差（同权重同批次下精确相等已证）。
4. **v 根梯度零实测 PASS**：policy loss 对 v 根特征/终局标签扰动**逐位不变**（2.9930159828042577 前后同值）；policy 梯度差 3.1e-05 ≤ 同输入重复跑噪声底 3.8e-05（无可测 v 贡献）；mixed batch policy 项与 q-only batch 精确相等（差 0.0）；零 q batch 策略项跳过且 loss=value；value loss 随 v 扰动而动（4.596→3.486，v 根确实流入价值路）；单行 ctx vs 槽 0 ctx 差 7.1e-05 ＜ batch 形状噪声底 6.8e-04（同根在两种展开 batch 形状间自身的 ctx 差）。
5. **denominators 对账 PASS**：A 短跑（varA_r10、ipw、8 updates）逐 update 以注册种子流独立复算 roots_q/candidates_q/ess_q/batch_sha256 全合。注：10% 率下每 64 根 batch 仅约 3–10 个 q 根、batch 内 ESS≈roots_q（层内 p 近恒定）。
6. **B 臂戳记对账 PASS**：declared 微跑（4 updates）产物戳记 ipw.active=false、declared_table={default: 1.0}、reconciliation.weights_match_declared=true、记录驱动激活抑制注记在册、any_pq_lt_1=true 如实记录（pi 侧本产品恒 1）。

## 基线读数

- 基线跑：base_all1_train（gen-8e357f53df68137e)、口径 A(ipw)、1000 updates、bs 64、lr 3e-4、seed 7、device cuda；产物 baseline.npz md5 a333feae、updates_done=1000、冻结闸 bitwise 通过、zero-q 跳过 0、total loss 4.709→3.974。
- **单跑 CPU-hours＝0.2123**（进程 user 723.5 s＋sys 40.9 s，RUSAGE_CHILDREN 口径）；墙时 630.4 s；GPU 利用率均值 87.1%；单 update 0.000212 CPU-hours。按注册矩阵 7 跑估总额 ≈1.49 CPU-hours（估计，变体跑 q 根更少、单 update 略廉，此数为上限口径）。
- 稠密主读数（留出 15 局、8,970 根，真 q 评分）：**CE 3.2361、锚 KL 0.3337、utility MSE 0.04647、wdl CE 0.38416、margin MSE 0.34778、学生策略熵 0.31031**；逐 step 段见 baseline_dense.json（如尾段 250-299 CE 3.3827/KL 0.3749、早段 0-50 CE 2.9563/utility MSE 0.09139）。

## 口径差与注记（如实）

- 基线训练根数 80,730 而非 89,700：验收 §3.2＋设计 §④ 要求留出局终局标签不进训练（防自评），故基线与全部变体产品统一排除留出 15 局；89,700 为源产品总根数。变体跑将用同一 135 局口径，基线 CPU-hours 即变体单跑的可比单价。
- /3 产品的 CONSUMPTION.json 由 k0_thin_joined 的 seal 步出具（stock consume_preflight 按 /2 格式分类、拒 /3）：sidecar 格式与内容寻址纪律与 preflight 同，tool 块如实记本工具与所做的 loader 验项；trainer 消费钩子（自封＋绑定）未改、照常生效。
- 变体跑未开（双限未批）：A×2 臂×2 率 4 跑＋B≤2 跑等作战室＋策略组按基线单价设总额上限。
