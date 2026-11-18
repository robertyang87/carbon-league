# 稀疏标签 /3 回放验证注册（工程组，2026-11-18，开跑前钉死）

依据：设计 c108965f、验收 a7da515d（判读三层、主口径 A、B 臂戳记对账、关键层只许 build 侧可观测量、双限纪律）。本注册在任何变体/基线开跑前钉死；跑后不许改。

## 采样器（join 层回放抽稀）

- sampler id：`q-sampler-replay/1`；种子 **20261118**。
- 根宇宙：源 joined 产品 gen-90faee2ef4221d59 中**训练局**的全部根（留出局整体排除，见下），按 (game, step, seat) 稳定序。
- step 段：[0,50) [50,100) [100,150) [150,200) [200,250) [250,299]。
- 抽法：逐层无放回，`np.random.RandomState(seed)` 按注册层序依次 `choice(n, drawn, replace=False)`；逐根 p_q＝该层实抽数/该层根数（精确包含概率），逐层实抽数入产品 meta。
- **甲臂（随机分层覆盖）**：层＝step 段×座（12 层），每层率 r∈{0.10, 0.25}，drawn＝round(n·r)。
- **乙臂（关键局面加标）**：关键集 K＝{step≥200 的根} ∪ {近分局的根}。近分局定义（build 侧可观测量：对局结果边车量，非 q/Q）：训练局集合上逐局 |terminal_margin|（该局 seat-0 根的终局分差绝对值，逐局恒定）的 25 分位阈值，|margin|≤阈值的局为近分局；阈值数值在抽稀时由训练局实算并写入注册记录与产品 meta。总预算 B＝round(N·r)；关键层抽数＝min(round(2.5r·|K|), floor(0.8·B))；余量均匀抽非关键根。p_q 分别按关键/非关键两层的实抽比记。
- **全覆盖（回归闸/基线）**：arm=full，has_q≡1、p_q≡1.0，采样块标 full coverage（验收 §1.2 钉死）。

## 留出局（稠密主读数专用）

- 种子 **20261119**，`RandomState(seed).permutation` 作用于 150 局名排序表，取前 **15 局**为留出局。
- 留出局**不进任何训练产品**（其终局标签不进训练——设计 §④ 价值校准防自评）；基线与变体共用同一留出集。
- 基线训练根数＝135 局×598＝80,730（源产品 89,700 根为留出前总根数，回执中明示此口径差）。

## 运行注册

- **基线跑**：稀疏 trainer（impl 副本），数据＝base_all1_train（全覆盖 /3），三身份同为教师件 cnet2_m3s2r1_heads_v2.npz（06a03d9c），口径 A（ipw），max_updates＝1000，bs＝64，lr＝3e-4，wd＝1e-4，seed＝7，device＝cuda。实测单跑 CPU-hours（进程 user+sys 累计/3600）＋墙时＋GPU 利用率，报作战室＋策略组设总额上限后才许开变体跑。
- **回归闸跑对**：stock trainer（仓 /3，5d28e2f7）跑源 /2 产品 vs 稀疏 trainer 跑 reg_all1_full（全覆盖 /3、口径 A），同 seed＝7、bs＝64、max_updates＝8、三身份同教师件；逐 update 比 batch_sha256 与 loss 逐项（浮点容差 1e-5 相对）。
- **B 臂戳记微跑**（正确性自证，非矩阵跑）：varA_r10 上口径 declared（声明权重全 1.0），max_updates＝4，验戳记 ipw.active=false＋权重表对账。
- 变体矩阵跑（A×2 臂×2 率＝4＋B≤2）**本单不开**，等双限批复。

## 稠密主读数（基线值）

留出 15 局上以真 q（源产品）为评分真值：学生 CE、锚 KL（逐 step 段分层）、utility MSE／wdl CE／margin MSE、学生策略熵。读数脚本 eval_dense.py，产物 JSON 落盘。
