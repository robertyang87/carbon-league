# Carbon League：基于 `master3s2` 热启动的 KataGo 训练与优化完整计划

- **目标指标**：在 0.58s/步 CPU 推理限时内，对 `master3s2.py` 达成 **胜率 > 90%** 且 **场均净胜 > +1000 分**，同时保持天梯泛化能力（0 超时、0 异常）。
- **算力基准**：单卡 NVIDIA L20（48GB VRAM）+ 多核 CPU 宿主机。
- **起点基座**：直接复用 `/home/yangjun/junyang/carbon-league/sources/master3s2/master3s2.py` 的 `_WEIGHTS`（6 层 96 通道 Global-Pooling ResNet，输入 54 通道）与 `_VALUE`（价值头）热启动，**不冷启动**；搜索框架基于已修复的 `/home/yangjun/junyang/carbon-gemini/ApexCarbon-V4TTA-Search.py`。

> **架构参数说明**：
> - `54` = 输入特征平面数（`NPL = 54`，形状 `(54, 15, 15)`，含碳分布、单位位置/载重、基地、树/树龄、5 方向必死通道 `25..29`、5 方向对撞通道 `30..34`、历史位置及经济标量平面）。
> - `96` = 网络隐藏层通道数（`stem` 将 54 通道映射为 96 通道，后续 6 个残差块均为 `96 → 96`）。

---

## 阶段 0：基础设施与基线准备（约 2~3 小时）

1. **提取 `master3s2` FP32 权重（Gen 0）**
   - 解包 `master3s2.py` 内的 `_WEIGHTS`（`stem`、`emb`、`6 × blocks`、`hw`、`hb`）与 `_VALUE`（`v1`、`v2`、`v3`）为标准 PyTorch `state_dict`（`gen0_m3s2.pt`）。
   - 数值对齐校验：确认解包后的 PyTorch 模型与 `master3s2.py` 的 `_WinoNet` 前向输出一致（最大误差 `< 1e-4`）。
   - 复用或封装 `pt → int8 + zlib + base64 → submission.py` 一键量化打包脚本，保证每代训练产物均可直接提交评测。

2. **固定标准考场（Exam Suite）**
   - 锁定 **200 局（100 种子镜像）** 对战 `master3s2.py` 的评测基准（如复用 `bigeval.py` 口径，固定种子块，单线程钉核）。
   - 每代统一记录：胜/负/和、场均净分及 95% CI、中位数分差、单步耗时 `mean / p95 / max`（红线：`max < 0.75s`、0 超时）、以及过程指标（`dep`、`tree`、`deathcargo`、`plants`）。

3. **盘点可复用语料（可选冷启动加速）**
   - 梳理 `carbon-league/data/` 中已有的高分天梯语料（如 `master6s_corpus`、`fablev8_corpus`、`krypton-earlyhome_corpus`、`sapling_top5_h2h_corpus`），作为 Gen 0 → Gen 1 热启动阶段的混合监督池，进一步压缩早期探索轮次。

---

## 阶段 1：Vectorized GPU 自博弈 + KataGo 训练框架（约 0.5 天）

### 1.1 网络结构（100% 兼容现有提交接口）

- **推理主干（提交保留）**：
  - `stem (54 → 96)` + `6 × GlobalPoolResBlock (96 → 96)` + `hw (worker 5-way)` + `hb (base 3-way)` + `Value Head (v1/v2/v3)`。
  - **可选无损扩容**（仅在 6×96 收敛见顶且 CPU 耗时允许时启用）：在原 6 块后插入 2 个 **Zero-Init Residual Block** 扩展为 `8 × 96`（初始输出为 0，第 0 步完全等价于 `master3s2`）。

- **训练期外挂 KataGo 辅助头（打包提交时直接剥离，零推理开销）**：
  - `aux_short_delta`：从全局特征预测未来 `+15 / +40 / +80` 步净碳差，显著降低 300 步终局回报方差；
  - `aux_fog_carbon`：1×1 卷积预测 `15 × 15` 全图真实碳分布（类似 KataGo ownership 头），迫使主干学会穿透战争迷雾推理资源；
  - `aux_win_logit`：预测终局胜率，与分差效用头共同稳定梯度。

### 1.2 Score Utility（胜率 + 分差联合效用）

为同时对齐 **>90% 胜率** 与 **>+1000 均分**，在搜索估值与训练标签中采用 KataGo 风格的联合效用：

$$U(\Delta) = 0.6 \cdot \tanh\left(\frac{\Delta}{600}\right) + 0.4 \cdot \text{clip}\left(\frac{\Delta}{2500}, -1, 1\right)$$

- 第一项在 $\Delta \in [-600, +600]$ 梯度陡峭，确保高胜率；
- 第二项在大幅领先（+500 ~ +2500）区间保持清晰正梯度，避免领先后躺平。

### 1.3 Vectorized GPU 自博弈引擎（L20 高吞吐核心）

- **并发架构**：宿主机多进程 CPU 负责环境步进与特征编码（并发 `128 ~ 256` 局），所有并发局的推理请求汇总到 L20 GPU 做 **动态大 Batch BF16/FP16 推理**（batch=128~512，单次前向仅 2~5 ms），单卡日吞吐可达 `10,000+` 局。
- **Playout Cap Randomization（变深度采样，提速 3~4×）**：
  - **75% 快回合**：浅搜索（1-ply 或带轻噪声采样）快速推进，为 Value 和辅助头提供低成本高覆盖的终局样本；
  - **25% 深回合**：完整宽/深搜索（`K = 24~32`、`D2 = 8`、双对手 `opp[0]/opp[1]`），产出高质量搜索估值 $Q(a_i)$ 作为策略蒸馏目标。
- **同时行动的边缘化策略目标（Marginalized Policy Target）**：
  - 针对 Carbon 多单位联合动作空间，对深回合的候选联合动作 $\{a_i\}$ 按 softmax 权重 $w_i = \text{softmax}(Q(a_i) / \tau)$ 加权，边缘化投影回每个己方 worker（5 类）与 base（3 类）的单单位分布 $\pi_{\text{target}}$，作为策略头交叉熵目标，并加轻度 KL 锚定防策略崩塌。
- **对手池配比**：
  - `50%` 当前最新模型自博弈（抬高策略上限）；
  - `30%` 对战固定考官 `master3s2`（定向收割弱点、积累大比分差样本）；
  - `20%` 对战历史最强 checkpoint 池（Hall of Fame，防循环克制与灾难性遗忘）。

### 1.4 V4 对称增强与训练超参

- **V4 群对称增强**：DataLoader 随机施加 4 种 V4 对称变换（`g ∈ {0, 1, 2, 3}`），同步变换 54 通道特征、方向危险通道 `25..34` 与 `hw` 策略目标。训练完成后权重天然具备 V4 等变性，可在推理端安全启用 V4-TTA。
- **Sliding Window Replay Buffer**：保留最近 3 代样本（约 300 万~600 万条状态），每代在 L20 上训练 2~3 epoch（AdamW，`lr = 1.5e-4 → 3e-5` cosine decay，BF16，batch=512，每代训练约 20~30 分钟）。
- **联合损失函数**：

$$\mathcal{L} = \mathcal{L}_{\text{policy\_w}} + \mathcal{L}_{\text{policy\_b}} + 1.5\,\mathcal{L}_{\text{value}} + 0.3\,\mathcal{L}_{\text{aux\_delta}} + 0.2\,\mathcal{L}_{\text{aux\_fog}} + 0.2\,\mathcal{L}_{\text{aux\_win}}$$

---

## 阶段 2：迭代自博弈进化（5~8 代，总计约 20~36 小时）

每代闭环（单代约 3~5 小时）：
1. **采样**：L20 向量化生成 `3,000 ~ 5,000` 局对局（含自博弈 + 对 `master3s2` + 对历史池）；
2. **训练**：L20 上更新主干、策略头、价值头及辅助头（~25 分钟）；
3. **打包 & 考场评测**：剥离辅助头、量化打包为单文件，跑 200 局标准镜像对战 `master3s2`；
4. **晋级门控**：仅当均分较上一代显著提升且无耗时/稳定回归时，晋级为主模型并入池。

### 里程碑预期

| 代数 | 累计墙钟时间 | 预期对 `master3s2` 胜率 | 预期场均净分 | 核心收益来源 |
|---|---|---|---|---|
| Gen 0（基线） | 0h | ~50%~55% | 0 ~ +50 | `master3s2` 权重 + 修复后搜索 |
| Gen 1~2 | ~8h | 70% ~ 80% | +350 ~ +600 | Critic 纠偏 + 辅助头 + V4 等变 + 深搜策略吸收 |
| Gen 3~5 | ~20h | 82% ~ 90% | +700 ~ +1000 | 开局树经济、抢树时机与残局回城系统性压制 |
| Gen 6~8 | ~32h | 88% ~ 93% | +900 ~ +1200 | 高水平自博弈策略收拢 + Score Utility 扩分差 |

---

## 阶段 3：蒸馏极速 Rollout 小网络 `_ROLL`（约 2~3 小时）

在 0.58s CPU 限时下将胜率锁死在 **>90%**、均分推过 **>+1000** 的关键放大器：

1. **瓶颈依据**：当前搜索约 85% 耗时花在中间层 rollout 推演（使用 6×96 主网算策略）。`master3s2` 原框架已预留双网络接口（主网 `_WNET` + Rollout 小网 `_RNET`）。
2. **蒸馏结构与方法**：
   - 设计 **2~3 层、32~48 通道** 极速 Rollout ResNet（仅保留 `lw/lb` 策略头，不含 Value 头），CPU 推理速度提升 **4~5 倍**；
   - 以阶段 2 最强主模型（Gen N）为教师，在最后 2 代回放池上做温度 KL 蒸馏（`T = 1.5`），L20 上 1 小时内完成收敛。
3. **搜索侧同步升级**：
   - 中间 rollout 层全部切到 `_RNET`，仅根节点与叶节点使用强主网 `_WNET`；
   - 同 0.58s 预算下放宽候选宽度（`K = 32 → 48~64`），让双对手验证（`opp[0] + opp[1]`）覆盖率接近 100%，并在叶节点开启已具备 V4 等变性的 `2× / 4× V4-TTA` 估值。
   - **预期额外增益**：在阶段 2 基础上再涨 **+150 ~ +300 分**。

---

## 阶段 4：最终验收与泛化防过拟合检查（约 2 小时）

1. **400 局大样本验收（200 种子镜像）vs `master3s2.py`**：
   - 胜率 `> 90%`，场均净分 `> +1000`（95% CI 下界稳过线）；
   - 单步耗时 `p95 < 0.55s`、`max < 0.75s`、`0` 超时、`0` 异常；
   - 打包后单文件体积约 `1.1 ~ 1.4 MB`，符合平台提交限制。
2. **跨对手泛化测试**：
   - 对 `carbon-league/sources/echo6/echo4_search_int8_d1_homefix.py` 及历史强 checkpoint 池各跑 50~100 局，验证不是只靠针对 `master3s2` 的过拟合，具备完整天梯冲榜实力。

---

## 工程落地文件规划（建议目录）

```text
carbon-league/
├── reports/designs/
│   └── katago_warmstart_m3s2_plan.md   # 本计划文档
└── tools/katago_m3s2/                  # 建议新建训练工程目录
    ├── extract_m3s2.py                 # 阶段 0：解包 master3s2 权重 -> gen0_m3s2.pt + 数值校验
    ├── model.py                        # 阶段 1：主干 + 训练期外挂辅助头 + 轻量 _ROLL 网络定义
    ├── vec_selfplay.py                 # 阶段 1：多进程环境 + L20 动态大 Batch 推理 + Playout Cap
    ├── train.py                        # 阶段 1：V4 对称增强 + Sliding Window + 多任务训练
    ├── distill_roll.py                 # 阶段 3：主网 -> 轻量 _ROLL 蒸馏脚本
    ├── pack_submission.py              # 打包：剥离辅助头 -> int8+zlib+b64 -> 单文件提交脚本
    └── run_loop.sh                     # 阶段 2：采样 -> 训练 -> 打包 -> 200局评测 -> 晋级闭环
```
