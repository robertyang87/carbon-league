# Sapling 120 万参数版

**实际可训练参数 1,206,476 个。** 棋盘骨干为 93 通道、6 个残差块，每个卷积 GroupNorm 分 3 组。原先的 96 通道设计实算 1,269,299 个参数，改为 93 通道后距目标 120 万仅多 6,476 个。棋盘输入 33 平面，标量输入 20 维，实体历史 8 步，实体注意力宽 96、1 层、4 头，动作解码器仍按前缀逐槽解码。与 55 万版相比，卷积容量和归一化分组一起改变。[模型初始化](../../code/init_sapling.py) · [旧版配置与对照](../SAPLING_551K_20261118.md)

| 制品 | 用途 |
|---|---|
| [int8 单文件 agent](../../data/weights/sapling_1200k/agent_int8.py) | 可直接导入其 `agent(observation, configuration)`。1,728,363 字节，风格槽 2，守卫开关与历史 Sapling 基线一致。 |
| [训练后权重](../../data/weights/sapling_1200k/sqrt64.npz) | 4,860,662 字节，64 次策略更新后的 FP32 权重，可继续训练。 |
| [初始权重](../../data/weights/sapling_1200k/initial.npz) | 4,854,182 字节，使用种子 551187 建网，供独立初始化和基线对照。 |

## 训练过什么

4 局自有规则策略示范，各取前 120 步、双席共 960 行，按局隔离的另外 2 局共 480 行验证。与 55 万平方根加权对照逐个核过六个分片的全部数组及有效权重，训练采样种子、更新数、批大小和学习率一致。两种结构的初始化、旋转增强随机流可能不同。训练仅使用策略损失。价值、胜平负和分差头留待真实终局标签训练，主干已随策略更新。当前权重未消费环 1 的 150 局或环 2 的 250 局教师标注。[训练记录、校验值和三件制品身份](training.json)

验证策略交叉熵从 1.3312 降至 1.1130。相同验证集上有 254 个允许招募的机会，示范实际招了 55 次，大网预测招 10 次，55 万对照预测招 54 次。这是教师动作前缀下的离线观测。[训练记录](training.json) · [55 万训练对照](../sapling_bootstrap/training.json)

## CPU 成本和短场表现

单核亲和、BLAS 单线程，在两份持久 agent 上交错送入相同观察。热调用每模型 472 次，冷首调用每模型另计 8 次。行为场使用另一批 4 张新地图，各换席一次，以相同规则策略为对手。下表中的 ms 是本机公开 agent 入口时间，CPU 小时的实际整场吞吐还包含考官和环境开销。[计时与行为记录](runtime/comparison.json) · [计时程序](compare_runtime.py)

| 实测项 | 55 万平方根加权 | 120 万平方根加权 |
|---|---:|---:|
| 单文件大小 | 845,288 B | 1,728,363 B |
| 热调用均值 | 5.261 ms | 10.352 ms |
| 热调用 p95 | 6.369 ms | 11.916 ms |
| 截至 t30 完成采集并交回 | 2/8 局 | 4/8 局 |
| 截至 t60 完成采集并交回 | 4/8 局 | 4/8 局 |
| 截至 t60 向本方基地正额交付总碳 | 184.58 | 90.62 |

大网平均推理耗时约为小网的 1.97 倍。它在这组地图上较早出现交付，至 t60 成功局数相同，大网交付总量更低。两组共 16 局均正常推进至截止步。[逐局行为](runtime/behavior.json)

## 数值与训练入口

新初始网、训练后大网及旧 55 万网都在带真实历史与动作前缀的同组局面上验证。三个工件的实际 int8 包与对应反量化推理共 1,770 次调用，动作差异 0。训练后的大网在所测 119 个有效槽上 FP32 与 int8 贪心动作一致，大网初始权重有 8 个槽因量化改选。正式蒸馏 CLI 的小样本兼容测试及其合成标签身份另列。[推理验证](parity.json) · [蒸馏入口烟测](distill_smoke.json)

从仓库根目录复现训练：

```bash
python code/init_sapling.py --preset sapling-1.2m --seed 551187 --out runs/sapling120-initial.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python code/sapling_bootstrap.py \
  --init runs/sapling120-initial.npz --out runs/sapling120-warmup \
  --balance sqrt-role-action --updates 64
python code/cnet2_pack.py runs/sapling120-warmup/trained.npz runs/sapling120-agent.py 2
```

正式评价沿用考官注册的整场分差，同时报告双方官方分数、存活和时长。先前 55 万三个变体的千局整场分差均低于随机初始基线，短场的交付改善没有带来更高整场分差。[55 万固定集报告与逐局证据](../EXAM_SAPLING_VARIANTS_VS_MASTER3S2_20261119.md) · [考官注册](../REGISTRATION_MASTER3S2_EXAMINER_20261116.md)
