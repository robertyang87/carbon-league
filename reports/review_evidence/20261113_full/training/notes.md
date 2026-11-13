# 第五轮：训练、联合动作与推理链复核

固定源码 **98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c**。这是该提交的独立结论，不覆盖随后新到的 heads/model/train 修改。只读业务源码；无优化器更新、真实训练、云任务、远端调用或部署。AWR 定向测试运行实际生成器的固定循环，策略由确定性动作函数替代；并非训练或强度测量。

## 主要缺陷：AWR 生成与重打分未接入灭绝终止（与 data 的 DAgger/LB 合并）

**P2，确定可达。** `code/awr_gen.py:65` 始终写 ACTIVE，`:68–71` 无条件推进至配置步数，`:81–82` 以最后现金差当整局奖励；`code/awr_weights.py:43–45` 又按回放长度判断 whole 并从最后现金重算，`:54` 将零分差当 mirror 剔除。

复用 data/recheck_labels.py 已经由原版引擎验证的合法全局动作序列：我方 step0/1 都向第一个基地招募 collector，对方 step0/1 分别向两个基地招募；原版在 step2 终止，奖励 `[-299,50]`，我方分差 **−349**。本探针不替换 FastCarbon、不改生成循环，只替换模型调用为上述确定动作，调用实际 awr_gen.play，并接实际 builder/scorer。

| 链路 | 本轮实际结果 |
|---|---|
| awr_gen，AWR_STEPS=299 | 走到 step299；保存 300 个状态且全部 ACTIVE；reward=0 |
| cnet2_build.build_game | 产 299 行，其中 297 行来自本应已结束的 step≥2；value 全 0 |
| awr_weights.rescore | reward2=0，mirror=True；真实败局被当作无信息平局排除基线并赋零游戏权重 |

因此 PPO 本体 F01 修好不等于所有造数入口修好；这是旧入口的独立遗漏，未在本轮差异中新增。修复应统一终止检测、原版终局奖励、terminal/truncated 原因和 builder/scorer 读标签口径；提前灭绝局不能按 `len<300` 自动当作未结束的身家片段。与 data 的 dagger_gen/lb_relabel 同一个根因合并报告。没有统计其在现有语料中的自然发生率。

证据：`awr-results.json/terminal_semantics`；`check_awr_pipeline.py`。原版合法序列权威结果引用 `../data/label-results.json/reference`，未重复跑原版环境。

## 推理服务失败契约：权重加载异常未回传给 worker

**P2，失败路径。** `code/rl2_ppo_server.py:87–92` 的 load_npz/make_model/to(device) 位于 `try`（:94）之前。模型文件缺失/损坏或构造失败会直接退出 `_serve`，不会执行 :101–103 的错误响应；Client.collect 的 :129 默认等 600 秒才能因队列超时退出。同服务其他待处理请求也收不到响应。

用真实 `_serve`、真实 CNet2 adapter/loader 与内存 queue，向不存在的临时权重路径请求，结果为 FileNotFoundError，响应队列为空。未启动进程、未等待 600 秒、未操作真实权重。它证明异常处理缺口，不证明当前训练机器发生过文件丢失。建议把模型加载和缓存更新包含在逐组 try 内，向该组每个 rid 返回明确错误；worker 另检查 server 存活，避免因一个不可恢复加载错误等待完整超时。

证据：`network-ppo-results.json/server_missing_weight`。

## AWR 精度边界：回应“先查实际范围即可”

Muse 未动这项代码、也未宣称发生过事故，当前证据支持这个事实边界。区别是：**当前数据范围检查可以放行一个固定数据制品；现有 AWR 算法并不保证未来权重安全。**

`awr_weights.py:71–79` 仅有 `wmax=20` 上界与 ESS≥20% 要求，正权重没有下界。ESS 控制大权重集中，不能排除个别极小正 w；`awr_collector_anchor.py:18–20` 用 mean(w)/w 后，`cnet2_train.py:200–213` 把 w 与 uw 分别压 half，可能破坏本来有限的常数乘积。

本轮补了实际入口证据，不再直接喂极小 w：构造 40 个 seed 组、每组 5 个分差标签；199 局 +100，一局 −100，全部非平局。用实际 awr_weights CLI，所有超参数默认，得到 ESS=0.211（通过）、min_w=3.2104e−6、max_w=20。再接实际 anchor CLI，finite=True，但 max_uw=428293.7；经过原 cached 转换后有 inf，提取的原 batch_loss 输出 NaN。

**限制：** 这是终局标签分布的代数 fixture；供 scorer 读取的早期回放项是占位，不声称这些 fixture 是完整合法轨迹或来自当前语料。它证明权重计算入口没有安全范围保证，不是生产事故/频率证据。已有实际数据 min positive w/max uw 未取到，仍待运行前核验。

最低可接受守卫：每份固定 dataset 记录 hash，检查实际转换后的 w/uw 及有效乘积有限，检查原正 w 是否下溢为 0；不只检查原 float32 的 finite=True。长久方案是权重保留 float32，或先在 float32 合成有效单位权重再保存，不靠随意截断 uw 改变目标。若批次构建/加权会更新，旧制品的范围检查不能自动批准新制品。

证据：`awr-results.json/weight_precision`。未将此边界升级为已发生训练事故。

## 重新验证通过的热点

本轮没有只以 Git hash 代替测试。用随机但固定的小型真实 CNet2（C=16、NB=1、D=16、L=1、TA=1），主动给 TA 非零权重，测试 28 个局面：双座位合法开局推进、step239/240/298 的构造 style 边界、10 工人+2 基地的构造满槽状态。小网络验证接口与数值实现，不代表最新大权重的资源或强度认证。

| 检查 | 结果 |
|---|---|
| NumPy actor 记录完整联合 logp → PackedTraj/Buffer → Torch evaluate | 最大误差 4.77e−7 |
| value NumPy↔Torch | 最大误差 1.22e−7 |
| forward 与 decode / packed forward_seq | 有效 logits 最大误差 5.96e−8；两条 GRU 路径梯度最大差 1.12e−8（未更新参数） |
| 非零 cell attention、单位 attention、prefix teacher forcing | 同上数值通过，覆盖有 padding 和全 12 个己方槽 |
| UID 历史随 cargo 排序换槽 | 确认 slot 顺序变化，历史前一位置仍跟随正确 UID |
| Prefix 现金 30、两基地依次招募 | 第一个合法，advance 后现金 0，第二个禁止 |
| style schedule 239/240/298 | Buffer teacher=[2,7,7]，与 actor 联合概率一致 |
| 同路径 learner/SNAP actor | 对象隔离，temp_kind 分别 true/false |
| 真实本地 server_encode 两个 teacher 分组 ↔ NumPy | token 最大误差 1.14e−6，ctx 最大误差 1.20e−7 |
| 变长 PackedTraj 与 GAE | 合并 28 行正常；长度1/3/9、gamma=lam=1 的终局回报逐步一致 |
| 真实 resume 分支 AST | 全继承、全覆盖、显式 KL=0、仅覆盖 LR，四组均通过 |

F01 终局奖励、S5a 完成局剔除及 F06 全局 rollout 修复对应代码在该 HEAD 未改，本轮沿用前轮已验证的合法终局与混合早停探针；没有新报为回退。此次新增 AWR 缺陷不否定 PPO 本体通过。

仍需区分旧训练目标与新设计：当前 PPO 的 dense/worth/terminal margin、win bonus、默认 gamma=.995 以及 critic target clamp 仍是既有配方，不能称为已经实现平台 E[U]；本轮没有把这些未迁移目标伪装成新增实现回归。

## 文件与运行

- `check_network_ppo.py` / `network-ppo-results.json`：真实小网络、特征/Prefix/两后端、server、角色、GAE、resume。
- `check_awr_pipeline.py` / `awr-results.json`：合法两步终局序列在实际 AWR 链上的扩展测试；独立的权重分布 fixture。
- `*.stdout.json`：同次运行 stdout 快照。

```bash
CARBON_REPO=/path/to/carbon-league CARBON_REVIEW_OUT=/tmp/carbon-training-r5 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/venv/bin/python check_network_ppo.py
CARBON_REPO=/path/to/carbon-league CARBON_REVIEW_OUT=/tmp/carbon-training-r5 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/venv/bin/python check_awr_pipeline.py
```

Python 3.12 / NumPy 1.26.4 / Torch 2.5.1 CPU；torch 单线程。源码未写入，工作区干净。所有本轮结果保持绑定 98ffadfa，不适用于之后的新代码。
