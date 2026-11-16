# C_PRECHECK — 测项 C（价值头互换配对场）构造可行性预检（只读）

2026-11-14 工程组预检。全程只读：未起对局、未改任何文件、未产出任何性能数字。
依据：PREREG_VALUE.md 测项 C（过签版）＋ HEAD_MAP.md 实证；仓库代码以 main 为准
（cnet2_search.py `94f7fb64`、cnet2_np.py `d8c15ad4`、packs/cnet2_search_pack.py `48392e98`，
均本趟 gh_api 读回验 md5）；L20 只读实测 ckpt_95.npz。

## 结论先行

**C 按构造作废停报**（过签「构造作废条款」的字面情形：仓库 search 玩家的决策可证
不消费任何价值头）；同时不可构造骑缝同立（见 §3 理由原文，可直接进终审输入）。

## ① 玩家形态：唯一的 search 玩家构造路径，及其叶评估实证

构造路径（仓库内唯一一条测试期 search 玩家路径，也是天梯提交形态）：

- 打包器 `code/packs/cnet2_search_pack.py`：把 main 的四模块源码
  （cnet2_feat/cnet2_np/fastsim/cnet2_search，N01 闸强制与 code/ 逐字一致）
  与权重一并嵌成单文件；包内 `agent()` 调
  `_S.make_search_agent(CONF, weights=_W[0], rweights=_W[1], oweights=_W[2])`
  ——三个权重槽分别是主策略、rollout 策略、开局策略，**没有第四个槽、没有任何
  critic/vnet 输入口**。
- 工厂 `cnet2_search.py:618 make_search_agent(conf, weights, rweights, oweights)`：
  三个 Policy 全部由 `load_weights()`（:42，仅 np.load npz→dict）构造。

决策构成（读全文后归纳，逐项有行号）：

- 默认动作＝主策略贪心解码＋规则层/破产守卫；`SEARCH` 为假时在
  `cnet2_search.py:1022` 直接短路返回默认动作，连 search 层都不进。
  （注：13 键 CONF 模板 PACK_CONF_TEMPLATE.txt 即 SEARCH=False；it95 冻结包
  9e16085a 的 CONF 经 r4 复核为 SEARCH=false/guard-on——本趟未读到该包本体，
  其不在 L20，此项引 r4 复核结论。）
- SEARCH 开时的两个模式（MODE="sweep"→`_search`、"mcts"→`_mcts`）：
  分支由策略解码生成（根部 `pol`、rollout `rpol`，批量路径
  `act_b/decode_b` 只返回动作标签，`trunk_b` 不算任何值头读数），
  分支打分**全部**出自 `leaf()`：`_mcts` 在 :1444、`_search` 在 :1649，
  全文仅此两处调用。
- **`leaf(sim, me, left)`（:663）是纯手写启发式**：终局按原版计分，
  非终局＝现金差 ＋ CARGO_W·可交付 cargo ＋ WORKER_V·工人数 ＋ TREE_V·树
  剩余寿命（含若干 cfg 开关的变体项），输入只有 FastCarbon 仿真状态与 cfg
  常量，**函数体内不引用任何网络对象**。`LEAF_Q`（:1465-1468）是对 leaf()
  输出分布取分位，不是网络量。

价值头在该玩家里的实际地位：

- `NP.Policy.act()`（cnet2_np.py:175-221）确实顺带算 tanh 值头
  （:184 `value = tanh(ctx @ value.weightT …)`，读的是**该权重字典自身**
  的 value 头），utility 头 u 仅在 `want_u=True` 时才算（:187-188）。
- 但 cnet2_search 内全部四个 act 调用点都把值返回值丢弃：
  :837 `y, lg0, _ = pol.act(...)`、:856 `y, lg0, _ = opol.act(...)`、
  :863 `y, lg0, _ = (epol or pol).act(...)`、:1558 `yo, _, _ = pol.act(...)`；
  `want_u` 在 cnet2_search.py 全文零命中。即：值头输出被算出（或根本不算）
  后从不进入任何决策量——分支生成、分支打分、根部先验（用的是 logits 的
  softmax，:1235-1241）无一消费它。

**①结论：该玩家的决策可证不消费任何网络价值头（tanh 头与 utility 头皆然）。**
「价值头作叶评估」这一玩家形态在仓库中不存在；C 的自变量（叶评估所用价值头
的来源）在该玩家上无定义。

## ② 移植忠实性：四重独立阻断（任一即足）

(a) **无消费槽位（接口层）**：如 §1，make_search_agent 与打包器都没有注入
外部 critic 的参数/配置键/文件口。要让玩家「消费 it74 的价值」，只能改
cnet2_search 的 leaf/打分代码并新造加载通路——产物是新玩家，不是预注册
所指的「仓库 search 玩家」，且两臂同配置（只差价值来源）的前提随之消失。

(b) **工件形态不对（连 P_orig 臂的名义价值也不存在）**：本轮受训 critic 是
独立 vnet、只存在于 trainer state `.pt`（it74 = state_pre_termU_it74.pt、
it95 = state_previous.pt，HEAD_MAP §1/§5）。玩家构造只读 npz actor 权重；
而 actor 自带的头从未受训——本趟在 L20 对 ckpt_95.npz 实测：187 键、
`value.weight` (1,128) f32 的 md5 = `12e9d6f7313f2b990b246b1377391b7f`，
与 HEAD_MAP §3 记的 init 逐字同值一致（六个 ckpt/state 恒同）；utility.weight
为冷启动常量（56e7762d…，HEAD_MAP §3；init 权重甚至无 utility 键）。
即「it95 策略＋it95 价值」按玩家输入根本拼不出 it95 的受训 critic——C 的
对照臂自身就不成立，不只是互换臂的问题。

(c) **表示不对齐（头与 trunk 绑定）**：值头读的是**自身网络 trunk** 的 ctx：
torch 侧 `self.value = nn.Linear(D,1)`、`v = tanh(self.value(ctx))`，ctx 来自
本模型 encode（cnet2_model.py :161/:209-222）；numpy 侧同构（cnet2_np.py:184
的 ctx 来自 self.encode）。vnet 与 actor 是两张独立训练的网（sep_value：
actor 出策略、vnet 只经价值损失训练，其策略头不吃策略梯度，HEAD_MAP §1）。
把 it74 vnet 的 value.weight 搬到 it95 actor 的 trunk 上，得到的函数 ≠ it74
critic（头是对着 vnet 自己的表示训的）；反过来整只 vnet 当玩家权重，则策略
头也被换掉，违反「it95 策略固定」。两种拼法都不忠实。

(d) **读出口径/单位无现成映射**：it74 头是 margin 奖励尺度上的 tanh 读出
（Terminal-U 切换**之前**的奖励下训成，HEAD_MAP §2 注记）；诊断里
v̂=(tanh+1)/2 只是测项 A/B 的读数约定，任何玩家代码都不实现它；it95 时代
的 utility u 是 [0,1] 的 E[U]（sigmoid）；leaf() 的分数是现金尺度差值
（与 MARGIN=20.0 现金单位比较）。三种尺度之间仓库内不存在映射函数，
新造映射即是即兴组件，按骑缝不许。

## ③ 结论件（可直接进终审输入）

> 测项 C 按构造作废停报。仓库唯一的测试期 search 玩家
> （code/cnet2_search.py @94f7fb64，经 code/packs/cnet2_search_pack.py 打包）
> 的决策不消费任何网络价值头：其叶评估 leaf() 是对仿真状态的手写启发式
> （现金＋货值＋工人＋树值），两个 search 模式（sweep/mcts）的分支打分全部
> 出自 leaf()；NP.Policy.act 虽顺带计算 tanh 值头输出，cnet2_search 内全部
> 调用点均将该返回值丢弃，utility 头（want_u）从未被请求。故「以价值头作
> 叶评估」的玩家形态在仓库中不存在，C 的自变量（叶评估所用价值头来源）
> 在该玩家上无定义。此外，即使另造通路亦无法忠实构造：本轮受训 critic 是
> 独立 vnet、只存于 trainer state .pt（it95 的 state_previous.pt、it74 的
> state_pre_termU_it74.pt），玩家构造只接受 npz actor 权重，而 actor 自带
> value 头与 init 逐字同、utility 头为冷启动常量（均未受训，本趟对
> ckpt_95.npz 实测 value.weight md5 12e9d6f7… 与 init 一致）；且值头读其
> 自身 trunk 的表示，与 actor trunk 不可互换，it74 头的 margin-tanh 读数与
> utility 口径、leaf 的现金尺度之间亦无现成映射。按过签构造作废条款与
> 不可构造骑缝，C 停报，不拼替代臂；价值通路的因果验证如需进行，按预注册
> 改走训练侧探针、另行授权。

## 附：证据与边界

- 代码证据均出自 main 读回件（md5 如卷首）；行号以 cnet2_search.py
  94f7fb64 / cnet2_np.py d8c15ad4 本版为准。
- 未做事项：未起任何对局；未运行 A/B/C 任何数值；未读 it95 冻结包本体
  （不在 L20，其 CONF 的 SEARCH=false 引 r4 复核在册结论）；未改仓库与
  L20 任何文件（本笔记为本地 workspace 文件）。
- 范围说明：仓库内另有 gate 的续弈路径（stage1_teacher_gate，纯策略贪心，
  见 duel 趟保真证据）与 K0 标注的 evaluate_root，均非「以价值头作叶评估」
  的 search 玩家，不构成 C 的替代构造。
