# HEAD_MAP — PPO-U 价值侧诊断：各检查点 critic 张量归属（实证）

结论先行：**六个 ckpt_*.npz 里都没有受训过的 critic**。本轮全程
`sep_value=1`，critic 是与 actor 分开的第二张 CNet2（trainer state
里的 `vnet`），只存在于 trainer state `.pt` 文件中；ckpt npz 只导出
actor（`net`）。actor 自带的 value/utility 头是冻结的初始化工件
（见 §3 行为证据）。v̂ 一律从各检查点配对 state 文件的 `vnet` 读，
读出方式按时代分两段——**分界在 it79，不是表面上的 it78**：

| 检查点 | critic 载体（state 文件，内嵌 it 已验） | 读出 | 所读张量（vnet 内） | v̂ 映射 |
|---|---|---|---|---|
| 74 | state_pre_termU_it74.pt（it=74） | tanh 值头 | `value.weight` / `value.bias` | v̂ = (tanh(v)+1)/2 |
| 78 | state_pre_vhead_it78.pt（it=78） | tanh 值头 | `value.weight` / `value.bias` | v̂ = (tanh(v)+1)/2 |
| 79 | state_pre_r18_it79.pt（it=79） | utility 头 | `utility.weight` / `utility.bias` | v̂ = u（sigmoid 输出本身） |
| 82 | **工件缺失**（见 §6） | （归属已证：utility 头） | `utility.weight` / `utility.bias` | v̂ = u |
| 85 | state_frozen_it85.pt（it=85） | utility 头 | `utility.weight` / `utility.bias` | v̂ = u |
| 95 | state_previous.pt（it=95） | utility 头 | `utility.weight` / `utility.bias` | v̂ = u |

state 文件 md5（备制时实测；其中 74/78/85 三个与 watcher 日志里记的
md5 逐字合）：74 `683d8758…`、78 `87cde228…`、79 `0b643177…`、
85 `fd9df203…`、95 `414b3f20…`（state_previous.pt 的副本）。

## 1. 架构事实：critic = 独立 vnet，值只从 vnet 出

- 现行 trainer（L20 副本 cf572ec1）:644 `vmodel = vnet if vnet is not
  None else net`；:656 rollout bootstrap 值 `ad.evaluate(vmodel, …)`；
  :712-713 更新期 critic 值 `ad.evaluate(vnet, …)`。同一行在两个时代
  备份里同位出现：pre_vhead trainer（92e97652）:452/:464、pre_termU
  trainer :427/:439——三代代码口径一致。
- ckpt 导出恒为 actor：现行 trainer :786
  `AD.save_npz(ad.export(net), … ckpt_<it>.npz)`。
- 各 state 均含 `vnet` 与 `vopt` 键且 `training` 契约记
  `sep_value: 1`（it74–it96 全部实测）。

## 2. 读出分工：adapter 的两条路

- 换头前 adapter（`rl2_ppo_cnet2.py.pre_vhead`，7a68d699）`evaluate`
  :254-258 **没有 utility 分支**：`lg, v, _ = model(b)`，v 只能是
  `forward` 的 tanh 值头输出（cnet2_model.py :160/:219，值头
  `Linear(D,1)` + tanh ∈ [−1,1]）。
- 换头后 adapter（62a9adc7，即现行）`evaluate` :258-267：模型带
  `_value_head == "utility"` 时走 `forward_heads` 取 u（sigmoid
  utility 头，cnet2_model.py :161-162/:209-222），并按 `v = 2u − 1`
  映回奖励尺度——即 u 本身就是 [0,1] 的 U 预测，utility 时代
  v̂ = u 无需再变换。
- tanh 时代的映射依据：tanh 值头与 utility 头活在同一奖励尺度
  [−1,1] 上（adapter 对 utility 的 `2u − 1` 即此尺度的自家约定），
  故 [0,1] 化唯一仿射式 v̂ = (tanh(v)+1)/2，与预注册「v̂ ∈ [0,1]」
  口径相容。注记：it74 的值头是在 Terminal-U 切换**之前**的奖励
  （margin 制）下训的——这是该检查点的史实，A 表只读跨检查点差值，
  口径影响在判读时由策略组掌握，本备制只记录事实。
- trainer 置位：pre_r18 trainer（1f1430b4）:342-351 在
  `--value-head utility` 时对 net 与 vnet 同时执行
  `model._value_head = "utility"`；现行 trainer 的
  `apply_utility_source` 同义。

## 3. 行为证据：哪张头在哪个时代真的被训（数组 md5）

actor（net）的两张头全程冻结：`value.weight` 六个 npz 与六个 state
恒为 `12e9d6f73…`，且与 init 权重 `cnet2_m3n1L.npz` 的同名数组逐字
同（init 根本没有 utility 键；net 的 `utility.weight` 恒为冷启动
常量 `56e7762d…`）。**actor 头从未被本轮训练触碰，ckpt npz 读不出
任何 critic 信息。**

vnet 的两张头（按 state 内嵌 it 顺序）：

| it | vnet utility.weight | vnet value.weight |
|---|---|---|
| 74 | `18e0eecc…`（冻结） | `43766c44…` |
| 78 | `18e0eecc…`（仍冻结） | `0dd04536…`（it75–78 间变） |
| 79 | `7d5ca21e…`（首变） | `0dd04536…` |
| 81（pre_r19 备份） | `d749fb48…` | `0dd04536…` |
| 85 | `3115ce33…` | `0dd04536…` |
| 95 | `e67714ed…` | `0dd04536…` |
| 96 | `7c5bb39a…` | `0dd04536…` |

读法：值头在 tanh 时代（≤78）持续受训、utility 时代开始即冻结；
utility 头在 ≤78 从未动过、it79 起每轮都变。**utility 读出实际生效
的第一个迭代是 it79**（换头 watcher 在 it78 边界落备份、重拉的
`it0=78` 指已完成迭代数，第一轮 utility 迭代即 it79；遥测同向：
it79 vl 0.0046→0.0022、v_mean +0.008→−0.060）。这修正了预注册
「78 起 utility」的表面分界：78 点的 critic 是旧值头训到的最后一版。

## 4. r19 的 utility-source=checkpoint 对 82/85/95 的含义

r19 trainer（cf572ec1）`utility_source_plan` :327-382：sep_value 时
来源标签恒为 `vnet`（:356），resume 时把来源 critic 的 utility 头
sha256 与整机 sha256 写进 `utility_provenance`；`apply_utility_source`
:384-405 在重拉时按哈希断言恢复的正是那台 critic。
实证闭环：r19 重拉日志（r19_swap_status.txt）记 head sha256
`a592d452…`、critic sha256 `bb700cea…`；备制时用 trainer 自家方案
对 pre_r19 备份（内嵌 it=81）的 vnet 复算，两值**逐字复现**——
即 it82 的 critic 就是 it81 的 vnet 续训，读出仍是 vnet utility 头，
无换源、无换头。85/95 同为该 vnet 的后续迭代（state 链连续：
frozen_it85 内嵌 it=85、state_previous 内嵌 it=95）。

## 5. 配对锚定：state 与 ckpt 同迭代

四个可用配对 + Stage 1 的 it74，逐数组比对 `state["net"]` 与
`ckpt_<k>.npz`：**186/186 数组全等、diffs=0**（78/79/85/95 于备制
时实测，74 于 Stage 1）。故每个 state 的 vnet 与同号 ckpt 是同一
迭代的 critic/actor 配对，提取时模型结构 cfg 取配对 npz 的 meta、
权重取 state 的 vnet 字典（extract 脚本内嵌此恒等断言，执行时复验）。

## 6. ckpt_82 的工件缺口（如实记录，不替代）

82 的 critic **归属**已证（§4），但**数值工件不存在**：it82 终局的
trainer state 没有任何存留副本。盘点（运行目录 176 项 + 全盘
`*.pt` 扫描）：备份点为 74/78/79（各代码/奖励切换）、85 是所有者
冻结的一次性 copy-only 副本、kl012/kl010 备份内嵌 it=86/91；
滚动备份 `state_last.pt.prev1` 在 r19 的 s19 watcher 写入后冻结在
it81（与 pre_r19 备份 md5 全同 `3f7fba2b…`）、prev2 已 rotate 掉；
`state_last.pt` 已是 it96；league/eval 导出皆只有 actor。
后果与处置（脚本已按此实现，最终判读归策略组）：
- A 表：82 列的 bias/MAE/相关不产出；单步变化 79→82、82→85 记
  不可观测；**r19 步组的合计分享率不受影响**——两步带符号变化之和
  恒等于 m85 − m79（望远镜式），脚本按合计口径算组分享率并标注。
- B 表：不涉及 82（判读线只用 79 与 95 对 74 的符号率）。
- 若策略组裁定必须补 82 点，所需证据是 it82 终局的 trainer state
  `.pt`（含 vnet）——该文件已确认不存在于 L20 任何目录。
