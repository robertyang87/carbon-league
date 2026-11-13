# 新提交追加审查：62e0dee + 57557a8

本文件与前面的 98ffadfa 结果分开。模型/BC/smoke 的实测源码为隔离工作树 **62e0dee16419725ece6f18a22f095c83bb6fcd0f**；新增完整训练器从 Git 对象 **57557a83a73cd4d6b26b56ebebb4c0df8e13a155:code/stage0_train_heads.py** 只读加载，它依赖的模型、BC、smoke 源码与 62 相同。所有探针只在临时目录写小权重/包；没有真实训练、优化器更新、远端动作或业务源码修改。

## L01 — P1：无条件注册新头，旧 NPZ / PPO checkpoint 重启必失败

位置：`code/cnet2_model.py:161–164`，消费端 `code/rl2_ppo_cnet2.py:214–220`、`code/rl2_ppo_train.py:268–272`。

新增 utility/wdl/margin 六个参数键是无条件注册，只有 `cnet2_train.py:339–342` 的 BC_INIT 放宽了 missing-key allowlist。PPO adapter 仍严格加载；旧 checkpoint 的 net/vnet 也严格加载。

真实验证包括两条：

- 从原 98ffadfa 类实例化、导出的旧结构小权重，被最新实际 `Cnet2Adapter.make_model` 拒绝。
- 前轮已使用的真实历史训练 b128 NPZ（C128/NB8/D128、metadata epoch13，hash 在 JSON）也被同一入口拒绝；不是只有随机 fixture 才触发。

错误均为：`Missing key(s): utility.weight, utility.bias, wdl.weight, wdl.bias, margin.weight, margin.bias`。因此旧权重启动 PPO、server 首次加载旧模型，以及旧 checkpoint 恢复受阻。**P1 范围是新的启动/恢复，未推断已加载模型的在线进程会中断。**

仅改 `strict=False` 不足够：模型受控补齐新键后，旧 Adam 参数组仍无法加载。小模型原 59 个参数对象，新模型 65 个，真实 `newopt.load_state_dict(oldopt.state_dict())` 抛 `ValueError: parameter group ... doesn't match the size ...`。还需迁移 net/vnet 两个 optimizer；不能 silently 丢弃历史动量。

建议用显式架构/头版本保留 legacy 模式，或集中实现旧→新加载契约：只允许六个新键缺失、初始化新头、按参数名称/shape 迁移旧 optimizer 状态、新参数新建状态，并记录迁移来源。所有 CNet2 加载入口共享同一规则。`cnet2_add_ta.py:17` 和 `cnet2_parity.py:15` 也仍严格加载旧结构，具有同类兼容缺口。`stage0_heads_patch.py:74–86` 的验证仅实例化和检查 heads shape，没有实际加载旧文件，因此其 PATCH-OK 不覆盖兼容性。

证据：`latest-heads-results.json` 的 `old_npz_ppo`、`existing_trained_npz_ppo`、`old_checkpoint_net`、`old_optimizer`。

## L02 — P2：smoke 和新 full-head trainer 不把非有限结果当失败

62 smoke 位置：`code/stage0_smoke_heads.py:110–118`。finite/decreasing 只打印，没有断言/非零退出；finite 只覆盖训练 hist，不覆盖 validation。

真实 main 控制流小探针（stub loader/loss、无更新 optimizer）：

| 注入情况 | 实际结果 |
|---|---|
| constant loss | decreasing=false，正常返回（CLI 会 exit0） |
| 训练 loss NaN | finite=false，仍正常返回 |
| 验证 loss NaN、训练有限 | finite=true，val_l_u=NaN，仍正常返回 |
| 旧 init 缺 stem.weight | 只打印 missing，继续 smoke 到 summary |

缺旧主干参数问题位于 smoke 的 :69，strict=False 后没有参考 BC_INIT 的 allowlist。它可能把随机初始化的主干误作“冻结的已有模型”。smoke 的有限/下降检查应明确何时判 PASS，验证集也应纳入有限性；短于 50 steps 时 first50/last50 重合，不应伪装成独立下降检测。

**575 更新：现在已经有独立 `stage0_train_heads.py`，会真实训练并保存新头，不能继续说“只有不保存的 smoke”。** 该文件 :20/:81 复用 smoke 的 head_losses，:89–92 保存 NPZ 并打印 TRAIN-DONE。但同样没有 loss/validation/参数的有限性失败门。用真实 main、无更新 optimizer 和明确 NaN loss/validation 注入，仍写 out.npz、打印 TRAIN-DONE、正常返回。这里没有执行实际优化，证明的是发布/退出契约而非某轮自然训练已产生 NaN。

建议每步在更新前检测训练 loss/梯度有限；保存前验证所有新头参数、验证指标与必要数据范围，并将未通过设为非零退出且不发布正式 out。下降/校准门槛应与当前阶段承诺一致，不把无条件 TRAIN-DONE 当验收。

证据：`latest-heads-results.json/smoke_*`、`latest-full-heads-results.json/nan_run`。

## 新头到底接在哪里：目前有专用 trainer，旧 BC/PPO/NumPy 仍走 legacy value

实际 `cnet2_train.losses` backward 后，新增六个参数 grad 全 None，stem 与旧 value 有梯度。其 `_fwd` 仍调用原 `forward/forward_seq`，没有 head_losses。PPO adapter/server 也读取旧 tanh value；NumPy 只读旧 value，未计算新的 U/WDL/margin。

专用 `stage0_smoke_heads.head_losses` 在冻结旧网时六个新参数均有梯度，主干没有；575 的 dedicated trainer 复用这个路径且会保存合并权重，已具备专用的新头训练入口。**这不是“头从未被训练”，而是现有 BC/PPO 主流程尚未接入它们，裸网策略本身也没有因此变化。** 不应把注册了 heads 或专用头训练结束等同于已有 PPO 已优化 E[U]。

数学口径由 method 追加审查详述：现 head_losses 对 margin 用 Huber，不能称条件均值；仍从旧量化 tanh 标签反推 M/U，与真实 raw margin/U schema 未贯通。producer 保存 raw 值而 builder 尚未落入训练 fields 是文档明确的暂缓项，不伪报为已经完成的数据迁移。

## 已通过和边界

- 按六键 allowlist 迁移模型权重后，旧 policy logits / legacy value 完全相同；`forward_heads` 的前三项与旧 forward 一致，输出 heads shape 正确。
- 新 NPZ 可以被 NumPy Policy 接收；将 utility/wdl/margin 参数全部改成大数，策略 logits 与所返回 legacy value 都不变，证明新头当前被部署路径忽略。它们只是多保存的参数，不会自动切换部署价值语义。
- 实际 cnet2_pack 写出含新头的小模型单文件包，首次行动成功；不代表真实大包性能/整局认证。
- 使用新源码重新运行 28 状态接口探针，结果在 `latest-parity/`，没有覆盖 98 的旧快照。联合 logp、Torch/NumPy、packed GRU、UID 换槽、Prefix 招募预算、style schedule、role、GAE 等继续通过；这用的是完整含新键的权重，不否定旧权重加载失败。
- 575 full trainer 增加“missing 必须恰好六头键”的断言，已拒绝缺 stem 的旧权重，比 smoke 严格。但它也拒绝自己的完整输出作为下一次 init（missing=[]）；若只支持一次性 legacy→heads，应在 CLI 声明，若要续训应接受无 missing 并定义 optimizer/resume。
- 575 输出原样复制旧 metadata，未记录新目标/新头 schema/这次训练信息。后续制品登记至少应带新训练源码、目标与数据版本；不要用旧 epoch/val 字段解释新头训练结果。

## 复现文件

`latest-heads-probe.py` / `latest-heads-results.json`：62 实际加载、loss 梯度、NumPy/pack 与 smoke 退出路径。`CARBON_OLD_WEIGHTS` 可指定真实旧权重，未指定时仍跑旧源码实例化的结构 fixture。

`latest-full-heads-probe.py` / `latest-full-heads-results.json`：575 源码通过 git show 读取，实际 main 的保存/退出契约；伪优化器不会更新网络。

`latest-parity/network-ppo-results.json`：在 62 下重跑先前 `check_network_ppo.py` 的新结果。

```bash
CARBON_REPO=/path/to/isolated-latest CARBON_OLD_WEIGHTS=/path/to/old.npz \
  CARBON_REVIEW_OUT=/tmp/latest-review BC_DEVICE=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/review-venv/bin/python latest-heads-probe.py
CARBON_REPO=/path/to/isolated-latest CARBON_REVIEW_OUT=/tmp/latest-review \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/review-venv/bin/python latest-full-heads-probe.py
```

环境 Python3.12 / NumPy1.26.4 / Torch2.5.1 CPU，单线程。隔离工作树和主工作树均未被本探针修改。
