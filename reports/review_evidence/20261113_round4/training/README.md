# 第四轮训练与新增入口复核

源码 `5463f504048bcca28868ebce0a063ac8d6c51b8e`。报告基线 `14d53320b20d58bdb82676f7fa866b8e11a55062`，另核对设计提交 `8c615a912430a0e5355c2ece5a166aa458e23cb5`。仅只读源文件与 Git 对象，在临时目录运行小型数据转换和参数拒绝探针；没有运行 cloud 脚本、训练、完整对局、远端命令或进程清理。

## 结论

旧 F01、F06、S5a 对应的 PPO 核心没有被这次合并回退。新增 cloud 脚本存在一个确定的入口兼容回归：继续使用旧的 `EVO_SEEDS + N=0` 调法，已经不能通过修复后的 bigeval 检查，但队列仍可写完成标记。

AWR/单位权重两个小工具的通常输入探针通过；另确认一个小正权重的缓存半精度边界，未取得实际数据极值，单列为待评估边界，不把它描述成当前线上/训练已发生的故障。

## T01 — P2：新增显式种子队列必然被新局数检查拒绝，仍能标记完成

主要位置：`code/cloud/q_h14d.sh:7`、`:8`、`:10`。同类完整列表如下（检查了 HEAD 下全部 cloud 文件，而不仅是工作区搜索结果）：

| 文件及调用行 | 显式种子数 / 命令 N | 本地真实 bigeval 结果 | 完成标记位置 |
|---|---:|---|---|
| `code/cloud/q_h14.sh:11`、`:13–14` | 动态缺局集合，0–100；N=0 | 构造原范围完全缺局的 100 seed 列表，exit 2 | `:15` 写 `SX_DONE`，随后继续 fork 阶段 |
| `code/cloud/q_h14d.sh:7` | 源码实际 57 个不同 seed；N=0 | exit 2，未启动 worker/加载 agent | `:10` 写 `CB_DONE` |
| `code/cloud/q_h14d.sh:8` | 源码实际 30 个不同 seed；N=0 | exit 2，未启动 worker/加载 agent | 同上 |
| `code/cloud/q_h15.sh:15` | 源码实际 57 个不同 seed；N=0 | exit 2，未启动 worker/加载 agent | `:18–20` 等待进程消失后写 `S50_DONE`/`CB_DONE` |
| `code/cloud/q_h15.sh:16` | 源码实际 30 个不同 seed；N=0 | exit 2，未启动 worker/加载 agent | 同上 |

`code/bigeval.py:22–24` 现在明确要求 `len(EVO_SEEDS)==N`。该修复本身防止缩水考试冒充预定规模，正确；本轮导入的旧调用方没有迁移。实际输出示例：

```json
{"tag":"review_seed_contract","ERROR":"EVO_SEEDS has 57 seeds but N=0; refusing inconsistent exam size"}
```

两个 camper 队列各循环 3 个版本，`3×(57+30)=261` 个原定评测均在参数阶段拒绝。q_h15 中独立的 S50 常规 50 局调用不受这一缺陷影响，不能把整份脚本都说成没有任何有效任务。

完成标记的判断也不成立：脚本没有按 PID 收集子任务状态，`wait` 不带参数只等待所有子任务，不传播每个失败状态。独立 Bash 语义微例 `(exit 2) & wait; printf ...` 输出 `WAIT_RC=0` 与 `CB_DONE`、自身退出 0。该微例不是执行仓库脚本，没有写实际完成文件。q_h15 的 `ps` 轮询只证明进程消失，也不能区分成功与 exit 2。

**触发范围：** 运行这三个新导入的历史队列，且相应权重/前置标记就绪。没有证据表明当前云机正在运行这些队列；这是 HEAD 可执行入口与当前检查契约的确定不兼容，不是远端运行事故声明。

**建议：** 固定列表传真实 N=57/30；动态补局在同一变量中冻结列表并传真实长度，空集合显式跳过。以 PID 逐项 `wait "$pid"` 汇总退出状态并检查结果健康字段，仅全成功后写完成标记。保留 bigeval 严格局数门，不恢复旧的隐式 N=0 含义。

## 已保留 / 已通过

- `rl2_ppo_train.py`、`rl2_ppo_rollout.py`、`rl2_ppo_adapters.py`、`rl2_ppo_cnet2.py`、`rl2_ppo_server.py`、`rl2_actor.py`、`rl2_common.py`、`fastsim.py`、`cnet2_feat.py`、`cnet2_model.py`、`cnet2_np.py`、`cnet2_train.py` 共 12 个文件，在报告基线、设计提交和 HEAD 的 Git blob 完全相同。逐文件哈希见 `results.json/core_blobs`。
- 因而 F01 终止前现金/终局奖励、F06 learner/SNAP 角色键及温度隔离、S5a 只推理未完成局和变长轨迹/GAE、显式 resume 参数继承/覆盖，没有在这些实现中被新合并回退。未重复跑上轮完整 PPO 探针；此结论依据代码恒等与上轮已保存的通过证据，不能替代最新完整制品的运行认证。
- `awr_collector_anchor.py` 的通常输入 `[w=.5,2,0]`、默认 SCALE=1：正权重 collector 的有效 `w×uw` 都是 mean(w)=0.8333333，零权重行仍为 0，原数据的 uw 不变。临时 OUT 的其他文件按设计链接原输入。
- `m3s_uw.py` 用初始全 .4 的单位权重：collector 清零、有效 planter/base 置 1、padding 清零，并把原 .4 权重精确保留为 uw_build.npy。没有发现这条常规转换自身的新错误。

## 边界项：AWR 倒数权重经过现有 cached trainer 的 float16 转换后可能 NaN

位置：`code/awr_collector_anchor.py:18–20`；消费端 `code/cnet2_train.py:200–213`、`:242–246`。新增脚本使用 `uw=mean(w)/w` 抵消游戏权重；cached trainer 把 w 与 uw 分别压成 float16，再转回 float32 相乘。独立压缩破坏了原本有限的乘积。

默认 SCALE=1，构造合法 float32 权重 `[1e-6,1]`：

- anchor 实际脚本 exit 0，输出 `finite=True`；collector uw 约 `[500000.5,0.5000005]`。
- 按现有缓存路径各自 half→float 后，前者成为 inf。
- 从真实 cnet2_train.py AST 提取未修改的 `train_cached.batch_loss`，只以固定零 logits 替代网络前向，得到 `loss=nan`。没有创建优化器或执行训练步。

该问题要求实际 AWR 权重足够小；本轮没有下载/扫描真实大数据，因此未证明目前数据触发。`code/cloud/q_l20b.sh:5` 确实配置 BC_CACHE_TRUNK=1，但脚本是历史任务，不据此断言已经发生 NaN。建议下次复用该路径前统计 `min positive w` 和 `max uw`，保持损失权重 float32，或在 float32 中先合成有效单位权重，并在开训前拒绝非有限损失权重。不要靠截断 uw 悄悄改变常数 anchor 的目标。

## 额外静态使用边界（不列作已发生回归）

`code/cloud/dgx_full.sh:5` 会用工作目录中未版本化的 `cnet2_train.py.d0dd809` 覆盖正式 `cnet2_train.py`，且复制失败后不退出。仓库核心 blob 未变的结论不覆盖执行该历史脚本后的工作区。今后若复用，应通过隔离 checkout/固定版本运行，不能在已验收源文件上覆盖旧副本；本轮没有执行或检查任何远端副本。

## 证据与复现

- `check_training_round4.py`：读取 blob 与 cloud 调用；真实 bigeval 参数拒绝路径；独立 Bash wait 语义；两种临时数据转换；AST 提取的真实 cached loss 边界。
- `results.json`：本轮原始结果，不覆盖上轮快照。
- 环境：Python 3.12.14、NumPy 1.26.4、PyTorch 2.5.1+cpu；torch/BLAS 单线程。

```bash
CARBON_REPO=/path/to/carbon-league CARBON_REVIEW_OUT=/tmp/carbon-training-review \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/review-venv/bin/python check_training_round4.py
```

脚本不导入训练入口执行 main，不执行 cloud shell 文件，bigeval 都在创建 worker 和结果文件前拒绝参数；数据转换仅写临时目录。仓库工作区在探针后仍为空改动。
