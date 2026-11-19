# 346 席 1 双容器实时计时复核

**本机停在资源预检。两套策略加载 0 次，各调用 0 步，未产生本场胜负。** 登记选的是已有首胜的地图 `621758346`、候选席 1，目标是测完整实时交互的超时风险。先前这一席的得分仍属于当时的开发队列。它不能证明平台资源下本场也能获胜。[预登记](protocol.json) · [本机容器实读](preflight/receipt.json) · [首次队列](../direct_original/README.md)

| 项目 | 要求 | 候选和原版两个容器的实际读数 |
|---|---|---|
| CPU | 每席独立 1.5 核 | `cpu.max=400000 100000`，合计上限 4 核；两个容器的 cgroup 根 inode 都是 148 |
| 物理内存 | 每席 512 MiB | `memory.max=17179869184`，16 GiB；两个容器共享该组 |
| 交换空间 | 本地复核限制为 0 | `memory.swap.max=max` |
| 网络 | 断网 | 两个容器均只有 `lo`，Docker 网络模式为 `none` |
| 基础依赖 | Python 3.12、NumPy 1.26.4 | 3.12.16、1.26.4 |
| 原版搜索依赖 | PyTorch 2.5.1+cpu、官方 helpers | 本机基础镜像均缺少 |
| 包体上限 | 每份 10,000,000 B | 候选 6,075,585 B；原版 2,897,684 B，均合格 |

这台主机的 Docker 运行在 rootless 模式，cgroup 驱动报告 `none`。`docker inspect` 显示两个独立的容器 ID，两边的请求值也确实为 1.5 核、512 MiB，但进程自身读到的是上表中的共享配额。预检耗时约 7.15 秒，其中没有策略冷启动或决策；这个时间不能用于判断任何一方会不会超时。另线对候选包的四场历史动作复演已有 1,196 次计时，属于另一种测量。[候选复演](../ladder_timing/README.md) · [平台指南当日读回](../ladder_timing/live_rules.json)

## 在符合条件的机器上复核

从仓库根执行。先按 `ladder_timing` 准备 `sapling-ladder-312:local` 基础镜像。下列构建只传官方环境源码和 Dockerfile。两套策略仅在开局时分别以只读文件挂入，运行时禁网。

```bash
lane=reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/ladder_live
context_dir=$(mktemp -d)
mkdir -p "$context_dir/code/env_src/carbon_challenge-master"
cp -R code/env_src/carbon_challenge-master/zerosum_env "$context_dir/code/env_src/carbon_challenge-master/"
cp "$lane/Dockerfile" "$context_dir/Dockerfile"
docker build -t sapling-ladder-live:cpu "$context_dir"
python "$lane/live.py" --image sapling-ladder-live:cpu --out "$lane/eligible_preflight"
```

输出 `eligible_probe_only` 后，在装有仓库官方环境依赖的 Python 下执行一次登记局。`--cpus` 是可交替使用的两个逻辑核号。两方对同一份推进前状态依次作答，每次独占这两个核的可用时段，各自的内核配额仍限制为 1.5 核。父进程在两份动作收齐后一次性推进官方解释器。

```bash
python "$lane/live.py" --image sapling-ladder-live:cpu --cpus 0,1 --out "$lane/registered_game" --play
```

`live.py` 使用容器镜像 ID、双方源文件 SHA-256、权重 SHA-256 和地图物理哈希锁定身份。资源检查发生在加载之前；两边随后各通过 `exec` 启动全新的 Python 进程，20 秒初始化计时从发出加载指令开始，包含解释器启动、编译、导入和读取策略。1500 秒整场时钟另从创建比赛容器前计起。双方隔离持久运行，收到自己的公开碳图和剩余超额时间。裁判从主进程序列化观测之前开始计步，包含来回传输、策略执行和解析返回动作。超时按每席独立判定，记录第 4 个慢步、最多 5 秒累计超额、单步 3 秒硬顶中的最早限制。单方违规判负；两边异常或整场任务失败单列。逐步动作、耗时、搜索计数和最终官方结果分别保存为 `turns.jsonl` 与 `receipt.json`。

原版在缺少 PyTorch 时会自动走弱化路径，仍可能返回合法动作。实时执行器要求原版在加载后读到 `_SEARCH_OK=True`、`_TEMP=1`、`_SEED=None` 和 `_S_ABORT=0.58`。本机已用合成进程检验往返、迟到回复和判负分支，并重新生成登记地图检查双方公开观测；目标资源下的真实双策略冷启动、RSS 和整局耗时仍待测。[执行器](live.py) · [资源预检和冷启动](worker.py) · [合成验收](verify.py)
