# 346 席 1 双容器实时计时复核

首胜包 `8da4b465` 已在两个独立限额容器内打完 621758346 席 1。该包 299 次公开请求全部及时返回，冷启 0.245 秒，P99 为 33.26 毫秒，最慢 82.72 毫秒。该局官方比分为 Sapling 7,016.02、原版 Master3s2 8,647.47。主机、容器内实读配额、双方逐步收据及完整复演均已归档。[托管复测](hosted_attempt/README.md)

### 本地主机的两次预检

本地预检完成后，本机仍共用 CPU、内存和交换空间 cgroup，未在这台主机启动登记局。[预登记](protocol.json) · [本地双容器读数](preflight_runtime/receipt.json)

| 项目 | 要求 | 候选和原版两个容器的实际读数 |
|---|---|---|
| CPU | 每席独立 1.5 核 | `cpu.max=400000 100000`，合计上限 4 核；两个容器的 cgroup 根 inode 都是 148 |
| 物理内存 | 每席 512 MiB | `memory.max=17179869184`，16 GiB；两个容器共享该组 |
| 交换空间 | 本地复核限制为 0 | `memory.swap.max=max` |
| 网络 | 断网 | 两个容器均只有 `lo`，Docker 网络模式为 `none` |
| 基础依赖 | Python 3.12、NumPy 1.26.4 | 3.12.16、1.26.4 |
| 原版搜索依赖 | PyTorch 2.5.1+cpu、官方 helpers | 首次基础镜像缺少。自足镜像两席均导入并运行通过 |
| 包体上限 | 每份 10,000,000 B | 候选 6,075,585 B；原版 2,897,684 B，均合格 |

这台主机的 Docker 运行在 rootless 模式，cgroup 驱动报告 `none`。`docker inspect` 显示两个独立的容器 ID，两边的请求值也确实为 1.5 核、512 MiB，但进程自身读到的是上表中的共享配额。首次基础镜像预检耗时约 7.15 秒，其中没有策略冷启动或决策；这个时间不能用于判断任何一方会不会超时。另线对候选包的四场历史动作复演已有 1,196 次计时，属于另一种测量。[候选复演](../ladder_timing/README.md) · [平台指南当日读回](../ladder_timing/live_rules.json)

## 自足镜像的第二次预检

镜像 `sapling-ladder-live:cpu` 从公开的 `python:3.12-slim` 构建，ID 为 `sha256:17f48d27456c3eeaf3eb5671f502bea48fb46ab88e40cdbcf7414c9fe8a5533d`。首次构建时，容器默认的信任库未识别本机网络代理颁发的 TLS 证书。第二次构建通过 BuildKit 在安装步骤临时挂入宿主可信 CA，保持 TLS 验证，成功从 PyPI 和 PyTorch CPU 索引下载依赖。镜像内已有 Python 3.12.16、NumPy 1.26.4、PyTorch 2.5.1+cpu 和仓库官方 Carbon helpers。[Dockerfile](Dockerfile) · [构建回执](preflight_runtime/build.json)

用该镜像分别启动两只断网容器，实际导入官方 `Board`，NumPy 与 Torch 都把同一矩阵和向量相乘，得到 `[8, 18]`。两席实际加载的 NumPy 二进制、Torch 扩展、CPU 后端库和 helpers 的 SHA-256 各自相同。正式执行器再次逐容器实读，剩下七条记录全指向共享 cgroup 的 CPU 配额、内存、交换空间和 cgroup 身份。运行依赖能力测试没有挂入任何参赛包，正式执行器在加载策略前终止。[镜像内实际导入与计算](preflight_runtime/image_imports.json) · [完整预检](preflight_runtime/receipt.json)

另做一次独立诊断，用 `python:3.12-slim` 只读挂载宿主 Python 包和官方环境源码。两个容器都导入成功，数值、四件关键文件哈希及内核阻断项与自足镜像相同。该诊断的 requests 和 jsonschema 版本随宿主环境，镜像内的构建证据另行保存。[只读挂载诊断](preflight_runtime/read_only_mount.json) · [首次基础镜像回执](preflight/receipt.json)

## 在符合条件的机器上复核

从仓库根执行。下列 BuildKit 构建直接使用 `python:3.12-slim`，上下文只需要仓库内的官方环境源码和 Dockerfile。两套策略仅在开局时分别以只读文件挂入，运行时禁网。

```bash
lane=reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/ladder_live
context_dir=$(mktemp -d)
mkdir -p "$context_dir/code/env_src/carbon_challenge-master"
cp -R code/env_src/carbon_challenge-master/zerosum_env "$context_dir/code/env_src/carbon_challenge-master/"
cp "$lane/Dockerfile" "$context_dir/Dockerfile"
DOCKER_BUILDKIT=1 docker build -t sapling-ladder-live:cpu "$context_dir"
python "$lane/live.py" --image sapling-ladder-live:cpu --out "$lane/eligible_preflight"
```

若构建出口使用组织认可的 TLS 代理，可给上述 `docker build` 增加 `--secret id=pip_ca,src=/etc/ssl/certs/ca-certificates.crt`，让安装过程使用本机可信 CA。

输出 `eligible_probe_only` 后，在装有仓库官方环境依赖的 Python 下执行一次登记局。`--cpus` 是可交替使用的两个逻辑核号。两方对同一份推进前状态依次作答，每次独占这两个核的可用时段，各自的内核配额仍限制为 1.5 核。父进程在两份动作收齐后一次性推进官方解释器。

```bash
python "$lane/live.py" --image sapling-ladder-live:cpu --cpus 0,1 --out "$lane/registered_game" --play
```

`live.py` 使用容器镜像 ID、双方源文件 SHA-256、权重 SHA-256 和地图物理哈希锁定身份。资源检查发生在加载之前；两边随后各通过 `exec` 启动全新的 Python 进程，20 秒初始化计时从发出加载指令开始，包含解释器启动、编译、导入和读取策略。1500 秒整场时钟另从创建比赛容器前计起。双方隔离持久运行，收到自己的公开碳图和剩余超额时间。裁判从主进程序列化观测之前开始计步，包含来回传输、策略执行和解析返回动作。超时按每席独立判定，记录第 4 个慢步、最多 5 秒累计超额、单步 3 秒硬顶中的最早限制。单方违规判负；两边异常或整场任务失败单列。逐步动作、耗时、搜索计数和最终官方结果分别保存为 `turns.jsonl` 与 `receipt.json`。

原版在缺少 PyTorch 时会自动走弱化路径，仍可能返回合法动作。实时执行器要求原版在加载后读到 `_SEARCH_OK=True`、`_TEMP=1`、`_SEED=None` 和 `_S_ABORT=0.58`。本机已用合成进程检验往返、迟到回复和判负分支，并重新生成登记地图检查双方公开观测；目标资源下的冷启动、峰值内存和整局耗时见页首托管实测。[执行器](live.py) · [资源预检和冷启动](worker.py) · [合成验收](verify.py)
