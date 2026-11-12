# PPO 修复复核证据

`results.json` 是对提交 `14d53320b20d58bdb82676f7fa866b8e11a55062` 的原始结果快照；本次发布整理没有重跑或改写结果。验证使用 Python 3.12.14、NumPy 1.26.4、PyTorch 2.5.1+cpu，以及仓库原版游戏环境所需依赖，限制 CPU/BLAS 线程为 1，未启动训练。

在含这些依赖的 Python 环境中运行：

```bash
CARBON_REPO=/path/to/carbon-league OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python review_ppo_fixes_v3.py
```

脚本优先使用 `CARBON_REPO`；未设置时，从脚本所在目录向上寻找 `code/fastsim.py`。结果输出到 stdout；checkpoint 测试夹具写入自动清理的临时目录。复核直接调用原版游戏解释器、真实 PPO rollout、actor 缓存与 decoder、轨迹打包、buffer、GAE，并提取执行训练器真实 resume 分支；小型合成策略替代昂贵模型。

边界测试构造了单方/双方灭绝及幸存者末步现金变化状态，这些构造本身不声称已由完整合法对局证明可达。另有独立测试使用 seed 42 生成的真实初局，通过连续两步在同一基地招募的合法动作序列触发灭绝，并逐项比较原版与 rollout 的终局分数、奖励增量和推理次数。覆盖同路径/不同路径 SNAP、SELF 镜像、提前终止与完整 299 步对局混合批次；通过结论限定于这些隔离探针。
