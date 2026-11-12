# 第四轮复核证据

审查基线：`5463f504048bcca28868ebce0a063ac8d6c51b8e`。原报告基线：`14d53320b20d58bdb82676f7fa866b8e11a55062`。

总报告：[REVIEW_FIXES_20261113_R2.md](../../REVIEW_FIXES_20261113_R2.md)。本目录为新的独立结果快照，不覆盖前几轮证据。

## 入口与范围

| 目录 | 内容 |
|---|---|
| [data](data/README.md) | 真实 build_top/set_uw、C02 默认/partial/零行/重试，抽取的 prelaunch gate、缺失 replay 与事务边界 |
| [simulator](simulator/README.md) | 原版环境对照、终局深度/混合/预算边界、packs 搜索副本、死亡分类 |
| [runtime](runtime/README.md) | C03、bigeval 失败门、真实 n1r 短测、双入口打包源/TTA、资源判定桩 |
| [training](training/README.md) | 核心 blob 恒等、旧 EVO_SEEDS 调用兼容、Bash 完成标记、权重转换边界 |
| [method](method/README.md) | D01–D07 新状态、Huber/胜率/U/预算数值反例与源文件 hash |
| [ecs/results.json](ecs/results.json) | 当前 ECS 工具 6 项纯 mock，无网络/资源操作 |
| [syntax.json](syntax.json) | 72 个变更 Python/Shell 文件的 ast.parse/bash -n 结果 |

`results.json` 记录的是观测，不是统一的 PASS 套件。某些脚本成功退出表示它成功复现了缺陷；C02 的21项检查也包含对不正确行为的观测断言。

## 运行约定

使用 Python3.12、NumPy1.26.4、PyTorch2.5.1 CPU 及仓库已有引擎依赖。给每次复跑指定**新的仓外输出目录**：

```bash
export CARBON_REPO=/path/to/carbon-league
export CARBON_REVIEW_OUT=/tmp/carbon-review-new-run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

数据、模拟器、训练与方法子目录的 README 给出具体入口。模拟器脚本每次创建新的临时子目录；data/training 在临时输入上操作。新增 runtime 脚本要求显式 CARBON_REPO，默认输出位置仍须用 CARBON_REVIEW_OUT 覆盖，避免写回已提交快照。

runtime 当前包错误门复用前一轮的 `reports/review_evidence/20261113_fixes/runtime/{prepare,timing_short,gates,first_error_order}.py`。先在新输出目录运行 prepare，生成历史 b128 的 baseline.npz 和合法/非法探针包；随后运行相应入口。`pack_source_probe.py` 的 CARBON_WEIGHTS 默认是新输出目录下的 baseline.npz，也可显式指定；权重来源见 fixture_provenance.json。

`bigeval_boundary_probe.py` 依赖 prepare/gates 生成的短局探针；`n1r_reference_smoke.py` 读取仓内真实 reference；`timing_resource_gate_probe.py` 只用本地合法空动作桩实际等待约3秒，借 AST 把计时循环缩为1步。它不是模型 benchmark。

ECS 重跑入口仍为 `reports/review_evidence/20261113_fixes/ecs_checks_round3.py`：它替换 API 与时钟，只输出 mock 结果。不要直接执行真实 ecs_release 入口作为复核。

新增 runtime/method 探针归档时调整了默认本机路径，并将 method 来源 hash 固定到审查提交，之后做静态语法/链接检查；method 数值脚本已重跑并与原 JSON 逐值一致；其余没有为路径整理再次运行所有动态测试。原始结果内的绝对临时路径保留当次运行语境。所有新脚本源码和结果可用本目录 [manifest.json](manifest.json) 文件 hash 清单核对。

## 明确未做的工作

未启动训练，未执行 cloud shell 队列、杀进程、云 API、部署或天梯提交；未测完整胜率或 1.5CPU/512MiB 平台资格。无大模型权重、生成包或数据数组入库，仅保存复现脚本、小型原始结果与审查说明。
