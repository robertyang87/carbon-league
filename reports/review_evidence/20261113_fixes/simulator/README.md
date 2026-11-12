# Simulator / evaluation / search 复核证据

原始测量版本：`14d53320b20d58bdb82676f7fa866b8e11a55062`，与 `a8eaff21` 比较。
本目录两个 JSON 保留原始测量结果。两个脚本仅做了路径和输出目录的可移植整理，随后通过 Python 语法编译检查；整理阶段没有重跑实验。

## 文件与测量范围

- `review_simulator_r3.py` / `results.json`：8 个完整原版 `env.run` 对拍，加 6 个终局 leaf 边界例。原始测量全部通过。覆盖灭绝时胜者保留前态 reward、首次转移灭绝、双方灭绝、自然到时、最后合法一步花钱和最后一步灭绝，以及分类招募/死亡计数。leaf 检查从被审源码提取 `_extinct`、`_terminal`、`leaf` 的原样函数定义，与真实原版规则比较；它不是完整神经模型强度测试。
- `sweep_terminal_depth.py` / `sweep-terminal-depth.json`：使用被审源码中原样 `_search` 与终局 helper、实际 FastCarbon 和 Feat，仅将神经网络输出固定为可解释的候选动作。没有修改游戏转移、特征解析、搜索控制流或 leaf 计算。

第二项是构造局面：默认动作 `UP` 令空载采集员进入敌方基地而灭绝；候选 `NONE` 截获敌方最后一个载碳采集员，使敌方灭绝。两条分支在根步均已终止，正确 leaf 分别为 `-220`、`+220`。

默认 `MIN_DEPTH=3`（该版本 `code/cnet2_search.py:35–36`）时，`all(term)` 在第 1529 行提前结束推进，但第 1573–1575 行又以深度不足返回默认的必败动作。仅把 MIN_DEPTH 改为 1 作诊断后，会选择 `NONE`，记录 gain 440。此证据确认 sweep 分支控制流问题，不声称该局面的自然发生频率，也不声称已经进行完整自然对局或测量线上胜率；mcts 没有这段最低深度闸。

主测量还记录了一项较小残留：出生当步死亡的种植员已纳入 `pla_dead=2`，但死因类别合计仍为 1（`pd_clash_stay=1`）。对应 `eval_v53.py:268–282`。

## 复现命令

在具备该仓代码所需依赖的 Python 环境运行。原始测量使用 Python 3.12、NumPy 1.26.4，以及原版环境导入所需的 jsonschema / requests。无需下载权重，不训练、不部署、不访问远端游戏。

```bash
CARBON_REPO=/absolute/path/to/carbon-league \
CARBON_REVIEW_OUT=/tmp/carbon-review-evidence \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python /absolute/path/to/evidence/simulator/review_simulator_r3.py

CARBON_REPO=/absolute/path/to/carbon-league \
CARBON_REVIEW_OUT=/tmp/carbon-review-evidence \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python /absolute/path/to/evidence/simulator/sweep_terminal_depth.py
```

`CARBON_REPO` 优先；未指定时，从脚本目录和当前工作目录分别向上寻找 `code/fastsim.py`。如无法找到，脚本明确报错，不猜测其他仓库。

`CARBON_REVIEW_OUT` 指定临时结果目录的父目录；未指定时使用系统临时目录。每次运行均创建独立的 `carbon-simulator-r3-*` 临时子目录，将新 JSON 写入其中，并在 stderr 打印完整路径；不会覆盖本目录或之前的测量文件。策略探针也仅写在临时目录，业务源码保持只读。

`review_simulator_r3.py` 使用断言比较当前代码与原版结果，失败会非零退出。`sweep_terminal_depth.py` 是缺陷对照探针：退出 0 只表示两组配置成功运行，应读取返回动作、shallow、changed 及分支分值判断问题是否仍在，不能把它的退出 0 当作“修复通过”。
