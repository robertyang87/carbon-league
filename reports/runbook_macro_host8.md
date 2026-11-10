# 执行单：八号机上的旋钮搜索 + 第一轮整局对局训练（2026-11-10）

给本地 Mac 上的代理。按顺序做，每一步后面是它的验证；验证不过就停在那一步并回报，不要往下走。
方案背景见 `reports/plan_macro_training.md`。代码分支 `cnet2-search-v12`，要求拉到 `cb5ae6a` 或更新。

机器：八号机 `root@47.101.31.141`（64 物理核倚天 710，Arm，128 GiB，盘 100 GiB，抢占式、回收即释放、盘不保留）。
训练在 DGX Spark。今晚 00:05 的探针提交照原计划，不依赖本执行单；第 2 步过线的配置才叠进探针。

## 第 0 步：部署和防回收（开跑任何长任务之前）

```bash
# 八号机
cd ~/carbon && git fetch origin cnet2-search-v12 && git checkout cnet2-search-v12 && git pull && git log --oneline -1
python3 -c "import numpy, platform; print(platform.machine(), numpy.__version__); numpy.show_config()" | head -20
```

- 这是 Arm 机器：不要拷四到七号机的虚拟环境，numpy 重新 `pip install`。
- 在 Mac 上起一个每 10 分钟的增量同步（只拉新增文件，不要反复整目录传，出向流量 ¥0.8/GB）：
  `rsync -az --ignore-existing root@47.101.31.141:carbon/data/{ks_full,sp1_*} <Mac 上的目录>/`

- 回报两个替身的现状：Master2s 替身和 M2 替身（m2s2）最近一次考试对真身的还原度（一致率或对同一批对手的分差）。
  替身明显弱于真身的话，本地分差要打折看，以探针在天梯上的账为准。

**验证**：`git log` 显示 `cb5ae6a` 或更新；`platform.machine()` 是 `aarch64`；同步跑过一次且 Mac 上能看到文件。

## 第 1 步：测速（约 2 分钟）

```bash
cd ~/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export AWR_STEPS=300 AWR_OPEN_STEPS=300 AWR_KINDS=planter,base
export AWR_US_WEIGHTS=<Grove1 策略网权重 .npz>
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
N=$(nproc); time python3 awr_gen.py ../data/sp_bench awr_us_env.py <Master2s 替身的对手文件> 5000 $N 1 0.5 $N
```

共 2×N 局。`OMP_NUM_THREADS=1` 必须设，否则多进程抢线程会卡死。

**验证**
- `real` 时间 ÷ 2 = 满载时每局耗时。云端 4 核 x86 上是 22 秒；超过 45 秒先查线程数和 numpy，再往下走。
- `python3 ../data/unit_ledger.py corpus ../data/sp_bench awr_us | grep 'TREE BUSINESS'` 里 income 不为 0（回放里带了树的吸收量）。
- 同一条命令再跑一次，应在 1 秒内结束（续跑不重复打）。
- 回报：`nproc`、每局耗时、`du -sh ../data/sp_bench`。

## 第 2 步：旋钮完整搜索（阶段 A，约 4–5 小时）

先写工厂文件 `code/exp/knob_us_grove1.py`（`code/exp` 不入库，各机器自己建）：

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cnet2_search as S
CONF = {...}   # Grove1 上榜包里的完整配置，原样照抄，再加 "SEARCH": False
W = "<Grove1 策略网权重 .npz>"; RW = "<Grove1 滚动网权重 .npz>"
def make(**over):
    c = dict(CONF); c.update(over)
    return S.make_search_agent(c, weights=W, rweights=RW)
```

```bash
cd ~/carbon/data
nohup python3 knob_search.py ks_full ../code/exp/knob_us_grove1.py \
  m2s=<Master2s 替身.py> m2=<M2 替身 m2s2.py> grove1=<Grove1 纯网络.py> scout1=<Scout1 或 v17d2 纯网络.py> \
  --workers $(nproc) > ks_full.log 2>&1 &
```

- 对手池（2026-11-10 改）：不用 1C-v9。它只有 2269 分，Grove1 纯网络对它的基准分差约 +1200，分不出配置好坏；
  1C-v10 / v11 源码不公开，本地没有。四个对手：Master2s 替身、M2 替身、Grove1 自己、Scout1 或 v17d2。
- 对手文件要求：模块里有 `agent(obs, config)`，或有 `make()` 返回代理。对手必须是确定性的（同种子同结果）。
- 默认三级淘汰：120 组各 40 局 → 前 30 组各 120 局 → 前 6 组各 400 局，四个对手，共约 4.4 万局（约 4–5 小时）。
- 被回收后换机器：把 `ks_full/` 拷回去，同一条命令续跑。
- 跑的过程中看中期结果：`python3 knob_search.py ks_full x m2s=x m2=x grove1=x scout1=x --only-report`。
- 不要同时打开上游的 TG / RG 和这里的 `MACRO`，两套管的是同一件事。

**开跑前的验证（10 分钟，必须做）**
1. 关闭时逐步相同：工厂 `make()`（不带 MACRO）和现在的 Grove1 纯网络，同一批观测 300 步动作完全一致。
2. 先跑一个小的：`--configs 4 --stages "8:1"` 到另一个目录，日志里没有 `!! agent errors`，`games.jsonl` 里配置 1–4 的 `mac` 计数不全为 0（旋钮真的生效了），配置 0 的 `mac` 全为 0。

**结果验收（`plan_macro_training.md` A3）**
1. 前 6 名对四个对手的配对分差都不为负，最差对手上过 2 倍标准误（日志里每个对手一列 `均值±标准误`）。
2. 账要跟着动：`tree-biz` 为正、`planters` 为负。分差涨而账不动，判为噪声。
3. `opp-tree-steps` 相对基准的增幅不超过 +15%。
4. `dep`（入库）不为负。
5. 最好的一组加上搜索，固定预算对 Grove1 头对头 200 局过 2 倍标准误，对 Master2s 替身不为负。过了才叠进探针。

**止损**：前 6 名没有一组过第 1 条，说明旋钮表达不了差距。回报结果，不叠进探针；第 3 步照做（它不依赖旋钮结果）。

## 第 3 步：第一轮整局对局（1 万局，约 1–1.5 小时；可在第 2 步跑完后接着跑）

先小样定温度（每个温度 200 局，几分钟）：

```bash
cd ~/carbon/code    # 环境变量同第 1 步
for T in 0.3 0.5 0.8; do python3 awr_gen.py ../data/sp_tau$T awr_us_env.py <Master2s 替身.py> 3100000 40 4 $T $(nproc); done
```

每条最后打印一行 JSON。选温度的规则：`adv_mean` 不低于 −150，在此前提下 `adv_sd` 最大的那个。
（云端试过 1.0：采样局比不采样局差 1371 分，太乱。三个都不满足就用 0.3。）

正式跑，四个对手各 2500 局：

```bash
for O in m2s m2 grove1 scout1; do
  python3 awr_gen.py ../data/sp1_$O awr_us_env.py <对应的对手文件> 3000000 500 4 <选定温度> $(nproc)
done
```

**验证**
- 每个目录有 `index.json` 和 `awr.jsonl`，局数 = 500 × 5 = 2500。
- `python3 ../data/unit_ledger.py corpus ../data/sp1_m2s awr_us`：能出完整报表，"TREE BUSINESS" 的 income 不为 0。
- 磁盘：每局约 180 KB，1 万局约 2 GB。放大到 10 万局以上之前先回报，100 GiB 的盘装不下 50 万局。
- Mac 上的同步目录里文件数和八号机一致，再继续。

## 第 4 步：训练（DGX Spark）

沿用 `data/awr_round_multi.sh` 的流程，三处不同：

1. 建库用整局：`CNET2_STEP_MAX=300`（原来是 100）。
2. 初始权重用 Grove1 的策略网，`COPY_TEACH` 传 `none`（保留 Grove1 自己的风格行）。
3. `awr_weights.py` 不用改参数：遇到 300 步的回放自动按最终分差算优势。

```bash
cd ~/junyang/carbon/code
CNET2_STEP_MAX=300 BUILD_WORKERS=14 ../.venv/bin/python build_awr.py ../data/cnet2_ds_sp1 ../data/sp1_m2s ../data/sp1_m2 ../data/sp1_grove1 ../data/sp1_scout1
../.venv/bin/python awr_weights.py ../data/cnet2_ds_sp1 ../data/sp1_m2s ../data/sp1_m2 ../data/sp1_grove1 ../data/sp1_scout1
../.venv/bin/python concat_ds.py ../data/cnet2_ds_anchor170k ../data/cnet2_ds_sp1 ../data/cnet2_ds_sp1A 1
BC_DEVICE=cuda BC_RAM=1 BC_THREADS=8 BC_SAVE_EPOCHS=0 BC_INIT=<Grove1 策略网权重> \
  BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1 BC_LR=1e-4 \
  ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_sp1A ../data/weights/cnet2_b128sp1.npz 2 '{"C":128,"NB":8,"D":128}'
```

**云端小规模验证过（2026-11-10 深夜，120 局整局对局，CPU）**
- `build_awr.py`（`CNET2_STEP_MAX=300`）、`awr_weights.py`、`cnet2_train.py`（只训解码器、从 Grove1 权重出发）三步都跑通：
  120 局 → 35880 行（每局 299 行），`ess_frac` 0.33，训练 1 轮正常结束。没有混锚定集（云端没有那份数据）。
- 数据集大小：每局约 3.2 MB（主要是 `maps.npy`）。1 万局约 32 GB、约 300 万行；主干缓存约 11 GB。
  Spark 的 128 GiB 内存放得下，但传输和建库前先确认磁盘空间。
- `awr_weights.py` 打印的 `ess_frac` 应不低于 0.2；`w_max` 会顶到上限 20，属正常；`mirror_dropped` 是被剔除的镜像局数
  （Grove1 对一模一样的自己会打出分差正好为 0 的对称局，只出现在 `sp1_grove1`）。
- 没验证的：1 万局规模下的耗时和内存，以及混入锚定集（`concat_ds.py`）这一步。
- 这一轮是"按整局分差加权"，只对种植员和基地槽位有权重（车的权重为 0）。

## 第 5 步：验收新模型（`plan_macro_training.md` B5）

1. 纯网络对四个对手，同种子减训练前：最差对手不为负。
2. `unit_ledger.py corpus` 的三项账（树的生意净值、零产出种植员比例、50 以上价格种树的占比）只当诊断记录，不当过线条件：
   今晚的旋钮结果说明我方的账变好不等于分差变好（见 `plan_macro_training.md` 0.1 节）。
3. 对方树·步增幅不超过 15%；我方入库不降。
4. 车的一致率相对训练前下降不超过 0.3 个点（防遗忘）。
5. 加搜索，固定预算对 Grove1 头对头 400 局过 2 倍标准误；对 Master2s 替身不为负。
6. 限时包量最慢一步，过了才上榜；上榜后用 `unit_ledger.py ladder` 重算对榜首三类的账。

**止损**：第 1 条不过，不上榜，回报数据；不要靠调学习率、加轮数反复试，先看是数据（温度、局数）还是标签的问题。

## 每一步回报的格式

一步一条，包含：做了什么（命令）、验证各条的实际数字、过 / 不过、下一步。不过的那条附日志原文最后 20 行。

## 不要做的事

- 不合 `main`，不删、不替换天梯上已有的条目。
- 别人公开的源码只当本地对手，不进训练集、不进提交包。
- 天梯密码不落盘。
