# M9XJ 三臂复刻（Line A 正式载体）预置备忘

- 预置探针：2026-11-12 19:06–19:07 CST，标记 `M9XJPRE1905A7B`（cloud-assistant run，L20 → DGX ssh + Studio ssh，双段均已认领回执）
- 触发规则（作战室 17:56 裁定）：it260 / it320 谁先满 100 局＋语料齐先训谁；都满则训 it320；同时只训一个；本备忘只预置、不建集、不开训
- DGX 仓库已拉到最新：`2f0e75d665964a2dcd62af20a91b56c919af3cc0`（19:06:15，含 Line B 预置备忘＋`code/lb_relabel.py`）

## 1. 两候选当前数（DGX 直读＋仓库文件逐字核验，19:06–19:07）

| 候选 | 清单局数 | 语料文件 | 缺口到 100 | 胜负 | 先手/后手（按 order 字段） |
|---|---|---|---|---|---|
| **it260**（`m9xj`） | 82 | 84 | **18 局**（按清单；按语料 16） | 76W–6L（a 37-3，b 39-3） | 43 / 39，均衡 |
| **it320**（`m9xj320`） | 45 | 5 | **55 局** | 44W–1L（a 39-1，b 5-0） | 24 / 21，均衡 |

- 教师名（逐字核验，清单 100% 命中）：
  - it260：`RyanX-AIPowercore-M9XJ-it260`（82/82 局；name_a 40、name_b 42）
  - it320：`RyanX-AIPowercore-M9XJ-it320`（45/45 局；name_a 40、name_b 5）
- 速率实测（18:32→19:07，35 分钟窗口）：两候选各 +9 局，约 0.26 局/分 → **it260 触发估计约 20:17，it320 约 22:41**。it260 先触发是大概率事件；it320 只有速率翻约 3 倍才反超。注意：该窗口含一次 API 收获补录（commit `fd78398` "pull m9xj replays via ladder API"），速率是估计值，触发前以侦查兵最新刻度为准。
- 榜单对照（侦查兵 18:42）：it320 第 1（2877±59，40/41，仍定位期）、it260 第 3（2848±39，72/78）。仓库清单（82/45）比榜单快照（78/41）领先约 4 局，属收获时滞，正常。
- **it320 先手/后手纠正**：18:32 观察名单写「清单 36 局全为 as_a 一侧」指的是 name_a 槽位，不是出场先后。按 `order` 字段实测：it320 先手 24、后手 21，均衡——没有先手偏科问题。it260 同样均衡（43/39）。

## 2. 三数对齐状态（触发时链内断言的输入）

- it260：清单 82 ＝ 语料 84 **不对齐（语料多 2）**。多出的 2 个回放文件（无清单条目，疑似收获时滞）：
  - `7b102de1f06443941cb6236fd57f447e-1.json.gz`
  - `afb6cf9d90956b3b66732212547c7782-1.json.gz`
  - 其余 82 个语料文件逐个对上清单 id（去 `-1` 后缀比对，82/82）。触发时链内修复路径：`ladder_corpus.sync()` API 重拉补齐清单（预期 82→84），再断言三者相等；修不好则链 FAIL 停住，不许带孤儿往下走。
- it320：清单 45，语料仅 5（全是早期 precache，5/5 对上清单 id）。**40 局回放缺失**——触发时必须 API 全量拉取后才能建集。
- 仓库语料目录均无 `index.json`（一贯如此：index 是建集时 `ladder_corpus.py` 生成，不进仓库）。
- **工作区语料是空的**：`/home/arm/junyang/carbon/data/m9xj_corpus` 与 `m9xj320_corpus` 均 `replays=0`、无 index。触发时必须先把仓库语料＋清单同步进工作区（`replica_chain.sh` 的 `REPO_DIR`/`REPO_MATCHES`＋`SYNC_CMD` 机制就是干这个的），不是可选项。

## 3. `replica_chain.sh` 与三臂开训命令（已写好，未执行）

- 链脚本现况：`code/replica_chain.sh`，commit `3053378b14f9b6ea9075dbcde16ee533b2a1fca9`（2026-11-12 18:19:12），md5 `51081dae1c0d37c95ef4235afd86b24f`——**当前版**。`build_top.py`、`set_uw.py`、`ladder_corpus.py` 均在位。
- 注意：链脚本本身是**双臂**（r1 注意力＋g1 对照）。三臂载体要这样接（A1＋A3 走链，A2 在 ds_w 建好后按同一配方追加启动；三臂同批、同机、同配方，只差数据集与 BC_INIT）：

### 3a. DGX 侧建集（无论训练机是谁，建集都在 DGX）

```bash
cd /home/arm/junyang/carbon-repo && git pull --ff-only   # 触发时再拉一次
cd /home/arm/junyang/carbon/code
export DATA=/home/arm/junyang/carbon/data
# 先把工作区语料同步到 100 局（仓库拷贝 + ladder_corpus.sync API 补齐），三数断言 replays == matches == index == 100
# A1+A3 标准数据集 + 双臂启动（链内含计数断言 → build_top → set_uw 1 1 1 → 双臂启动 → 自证）：
PREFIX=m9xj TEACHER='RyanX-AIPowercore-M9XJ-it260' \
  CORPUS=$DATA/m9xj_corpus MATCHES=$DATA/m9xj_matches.json DS=$DATA/ds_m9xj \
  EXPECTED=100 ./replica_chain.sh 2>&1 | tee $DATA/dq_logs/replica_chain_m9xj.log
```

- 链内启动的两臂（DGX/CUDA 配方，`BC_DEVICE=cuda BC_RAM=1 BC_THREADS=5 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4`，20 epochs）：
  - A1：`BC_INIT=$DATA/weights/cnet2_arch_n1_last.npz` → `cnet2_m9xjr1.npz`，cfg `{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}`
  - A3：`BC_INIT=$DATA/weights/cnet2_b128m2t3.npz` → `cnet2_m9xjg1.npz`，cfg `{"C":128,"NB":8,"D":128}`

### 3b. A2 加权数据集 ds_m9xj_w（gap-plan v1 定版口径，只动数据集、不动训练器）

- 做法：复制 `ds_m9xj` → `ds_m9xj_w`，然后**终局行（step≥270）复制 ×3 入集；老师败局与 500 分内局复制 ×2**；验证集与 `ds_m9xj` 逐字相同（m3s2_w 试点先例：总 41,488；训练 38,199；验证 3,289；终局行 3,103→9,309）。
- **缺口（诚实标注）**：仓库里没有加权构建脚本，DGX 上也没有 `ds_m3s2_w`（根本不在 DGX 上）及任何加权构建日志——`ds_m3s2_w` 只在 Studio 上（433M），其构建方法是试点代理当场做的、未留档。触发时执行代理必须按上款口径**现写加权步骤**：先读 `ds_m9xj/meta.json` 与文件布局确认行格式，改完核对行数（终局行数 ×3 前后），验证集 md5 与标准集一致，再用真实训练器跑 1 个 epoch 冒烟，过了才全量启动。不要凭印象拼脚本。
- A2 启动（与 A1 同配方、换数据集）：
  - `BC_INIT=$DATA/weights/cnet2_arch_n1_last.npz` → `cnet2_m9xjwr1.npz`，cfg 同 A1，数据集 `ds_m9xj_w`

### 3c. Studio 侧三臂（首选训练机；K4v2 模板已验证）

```bash
D=$HOME/junyang/carbon/data; DGX=arm@10.188.0.23
# 1) 等训练机空闲（wg1 须已结束；pgrep cnet2_train.py 为空才继续）
# 2) DGX → Studio 同步 ds_m9xj 与 ds_m9xj_w（tar over LAN ssh，禁 rsync），逐文件 md5 聚合校验对上 DGX
# 3) 确认初始权重在位：$D/weights/cnet2_arch_n1_last.npz、cnet2_b128m2t3.npz
cd $HOME/junyang/carbon/code
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_m9xj   $D/weights/cnet2_m9xjr1.npz  20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/m9xjr1.log  2>&1 < /dev/null &
BC_INIT=$D/weights/cnet2_b128m2t3.npz  nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_m9xj   $D/weights/cnet2_m9xjg1.npz  20 '{"C":128,"NB":8,"D":128}'                          > $D/m9xjg1.log  2>&1 < /dev/null &
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_m9xj_w $D/weights/cnet2_m9xjwr1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/m9xjwr1.log 2>&1 < /dev/null &
# 4) 三臂 ep0 出齐后发三件套回执（进程＋GPU/MPS 占用＋日志 mtime）；训完 ship-watcher 等三 PID 退出，scp *_ep19.npz → DGX 作 *_fromstudio.npz，双侧 md5 一致才算落袋
```

- 权重命名：训练机本地 `cnet2_m9xjr1 / cnet2_m9xjg1 / cnet2_m9xjwr1.npz`；Studio 回传 DGX 后加 `_fromstudio` 后缀（K4v2 / m3s2wr1 先例）。
- 若 it320 先触发（小概率）：`PREFIX=m9xj320`、`TEACHER='RyanX-AIPowercore-M9XJ-it320'`、`DS=$DATA/ds_m9xj320`，权重 `cnet2_m9xj320r1/g1/wr1.npz`，其余命令逐字替换。
- DGX 备选：若 Studio 在触发时仍忙，DGX 训三臂——A1＋A3 走链脚本，A2 追加；三进程同卡会抢 GPU（M6 双臂已占 96%），启动前须确认 M6 已结束且 GPU 空闲，线程数按实测下调， launch-time 由执行代理定，不在此预置里硬编。

## 4. Studio 就绪（19:07:24 只读核验）

- `mac-studio.local`，up 14 天，load 1.56；venv `torch 2.15.0`、`mps True`，完好。
- **无训练进程**（19:07 时 wg1 尚未启动——所有者 19:04 下令的加权对照臂 `cnet2_m3s2wg1`，派发在途，本探针未干预）。`ds_m3s2_w` 433M 已在 Studio 上，wg1 输入就绪；无 wg1 日志。
- 磁盘：926Gi 总，**425Gi 空闲**，无压力。
- `ds_k4v2` **651M / 125 文件，结论：留着，别删**。K4v2 双臂权重已回传 DGX（r1 `344cfde9…`、g1 `a594efd6…`，双侧一致），但定标还没跑；651M 对 425Gi 空闲不构成压力，留作重建余地。等 K4v2 定标落地后再议清理。
- Studio 数据集现状：ds_m3s2 368M、ds_m3s2_w 433M、ds_k4v2 651M、ds_fable 1.1G、ds_distill 2.2G、weights 1.9G；最新权重 `cnet2_k4v2r1_ep19.npz`（18:42）。
- Studio 上 `/Users/arm/junyang/carbon/*.sh` 无启动脚本（ls  glob 未命中；ship 脚本在 `data/` 下：`ship_k4v2.sh`、`ship_m3s2wr1.sh`）。K4v2 启动是内联命令，M9XJ 同理按 §3c 内联执行。

## 5. DGX 就绪与冲突

- M6 双臂**正在训**：PID 218153（r1）/ 218154（g1），19:06 时 etime 01:07，GPU 96%。约 19:05 启动，20 epochs、DGX 约 1.5 分钟/ep → 估计约 19:35 完训。DGX 在触发前不空闲，**M9XJ 不许现在上 DGX**。
- DGX 磁盘：3.7T 总，1.9T 空闲（47% 已用），无压力。
- **前提变更（任务书发出后）**：Studio 的下一个活已不是 M9XJ——所有者 19:04 催办 wg1（`cnet2_m3s2wg1`，ds_m3s2_w＋无注意力），不等 M9XJ 触发。M9XJ 训练机＝触发时 first-free（Studio 在 wg1 之后约 19:30 空闲，或 DGX 在 M6 之后约 19:35 空闲），触发当场再定，不提前锁定。

## 6. 验收口径（触发后执行代理直接用，不用再翻方案）

- Line A 预注册（`reports/live/replica_gap_plan.md`）：A2 vs A1，共同考官 Master2 真身＋E0 各 200 局；**主指标＝行为统计**（种树量与本尊剖面距离缩小、终局残树下降），副指标＝考试均分不回退超 1 SE；主指标同向改善且副指标不回退 → 加权升为后续复刻（it320 起）标准配方。
- Echo2 桥合格判定（所有者 18:59 定规）：替身对 Echo2 的考试分差 vs 本尊天梯直打分差——差在 1 SE 内＝合格，低 1–2 SE＝陪练可用（缺口量化），低逾 2 SE＝不合格。**本尊锚（已提取，`hidden_files/echo2_ladder_h2h.md`）**：
  - it260 本尊 vs Echo2：5-0，**+1765 ± 339**（逐局 +878, +2979, +1609, +1734, +1624）
  - it320 本尊 vs Echo2：2-0，**+1635 ± 801**（仅 2 局，方向参考）
- 本尊行为剖面（种树量/终局残树）须用本尊自己的回放实测，不许借 Master3s2 的 40.2/2.1——**尚未提取**，触发时（或之前）用 `hidden_files/behavior_stats.py` 对本尊语料跑一趟，作 Line A 主指标的对照基线。

## 7. 触发前仍缺的东西（清单）

1. ⏳ 触发本身：it260 还差 18 局（估计约 20:17）；侦查兵须持续刷新清单＋语料（50/80/100 刻度保险拉取别断）。
2. 🔧 工作区语料同步（两候选皆空）：触发时建集前必须先落地，链内 `SYNC_CMD` 覆盖。
3. 🔧 it260 的 +2 语料孤儿：触发时 API 重拉补清单，断言修平；修不好链停。
4. 🔧 it320 语料缺 40 局：只有它先触发时才需要补；it260 线不受影响。
5. 🔧 加权构建步骤：无现成脚本，触发时执行代理按 §3b 口径现写＋行数核对＋1 epoch 冒烟验证。
6. 🔧 本尊行为剖面基线：尚未提取（§6 末）。
7. ⏳ 训练机定夺：触发当场按 first-free 定（Studio wg1 后 / DGX M6 后），本备忘不预锁。
8. ✅ 已齐：链脚本现版（3053378b）、三臂命令模板（§3）、教师名逐字核验、Studio venv/磁盘、ds_k4v2 留存结论、Echo2 桥验收锚。
