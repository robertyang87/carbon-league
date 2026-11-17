# SOP：标注后链标准流程（merge → build → join → prescan → 开训）

版本：v1.1（2026-11-18 修订：build/join/prescan 座位改 L20，所有者 00:33 裁定；v1.0 于 2026-11-18 立，源于 leg-1 241 局后链实战复盘）
适用：任何一批标注完成后的数据后链。执行人一名、端到端；作战室发 CHAIN_GO 后整链连跑，步间不许停顿等指令。

## 0. 总则

1. **单一执行人**：一条链一个具名执行人，从 collect 到 prescan 一口气跑完。禁止把执行拆成多线中转（DGX 只作档案/登记，不作执行中转）。
2. **步间不停**：CHAIN_GO 落盘即整链授权。每步完成自动接下一步；步间停滞超 15 分钟＝异常，当趟报作战室。回执在每步完成时发，不等整链结束。
3. **每步回执三件套**：命令原文、RC、产物身份（gen 号／md5／计数）。断言「在跑」必须进程在册＋日志 mtime 新鲜，只看日志尾不算。
4. **估计口径**：ETA 只许用同窗同并行度的实测速率推，标「估计」；bench 单价不许线性外推整局/整批墙时（并行争用与步深都会抬价）。
5. **产物登记**：每件产物落地当趟登记入 `data/artifact_registry.json`（名称、md5、存放机、路径、责任线）；找件先查册。

## 1. 前置条件（开链前逐项验，缺一不开）

- [ ] 标注分片全量回拉并逐片 md5 验合（计数 = 计划局数，分片根数 = 局数 × 窗口根数）。
- [ ] 标注前完整性检查已做（SOP 既有项）：逐局步数、步连续、两边 statuses DONE、终局 rewards 在位；异常局已列单、处置已定。
- [ ] 工具钉版齐全且座上直读 md5 合（现行钉版见附录 A；同名不同代以任务件单为准，错代即退）。
- [ ] order 文件（局序清单）在册，局集与冻结清单逐项对齐，test/train 不混装。
- [ ] L20 在位且处于开训前窗口（build/join/prescan 落 L20，见 §4–§6）；持有唯一产物的座位释放闸已按本批调整（见 §8）。

## 2. 步骤一：collect（收集封存代）

- 首选：座位在册时整包拉 `annot_train` 根目录（含 `current.json` 的封存代），拉后逐片 md5 验合再放行释放。
- 座位已释放、只剩分片时走**重封路径**（2026-11-17 实证）：
  - 分片 `.npz` 自带完整 `_meta`（身份＋逐字段 sha），`k0_annotate` 对已验分片 skip 并经 `publish()` 确定性重封。
  - ⚠️ `meta.teacher.weights` 存的是权重文件**路径字符串**，gen 号随路径变。重封必须在能解析出**与原座位相同路径**的机器上做（本批口径：CPU 座 `/root/carbon/data/weights/...`）。
  - **重封证明**：至少两包与已知参照 gen 号逐字复现（活参照＝座上现成 current.json；记录参照＝数据组封存记录），才许信任无参照包；试封产物一律作废、不得作 merge 输入。

## 3. 步骤二：merge（合并，闸 ②③⑥ 不过不许继续）

- 工具：`merge_annot_gens.py`（钉版见附录 A），按 order 文件合并各源封存代。
- 闸② 身份一致：各源 config_sha256、权重 md5、七件代码 md5、采样口径全等，任一不合该源拒收、点名。
- 闸③ 试合并：先 1 局试合 RC=0 再全量。
- 闸⑥ 对照：抽 1 局与源分片逐字段比对。
- 合并后必做**全量字节审计**：合并代逐行与源比对（除 gid 按合并局号重排外，其余字段逐字节同），0 不符才算过。本批实测：144,118 行 0 不符。
- 回执：merged gen 号、总根数、局数、manifest sha256。

## 4. 步骤三：build_src 组装 + build

- build_src：index 条目与 order 文件**逐项同序**；replays 件数 = 局数（逐件 md5 对拷源）；断言 0 短局（全 299 步）。
- build：`cnet2_build.py`，环境 `CNET2_LADDER_FRAME=1`。**build 落 L20 跑**（所有者 2026-11-18 裁定：build+join 在 L20，产物就地供训练消费，省跨机搬运；只许开训前窗口、不与训练并发）。
- 回执：build gen 号、样本数、cfg_fp、RC。

## 5. 步骤四：join（落 L20）

- **join 落 L20 跑**（所有者 2026-11-18 裁定，取代 2026-11-17 21:52 的 CPU 座口径）：build 与 join 同机连跑，joined 就地供 L20 训练消费。护栏：只许开训前窗口执行，不与训练并发（防抢训练读数据的 CPU）。标注等其余 CPU 重活仍走 CPU 机队，本条只改 build/join/prescan 三步。
- 开跑前机上直读：`k0_join.py` md5 必须 = 执行钉版（附录 A）；`PYTHONPATH=<L20 上 code 目录>`（join 的发布预检从脚本同目录 import k0_distill，缺 PYTHONPATH 必死）。
- 命令：`k0_join.py --annot <merged> --build <build> --out <joined> --verify`，`--verify` PASS 才算完成。
- 回执：joined gen 号、manifest、根数、verify 结果。

## 6. 步骤五：prescan 硬闸（所有者 2026-11-17 19:10 定规，无报告不开训）

- 座位：prescan 随 joined 在 **L20 就地跑**（读全集，不再搬运）。
- 用训练器本人的拒训判据（k0_distill features 校验：root cash 在 f16 下跨 60.00 阈值不可判定一类）对 joined **全集逐根全扫**，不许抽样、不许只扫触发条件的一部分。
- 报告必含：全集根数／问题根数／问题局清单。报告不出，训练不开。
- 有问题根：其所在局**整局一次性全排除**（不论本代是否真触发），回步骤二重走 merge → build → join → prescan，全清才过闸。
- 起因存档：快线首训死于第 600/3,526 步（第 55 局 root 32918 现金恰压 60.00），600 步算力作废——此闸即为此设。

## 7. 步骤六：开训（作战室单独发令）

- prescan 过闸后，执行人**不得自开训练**，等作战室开训令。
- 配方按当线注册件；步数口径：`round(5.17 × joined 实际根数 / 128)`（剂量按注册值，变更须策略组裁定）。
- 开训三件套回执：进程 PID、命令原文、日志首行＋权重输出路径。

## 8. 释放闸（与后链并行执行）

- 持有唯一产物的机器，释放前必须齐：产物全量回拉验合＋产物责任线书面「已回收」＋作战室确认。缺一不释放（spot-48 教训：尾窗 60 局产物随释放全损）。
- 回拉/验合脚本的产物后缀必须按**本批实际产物**写（.npz ≠ .json.gz），冒烟必须覆盖每个分支（WAITING/NEW_VERIFIED/MISMATCH/ALREADY），只跑空分支不算验过（test118 误报案）。

## 附录 A：现行钉版（2026-11-17 版，变更走策略组裁定＋入仓登记）

| 件 | md5 |
|---|---|
| k0_annotate.py | 7f567a2e |
| stage1_teacher_gate.py | e922d9d8 |
| stage0_teacher.py | c6ade5ce |
| cnet2_feat.py | 2a45b08b |
| cnet2_search.py | 94f7fb64 |
| cnet2_np.py | d8c15ad4 |
| fastsim | aefe6c83 |
| 教师权重 cnet2_m3s2r1_heads_v2.npz | 06a03d9cfeec239af3065a49fefec382 |
| merge_annot_gens.py | 821fa780 |
| merge（排除版） | 57cc1310 |
| cnet2_build.py | 5c2dd58e |
| **k0_join.py（唯一执行钉版，仓 fd7781a8）** | **fac55acbcbffdfd6582c9449e535aa2f** |
| k0_distill.py | 2728fee6 |
| prescan.py | 27aad536 |

config 口径：全窗 [0,299) = 3cc2e67e66cff9a1d901b663a579abc2d62246d6bdf76017f00bcf998aea1ff3；尾窗 [200,299) = 9a2b2440。
注：DGX 自行版 k0_join（49e5c9a4…）只存档、禁止执行。

## 附录 B：本 SOP 的来源事故（防重犯索引）

1. 快线首训 600 步死于坏根 → §6 prescan 硬闸。
2. spot-48 释放丢 60 局尾窗产物 → §8 释放闸。
3. leg-1 collect 脚本按「整包已在 DGX」写、实际只有分片 → §2 重封路径＋路径字符串陷阱。
4. test118 验合脚本后缀照搬误报 MISMATCH → §8 后缀/冒烟规矩。
5. 后链步间停顿（所有者三问 join）→ §0 总则 2：整链连跑、步间 15 分钟线。
