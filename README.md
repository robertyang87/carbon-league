# Carbon League 策略研究 — 交接文档

> 本仓是 Carbon League（捕碳竞技场，https://132.232.176.153）策略研究的完整工作快照：
> 代码、实验报告与实验数据。**私有仓**——含第三方公开策略源码与未发布的竞争性研究，不要公开。
> 快照时间：2026-11-03 约 20:10 CST（源目录 `~/workspace/carbon-league/` 在快照时仍有 agent 在写案 C 文件，本仓为只读拷贝；文件清单与排除项见 `MANIFEST.md`）。2026-11-03 21:40 CST 增量刷新：案 C 全部 `casec_*` 文件已同步至结案状态（G1 判负＋P1.5 判负，见下），明细见 MANIFEST 刷新记录。

## 目标

- 打败大文件榜（单文件 ≥250 KB）榜首 **claude-AlphaCarbon-2**（总榜 #1，评分约 2600、59 胜 3 负、均分 9099），做到全场第一。大小榜别不为体量扭曲设计，NN 路线文件自然 ≥250 KB。
- 参赛账号：`muse-spark1-3`。**至今未提交任何策略**——提交是最后一步，须所有者确认后才做。
- 目标对象画像与关键步骤：见 `reports/alpha_analysis.md`（开局模板、养树流编制、终局基地口碰撞整合、三场败局克星档案）。

## 当前状态（三行账，数字以各报告磁盘原文为准）

1. **规则线（可提交的门面）**：基座 `code/base_v53.py`（= 站内公开的 claude-opus-v53，纯规则 + 全图 DP；标杆数字见 `reports/BASELINE.md`）。我方 fork：`code/fork_v0_1.py`（仅改持货成本 rho 0.02→0.08）对基座配对 8 局 6 胜、场均 **+1184**；`code/fork_v0_2.py` 是集成实验的出厂定版（全栈叠加经析因验证后全量回滚，出厂行为与 fork_v0_1 逐分一致，代码保留、开关关断）。规则线在天梯的估位约 **#80–180 / 1700–1950 分**（推断，非实测）。规则层已被六条支线（树估值/姿态切换/终局相位/camp/安全定价/参数激活扫描）证明是局部最优，便宜缝隙全部验过，不要重开。
2. **NN 旧线（已封存）**：逐工人独立 MLP + PPO 路线。冠军 = `code/caseb_rl_ckpt/policy_aux_seg2_zero.py`（权重 `aux_seg2_zero.npz`），固定探针合并 16 局 **−9,615.8**（对 base/fork 池），登记于 `code/caseb_rl_ckpt/state_champion.json`。蒸馏权重在 `code/distilled_v1_weights.npz`（652 KB）。五种机制级干预已逐一预注册证伪、方向关闭，不要重试：威胁特征入策略（消融净贡献 −1,391）、辅助头预热＋威胁列高 LR（转移路径毁生产行为）、预测器转硬盾（丢碳降幅 −7.7，未达 ≥300 判据）、教师种植代行（树收入 2.06× 但分差 −1,356，被阵型税吃掉）、停留信用价格塑形（税收先行）。全过程见 `reports/caseb_rl_report.md`（§7–§12）与 `reports/case_b_distill_report.md`。
3. **案 C（已结案）**：换策略类——集中式分配层＋确定性路由（设计见 `reports/casec_design.md`）：信念图感知 → 每工人 ≤14 个候选意图 → 学习型评分＋匹配解算强制无冲突 → 确定性路由＋四盾 → 基地头管配比。进度（以 `reports/casec_report.md` 为准）：G0 已结案（事件真值三项：PLANT 召回 98.7%、STEAL 100%、存碳精度 100%；双路一致率项经两轮裁定退役，属标签语义差而非正确性问题，见报告 §6–§7.1）；300 局语料已产完（450 个教师 player-game、730,136 样本、可训 611,646，见 §7.7；语料本体未进仓、清单见 `data/casec_corpus/manifest.json`）；BC 权重已出（`code/casec_ckpt/`）。**G1 判负**（§7.9：对 base 迷你探针场均 −14,023，闸 ≥ −6,000，四项全灭）；有界修复 P1.5（种植承诺锁，单变量 A/B）**复验判负**（§8：锁开场均 −15,051、灭绝 4/4、种树仍 2.5/局；归因＝约束在种植员存活与现金窗口、均在分配持续性上游，「承诺缺失」假说证伪）。**案 C 到此结案、不进 P2**，证据见 `reports/casec_report.md` §7–§8。至此规则 fork 之外的两条学习路线（案 B 蒸馏、案 C 分配层）均已结案并留全套证伪记录；存量＝fork-v0.1（可提交门面）＋aux_seg2_zero（冻结研究存量），暂无在跑攻坚线。

本地强对手基准已定为 nocamp_recon（对 base +3,351.7/8 局）；存档冠军对 no-camp −14,414.6、fork-v0.1 对 no-camp −3,024.3，详见 reports/nocamp_bench_report.md

## 目录地图

- `code/` — 全部策略与工具代码（根目录平铺，保持源目录相对引用关系）：
  - 基座与 fork：`base_v53.py`、`base_v53_probe.py`、`fork_v0_1.py`、`fork_v0_2.py` 及全部 `fork_*` 析因/消融变体、`policy_v*`（早期自研规则线 v0.1–v1，已关闭）、`switcher_*.py`（三型变速陪练）。
  - NN 旧线：`distilled_v1.py` / `distilled_v1_hybrid.py` / `distilled_v1_weights.npz`、`caseb_*`（RL 训练/探针/演员/威胁/价值/H4 脚本）、`caseb_rl_ckpt/`（全部段权重与生成策略＋`state_champion.json`）、`rl_policy_*.py`、`shadow_*.py`（榜首/DeepSeek 行为克隆影子，已判不够格当陪练，留作教材）。
  - 案 C：`casec_*.py`（标签/语料/BC/策略/探针）、`casec_ckpt/`、`casec_v53_probe.py`。
  - 基础设施：`fastsim.py`（与原版环境逐步精确一致的快速模拟器，约 7×）、`record_trace.py` 及 `record_*_traces.py`、`eval_*.py`（各线评测）、`pred_eval*.py`＋`opponent_predictor_tables.npz`（对手行为预测器：出击 AUC 0.956–0.996，资产已验、两条应用路线均已证伪关闭）。
  - `reference_*.py` — 站内公开源码的他人策略（教师谱系、陪练多样性、教材；注意各自来源与授权状态，见 `reports/public_sources.md`）。
  - `env_src/` — 2021「智荟杯」原版竞赛环境源码（本地运行/校验用；跑原版环境用系统 python3，venv 跑不了 zerosum_env）。
- `reports/` — 全部实验报告与分析（索引见下）。
- `data/` — 实验数据：根目录全部 `*_results.jsonl/json`、训练日志、`leaderboard*.json`、`matches_alpha*.json`、预测器数据集 `weights/pred_data_*.npz`、RL/影子权重 `weights/`、`traces/INDEX.md` 与 `traces/` 本体（逐局 301 帧留痕）、`summaries/`（榜首回放摘要）。
- `MANIFEST.md` — 本仓最终文件树、排除清单及理由、快照口径。

## 运行口径（评测的可复现性全靠这一节）

- **快速模拟**：`code/fastsim.py`，与原版环境 36/36 局逐步逐字段精确一致；正式对局留痕走 `code/record_trace.py` 写入 `traces/` 并登记 `traces/INDEX.md`，复跑核对终分须 0.00。
- **固定探针**：候选版本对 base_v53 配对 4 局＋对 fork 配对 4 局（筛选），定论 ≥8 局、关键晋升 16 局双批确认；同 seed 同图交换先后手。
- **规范播种（全项目统一）**：`random.seed(7919*seed + 104729*side + 31)`，且以被测方（policy 方）为播种对象——v53 消费全局随机，播种对象不同会改变整局，跨口径数字不可直接比。
- **噪声带**：单局地图未播种，探针 Δ 在 ±1,000 内只记方向、不记功劳；晋升须 Δ>0 且生产账本（种树/树收入/死亡丢碳/终局滞留/灭绝）不触红线，噪声带内加跑新 seed 确认批、双批同向才晋升。
- 常用入口：探针 `code/caseb_rl_probe.py`（旧线）/ `code/casec_g1_probe.py`（案 C）；评测骨架 `code/eval_*.py`；基座冒烟 `python3 code/fastsim.py` 相关 test 见 `code/test_fastsim_parity.py`。

## 方法论纪律（所有者定调，违反即返工）

1. **有的放矢**：每步先有针对性的、可证伪的假设与**预注册**的验证口径和阈值（写进报告再执行），一次只动一个变量；归因靠消融/A/B 对照，不靠事后叙事。
2. **数字只认磁盘**：汇报通道送达的数字与磁盘（日志/结果文件/INDEX）不符时一律判损坏弃用——本项目已发生 5 起同类事件。声称完成必须有落盘产物与读回验证。
3. **闸不过不往下**：每阶段预注册生死闸（如案 C 的 G0–G3）；闸口径若被证明测错对象，修正须留数学证明与书面裁定（如 G0 一致率项），不许看数后放宽。
4. **存量只读**：定版文件（`base_v53.py`、`fork_v0_2.py`、`distilled_v1*.py`、`caseb_rl_ckpt/`）只许引用、不许原地覆盖被测文件（曾有覆盖导致留痕错位的教训）。
5. 算力：本机 4 核 7 GB（与其他项目共用）；Carbon League 侧 RSS ≤3.5 GB，数据分片落盘，长进程 setsid＋断点续跑。

## 关键报告索引（一句话/份，全文在 `reports/`）

- `BASELINE.md` — 基座标杆登记：v53 对 juzi +5575 的账本与参数地图入口。
- `alpha_analysis.md` — 榜首 AlphaCarbon-2 全量 62 局＋深挖 12 局拆解：开局步级时间线、养树编制、终局收口、三场败局。
- `v53_structure_roadmap.md` — v53 八病灶结构路线图与六线总账（规则层局部最优的完整证据）。
- `lineage_diff_report.md` — v18→v53 谱系考古：经济引擎逐字节冻结、增量全在送货战争层。
- `public_sources.md` — 全站公开源码普查结案：26 份全为手写规则、零 NN，无捷径。
- `case_b_distill_report.md` — 案 B 第一阶段：蒸馏保真闸未过（−15,864），死因是可表征性而非数据/管线。
- `caseb_rl_report.md` — 案 B 第二阶段全记录（§1–§12）：RL 周期、价值头、协调通道、威胁三部曲、生产分解、H4，及全部证伪裁决。
- `casec_design.md` — 案 C 设计与实测算力账：分配层架构、四道闸（G0–G3）数字、死刑条款。
- `casec_report.md` — 案 C 执行报告（已结案）：G0 两轮重验与裁定、300 局语料总账、§7 G1 总裁决（判负）、§8 P1.5 承诺锁 A/B 复验（判负）与归因。
- 其余专题报告（`s1_*.md`–`s4_*.md`、`integration_report.md`、`c_pricing_report.md`、`guard_steal_report.md`、`activation_scan_report.md`、`opponent_predictor_report.md`、`bc_shadow*_report.md`、`rl_stage*.md`、`v53_verify_report.md`、`policy_v*_report.md` 等）为各支线的单项裁决存档，按文件名检索即可。

## 合规与约束

- 提交形态：单文件 `submission.py`，顶层 `def agent(observation, configuration)`，≤10 MB；运行环境 Python 3.12 + NumPy 1.26.4 + PyTorch 2.5.1 CPU；初始化 ≤20 s、每步基准 1 s（最多 3 s）；策略名全站唯一；源码默认不公开。
-
...[truncated 806 chars]