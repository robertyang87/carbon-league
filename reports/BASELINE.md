# BASELINE —— 基座标杆登记（2026-11-03）

用户 2026-11-03 09:29 拍板：**claude-opus-v53 定为基本标杆**，我方在其之上做优化。

- 基座文件：`base_v53.py`（= `reference_claude_opus_v53.py` 逐字节复制，md5 `1779b453672f5b85d6df1cf473e64ab0`，2071 行，115,897 B；参考原件不动）
- 来源：站内公开源码，作者 "Claude"，源地址 https://132.232.176.153/source/97898ce7047c0bb1dd7a13f02eff0dac
- 定位：纯规则 + 全图 DP 规划（无 NN）。架构 = 四重对称信念碳图（Belief 类）+ 采集员 H=12 步全图 DP（持货成本 rho）+ 一步硬安全/软风险 + 种植/抢树边际估值 + 扎营/猎杀/围攻/冲门等子系统（PARAMS 共 167 项）。

## 标杆数字（第一把尺 / 第二把尺）

评测口径：FastCarbon 同 seed 同图、换边配对（eval_v53.py，与 eval_v0_8 同方法学，账本由状态差重建）；终分经 traces 复跑核对（见 traces/INDEX.md 的 v53 节）。

| 对位 | 局数 | 战绩 | 场均分差（基座视角） | 备注 |
|---|---|---|---|---|
| **base_v53 vs juzi（历史标杆）** | 8（seeds 1000–1003 ×2） | **7 胜 1 负** | **+5575** | 逐局：+4337 / +7254 / +3551 / −162 / +5458 / +5045 / +8124 / +10993。用户 2026-11-03 09:46 指令：juzi 停用为对弈对手（已大幅过线且其 ~1.3 s/步为最贵对手），此行仅作历史标杆引用，今后不再新打 |
| base_v53 vs 我方 v0.5 | 4（seeds 1000–1001 ×2） | 4 胜 0 负 | +10768 | |
| base_v53 vs example | 2（seed 2000 ×2） | 2 胜 | +17375 | 冒烟 |
| GloriousEvolution 版（rho 0.06）vs base_v53 | 4（seeds 1000–1001 ×2） | GE 1 胜 3 负 | **−578（GE 视角）** | GE 自称本地 +576/73 胜 120 局；我方口径下未复现（见 tuning 报告） |

**今后自家版本的第一把尺 = 对 base_v53 的配对分差；NN 型参照尺 = 对 shadow_alpha_v2（及后续影子 DeepSeek）的配对分差。juzi 自 2026-11-03 09:46 起停用（历史成绩保留引用）。**

### 关键账本（对 juzi 8 局场均，历史标杆）

| 项 | base_v53 | juzi |
|---|---|---|
| 存款 deposits | 6126 | 3562 |
| 树收入 tree_income | 5819 | 3197 |
| 死亡数 deaths | 18.6 | 31.1 |
| 死亡丢碳 death_cargo | **43** | 1461 |
| 终局滞留 stranded | **36** | 1700 |
| 种树 / 抢树（次） | 26.4 / 14.8 | — |

### 速度

- 对 juzi 局：29–42 ms/步（4 进程共抢口径），单步最差 292 ms。离站点 1 s/步均值线很远，速度不是约束。

## 参数地图（PARAMS，L271–282，共 167 项）

重点参数机理（通读 base_v53.py 后）：

- **rho = 0.02（持货成本，本轮第一刀）**：采集员 DP 中每步持货惩罚 `lam = rho × 携带量`（L1773），同时决定终端项 `term = −lam × 距基地距离 × term_k`；rho 越大越倾向早存。仅在中盘生效——开局相位（`early`：编制 <10 且 R>60）用 **rho_early = 0.05**，终局（R ≤ endgame）lam 强制 0。
- **mu = 0.05 / mu_early = 0.3**：基地格价值加成 `base_val = Wfree + mu × 携带量`（L1780），是「往家走」的引力，与 rho 对偶。
- **H = 12, S = 5, gamma = 0.92**：DP 规划步长 / 停留状态数 / 折扣。H 是算力与远见的主钮。
- **安全组**：`avoid1 = 0.5 / avoid2 = 0.15`（下一步/再下一步可被更轻敌采集员到达的格子，按 (携带+30) 计罚，C1/Cr 成本）；`soft_w = 1.0`（距离 2 内敌采集员的软风险）；`tie_cost = 12.0`（同格对峙的并列死亡成本）；`kill_c = 40.0 / kill_p = 45.0`（猎杀采集员/种植员的击杀奖金基数，实际奖金 = 目标携带 + 基数）。
- **种树组**：`plant_min = 12.0` 与 `plant_ratio = 1.6`（种树边际价值 TV 须 ≥ 价 + 12 且 ≥ 1.6×价 才立项，L1000 附近）；`hold_p = 0.8`（TV 折扣）；`steal_w = 0.6`（抢/防抢权重，同时出现在 TV 与抢树估值 SV 中）；`seize_min = 8.0`（抢树门槛，抢价 = (1+steal_w)×SV − 20）；`pdisc = 0.85`（种植员赶赴任务的距离折扣）；`pp_cost = 10.0 / pp_stand = 28.0`（种植员赴任/站位成本）；`guard_min = 20.0 / guard_w = 0.5`（守树任务门槛/权重）。
- **招募/终局组**：`open = "CP"`（第 0 步两基地开局序列：先 C 后 P）；`planter_from = 4`（采集员数 ≥ 4×种植员数 才许再招种植员）；`early_planters = 2`（前期种植员目标数，由 block_ok/存量条件切换到 retire_keep）；`recruit_until = 40`（R ≤ 40 后只许招「岗哨型」晚招，须 recruit_post 且缺岗且预算 ≥150）；`endgame = 14`（R ≤ 14 进终局模式：lam=0、非基地格终端值 −1e4、存入即全额计分）；`max_units = 10`；`retire_from = 85 / retire_stock = 2400 / retire_entrees = 2 / retire_min_left = 50`（步 ≥85 且全图存量 <2400 且敌树 ≤2 时裁种植员换采集员）。
- **环境常数（非 PARAMS，属规则复刻，勿动）**：REGEN 1.03、CAP 100、ABS 0.0375（树吸收率）、COLLECT 0.3、种树价 20 + 5×1.235^存活树数（L751）、停留先验 pw = 0.7^s（L787）。
- 子系统参数（camp/hunt/siege/sb/run/shadow/gate 共约 120 项）见下表分组清单；本轮不动，先扫 rho。

### DP核心（8）

| 参数 | 现值 |
|---|---|
| H | 12 |
| S | 5 |
| gamma | 0.92 |
| rho | 0.02 |
| rho_early | 0.05 |
| mu | 0.05 |
| mu_early | 0.3 |
| term_k | 1.0 |

### 安全/战斗（8）

| 参数 | 现值 |
|---|---|
| tie_cost | 12.0 |
| soft_w | 1.0 |
| kill_c | 40.0 |
| kill_p | 45.0 |
| camper_max | 20.0 |
| displace | 1 |
| avoid1 | 0.5 |
| avoid2 | 0.15 |

### 信息（2）

| 参数 | 现值 |
|---|---|
| info_c | 2.0 |
| info_p | 3.0 |

### 种树/抢树（21）

| 参数 | 现值 |
|---|---|
| open | "CP" |
| planters | 1 |
| planters_late | 1 |
| planter_from | 4 |
| plant_ratio | 1.6 |
| plant_min | 12.0 |
| hold_p | 0.8 |
| steal_w | 0.6 |
| cell_loss_w | 0.5 |
| seize_min | 8.0 |
| pdisc | 0.85 |
| pp_cost | 10.0 |
| pp_stand | 28.0 |
| lunge | 1.0 |
| guard_w | 0.5 |
| guard_min | 20.0 |
| p_extra | 0 |
| p_extra_trees | 5 |
| no_p_T | 60 |
| early_planters | 2 |
| lunge_floor | 0.0 |

### 招募/终局（9）

| 参数 | 现值 |
|---|---|
| recruit_until | 40 |
| endgame | 14 |
| retire_entrees | 2 |
| recruit_post | 20 |
| retire_stock | 2400.0 |
| max_units | 10 |
| retire_from | 85 |
| retire_keep | 0 |
| retire_min_left | 50 |

### 扎营 camp（17）

| 参数 | 现值 |
|---|---|
| campers | 4 |
| camp_from | 6 |
| camp_min | 0 |
| camp_div | 99 |
| camp_decay | 80 |
| camp_orbit | 2 |
| camp_late | 60 |
| camp_from_late | 3 |
| camp_occ | 0.0 |
| camp_lunge | 0.0 |
| camp_end | 4 |
| camp_end_from | 28 |
| camp_planters | 1 |
| camp_tie | 40.0 |
| camp_w | 30.0 |
| camp_keep | 100.0 |
| camp_until | 0 |

### 猎杀 hunt（54）

| 参数 | 现值 |
|---|---|
| eval_T | 30 |
| eff_hi | 0.6 |
| eff_lo | 0.35 |
| eff_rate | 2.3 |
| reprobe | 0 |
| kill_win | 30 |
| kill_n | 3 |
| lost_max | 6 |
| probe_n | 0 |
| probe_T | 12 |
| probe_kill | 2 |
| probe_win | 20 |
| orbit_kill | 3 |
| orbit_win | 12 |
| r2_hi | 0.75 |
| r2_lo | 0.45 |
| r2_kill | 0.6 |
| hunt | 1 |
| hunt_min | 150.0 |
| hunt_R | 5 |
| hunt_n | 3 |
| hunt_hmax | 1e9 |
| hunt_camp | 0 |
| hunt_Rc | 5 |
| hunt_targets | 3 |
| hunt_pat | 8 |
| hunt_cool | 15 |
| hunt_fw | 0.5 |
| hunt_fcap | 250.0 |
| hunt_sw | 1.0 |
| hunt_scap | 600.0 |
| hunt_fmin | 0.99 |
| rovers | 0 |
| rover_d | 3 |
| rover_from | 8 |
| hunt_ct | 1 |
| hunt_ct_d | 2 |
| hunt_ct_r | 4 |
| hunt_gate | 99 |
| hunt_ct_solo | 1 |
| hunt_tab | 1 |
| hunt_eps | 0.3 |
| hunt_vmin | 0.25 |
| hunt_tabfile | None |
| lone | 1 |
| lone_R | 5 |
| lone_stick | 10.0 |
| mark | 0 |
| mark_min | 40.0 |
| mark_R | 4 |
| mark_reach | 6 |
| mark_n | 4 |
| mark_chase | 0 |
| orbit_tie | 40.0 |

### 围攻 siege/sb（15）

| 参数 | 现值 |
|---|---|
| siege_hold | 10 |
| siege_break | 1 |
| sb_T | 20 |
| sb_carry | 1500.0 |
| sb_off | 500.0 |
| sb_gate | 1 |
| sb_recall | 1 |
| sb_guards | 2 |
| sb_gap | 25 |
| sb_gap_p | 60 |
| sb_static | 1 |
| sb_still | 3 |
| siege_spawn | 1 |
| siege_T | 3 |
| siege_carry | 150.0 |

### 冲门 run（12）

| 参数 | 现值 |
|---|---|
| run | 1 |
| run_mode | 3 |
| run_ev | 1.0 |
| run_small | 120.0 |
| run_T2 | 8 |
| run_n0 | 6 |
| run_b1 | 150.0 |
| run_b2 | 300.0 |
| run_T | 30 |
| run_k0 | 0.25 |
| run_k1 | 0.05 |
| run_sg | 15 |

### 影随 shadow（10）

| 参数 | 现值 |
|---|---|
| shadows | 2 |
| shadow_pen | 3.0 |
| shadow_minfree | 2 |
| shadow_post_d | 0 |
| shadow_swap | 1 |
| shadow_hyst | 2 |
| shadow_from | 8 |
| shadow_start | 20 |
| shadow_trees | 0 |
| shadow_end | 20 |

### 门 gate（11）

| 参数 | 现值 |
|---|---|
| carry_ok | 100.0 |
| def_always | 0 |
| gate | 1 |
| gate_min | 60.0 |
| gate_w | 0.5 |
| gate_cap | 150.0 |
| predict | 1 |
| sweep_w | 0.3 |
| sweep_base | 10.0 |
| sweep_max | 80.0 |
| base_def | 20.0 |


## 本轮调优产出（2026-11-03 第一轮，详见 v53_tuning_report.md）

- **fork-v0.1 = 基座 + rho 0.02→0.08**：对基座 8 局 6 胜、场均 +1184；对 shadow_alpha_v2（NN 型参照）4 局 4 胜、场均 +14603（陪练太弱，仅作不翻车证明）。已知代价：镜像局终局滞留 1686 vs 基座 401（终局相位问题，S3 候选）。
- GE 的 rho=0.06 本地同口径复测为负（−578），其自称 +576 不成立。
