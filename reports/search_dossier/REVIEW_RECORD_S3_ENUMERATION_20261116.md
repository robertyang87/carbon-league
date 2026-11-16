# S3 枚举脚本审查记录（§10.4 审查门）

> 状态：**本地审查记录，未入仓、未发策略组**——待父代理裁定未决事项（§8）后，再定
> 是否以此脚本开跑全量枚举。
> 审查人：搜索组（子代理）。审查日期：2026-11-16。
> 依据：接受版协议 `S3_ROOT_LIST_FREEZE_PROTOCOL.md`（仓内 commit ebed48f4，
> md5 4212f13a）；触发清单 `S3_TRIGGER_LIST_EXEC.md`（afcb0ed1，md5 656e279f）；
> 附则（8167b5e5）与修订（dc95e2c2）。

## 0. 审查对象与结论总览

| 件 | 路径 | md5 |
|---|---|---|
| 枚举脚本 | `~/workspace/search-group/s3_freeze/enumerate_roots.py` | `1218d573e81f6f9b614a61597968ea45`（试跑启动时与审查时一致，未改动） |
| 试跑根单 | `s3_freeze/trial_out/roots_trial.json`（4,513,552 B） | `e6e2ba1453c526f63868da66f83a9236` |
| 试跑根单 JSONL | `s3_freeze/trial_out/roots_trial.jsonl`（3,693,068 B，3920 行） | `11f0baa9cc2cc172fc706413d302a9d5` |
| 试跑局总体册 | `s3_freeze/trial_out/register_trial.json`（1,777 B） | `58a39ad8bd036789f97fc75423bd72cd` |
| 试跑报告 | `s3_freeze/trial_out/report_trial.json`（1,166 B） | `ab3828ad9c227e17cd36a45d2fb565b4` |
| 试跑见证 sidecar | `s3_freeze/trial_out/witness_trial.jsonl`（19,881,352 B，3920 行） | `4609087301209eefa902067758d509eb` |
| 试跑日志 | `s3_freeze/trial_run.log` | `40b34567061a7d1fa2bbef0bcf1c9981` |
| 结算验证探针 | `s3_freeze/probe_settle_298.py` | `86b20725e2b67d10541a4c9be521fd00` |
| 探针日志 | `s3_freeze/probe_settle_298.log` | `9a20278634f6aa41a1391f317597e45b` |

**§10.4 三门结论：门 1 通过；门 2 通过（附一项未覆盖说明，见 §2.3）；门 3 通过
（18 根人工抽查全合＋结算独立验证 8/8）。全量开跑的前置条件只剩 §8 的四项
未决事项待父代理裁定。**

试跑口径与考位口径的差别（任务书要求写明）：试跑在本 VM 以普通进程跑，
**不设 0.7 秒 decide 预算约束**（枚举不做比较器决策，预算在测试期才绑定，
协议 §10.2 明文）；不设 systemd scope（CPUQuota=150%／MemoryMax=512M），
不钉 BLAS 线程数；其余口径与考位一致（基线包 md5 819250bf、生产 CONF 原样、
agent 局内持久／跨局重建、打满至权威终局 step 299）。试跑 3 局计入全集
（同一脚本、同一批，§10.4 门 3）。

## 1. §10.4 门 1：触发判据逐行核对

核对源：本地 vendor `proto/vendor/cnet2_search.py`（钉版 md5 94f7fb64 的
内容；文件头 3 行只读注释除外，`real_env.verify_vendor()` 头除外 md5 全合）。
协议 §1.2 引用的行号有两处 off-by-one（见下表注），实质已核对无误。

| 判据项 | 协议引文 | 实测行（vendor） | 核对结果 |
|---|---|---|---|
| `key_units` 定义 | :386–478 | :386 `def key_units(obs, cfg):` | 合。基础段 :386–405 |
| 1a 谓词 | :399–400 | :399 `if e[1] <= cargo and cargo >= cfg["MINC"]:`；:400 `best = max(best, cargo * cfg.get("PRIO_THREAT", 1.0))` | 合 |
| 1b 谓词 | :401–402 | :401 `elif cfg["HUNT"] and e[1] > cargo and e[1] >= cfg["MINC"]:`；:402 `best = max(best, e[1])` | 合 |
| 排序 | 协议引 :404 | **实测 :405** `keys.sort(reverse=True)`（元组降序：stake 降、uid 降） | 合（行号引文差 1，实质合） |
| THREAT_FIRST 门 | :406 | :406 `if cfg.get("THREAT_FIRST"):`；生产 CONF 缺省 → 关 | 合（脚本 `assert_variant_gates_off` 启动时断言，试跑通过） |
| SPLIT 门 | :418 | :418 `if cfg.get("SPLIT"):`；缺省 → 关 | 合（同上） |
| KEY_LATE_HOME 门（类二唯一路径） | :437 | :437 `if cfg.get("KEY_LATE_HOME") and ...`；键不在包 CONF 亦不在 DEFAULTS → 关 | 合；类二触发不成立 |
| MAX_KEY_LATE 门 | :453 | :453 `if cfg.get("MAX_KEY_LATE") is not None and obs["step"] >= cfg.get("KEY_LATE_STEP", 999):`；键缺省 → 关 | 合 |
| KEY_LATE_CARGO 门（类三主段） | :455 | :455 `if cfg.get("KEY_LATE_CARGO") and obs["step"] >= cfg.get("KEY_LATE_STEP", 999):`；键缺省 → 关 | 合；类三触发不成立 |
| PLANTER_KEY 门 | :463 | :463 `if cfg.get("PLANTER_KEY") and ...`；缺省 → 关 | 合 |
| BASE_KEY 门 | :479 | :479 `if cfg.get("BASE_KEY") and ...`；缺省 → 关 | 合 |
| 截断 | 协议引 :463 | **实测 :462** `keys = keys[:mk]`；:452 `mk = cfg["MAX_KEY"]` | 合（行号引文差 1，实质合） |
| 常数 | KEY_R=2／MINC=10.0／HUNT=True／MAX_KEY=3 | 试跑 conf 摘要逐字合；脚本启动断言（`assert_variant_gates_off`）偏离即停 | 合 |
| KEY_LATE_STEP | 默认 999 | 试跑 conf 摘要 `"KEY_LATE_STEP": 999` | 合；CONF 的 `"LATE_STEP": 240` 是对手教师相位切换（:843–857），不是 KEY_LATE_STEP（触发清单 §3 已辨） |
| PRIO_THREAT | 未设 → 1.0 | 试跑 conf 摘要 `"PRIO_THREAT": 1.0` | 合；stake_1a ＝ cargo |

脚本内独立复算（非仅调用）：`independent_predicates`（:197）与
`independent_key_set`（:226）不用 `key_units`、从 obs 现值重算全集；
`_process_triggered_seat` 在**每一个触发席步**比对独立重算与 `key_units`
输出（stake 差 >1e-9、uid 集或序不符即整趟停）。试跑 3920 根、0 停，
即该交叉核对在全部触发席步上通过。另：脚本在启动时以生产 cfg 原样
复算（`make_search_agent` :621–623 口径），不设 step≥240 线，全步带
0–298 均筛（协议 §1.5）。

**门 1 判读：通过。**

## 2. §10.4 门 2：结算函数指针核对

| 指针 | 文件＋行 | 核对结果 |
|---|---|---|
| 叶内 F01 分支（权威源） | vendor `cnet2_search.py:666–676`（`leaf` 内 `_terminal` 分支：`pen = step_count − episodeSteps − 1`；存活席取 `prev_cash`） | 已亲读全文，合 |
| `make_terminal_score_fn`（verify3 同源） | `proto/s1_verify3.py:298–315`（逐席 rew 组装式，与叶 :666–676 同源；`_extinct` 取自叶闭包权威实现） | 已亲读全文，合 |
| 脚本 `terminal_scores` | `enumerate_roots.py:188–195`（F01 组装式：`pen`／`prev_cash`，谓词取自叶闭包） | 合（与 s1_verify3 同式） |
| 脚本 `_settle_one` | `enumerate_roots.py:409–434` | 合，见下 |
| 冻结记录指针 | 根单 `enum_provenance` 记 `key_units {file, lines 386-483, code_md5 94f7fb64}`；局总体册 meta 记 `settlement_fn {file code/cnet2_search.py, lines 666-676 (F01), score_fn_source s1_verify3.py:298-315, time_terminal F04}` | 合 |

`_settle_one` 两分支口径：
- **灭绝**（`terminal_fn(sim)` 真）：`terminal_scores` 逐席 F01 得分
  （灭绝席＝`step_count − episodeSteps − 1`；存活席＝入终局步前现金
  `prev_cash`），值＝本席分 − 对手分；并与冻结叶 `leaf_fn` 的 F01
  分差逐例交叉核对，不等即停。
- **时限**（`sim.step_count >= 299`）：值＝一步后双席现金差
  `cash[me] − cash[1−me]`（F04 终局现金口径，修订版 §1.1）。

### 2.3 未覆盖说明（非门失败，记入审查记录）

试跑 3 局全部以时限终局（reached 299），**无灭绝局**；故 `_settle_one`
的灭绝分支及其叶 F01 交叉核对在本次试跑中**未实际触发**。该分支代码与
s1_verify3 的 `make_terminal_score_fn` 同源（R2-02 修复后已由 battery
rerun 的 A6 B9 例验证：灭绝∩时限叶值恰 −102.0），且脚本在灭绝分支内
保留逐例交叉核对；全量枚举中一旦出现灭绝局，该核对即自动执行、不等
即停。审查结论：门 2 通过，此点记为「试跑未覆盖、代码同源已验」，不
构成开跑阻拦。

**门 2 判读：通过。**

## 3. §10.4 门 3：3 局试跑与人工抽查

### 3.1 试跑结果

试跑启动 2026-11-16 约 10:52 CST，进程 PID 9023，本 VM，`--games 0-2
--shard-id trial`，脚本 md5 1218d573（启动时与审查时一致）。
总墙钟 2241 s（约 37.4 分钟），退出 0，无 traceback，无 HALT，
§1.4 触发事件「无」。

| 局 | 种子 | 到步 | 止因 | 根数（席 0／席 1） | 墙钟 |
|---|---|---|---|---|---|
| 0 | 986200 | 299 | time_limit_299 | 1450（725／725） | 764.0 s |
| 1 | 986201 | 299 | time_limit_299 | 1030（515／515） | 690.6 s |
| 2 | 986202 | 299 | time_limit_299 | 1440（720／720） | 785.1 s |
| 合计 | — | — | — | **3920（1960／1960）** | 2241 s |

- 产根局 N＝3，零根局 M＝0（N＋M＝3，合）。
- 结算可辨根 10（局 0：6，局 1：4，局 2：0），全部在 step 298；
  `settlement_argmax` 全部为 0（A0）；非零结算差仅 1 例（局 1，
  一候选 −11.5，见 §3.3）。
- agent 调用 1794 次，均值 0.463 s、最大 1.323 s（VM 口径，非考位口径）。
- 根数按 50 步带分布（局 0 例）：0–49：48；50–99：246；100–149：300；
  150–199：270；200–249：298；250–299：288——根随局势发展增多，
  后期近每席步 3 根（MAX_KEY 上限），符合触发面随 collector 密度
  上升的预期。
- 早期 20 步冒烟（同一脚本、game 0 前 20 步）：10 根（5／席），
  两次独立进程输出逐字节一致（确定性已验）；试跑重放亦复现
  step-298 key 集逐字一致（见 §3.3）。

### 3.2 人工抽查（每种子 6 根，共 18 根，全合）

抽查法：从见证 sidecar 取每根 key 工人（pos、cargo）与全部敌方
COLLECTOR（pos、cargo），以手算在 15×15 环面重算 DIST
（pos→(r=pos//15, c=pos%15)；d＝min(|Δr|,15−|Δr|)＋min(|Δc|,15−|Δc|)），
再按触发清单逐字重算 1a／1b 谓词、stake_1a／stake_1b、子方向、stake，
与根单记录逐项比对。下表只列起决定作用的敌方（DIST≤2 者）与结论；
其余敌方均 DIST>2（已逐一手算核对，略）。

**种子 986200（6 根）：**

| 根 | key 工人 | 决定性敌方（手算 DIST） | 手算重算 | 根单记录 | 判 |
|---|---|---|---|---|---|
| 986200-0-45-0-player-0-worker-12 | pos 91 (r6,c1)，cargo 30.0 | pos 105 (r7,c0)：Δr1,Δc1→d=2，cargo 0≤30 → 1a | 1a=30.0，1b=0，sub 1a，stake 30.0 | 同左 | 合 |
| 986200-0-79-1-player-1-worker-27 | pos 102 (r6,c12)，cargo 0 | pos 117 (r7,c12)：d=1，cargo 134.5278>0 且 ≥10 → 1b | 1a=0（cargo 0<10），1b=134.5278，sub 1b | 同左 | 合 |
| 986200-0-125-0-player-0-worker-33 | pos 164 (r10,c14)，cargo 239.67654 | pos 150 (r10,c0)：Δc14→1→d=1，cargo 239.67654＝key → 1a（等值计入） | 1a=239.67654，1b=0，sub 1a | 同左 | 合 |
| 986200-0-173-1-player-1-worker-68 | pos 135 (r9,c0)，cargo 30.0 | pos 149 (r9,c14)：Δc14→1→d=1，cargo 30＝key → 1a | 1a=30.0，1b=0，sub 1a | 同左 | 合 |
| 986200-0-225-0-player-0-worker-81 | pos 188 (r12,c8)，cargo 0 | pos 186 (r12,c6)：d=2，cargo 0（1b 须 >0 且 ≥10，不成立）；pos 173 (r11,c8)：d=1，cargo 75.177681 → 1b | 1a=0，1b=75.177681，sub 1b | 同左 | 合 |
| 986200-0-274-1-player-1-worker-109 | pos 151 (r10,c1)，cargo 0 | pos 164 (r10,c14)：Δc13→2→d=2，cargo 158.9961 → 1b | 1a=0，1b=158.9961，sub 1b | 同左 | 合 |

**种子 986201（6 根）：**

| 根 | key 工人 | 决定性敌方（手算 DIST） | 手算重算 | 根单记录 | 判 |
|---|---|---|---|---|---|
| 986201-1-45-1-player-1-worker-6 | pos 103 (r6,c13)，cargo 0 | pos 133 (r8,c13)：d=2，cargo 124.815 → 1b | 1b=124.815，sub 1b | 同左 | 合 |
| 986201-1-77-0-player-0-worker-19 | pos 145 (r9,c10)，cargo 0 | pos 144 (r9,c9)：d=1，cargo 88.6158 → 1b | 1b=88.6158，sub 1b | 同左 | 合 |
| 986201-1-126-0-player-0-worker-31 | pos 162 (r10,c12)，cargo 0 | pos 178 (r11,c13)：d=2，cargo 68.56416 → 1b | 1b=68.56416，sub 1b | 同左 | 合 |
| 986201-1-173-1-player-1-worker-47 | pos 57 (r3,c12)，cargo 0 | pos 42 (r2,c12)：d=1，cargo 42.2127 → 1b | 1b=42.2127，sub 1b | 同左 | 合 |
| 986201-1-215-1-player-1-worker-51 | pos 15 (r1,c0)，cargo 75.8565 | pos 16 (r1,c1)：d=1，cargo 0≤75.8565 → 1a；pos 29 (r1,c14)：Δc14→1→d=1，cargo 75.8565＝key → 1a | 1a=75.8565，1b=0，sub 1a | 同左 | 合 |
| 986201-1-270-1-player-1-worker-27 | pos 45 (r3,c0)，cargo 116.9535 | pos 59 (r3,c14)：Δc14→1→d=1，cargo 116.9535＝key → 1a | 1a=116.9535，1b=0，sub 1a | 同左 | 合 |

**种子 986202（6 根）：**

| 根 | key 工人 | 决定性敌方（手算 DIST） | 手算重算 | 根单记录 | 判 |
|---|---|---|---|---|---|
| 986202-2-43-0-player-0-worker-8 | pos 223 (r14,c13)，cargo 0 | pos 14 (r0,c14)：Δr14→1,Δc1→d=2，cargo 0（不成立）；pos 209 (r13,c14)：d=2，cargo 0（不成立）；pos 222 (r14,c12)：d=1，cargo 27.8106 → 1b | 1b=27.8106，sub 1b | 同左 | 合 |
| 986202-2-76-0-player-0-worker-6 | pos 139 (r9,c4)，cargo 70.6545 | pos 124 (r8,c4)：d=1，cargo 0≤70.6545 → 1a | 1a=70.6545，1b=0，sub 1a | 同左 | 合 |
| 986202-2-126-1-player-1-worker-35 | pos 74 (r4,c14)，cargo 30.0 | pos 60 (r4,c0)：Δc14→1→d=1，cargo 30＝key → 1a | 1a=30.0，1b=0，sub 1a | 同左 | 合 |
| 986202-2-173-0-player-0-worker-16 | pos 224 (r14,c14)，cargo 30.0 | pos 210 (r14,c0)：Δc14→1→d=1，cargo 30＝key → 1a | 1a=30.0，1b=0，sub 1a | 同左 | 合 |
| 986202-2-225-0-player-0-worker-2 | pos 192 (r12,c12)，cargo 0 | pos 207 (r13,c12)：d=1，cargo 103.7349 → 1b | 1b=103.7349，sub 1b | 同左 | 合 |
| 986202-2-272-0-player-0-worker-65 | pos 182 (r12,c2)，cargo 81.0 | pos 196 (r13,c1)：d=2，cargo 24.66072≤81 → 1a | 1a=81.0，1b=0，sub 1a | 同左 | 合 |

**人工抽查结论：18/18 全合。**每种子 6 根（要求 ≥5 根），覆盖早／中／
晚步带、双席、双子方向。

### 3.3 全局完整性核对（试跑输出逐项复算）

| 核对项 | 结果 |
|---|---|
| `root_id` 唯一 | 3920/3920 唯一，合 |
| 去重键（seed, game, step, seat, key_uid）唯一 | 3920/3920 唯一，合 |
| `root_id` 格式 ＝ `<seed>-<game>-<step>-<seat>-<key_uid>` | 抽查 18 根全合；全局唯一性已验 |
| `cand_count` ＝ 见证 `legal_candidates` 长度 | 3920/3920 合 |
| `cand_count` ＝ 4×（该席步 key 数）＋1 | 0 违例（值域 {5, 9, 13}）——即试跑中**守卫零否决**（任一否决都会打破 4k+1 型） |
| `stake` ＝ max(stake_1a, stake_1b)；`sub_direction` ＝ 1a ⟺ stake_1a ≥ stake_1b | 3920/3920 合，0 错配 |
| key 工人 kind 全 COLLECTOR | 3920/3920 合 |
| `trigger_class` 全 "class1" | 3920/3920 合 |
| `settlement_decidable` ⟺ step＝298 | 10 根可辨全在 298；298 步根 10 根全可辨；非 298 步根 3910 根全不可辨（kind/value 全 null） |
| 同席步多根的结算值一致性 | 1358 个多根席步组，0 不一致（同一席步三根共享同一根对象与候选表，值自洽） |
| 根单 schema（协议 §2.1） | 15 字段齐全：root_id／game／seed／step／seat／trigger_class／sub_direction／key_uid／stake／cand_count／settlement_decidable／settlement_argmax／in_sample／pi_r_given_g／enum_provenance；`in_sample`／`pi_r_given_g` 为 null（冻结时填，D6）；`pi_g`／`sampled` 属局总体册字段（§3.2），不在根单——审查初查误以为缺失，已纠正，脚本无误 |
| 局总体册 schema（§3.2） | game／seed／root_count／root_count_by_seat／pi_g／sampled／zero_valid_roots 齐全（冻结时填 null） |
| 枚举报告 N／M | N＝3、M＝0、N＋M＝3，合 |

### 3.4 结算独立验证（探针 `probe_settle_298.py`，非冻结件）

背景：试跑 10 根可辨根的结算值多为 0.0（局 0 六根全 0.0），须排除
「结算代码恒返 0」的静默 bug。验证法：重放局 0（seed 986200）至
step 298（agent-only，不枚举），对每席的合法候选同时跑 (a) 冻结代码
`Enumerator._settle_one` 与 (b) 独立一步推进（全新 FastCarbon sim，
`load_observation(root.full)`，`_tid_counter` 同式 +500，
`step(候选，对手实际动作)`，直读现金差），比对两者并打印绝对现金。

结果：**PASS，8/8 一致**。
- 重放确定性：step-298 key 集与试跑根单逐字一致（双席均为
  worker-116／worker-20／worker-96）。
- 冻结值与独立值在 8 个候选上全等（均为 `(time, 0.0)`）。
- 绝对现金非零且真实：一步前 `[4027.98, 4027.98]`，一步后
  `[4192.85, 4192.85]`——0.0 是镜像平局的真实分差，不是恒返 0。
- 试跑数据本身亦证明结算有区分力：局 1 可辨根中一候选结算差
  **−11.5**（一步少存 11.5 现金），argmax 仍为 A0（0.0 为最大）。
  10 根可辨根的结算差向量：局 0 六根全 0.0；局 1 四根为
  `[0.0, 0.0, 0.0, −11.5, 0.0, 0.0, 0.0, 0.0, 0.0]`（9 候选）。
- 探针覆盖范围说明：探针取冻结函数 `_gen_candidates_raw` 的
  4 候选／席（一 key 的候选集）；试跑根单的 13 候选／席由脚本复制
  管线 `_full_candidates` 产出，其与冻结函数的截断一致性已在试跑中
  由每席步交叉核对验证（3920 根、0 停）；`_settle_one` 与候选无关，
  故 8 候选的验证覆盖该结算代码路径。

附记：verify3 附件中同一种子 986200 的 A0 臂终局分 `[5652.48, 5652.48]`
与本试跑活局终局现金 `[4192.85, 4192.85]` 数值不同——verify3 延续腿
是快照恢复后的重放续弈，其轨迹与活局自对弈在根步之后可以分叉，
终局现金不同不矛盾；结算字段定义在活局一步推进上，探针已独立验证，
此点记为观察、不构成对枚举的质疑。

**门 3 判读：通过。**

## 4. 协议逐条对照表

| 协议条 | 要求 | 脚本／试跑落实 | 判 |
|---|---|---|---|
| §1.1 对局形态 | 诊断块 986200–986499，300 局，基线包自对弈，双席，打满至权威终局，agent 局内持久／跨局重建 | 脚本 `--games A-B` 按块内局号切分，`SEED_BASE=986200`；试跑 0–2 局全打满 299；agent 局内持久（同一 agent 对象）、跨局重建（`make_fresh_agent`） | 合 |
| §1.1 不做比较器决策／不算 ΔU | 枚举期不做比较器决策、不算 ΔU | 脚本无 comparator 导入、无 ΔU 计算；grep 可验 | 合 |
| §1.2 触发筛查 | 每步生产 cfg 原样复算 key_units，全步带 0–298 | 见 §1；试跑 3920 根、0 停 | 合 |
| §1.3 子方向复核 | 每 key 重算 1a／1b 登记子方向；两谓词均不成立即 §1.4 | `independent_predicates` 逐根复算；不成立即抛错停（试跑 0 触发） | 合 |
| §1.4 类二／类三出现即停 | 自称类二／类三的根或复核不成立即停，不跳过、不改记 | 脚本无类二／类三分支；`trigger_class` 写死 "class1"；复核不等即停；试跑触发事件「无」 | 合 |
| §1.5 步域 | 根步 0–298，不设 step≥240 线 | 脚本无步数下限；试跑最早根 step 15，最晚 298 | 合 |
| §2.1 根记录 schema | 15 字段全量 | 见 §3.3，齐全；`in_sample`／`pi_r_given_g` 冻结时填（null） | 合 |
| §2.2 去重键写死 | `root_id = <seed>-<game>-<step>-<seat>-<key_uid>` | 脚本写死该格式；全局唯一已验 | 合 |
| §2.3 枚举期不做判读 | 不算 ΔU、不比候选、不排优劣；stake 只作审计 | 脚本不做判读；stake 仅记录 | 合 |
| §3.1／§3.2 局总体册 | 产根局入册，含零根局计数 | `register_<shard>.json` 按 §3.2；试跑 N=3、M=0；零根局计数在报告中 | 合 |
| §3.3 零根局报告义务 | 报告写明 N／M（N＋M＝300） | 报告含 producing_games_N／zero_root_games_M／N_plus_M_eq_games | 合 |
| §4 两阶段抽样 | 概率冻结时填（默认普查） | D6：枚举期 null，冻结时填；脚本不预设 B／C | 合 |
| §5 冻结形态 | 三件开测前落盘钉版、开测后只读 | 脚本产出根单／局总体册／报告三件（＋见证 sidecar 非冻结件）；`--merge` 合并验合 | 合（冻结入档时执行） |
| §6 结算可辨性下限 | 枚举时算出 settlement_decidable／argmax 随根单冻结；函数指针记入冻结记录 | 见 §2；试跑 10 根可辨、argmax 全 0；指针在 meta 中 | 合 |
| §7.3 镜像根计数 | 同一步双席两根分别计，不删其一 | 试跑双席根数完全对称（1960／1960），分别记录 | 合 |
| §9 测试执行口径（引用） | 叶①、J=1 等注册口径 | 枚举不涉测试；D4 的 J=1 结算对手动作与 §9 一致（待裁，§8） | 合 |
| §10.2 执行口径 | scope／BLAS／agent 持久／基线包／生产 CONF；0.7 s 预算枚举期不绑定 | 试跑口径差别已在 §0 写明；考位执行时按 §10.2 全套 | 合（试跑口径） |
| §10.2 分片 | 每机独立输出，合并验并集恰＝986200–986499 | `--games` 切分＋`--merge` 验合（并集／不交叠／不缺／不重／去重键唯一）；试跑单分片，合并逻辑未在多分片下实测（见 §8 备注） | 合（单分片；多分片合并待全量时验） |
| §10.3 脚本要求 | 基线自对弈／key_units 复算／子方向复核／§2.1 全量／§6 结算／分机输出与合并 | 全部实现；脚本 md5 入档 | 合 |
| §10.4 门 1／2／3 | 见 §1／§2／§3 | 三门通过 | 合 |
| §10.5 回执 | 开跑三件套先发；完赛回执要素；失败单独披露 | 脚本产出报告含完赛要素；三件套属执行方回执程序（考位执行时执行） | 合（脚本侧） |

## 5. 试跑中发现的两处脚本级事实（非缺陷，记入审查）

1. **D1 并列分支在现行 CONF 下不可达**：PRIO_THREAT＝1.0 时，
   stake_1a＝key cargo；而 stake_1b（若非零）＝某敌 cargo＞key cargo，
   故两向同非零时恒有 stake_1b＞stake_1a。试跑 3920 根中真并列
   （两向均＞0 且相等）0 例；296 根两向同非零者子方向全为 1b。
   D1 的「并列取 1a」在现行 CONF 下永不触发，写为防御性口径无害。
2. **见证 sidecar 只在分片末写盘**：`witness_<shard>.jsonl` 在
   `_write_shard_files`（分片末）一次性写出；若分片中途崩溃，已完
   局的见证丢失（根单 JSONL 逐局落盘不受影响）。见证非冻结件，
   可凭确定性重跑再生；是否改为逐局落盘见 §8 第 5 项。

## 6. 全量枚举规模预估（试跑实测外推，估计）

- 根量：试跑均值约 1307 根／局 → 300 局约 **39 万根（估计）**。
  普查测试不可行，B／C 两阶段抽样（协议 §4）为必需，冻结时须定数。
- 墙钟：试跑均值约 747 s／局 → 单机 300 局约 **62 小时（估计）**；
  须按 §10.2 分片并行（分片数由作战室定）。
- 文件：根单约 1151 B／根 → 全量约 **450 MB（估计）**；见证约
  6.6 MB／局 → 全量约 **2 GB（估计）**（见证为 sidecar，非冻结件；
  分片传输与合并存储须预作安排）。
- 以上均为试跑外推的估计，非承诺；实际以全量执行回执为准。

## 7. 审查程序性说明

- 审查期间发现并纠正的两处审查侧失误（非脚本问题）：(a) 初查
  误以为根单缺 `pi_g`／`sampled` 字段——实为 §3.2 局总体册字段，
  根单无此要求，脚本无误；(b) 等待试跑时一次 `pgrep` 读到
  "TRIAL-RUNNING" 为自匹配假阳性（检查命令本身含被查模式），
  后以 PID 9023 直查确认为已结束——同类 pgrep 自匹配教训已在
  AGENTS.md 有案，此处再记一笔。
- 试跑中修复的两处脚本 bug（冒烟期发现、试跑前已修、试跑用已修
  版 md5 1218d573）：(1) `independent_key_set` 对 PLANTER 工人
  调 `independent_predicates` 会抛错——改为跳过非 COLLECTOR，
  与 key_units 基础段一致；(2) `from smoke_real_s1 import fetch_pack`
  会触发其模块级 argv 解析——改为内联同序取包（--pack → /tmp
  缓存 → checkout → gh_api blob）。另补 stale-HALT 清理（分片开跑
  即清旧 HALT 文件）。

## 8. 未决事项（需父代理裁定）

1. **D1（子方向并列 → "1a"）**：建议接受。现行 CONF 下不可达
   （见 §5 第 1 条），防御性口径无害；若日后 PRIO_THREAT 改动，
   须重审。
2. **D2（"合法候选"＝合法性＋去重后、守卫准入后、K≤4 截断前；
   `cand_count` 为该集合大小；守卫否决的候选不入结算）**：建议接受。
   理由：被守卫否决的候选在测试时永不可入选，若计入 §6 的「全部
   合法候选」，结算 argmax 可能落在永不可入选的候选上，使 §6 下限
   在构造上不可满足。试跑中零否决（cand_count 全为 4k+1），该子句
   在试跑数据上无可观测效应，属定义性裁定。
3. **D3（`settlement_argmax` 下标空间＝合法候选表、生成序、A0=0；
   并列取先验序靠前者，A0（差 0）优先于一切非正差候选）**：
   建议接受。试跑中 A0 优先规则已行使（10 根可辨根全 argmax 0，
   其中局 0 六根全差 0.0）；非 A0 候选间的并列次序在试跑中未被
   行使（无两非 A0 候选并列最大正差），该子项为未行使口径。
4. **D4（结算一步的对手动作＝该根步实际执行的对手动作，J=1
   确定性延续，不用 agent 预测的 acts_o）**：建议接受。与 S3
   注册口径（§9：对手形态 J=1）及 verify3 延续腿口径一致。
5. **见证逐局落盘是否改**（§5 第 2 条）：二选一——(a) 维持现状
   （见证非冻结件，崩溃可重跑再生）；(b) 改脚本为逐局写见证后
   重试跑。选 (b) 则脚本 md5 变更，按 §10.3「脚本版本变更＝
   重新枚举」，试跑 3 局须以新脚本重跑（约 40 分钟）再行审查。
   建议 (a)——见证只是审计 sidecar，根单 JSONL 已逐局落盘，
   崩溃损失可控；但请父代理定夺。
6. **多分片 `--merge` 未在多分片下实测**：合并逻辑（并集验合、
   去重键唯一、局册不重复）已实现，试跑为单分片，未覆盖多分
   片合并路径。全量执行时首次合并即为实测；建议在冻结记录中
   写明「合并验合结果随冻结包同报」，不另设试跑。
7. **协议引文行号 off-by-one**（§1）：协议 §1.2 引 sort :404／
   截断 :463，实测 :405／:462。实质已核对无误，不影响脚本；
   请父代理决定是否在冻结记录中备注（不改协议本文）。

---

审查结论：**§10.4 三门通过**。脚本 `enumerate_roots.py`
（md5 `1218d573e81f6f9b614a61597968ea45`）可用于全量 300 局枚举，
前提是父代理裁定 §8 第 1–4 项（D1–D4）与第 5 项（见证落盘方式）。
全量执行仍须作战室派单（C1／C2＋Studio 分片），开跑三件套与完赛
回执按 §10.5 执行；冻结包（根单＋局总体册＋冻结记录，B／C 由
策略组定）报策略组确认冻结成立后才许开测。
