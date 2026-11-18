# R2 修复复测报告：冻结 SHA ＋ 同套实例复测

搜索组 2026-11-18。对象：Codex R2 复核（fc317156）六项派修 R2-01/02/03/05/06/07。
修复已于 2026-11-15 晚分两组落 main（组1 e501ee26/e123e816/d35d9962；组2 2cb48bbb/
8ba0ad84/6f933f83/024c4657；pack_bridge 另 bbf7dea4 钉版 blob 寻址）。复核报告正文
基准为更早的 b15b475d。本报告＝派单要求的「冻结 SHA 后在同一套双席＋有历史＋
有灭绝实例上复测」的回执：冻结对象为当前 main 版本，复测在本 VM 以组装 checkout
树执行（文件与 main blob 逐字一致，md5 见下）。

## 1. 冻结清单（md5 全值，2026-11-18 读回）

| 文件 | md5 |
|---|---|
| code/search_proto/real_env.py | f9bfacb72e112859459a57aceebbbf46 |
| code/search_proto/s1_verify3.py | 17310cc1dfc2f32519bef6521f38048b |
| code/search_proto/s1_battery2.py | fc9d35572a55d006b84a872287d494c1 |
| code/search_proto/s1_battery.py | 6a78101a5a1d4403f1f89a053c1c0453 |
| code/search_proto/smoke_real_s1.py | 2425f34066c1b2b56208bdd4a2b7ac10 |
| code/search_proto/pack_bridge.py | ad522153d32b7cbc5008ef0b59638c0f |
| code/search_proto/comparator.py（未动，钉版） | 53e5c1a1dd6ccc1a1f3343a47fc9b589 |
| code/search_proto/leaf_adapters.py（未动） | a5852c73f7187bb3e179428a81a6fba9 |
| code/search_proto/telemetry.py（未动） | 91a7121f02bc848045a5dcae9554ec0e |
| code/search_proto/tests_r2_fix.py | 30285847e8c1dc28a017650bd80153a8 |
| code/search_proto/tests_s1_fix_r2.py | 5206e4c3199dcbb3c2688f3a68d5bc22 |
| code/packs/echo4_int8.py | 819250bfc82632786063dabd46a616bd |
| vendor 钉版（search/feat/np/fastsim） | 94f7fb64 / 2a45b08b / d8c15ad4 / aefe6c83 |

## 2. 回归结果（自有回归，两文件均 exit 0）

- **tests_r2_fix.py：17/17 PASS**（R2-01 行为反例与归属矩阵、R2-02 权威逐席结算
  含 Codex 控制例、R2-03 在线/离线全等＋别名重绑、§5 小项三件）。
- **tests_s1_fix_r2.py：10/10 PASS**（R2-05 G1–G5、R2-04 残余 G6–G8、R2-07 G9、
  R2-06 G10）。

## 3. 复核方探针复测（证据包 REVIEW_SEARCH_IMPLEMENTATION_R2_PROBES_20261116）

探针均按「断言缺陷存在」写成；修复后缺陷断言翻转且数值落到正确值，才算复测通过。

| 探针 | 项 | 复测结果 |
|---|---|---|
| real_env/probe_r2（environment） | R2-01 | **翻正**：双席 actual 与 reference 逐项全等；行为反例叶值 −117 → **+209**（seat1 镜像 −209）；两席 guard lostw 均 0、prevw 各归本席；终局根拒绝、H1–H3 均止于 step 299 |
| verification/probe_extinction | R2-02 | **翻正**：valid=True，actual_delta **−56.0** ＝权威期望 −56.0（chosen −102 / A0 −46）。注：探针按旧签名调用 verify_root，固定版要求显式传 score_fn（生产主流程 603 行即如此传），复测副本按新口径补传 make_terminal_score_fn 后通过；无 score_fn 时固定版 fail-closed 报错停，不回退旧公式——此行为本身符合修复意图 |
| verification/probe_offline_memory | R2-03 | **翻正**：离线/在线 hist_lengths 全等、last_step 与 est 差异格 **0**、max 差 0.0；restore 后别名 before/after/propagates ＝ True/True/True |
| acceptance/probe_gate | R2-05 | **翻正**：带 B 腿 overrun 的「正例」A2/A3/A8 现 FAIL；NaN 结算误差 A5 现 FAIL；缺正典用例合成件 A2/A3/A4/A5/A7 现 FAIL；fixture 身份随 step/seat 改变而变（P3） |
| bridge/probe_bridge | R2-07/06 | **翻正**：clean_np_feat_same_object=true、stale_module_survives_in_np=false、np 侧 LOGC 6.5＝已验 feat 6.5、cargo100 特征 0.7100185 ≈ 钉版 0.71001854（旧缺陷值 0.355）；缓存哨兵被拒、smoke 取包成功 |

## 4. 附带发现（单独上报，不在本批处置）：successor 落地后钉版表漂移

当前 main 的 code/cnet2_search.py md5 为 **c3222eea**（cb5a5734，2026-11-15 23:46，
「land successor authority file per REGISTRATION_SUCCESSOR_AUTHORITATIVE_2」），而
pack_bridge 钉版表仍钉 **94f7fb64**（b15b475d 时该文件正是此值）。后果：
bootstrap_offline 对当前 checkout 在 cnet2_search 一件上 fail-close（其余三件全合）。
本复测的 bridge 探针因此按复核基准环境补入钉版 blob 后执行。94f7fb64 是生产身份
钉，改钉与否、以及 successor 与 S1 vendor 集的关系，属策略组/该注册所有者裁量，
本线不单方改钉；但在裁量前，「完整 checkout 离线 bootstrap 可跑通」这一 R2-06
结论只在钉版文件在场时成立，特此标注。

## 5. 未完项

- §5 第 2 步后半「再跑一次可核验的 S1」（正式 battery2 全量复跑：采集＋新判读器
  出正式裁定）尚未执行——旧正式工件经新判读器复算已正确 FAIL（G1），新一轮正式
  跑需占正式考场座位与墙钟，请作战室排座后本线执行。
- R2-04 的公开入口全链 deadline 属 §5 第 3 步（S2）范围；R2-08 属策略组裁量；
  R2-09/10 归工程组——均不在本批。
