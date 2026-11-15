# Echo4 搜索实现与策略更新：完整复核

审查日期：2026-11-16 CST。审查者：Codex，6 条并行审查线＋主审交叉复核。

代码主基线：`eeb1b4b93374a0aba2c61b8d2bf8e867f2c3b58c`；上次方案复核后的增量起点：`cbf4c4db565a593e294664141f2e6e1902521b63`。审查期间追加复核至 `21098bad35e5cd8ac6104b3eef8ad20edb56c043`，包括 `s1_addendum.py`、补测结果、切换计划及注册修订。后一个版本没有修改 comparator/real_env 的实现。

本次只读检查、轻量复现与报告入仓。没有修改业务代码、部署、触碰在跑训练或发起正式比赛。机器上的 vnet33、C1 资源认证原始工件未直读；对这些工件只确认仓内记录，不代替现场认证。

## 1. 结论

**原型确实接通了真实模型、候选比较与叶调用，但目前不能通过正式 S1，更不能按现有接线直接进入生产切换。** 两个最优先的问题是时间终局漏判、总预算漏计。另有正式电池误报 PASS、续弈不带生产守卫、首次 seat1 状态错配、数值失效未拒绝等可操作问题。

这是对新实现的判定，不是对搜索路线的否定。当前工作仍是影子比较：自然对局执行原 A0，比较器只记录建议。完整 critic 尚未接入；`VNetLeafAdapter` 是未注入 backend 的骨架。

上轮方案问题已实质吸收：**MS01、MS02、MS04–MS08 文档层通过；MS03 主条通过，仅覆盖表残留一句冲突文字。** 不再把这些修复列为“仍未修复”。新的 S4 固定 N、同口径整局 pilot、分离种子块，以及改变 S1 正例口径须先修正案，均予认可。

本轮合并去重后的清单为 **2 项 P1、9 项 P2**。P1 指在扩大实验／进入下一阶段前必须修的正确性问题；P2 包括代码缺口和明确的执行前置缺口，不意味着它们都已造成线上事故。

| ID | 级别 | 问题 | 性质／影响范围 |
|---|---|---|---|
| SI01 | P1 | 时间终局未冻结，测试把 step299→300 当有效结算 | 实包复现；H≥2 晚局评价改变，终局根未拒绝 |
| SI02 | P1 | 根构造／候选生成漏出总预算，耗尽后仍补 50ms | 确定性时钟复现；不能据当前遥测认证平台计时 |
| SI03 | P2 | “正式 S1 电池”正例失败仍 PASS | 原脚本实跑；最新 addendum 未冒称通过，须分别记账 |
| SI04 | P2 | H≥2 续弈未执行生产 BankruptcyGuard | 源码可证；现有有限样本未证实动作／收益翻转 |
| SI05 | P2 | fresh seat1 的对手快照席位错误 | 实包复现，错误方动作 ID 被对手执行路径忽略 |
| SI06 | P2 | 非有限叶值仍可提交，异常无有效决策记录 | 独立注入复现；不是声称自然样本已有 NaN |
| SI07 | P2 | 续弈验证入口与目标策略不一致 | 原晚局入口不可达；新补测仍重置历史，终局错误沿用 |
| SI08 | P2 | 仓内说明不足以复建钉版验收环境 | 四源桥只拉一源、本机 helper 依赖、缓存不复核 |
| SI09 | P2 | it35 判读分支同时给出回退与继续 | 注册条款可满足性反例，不是实际 it35 数据 |
| SI10 | P2 | ckpt30 完整恢复前置未建立 | 实际轮转函数复现；未断言远端不存在未记载备份 |
| SI11 | P2 | 新切换计划漏掉权威源／嵌入模块／NO_ROLL 接线 | 构建链静态核验，尚无已交付派生包失败之说 |

## 2. 代码与验收发现

### SI01：时间终局漏判及“精确终局”用例错位

定位：[`real_env.py:452`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/search_proto/real_env.py#L452)、`:465–483`；所借 `_terminal` 在 [`cnet2_search.py:660`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/cnet2_search.py#L660) 只判灭绝。权威口径见 `eval_v53.py:165` 与已修教师 `stage0_teacher.py:293`：最后有效动作观察为 step298，终态为 step299。

实包同根反例：root298，H1 到299，现金 `[114,100]`、叶值14；H2 却到300，现金 `[114,40]`、叶值74；H3 到301。额外一拍的对手招募改变现金，绝非只差一个无害的计数。root299、root300 也仍继续推进。

原电池 `s1_battery.py:167` 恰把根造在299，再比较推进到300后的现金，因此“leaf 与现金差误差0”只证明两个计算共享同一个越界时间轴，不能证明与权威终局一致。最新 `s1_addendum.py:264–286` 同样以 `<300` 循环；独立执行其未改控制流，从298推进到299后仍调用双方策略，再到300才标 complete。

修复要求：入口拒绝已终局根；每次联合推进后同时判灭绝和时间边界，终态冻结且零额外策略调用。结算沿 F01/F04 保留进入灭绝步前现金；正常到时使用真实终态现金。以权威规则重做297/298/299/300、双席、灭绝与到时重合案例。不能只改测试名字或把 left 截为0。

证据：`real_env/probe_real_env.py`、`real_env/results.json`；`comparator/probe_addendum_terminal.py`。

### SI02：公开调用预算没有贯穿整条链

定位：[`real_env.py:499`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/search_proto/real_env.py#L499) 先做 `_build_root` 和 `_generate_candidates`，`:505` 只扣 baseline，`:506–510` 把不足的余额补至0.05秒；比较器随后才重新起表。`:524` 记录的也只是比较器耗时。

确定性时钟探针：基线0.2秒＋根/候选0.9秒＋比较0.002秒，总计1.102秒，相对0.7秒预算已经超支，却提交新动作、`abort_reason=null`，遥测只报0.002秒。另一例基线已耗0.8秒仍分配0.05秒并改子。以上是控制流反例，**不是本机或平台吞吐测量**。

应由公开入口产生唯一绝对 deadline，覆盖快照、基线、对手推断、候选解码、guard、模拟、叶和返回。余额不足直接 A0，删掉补余额逻辑；记录总耗时及分阶段耗时。C_hi 整轮准入尚未实现，事后发现一次不可取消调用超时并返回 A0，不能追回已经消耗的时间。

因此目前“每根约1.3ms”只说明比较器内核那部分开销，不是完整搜索成本，也不能替代 S2。证据：`comparator/probe_comparator.py`、`results.json`。

### SI03：正式电池的可执行判据没有执行正式合同

定位：[`s1_battery.py:514`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/search_proto/s1_battery.py#L514)：唯一 PASS 条件是 `len(completed)>0`。B 的 `expect_met`、是否选中精确最优、叶与结算误差、异常、最低根数和局数均不参与裁决。

独立实跑原代码，只把自然采集缩为1根／1局，B例不改：P1、P2均 `expect_met=false`，N1为true，脚本照样 `BATTERY: PASS`、返回0。仓内48根结果中的三个正例也都未达。自然根 changed 只能证明有动作改变，不能替代“已知更优动作被发现”的正控制。

必须区分诊断完成与正式 PASS；正式模式应把全部注册断言纳入退出状态，并绑定源码、包、配置、fixture身份与 run_id。缺正例或边界失败不能报正式通过。

**认可现有诚实口径：**状态文档本来就写 S1 正例未达，并未宣称正式关账。新 addendum 的退出0明示只表示脚本跑完；48次补测 `success=null`，没有宣称正例通过。本条批评针对原电池的可执行闸，不把 addendum 的诊断退出码误读为造假。

证据：`acceptance/probe_acceptance.py`、`probe_result.json`、`battery_reduced.log/json`。

### SI04：深度续弈绕过生产守卫

定位：[`real_env.py:469`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/search_proto/real_env.py#L469) 到477：`act_b` 后直接 `to_actions→step`，没有两席独立 BankruptcyGuard，也不复制／演化 lostw、prevw、guard_base。根候选的 guard.copy() 否决不能代替后续步骤守卫。

这会把 H≥2 估值的续弈策略变成无守卫策略，与“生产策略续弈”的合同不同。H1 现有自然根比较没有后续策略调用，不能据此宣称此缺口已经污染全部48根。

补齐独立守卫历史，并验证连续失员、低现金危险动作、招募保护与双席状态互不串扰。有限实包对照中没有找到动作翻转，报告只主张源码合同缺失，不虚构负收益。证据：`real_env/findings.md`、`probe_guard.py`、`guard_results.json`。

### SI05：首次 seat1 观察会预测错方动作

定位：[`real_env.py:300`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/search_proto/real_env.py#L300) 到304：尚无 f/fo 时默认己方为0，创建 `Feat(1)`。实际己方为1时，对手应是0；`Feat.observe` 使用自身 seat，不会被 obs.player 自动纠正。

实包 fresh step100 根，`root.acts_o` 全为 `player-1-*`，传给 player0 后不起应有效果。相同根和 A0：错误分支对手现金100、零工人；正确 seat0 预测分支现金114、两工人。

应从当前 observation 明确席位初始化快照，并处理新局／换席。持续同席自然对局随后可自动回到正确 fo，且初始step0通常无战术候选；**本条不声称每个 seat1 自然根都错**，但 fresh晚局验收明确可达。

证据：`integration/probe_fresh_seat1.py`、`fresh_seat1_results.json`。

### SI06：数值与运行时失败不能作为完整比较提交

定位：[`comparator.py:132`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/search_proto/comparator.py#L132) 到161；`leaf_adapters.py:45–51`。

注入 `[0.1,+inf]` 会选中替代动作并记录完整比较；`[NaN,0.9]` 会把没有有效基线分数的比较记录为完成。启发式适配器把 NaN/Inf 当普通超域返回，未拦截。rollout/leaf 异常直接逃逸，`record_decision` 又在成功返回之后，故无有效失败行。

有限性应独立于尺度检查：已声明的原始分差可超出[0,1]，但不可为非有限值。任一候选坏值使整轮无效，保留 A0／上一完整轮，记录失败原因及已耗成本。诊断模式可以记录后报错，生产入口需要安全回退；调用合同本身错误可以继续 fail fast。

L-vnet33 尚未接入，不能声称已有 critic数值事故；未来 backend 应在 tanh前检查有限性，避免 Inf 被压成合法端点1。证据：`comparator/results.json`、`leaf_bridge/probe_leaf_bridge.json`。

### SI07：续弈验证的采集、记忆和完成性需统一

原电池 `MAX_STEPS_PER_GAME=200`，根最多199，却要求 `VERIFY_MIN_STEP=220`，晚局验证集合恒空。不是换种子多跑一次就会出现晚局根。

原 `verify_changed_root` 在墙钟中断后仍以 `final_diff_seat` 计算收益。注入时钟反例：两臂都只走到251，却报告“终局差值”44。**新 addendum 已修此项：其 incomplete 臂正确让 delta=null，予以认可，不再向新版重复报这个错误。**

但两版都从保存模拟器状态后新建双方 agent，没有恢复 Feat 信念／历史、守卫和对手模型记忆。两臂同样冷启动只保证两边待遇相同，不能证明被测策略就是持续运行的生产续弈策略。两版还沿用错误的 step300 终止（合并在 SI01）。

最新补测：48个构造无正例；扫描86根、40次changed，只到197，step≥220根为0；唯一step150验证，A0到300，chosen到255中断，收益null。它补上了显式未完成标记，但没有提供 S1 正例或晚局有效收益证据。

后续必须分配可到达的晚局采集配额、保存并恢复每席策略记忆、按权威终局和 U 计算结果。若刻意测试冷启动策略，需另命名口径，不能升级成生产收益。证据：`acceptance/findings.md`、`truncated_verification.json`、`comparator/addendum_notes.md`。

### SI08：验收环境不能仅凭入仓说明复建

README/状态表称 pack_bridge 拉取四源，实际 [`pack_bridge.py:40`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/code/search_proto/pack_bridge.py#L40) 只拉 search 一件；另外三件无准备路径。干净 checkout 又依赖 `~/workspace/gh_api`，仓内 helper 位于 tools 且依赖本机运行时。

离线探针仅替换远程传输、提供精确钉版 search 字节：实际只生成一个文件，加载因缺 cnet2_feat 失败。已有 vendor 缓存时 loader 不复核 hash，替换为临时哨兵模块也会执行。

应提供基于冻结 checkout 的离线 bootstrap，准备并全量校验四源，加载缓存同样核验。当前 real_env 的独立四源校验和整包 hash 仍在，**不据此指控已跑实验用错源码**。新切换计划已声明桥退役，生产构建完成后应由构建与 certify 接管；原型复现说明仍需补齐。

证据：`leaf_bridge/findings.md`、`probe_leaf_bridge.py/json`、`bridge_cli.log`。

## 3. 策略与切换计划发现

### SI09：it35 单次判读仍能产生相反操作

[`PPOU2_FILING_20261114.md:395`](https://github.com/robertyang87/carbon-league/blob/eeb1b4b9/reports/PPOU2_FILING_20261114.md#L395) 要求点估<0.0760回退ckpt30；`:398` 又要求 CI跨比较线时按中间分支从ckpt35继续，不记回退，未规定优先级。

合法构型点估0.075、CI `[0.062,0.088]` 同时满足两条相反指令。点估0.090、CI `[0.077,0.103]` 又同时被标“无害”和“未判”。固定2000样本、取消扩测已经修复；此处是新的决策函数歧义。

在读 it35 结果之前写互斥、完备、有优先级的决策表，并明确“所比读数线”是哪条。示例为条款反例，不是实际考试读数。证据：`strategy/evidence.json`。

### SI10：注册的 ckpt30 完整恢复缺少已核验工件前置

同一注册要求 `resume ckpt_30 state`。但 `rl2_ppo_train.py:931–945` 只按轮数导出 actor NPZ，完整 net/vnet/optimizer/voptimizer 等进入两槽轮转；`rl2_checkpoint.py:106–110` 不创建持久 state30。

直接调用实际 `commit_checkpoint`、以有限小张量跑30→31→32→33：到32时已不再有完整30，最终只剩32/33。同期期仓内完整state盘点也只列0/32/33，后来冻结的是33。

**未直读L20/DGX，不断言没有其他备份。** 执行前须找到并钉 hash 的真实完整it30 state及恢复所需清单；若找不到，须明确修订回退操作。actor NPZ重启、拿最新critic拼接，均不等价于所注册的resume。以后作决策锚的检查点应在轮转前原子冻结完整state。

证据：`strategy/probe_strategy.py`、`evidence.json.checkpoint_rotation_actual_code`。

### SI11：切换计划段0尚缺三项构建接线

新增 [`SWITCH_PLAN_20261116.md:10`](https://github.com/robertyang87/carbon-league/blob/21098bad/reports/search_dossier/SWITCH_PLAN_20261116.md#L10) 准备改 `code/packs/cnet2_search.py` 并引入 `code/cnet2_comparator.py`。按当前打包器不能只完成这两处就得到所述制品：

1. `code/cnet2_search_pack.py:135` 读取 code/ 同级权威源，只改 packs 副本会被忽略；改用 packs 打包器则在`:305–309`被 N01 权威源一致性门拒绝。应明确权威代码和镜像更新顺序。
2. 两个搜索打包器的 MODS 仍固定四件，不含 cnet2_comparator。新增模块还须进入单文件嵌入、依赖顺序、hash与验包清单；普通 cnet2_pack 本身不打搜索入口。
3. `SEARCH=True` 不能沿用现有 `CNET2_NO_ROLL=1`，断言要求 SEARCH=False。不用该标志则默认额外打入s64权重。需要明确 ROLL=self 的新合法配置并重算／实测包体。

这些是**计划漏列的实现工作**，尚非已交付新包的实测失败。请纳入段0验收，防止出现“改好了源但实际包仍嵌旧源”或缺模块。证据：`leaf_bridge/switch_plan_integration_note.md`。

## 4. 已通过事项、未完成事项与低优先修订

### 已独立核验的正向事实

- 原有桩级用例7/7通过。第一轮完整比较才提交、平局优先A0、首轮中断回A0、已有根候选guard否决路径均有可运行实现。
- Echo4整包 MD5 `819250bfc82632786063dabd46a616bd` 合；独立重算四件内嵌源码hash一致。
- 直接构造基线与公开pack.agent，两局×双席×12步共48次观察，动作零差异。另一条含影子搜索的连续60步×双席共120次比较也零差异，52根参与比较、20次建议改子。这只认证有限功能对齐，不认证性能或搜索收益。
- 根候选复用真实 Prefix 解码，不是直接拼接标签；根模拟 full_carbon 取 f.est，未发现该路径直接偷读真实隐藏图。尚未完成所有隐藏信息扰动／记忆隔离边界认证。
- it20 vnet缺失已明确关闭，it33登记为新叶而非冒充同源恢复；S4锁N与扩测问题已修。

### 明确未完成，不冒充已交付回归

- 当前 Comparator 每次只评一个H，只有候选循环，没有H1→H2→H3逐轮加深及上一完整轮结果。README／状态表称“中断回上一完整轮”过强。H1原型可保留，但必须标未完成；需要刻意构造H1已有改子、H2中断仍返回H1的测试。
- C_hi整轮成本准入、完整公开入口计时、生产单文件构建、完整vnet backend与四项验收尚待完成。成本表需要含候选／对手网络与未完成工作，不能只用比较器内核计时。
- 三类战术覆盖尚未证全：默认 key_units 主要是载货碰撞／逃逸，WIRING_MAP 自己已注明返航／清算依赖默认关闭的KEY_LATE_HOME等变体。应按实际CONF提交分层触发正反例或缩窄覆盖，不能把含邻近敌人的交付构造当成全部返航覆盖。
- C1 S0a原始资源证书仍未入仓，当前报告把init归为exec内解压构造，但公开Echo4入口存在懒加载；需原始工件解释初次调用计时归属。未因此判定C1已超时或认证作假。

### 文档收尾（P3）

- MS03主条已修，最新注册覆盖表仍写“H=0只报分布、不判读”；与S3主尺含真实A0零收益的规则统一，避免消费者误读。不是把MS03整体重新打开。
- `SEARCH_LINE_STATUS_20261116.md` **文件本体**在末尾写着 `...[truncated 1107 chars]`，上一行“根名单冻结法、无”断句。这不是审查工具截断，请恢复原文并核字节。
- λ0.92规则中“两条门都未触发”不等于“≈λ0.95”；可保留不采取行动，结论改成相对差异未确认。
- 原始分差叶与统一U接口文案不一致。J=1、同叶纯argmax下单调变换不改排序，本轮不把原分差直接判为动作选择bug；在混叶／多情景期望／阈值判定之前必须重新统一量纲。
- 原电池结果没有完整run_id／源码／配置／fixture身份，P3例为事后追加。已披露不等于一趟可复现正式验收；下一轮应输出自描述且原子发布的证据包。

## 5. 下一轮最短闭环

1. 先修SI01/SI02及SI03，冻结SHA：权威终局、公开deadline、正式判据必须成为可执行硬门。
2. 同一冻结版补双席初始化、H≥2生产守卫、非有限值／异常回退、正确策略记忆续弈；重跑本报告反例与正负控制。
3. 明确首轮只验H1，或补完多轮anytime后一次验全。把构建链SI11、bootstrap及成本拆解列成可读回工件。
4. S1通过再拿最终派生包做S2；L-vnet33按登记独立验收。收益仍由S3/S4判断，不由changed率、叶调用数或毫秒数替代。
5. 策略组在it35读数前固定SI09决策顺序，并核实SI10恢复工件；两件可与搜索修复并行。

## 6. 工件与复跑

本报告同目录的 `REVIEW_SEARCH_IMPLEMENTATION_EVIDENCE_20261116.json` 记录版本、文件hash、检查范围及证据文件清单；`REVIEW_SEARCH_IMPLEMENTATION_PROBES_20261116.zip` 包含分线笔记、探针、输出与复跑说明。无需访问评审VM私人目录。

测试环境：Python3.12、NumPy1.26.4、Torch2.5.1+cpu，BLAS单线程。测试为局部状态／小规模自然根／确定性时钟／临时checkpoint；没有运行正式S2资源认证或S4比赛。合成反例不是比赛效果数据；已有仓内补测结果与本次独立实跑已分别标明。

判定最终落点：**支持继续修原型；不签S1通过、不签搜索收益、不签部署准备完成。旧方案修订予以接受，新代码缺口以本轮SI编号逐项关闭。**
