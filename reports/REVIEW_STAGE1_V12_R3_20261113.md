# S1 v1.2 确认复核（第三轮，2026-11-13）

**结论：R22、R23 与到时 root/finite/功效的指定原反例已关闭；R21 尚余训练目标软场景收益的一处聚合遗漏，不能确认 S1 整体关闭。** 原锚点异号例已经正确变为 INDETERMINATE，本轮没有否定这一修复。其余保留两项 P3：验收数量/唯一身份的完成表述仍过宽，以及 du 缺隔离清单时出现新的诊断入口异常。

冻结源码 **`4538fddf37bcc0e8ac608e5aa9b5f9093eca0b28`**，对照 v1.1 `7b025ae5` 与第二轮报告 `91cab34d`。本轮审查五个变更文件及其调用契约；源码行号均指冻结版。提交前另核对 main `d8f94f33`，所审19个源码/规范/验收文件与冻结版相同；新增提交仅为比赛数据更新。便携复现与 hash 清单见 [R3 证据目录](review_evidence/20261113_stage1_r3/README.md)。

gate 实验执行真实源码中的 sampler、workers 内部汇总、main、统计与工件读写，耗时模型和终局结果使用明确的控制表/worker fixture；终局实验执行真实 teacher/FastCarbon/原版规则、仅固定网络推理。**这些是代码与裁决反例，不是实际教师强度测量。** 未训练、部署、远程验机或改变作战室任务；未把作者 Studio 正典验收声明冒充本轮独立复跑。

## 1. 对账与已确认关闭的范围

| 项目 | 独立复核结果 | 状态 |
|---|---|---|
| R21 原锚点异号例 | 硬/软锚点按局均值仍为 −.0133333/−.0082984，正式结果 INDETERMINATE；正例 PASS、负锚点 FAIL | 原例关闭，另有软场景收益遗漏 |
| R22 诊断 pilot | 2根裁剪 pilot、9局不足 pilot、未完成隔离的 pilot 均发布诊断级工件；B 拒绝升格 | 关闭 |
| R22 交叉隔离/版本 | B 的训练清单包含 pilot 局时拒绝；训练版本变化、目标温度错配均拒绝 | 关闭 |
| R22 零方差重跑 | 同路径成功重跑发布新 run_id 的诊断级 sidecar，旧工件被替换；B 拒绝 | 关闭 |
| R22 合格正路 | 10局33根合格 pilot → 50局167根 Stage B 仍 PASS；预期 run_id 正确通过、错误拒绝 | 通过，非“全拒绝” |
| R23 训练目标 | behavior_temp=2、默认 train_temp=1 时负原 q 判 FAIL；显式 train_temp=2 按变温目标 PASS，头部身份一致 | 原例关闭 |
| 到时 root P3 | 5个 step299/300 双席输入均 ValueError、零推进、零推理，输入不变 | 关闭 |
| 原 S01 回归 | 16个合法 root 终局案例、非终局深度例、两臂 Q=`[.2228574087,.876]` 选 NONE、双席分支隔离保持通过 | 保持关闭 |
| guard-on finite P3 | NaN、负概率等原反例被拒；正常输出可通过 | 原例关闭 |
| 功效字段 P3 | significance-only 与含业务门的 conjunction 分开报告，数值与声明的正态近似公式一致 | 指定口径关闭 |

业务文件身份也核对一致：teacher MD5 `77ba36773fb135c92a6e5f171f88bdbd`；gate MD5 `4567b31e83e18fa099698335a476f65f`。search、guided_match、np、model、Feat、producer 及 NP/search packs 副本均与 v1.1 逐字相同。因此不要求无差别重跑所有 runtime/A1；本轮对改变的 teacher 入口及下游终局路径做了回归。

## 2. R21 剩余 P1：训练目标软场景收益仍按 root 聚合

位置：**`code/stage1_teacher_gate.py:698–710`**。

硬场景收益、改善率、硬/软锚点已经按局计算；但新增训练目标字段仍为：

```python
res["q_soft_gain_train_mean"] = float(np.mean(
    [r["soft_gain_train"] for r in rows]))
soft_train = res["q_soft_gain_train_mean"]
```

这个逐根均值直接参与 FAIL/PASS。规范 §9 和本次交付声明说“全部裁决输入按局”，还有这一条没接齐。每局正好三个根时两种算法相同；允许短回放、每局抽取根数不同时才出现偏差，不能由前一轮等长输入的翻转推定已全面闭合。

本轮使用**原始 list_replays 和 sampler**读取仅提供长度结构的合成回放 fixture：25个300步长度输入各抽3根，75个90步长度输入各抽1根，总计150根、100个来源局。长局的教师软收益为正，短局为负；两类均有正的 argmax 收益。实际 `_q_scene`、`_anchor_task` 和 main 运行如下：

| 指标 | 结果 |
|---|---:|
| 训练目标软收益，逐根均值 | **+.0040765239** |
| 同一批结果，按局均值 | **−.0072474825** |
| 按局硬场景收益 | +.0325 |
| 按局严格改善率 | 1.0 |
| 独立软锚点按局均值 | +.0778554 |
| 驱动裁决 | **formal=true，PASS，exit0** |

独立终局表在这里故意设为所有锚点正收益，以隔离“主软场景收益”这一项；**不是声称真实终局收益为负，也不是锚点还在按根计算**。按已冻结的协议，本例按局软场景收益≤0，应 FAIL，却因逐根均值为正越过该条件。反例证明裁决单位仍不一致，不估计真实数据发生频率。

最小修复：`cluster_stats(rows, "soft_gain_train")` 得到训练目标软收益的 `game_mean`，FAIL/PASS 共用该字段；逐根值保留为描述性指标并明确命名。同时保留原锚点异号、正常正例、负锚点、behavior/train 温度分离测试，补入不同长度/不同根数的真实采样器反例。

证据：[方法探针](review_evidence/20261113_stage1_r3/method/probe_method_r3.py)、[结果 `remaining_scene_soft_aggregation`](review_evidence/20261113_stage1_r3/method/results.json)。这一项是 R21 的残余，范围已从锚点收窄到软场景评分，不另造一套收益协议。

## 3. R22 / R23 关闭说明与运行口径

R22 本轮28项观测中，原四条不合格 A→B 路径均已翻转，合格正路保持通过。旧 v1.1 工件按原归档内容重建后直接提交给 v1.2 verifier，因没有 formal grade 被拒绝；没有手改旧工件补 grade/hash 来制造路径。新 sidecar 含本次 `pilot_run_id`，零方差也发布明确的 diagnostic grade，与旧主输出保留策略不是同一问题。

`--expect-pilot-run-id` 依规范为可选；明确给出错误身份时已拒绝，正确身份通过。调用者需要钉住指定运行时应传该值。本轮不把已明确的可选设计再次判为原 sidecar 残留漏洞。

R23 主评分和全部候选真实续弈软锚点均使用归一化 `q ** train_temp`。前轮16臂例在 behavior_temp=2 时仍报告行为收益 +.00430850，但默认训练目标收益 −.01857149 导致 FAIL；传 train_temp=2 才能认证相应变温训练目标。证据不代表标注/建集消费端已经接入该温度，后续实际训练必须消费同一目标身份；这是后续接线条件，不在本轮虚构一个已发生的消费端错误。

对用户提到的 Studio 16:26 v1.1 pilot：**“它不能授权 v1.2 的正式 B”符合当前代码核验结果。** σ 和单价可以作为带来源标记的诊断参考，正式 N 应由新版本、隔离数据、相同完整配置的合格 pilot 锁定。本轮未登录 Studio，也未查看那趟实际工件，结论来自旧格式工件的拒绝路径及用户描述。

另外，34局隔离池足够满足当前 pilot 的≥10局/≥30根资格门（还需满足实际长度、方差等条件），**不足以同时覆盖正式 B**。N_games 下限仍为50，且 B 与 pilot 局表必须不交；若34局全用于 pilot，B 需另外至少50局，实际可能按锁定 N 增至200局。更换种子或席位不能把同一个来源局变成新的独立局。这个要求是现有规则的执行含义，不是新增关闭条件。

证据：[pilot 探针](review_evidence/20261113_stage1_r3/pilot/probe_pilot_r3.py)、[28项精简结果](review_evidence/20261113_stage1_r3/pilot/results.compact.json)。

## 4. 两项 P3 与已通过的验证边界

### P3-a：fixture“精确数量与唯一身份已锁”仍过宽

`test_s12_a3.py:73–78` 新增 compare 身份去重，原单条、63条重复身份均拒绝，真实 `drive_agent` 的 game/t 字段兼容，63条唯一记录正路通过。但数量只要求≥50，50条唯一记录仍通过，不是锁定63条正典清单。

`test_s12_a3_teacher.py:61–63` 只要求≥40根，没有 root 身份去重；40根或50个重复根仍可 PASS。guard-on 的 finite/非负/归一化/长度检查已补齐，NaN/负q原例关闭，不与数量问题混在一起。应落实正典 fixture 清单、数量及 root identity，或者把 §9 的措辞改成准确的“下限检查/部分身份去重”，不要声称两个脚本都锁了正典唯一身份。

这是断言与报告边界，不证明作者 Studio 数据实际有重复或缺失。本轮正向使用合成输入及真实 drive_agent + no-op policy 检查格式，不冒充正典网络验收。证据：[验收探针](review_evidence/20261113_stage1_r3/acceptance/probe_acceptance_r3.py)、[15项结果](review_evidence/20261113_stage1_r3/acceptance/results.json)。

### P3-b：du 省略隔离清单时不再正常返回诊断结果

`stage1_teacher_gate.py:794` 新增调用 `_verify_pilot_artifact(..., ds_ids=ds_ids)`，但 `ds_ids` 只在提供 `--ds-games-file` 的分支初始化。du 非 pilot 模式省略该参数，会在 workers 已运行后抛 **UnboundLocalError**；预期是按既定协议返回 DIAGNOSTIC ONLY。

这不会把不合格运行判为 PASS，也不影响带正确隔离清单的正式正路，按 P3 诊断入口回归记。建议在读清单之前初始化 `ds_ids=[]` 或 None，并保持缺清单时 formal=false，增加无清单的 du 流程检查。不要因为非零退出安全地阻止了放行，就把它误写成期望内的诊断结果。

### 功效字段和源验证

实际 pilot 工件已分别报告 `power_significance_only_at_locked` 与 `power_conjunction_approx_at_locked`。σ=.1、N196时约 .79955/.5；σ=.36、N200时两者约 .12013；说明明确是正态近似且后者含样本均值≥.02业务门。指定的解释缺口关闭，不把近似功效写成精确有限样本保证。

终局到时入口与原S01保持测试见 [terminal 脚本](review_evidence/20261113_stage1_r3/terminal/teacher_terminal_probe.py)、[结果](review_evidence/20261113_stage1_r3/terminal/results.json)。本轮未重复完整 runtime 探针，相关未变源码的 hash 对照已归档；teacher 的实际变更路径已独立回归。

## 5. 本轮关闭建议

先补 R21 的训练目标软场景收益聚合，在最终同一 SHA 上重跑本轮短/长回放混合输入及既有正负控制例；同时处理或准确登记两项 P3。R22、R23 和已通过的指定 P3 按上述范围关闭，不要求推翻重做。

代码复核通过之后，合格 pilot、正式 q 闸、独立硬引导 ΔU 闸、蒸馏后的冻结裸网收益仍各自需要真实运行证据。旧训练集 pilot 不得作为正式 N 的依据；本轮未执行部署或调度任何机器。
