# r4 复审 · k0_distill.py 三项修复回执（X01 / X02 / X04(b)＋X06 记注）

对象：`robertyang87/carbon-league` `code/k0_distill.py`（只动此一文件，三项各一独立提交）。
基线 md5 `d5fc464fa3e8acafd28f675f3f281091`（37,912 B，blob `609867d3e6a62157569c2fb32fb2a2bf09b7411b`，开工时 API 读回核对）。

| 提交 | commit | blob sha | md5（读回逐字节验合） |
|---|---|---|---|
| X01 | `a84c48ba2419817d9679479545f65c3d13556453` | `3ba87a5e4ed3b3606708a39736c382d81cf891b3` | `289c27e51b485a46327af3aec66ad90b` |
| X02 | `7461096b11b0c70f409ac680602e7774143157cc` | `20a6a8f814c76597a46d2baa0c2486a365f7803b` | `9a4442f038076db8eb52757d0db3e7d2` |
| X04(b) | `c067c8d000f210b56e7dc2fbafd2e0b0382031eb` | `b8fe891a11f8b52622bbd108b2a1a44efa919cab` | `2728fee63e16f4321c5639273a62631c`（终版 58,243 B） |

证据目录：`~/workspace/r4-fixes/x01/`（build_repro.py / build_r4.py / repro/ / r4/ / fx2/ / run_base|run_x01|run_x02|run_x04/）。

---

## X01（P1）：候选前缀特征不重算 → 损失爆炸且照常保存为完成

### 机制（亲读代码坐实）
原 `expanded_batch` 把根行按候选重复 K 次、只换 `y = arm`，`uf/mf/legal` 全沿用存档——存档特征是 `cnet2_build` 按**日志动作**对根内前缀重放的产物。动作效果派生、必须随候选变的成分恰五类：mf[·][·][0]（占位 reserved）、uf[·][6]（cash）、uf[·][7]（空位 slots）、uf[·][8]（plants）、基地槽 legal。其余成分只依赖根观测与槽序，存档值对所有候选正确。解码器 NEG 掩码 = −1e9：候选动作被存档 legal 掩掉时 joint logp ≈ −1e9，CE ≈ q_k×1e9。

### 复现（先于修复）
- 自建构造（cash 45，日志 (招募，停)、候选 (停，招募) 等 3 根，真实 Feat+Prefix 封存）：基线代码 update-1 total loss = **243,663,561.58**，且 `complete=True` 照常落盘；同根同臂以 Prefix 驱动特征独立算 joint_logp = **−10.3359**（CE 量级 O(1)），存档特征则 −1,000,000,009.03。
- 评审精确几何（step 200、cash 40、两基地、候选 (招募，停)/(停，招募)，存档为第一臂前缀；ds_r4a）：基线 CLI update-1 CE = **817,624,478.13**，退出码 0 且 SAVED complete True——与报告 CE 353,718,778.79 同机制同量级（数值差来自网络/目标权重不同），「照发 checkpoint＋complete」症状逐项复现。

### 修法
- 新增 `expand_candidate_batch`：逐候选按 `cnet2_feat.Prefix` 转移重放，只重算上述五类成分，其余透传存档；重算后整表过 f16 量化（对未触成分幂等）。
- 根 cash 精确恢复：fastsim 每步 round(cash, 2)，Feat 把 cash 四重编码进 scal（[2]/[19]/[12]/[13]，树价由 maps 树平面按 Feat 原式重建）＋第五方程——存档 slot-11 uf[6]（日志槽-10 动作后的 cash 编码，含/不含一次 30 扣款两分支必居其一）。一致集为空 → die 点名；集内跨 slot-11 合法阈值（cash 60）分歧 → die 点名；仅跨 uf f16 桶分歧 → 取中位并计入 provenance 披露（`candidate_features.cash_ulp_roots`；全部测试语料中该计数为 **0**）。
- 候选须在**自身前缀**下合法：基地槽按重算 legal 校验、末尾再对全槽做「所选动作不得被解码器实际施加的 legal 掩掉」自洽校验，不合 die 点名（根/game/step/候选/槽）。
- 附带堵保存：训练循环在既有有限性闸同处加量级断路器 `MAX_LOSS = 1e6`（total / ce_root.max / kl_root.max，依据：合法 CE/KL 的 per-root 量级界 ~1e2，1e6 是 NEG 掩码签名），走既有 `die_nonfinite` 通道（circuit_breaker 落账＋exit(3) 不保存）。

### 验证
- **特征逐位对照（独立重算）**：expand 输出 vs 逐 (根，候选) 独立 Prefix 驱动的参考特征——ds_x01 **27/27**、ds_r4 **24/24** 组 (uf/mf/legal) 全 `array_equal`；含同前缀正例（日志臂本身，误差 0）、占位冲突例（相邻采集者 80/81 格、候选抢占对方格）、共享预算例（cash 70 双招募，第二招募靠共享池恰扣一次才合法）。
- **CE 对账（评审几何）**：逐候选重放 CE = **10.139237666**；修复后 CLI update-1 CE = **10.139237665746773**（合至 ~1e-9）。
- 负例：ds_neg1（候选在自身前缀下非法：cash 45 双招募）→ 扩展期 die 点名、无检查点；ds_neg2（存档 worker legal 毒化）→ 初版被量级断路器拦下（`loss:total:magnitude`，值 1,000,000,008.8，exit 3、无 complete 标记），加全槽自洽校验后改为扩展期点名 die；训练发散路径（lr=30）→ 量级断路器以 5.5e14 触发 exit(3) 不保存。
- 真产物链：joined_out（gen-f18f5906）S/C 双臂短训（最终代码）：update loss 3.4104/39.6378（S）、3.4434/40.0565（C），有限、落盘 complete；2 根触发 cash 恢复、ulp 计数 0。

## X02：train_temp 记录未消费 → 按未 temper 分布训练

### 落点与读法
认证口径 = gate R23：`normalize(q ** train_temp)`（q 为 §13-C 以 q_temp=0.05 从 Q＋priors 的构造；标注存原始 q，temper 是消费侧变换）。实现（`_recorded_train_temp`＋`temper_target`）：
- 主记录 = **meta.teacher.config.train_temp**（两种产物形态都带：标注器直出 producer.config 亦有、meta.teacher.config 同值；join 产物 producer 无 config、只有 meta 一处＋`teacher_config_hash` 绑定）。
- producer.config 若在场，其 train_temp 须与主记录相等，否则 die 点名两值；在场但缺 train_temp 同样 die。
- 哈希互绑：producer.config_sha256 / producer.teacher_config_hash / meta.teacher.config_sha256 三者凡在场必须全等，否则 die 点名。
- 记录缺失、非数值、非有限或 ≤0 → die 点名。无 CLI 覆盖口（`--q-temp` 角色不变，仍是构造温度；合成关系写入 provenance `recipe.target_composition` 与 `recipe.train_temp`）。
- 存档 q 交叉检查保持对**未 temper** 构造（在 temper 之前）；S 臂 target 与 C 臂（targets_with_flatten 的置平构造）同施 temper。

### 两种口径的数值关系（按预注册要求写明，不默默选边）
本实现 temper 的对象是**构造 q**（先被存档交叉检查以 2e-6 容差认证的同一对象）：ds_tau2 上 target 与 `normalize(construct**2)` 逐根**差 0**；与 `normalize(q_stored**2)` 的最大差 **3.75e-07**，恰在交叉检查容差（2e-6）以内——两口径之差即存档 q 与构造 q 之差经 temper 的传播，无额外偏差。

### 验证
- tau=2 fixture（ds_tau2，join 产物形态的记录布局）：上行对齐成立；C 臂 root0 与 temper 后的 sqrt-priors 目标差 0；CLI 短训 provenance 显示 train_temp 2.0＋composition，loss 11.68→8.10 正常落盘。
- tau=1 逐位回归：同一 ds_a 以 X01 阶段代码与 X02 阶段代码分别 load，target 数组 `array_equal`；五项 fixture 在 X02 阶段的读数与 X01 阶段逐字相同。
- 负例（重封 doctored 代）：记录缺失 / producer 与 meta 不一致（3.0 vs 1.0）/ config 哈希不一致，三件全 die 点名。
- joined_out 以 X02 代码 load 成功（train_temp 读出 1.0）。注：标注器直出代（out1）直喂 distill 仍被**既有** format 闸拒（`k0-annotation-generation/1` ≠ 消费格式，与基线行为一致、非本项引入）；其记录本身双处齐备已亲读核对，实际链路经 join 透传后由 meta 记录供消费。
- 连带说明：旧五项 fixture 的封存代不带 train_temp 记录，X02 后会被按名拒收——这是预注册口径（缺失即 die）的应有行为；回归改用带记录的再生 fixture（见下）。

## X04(b)：verify 全量验封后二次按路径打开（工程组同类增补，与报告 X04 本体分开标注）

按 2026-11-14 16:41 映射更正：报告 X04 本体 = k0_annotate 教师权重加载＋k0_duel 双半边，归标注器趟与 X07 趟；本项是工程组先报四处同类增补中的 **distill load_annotation 一处**，续修并单独标注。
改法：`verify_annotation_generation` 每文件只读一次字节，size/sha256/行数全在该字节上验，`.npy`→数组、`.npz`→{名:数组}、meta 同理（读字节→哈希→json.loads），返回 `(gen_dir, man, meta, verified)`；`load_annotation` 改消费 verified，删全部二次打开。全部拒收消息字符串逐字保持。
验证：X02 阶段代码与 X04 阶段代码对 5 个数据集（ds_a/ds_b/ds_tau2/joined_out/ds_x01）load 结果**深比全等**（feats/keys/arms/target/info 等逐数组 dtype+shape+sha256 与 info 全等）；tamper 与直传封存代两负例拒收消息不变（"hash mismatch" / "via its dataset root"）。join 的 consumer preflight 只读 `ds["info"]` 四键，load 返回结构只做加法（verify 元组为内部变化），兼容。

## 五项 fixture 回归（三阶段一致）
原 fixture 的合成根不是诚实 Prefix 记录（scal 随机、无 train_temp 记录、base 臂可与重算 legal 冲突），在 `fx2/make_fixture2.py` 最小分叉再生：每根固定敲掉 worker 槽 3、8（slots_root=2）；scal 按 cash=500 诚实编码、maps 树平面清空（价 20）；base legal 取 Prefix 诚实的 [T,ok,ok,F,F]；记录双处写 train_temp＋单哈希绑定，另加 ds_tau2。t1/t2/t5 的 harness 自带 `expanded()`/replay 两边同用存档特征，是既有 harness 属性、非本补丁引入——真正的独立对照是上文 Prefix 驱动逐位比对。
三阶段（X01 / X02 / 最终）读数一致、全过：t1 max diff 2.28e-07（tol 1e-5）；t2 target 恒等、grad diff 2.32e-06（tol 1e-4）；t5 IPW ratio 2.0、同根 CE 逐字同；t4 tamper/直传双拒；t3 batch 序同、冻结头逐位不变、63 参数移动、marker sha 合；t4 parent 不符点名拒。绝对读数与原验收跑不同仅因 fixture 数据已按上表再生（检查均为内部一致性判据）。

## X06 记注（只记不改）
`load_annotation` 的 key-agreement 段逐行强制 policy bundle 的 game/step 与特征行 gid/step 相等（zip 式联结；跨代唯一身份本应是局键＋seat）。本趟三项均未触及它：X01 的重算只在根行内以 arms/tok_* 为键，不碰联结；X02 的记录读自同一代的 manifest/meta，不碰联结；X04(b) 只改字节取得方式，该段检查代码与其所比数组逐字不变。（join 侧 X06 已由策略组补充探针按报告口径结案；此处仅 distill 存量登记，供后续契约收口参考。）
