# k0_distill 冻结清单事实确认

- 依据：策略组裁定 `reports/RULING_CODEX_REVIEW_RESPONSE_20261117.md`（commit 50e00d1a）§1 第 4 点；`reports/PLAN_GLOBAL_20261118.md` §9 检查点 2026-11-18 12:00。
- 出具方：工程组。方法：main 分支代码直读（API 读回原文后逐行核对，非凭记忆）；md5 均为 2026-11-18 12:0x 由 API 读回字节逐件算得。
- 确认行：**钉版 k0_distill（md5 2728fee6）冻结 utility./wdl./margin./aux. 四头；损失＝策略 CE＋kl_beta·KL（默认 kl_beta＝1.0），无价值头监督项；主干（trunk/decoder）与策略头可训。**

## 1. 冻结清单（直读 `code/k0_distill.py` L137、L40–46、L978–990、L870–890）

- 冻结前缀（L137 原文）：`FROZEN_PREFIXES = ("utility.", "wdl.", "margin.", "aux.")`
- 冻结对象（L40–44 原文意）：CNet2 的 utility / wdl / margin 三个价值头＋aux（true-carbon）卷积——`requires_grad=False`、排除出优化器、checkpoint 发布前与父代逐位比对（moved 的冻结参数拒绝发布并点名）。
- 执行机制（L978–990）：启动时按前缀逐参数置 `requires_grad_(False)`；冻结名单为空即 `SystemExit` 拒绝启动；随后拍快照 `frozen_snapshot`；优化器只含可训参数（AdamW，lr 默认 3e-4、wd 默认 1e-4）。
- 发布闸（L877–890 `check_frozen`）：逐位比对快照，任何冻结参数字节移动即列入 `moved`、拒绝发布；文件原文定性为代码缺陷、非训练结果。
- 可训部分（L44–46 原文意）："Everything else (trunk, decoder, policy heads) trains."——**共享主干与解码器不在冻结之列**，故价值读数可经主干漂移，尽管价值头权重逐位冻结。这是裁定 §1 第 4 点所指机制的代码事实（此处只陈述代码事实，不作判读）。

## 2. 损失构成（直读 L8、L28–38、L907、L1078–1079）

- `loss_root = ce_root + kl_beta * kl_root`（L1078）；`--kl-beta` 默认 1.0（L907）；批量损失为 IPW 加权根均值（pi＝1 时 IPW 不激活）。
- CE 项：策略全联合动作交叉熵（教师 π_search 为目标），无硬标签 CE（L38 原文："No hard-label CE anywhere"）。
- KL 项：`KL(pi_student || pi_parent)`，逐候选逐有效槽分类 KL 求和、根内候选均值；方向为学生先、父代后（L28–37）。
- **无价值头监督项**：损失路径仅 CE 与 KL 两项；`VALUE_BUNDLE = "value_labels"`（L135）为数据集标签束名，不进入损失计算。utility / wdl / margin 在本文件内无 MSE/CE 监督项。
- 断路器：`MAX_LOSS = 1e6`（L142），1e6 量级损失即 NEG-mask 特征、触发 die。

## 3. 钉版文件清单（逐件 md5，main 读回）

| 文件 | 字节 | md5 | 备注 |
|---|---|---|---|
| code/k0_distill.py | 58243 | 2728fee63e16f4321c5639273a62631c | 钉版本件（blob b8fe891a） |
| code/k0_duel.py | 26855 | bead6367e7650894afdda7717ff649a5 |  |
| code/k0_join.py | 32537 | fac55acbcbffdfd6582c9449e535aa2f | 现行 main 值；旧值 27c11e16 作废 |
| code/stage0_teacher.py | 25186 | c6ade5ced42e2c54a77c1499163aa151 | 钉版教师代码 |
| code/cnet2_model.py | 19720 | 3bb5857f833a94797ac6cdc2a5a28963 |  |
| code/cnet2_search.py | 131590 | c3222eea76ea9d42c160626c2e239ca0 | **main 头值；钉版搜索为 94f7fb64，勿由 main 推断钉版行为** |
| code/cnet2_np.py | 14953 | d8c15ad4fdd8ee079363cf215d52daf3 |  |
| code/cnet2_feat.py | 22539 | 2a45b08b6f97508541e62867e408748e |  |
| code/stage1_teacher_gate.py | 51013 | e922d9d890c3e1eff6643e947289acb0 |  |
| code/fastsim.py | 36212 | aefe6c83826efae4de83bef8c643c80b |  |
| code/eval_v53.py | 37865 | 45792dfbea3101812836fe77e6752706 |  |
| code/k0_annotate.py | 49151 | 7f567a2e8544b78e51d16aa7660bc47b | gen-1 钉版标注器 |
| code/k0_distill_u.py | 67392 | 06826f5926c97ecccda8b8cbd18ff020 | **双目标变体，非钉版训练器**：解冻 utility/wdl/margin 并加 λu·MSE(U)＋λw·CE(WDL)＋λm·MSE(margin/1000)，不在本冻结清单内 |
| code/k0_join_u.py | 40262 | 9054aadca62a9770ea5b7df90d46cb73 | 同上，双目标 join 变体，非钉版 |

- 权重（不在仓内，部署树实测值）：`cnet2_m3s2r1_heads_v2.npz` md5 `06a03d9cfeec239af3065a49fefec382`。

## 4. 范围说明

- 本确认只覆盖钉版 `k0_distill.py`（2728fee6）本人的冻结与损失事实；价值校准口径与分支排序检查按裁定在 gen-2 注册时另行落字，不属本件。
- 12:00 检查点交出时间：本回执入仓 commit 见下；读回 md5 与本表逐字一致方为交出。
