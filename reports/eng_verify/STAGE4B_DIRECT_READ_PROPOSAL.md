# 备选口径读数提案：学生在教师 arm 动作上的直读 q 评估

- 状态：**提案，只写不执行**。触发条件：shard_compare v4 合计 compared_roots 94 < 100（裁定下限），按裁定 §3「另行提案」成文。
- 裁定依据：`reports/RULING_SMALL_TEACHER_COMPARE_ALIGNMENT_20261118.md`（commit `ce48e03e`）§2 备选口径。
- 须策略组批准后才可执行；本趟未执行、未写任何 harness、未动任何标注输出。

## 1. 口径定义（必须明示）

**直读口径 ≠ 搜索臂一致性口径。** 直读读数度量的是学生 trunk 在教师选定的动作上的价值保真度（value fidelity），不度量学生搜索走出的臂集与教师臂集的一致性。输出件中须逐件标注「直读口径」，Tier 判读以直读四项为准并标注口径变更；阈值本身不变（Tier 1：flips=0 / |bias|≤1e-4 / max|Δq|≤1e-3 / max|Δdu|≤1e-3；Tier 2：flips≤2% / |bias|≤1e-3 / max|Δq|≤5e-2 / max|Δdu|≤5e-2）。

直读通过与搜索一致性失败是两个独立的读数：直读过而搜索一致性不过，不等于产品过原 Tier 口径；两者须并列报告、各标口径。

## 2. 做法

1. **先读源码定映射（提案未定、先决步骤，不许想当然）**：读 `stage0_teacher.py`（c6ade5ce）`evaluate_root`，逐行确认每个 arm 的 q 是由哪些 trunk 调用构成的（子局面 u 直读、rollout 聚合、抑或其它）。直读 harness 必须逐项复现该映射，仅把 trunk 换成学生 trunk；映射未读清之前不许写 harness。
2. **逐根重建局面**：对附卷两局 994400/994401 的每根（`(step, seat)`，共 396 根），用密封回放文件经 k0_annotate 同款回放前缀重建根局面。
3. **取教师 arm 集**：该根的动作集取自教师侧分片 `arms`（16 个 12 维编码）——不展开学生搜索。
4. **学生直读**：对教师 arm 集内每个动作，用学生 trunk（`stage0_teacher_small.py` 5c0dc757 + `student.npz` b22020ff）按第 1 步确认的映射直读 q，不经学生搜索、不展开学生臂。
5. **四项度量同口径计算**：以教师分片的 q 为 base、直读学生 q 为 cand，用 v4 仪器（join 键退化为全集——每根的动作集即教师臂集，交集 = 全集）算四项；输出件须含每局 compared_roots（目标 396/396）、可比动作条目数（目标 396×16=6,336）、Tier 1/2 四项实测值。

## 3. 所用件（全部已定位、L20 在位）

| 件 | 位置 / 身份 |
|---|---|
| 教师侧分片（arm 集 + q 基准） | L20 `/root/small_teacher_50k/annot_teacher/shards/selfplay-99440{0,1}.npz`（config `9a2b2440`） |
| 附卷回放 | L20 `/root/small_teacher_50k/stage4/appendix/selfplay-99440{0,1}.json.gz` |
| 学生权重 | L20 `/root/small_teacher_50k/stage4/student.npz`，md5 `b22020ff`（已直读验合） |
| 学生件 | L20 `/root/small_teacher_50k/stage4/code_student/`（`stage0_teacher_small.py` 5c0dc757） |
| 教师件（映射来源） | `stage0_teacher.py` c6ade5ce（仓库在册） |
| 新 harness（待写） | `tools/direct_read_q.py`（本趟未写）；输出表建议落 L20 `stage4/direct_read/`，再以 v4/v5 仪器读数 |

## 4. 校验关（缺一不许采信读数）

1. **交叉核对**：在 v4 已有的 94 个可比根、228 条 (root, action) 上，直读学生 q 与学生搜索 q 逐条比对——两者应接近（同一 trunk、同一叶评估）；若系统性偏离，先停、查映射，不许直接采信直读数。
2. **确定性**：同参重跑一次，逐位一致（seed 12345，与标注同）。
3. **身份**：开跑前机上直读学生 npz md5（须 = b22020ff）、config 重算（须 = 9a2b2440）、回放 sha 与注册 v2 对账。
4. **口径标注**：输出件文件名与件内均写明「直读口径」，与 v4 搜索一致性件区分存放，不混入同一 generation。

## 5. 座位与工作量（估计）

- 座位：L20（GPU 在位、代码与数据已在机上；须在派单时核 L20 空闲窗口，不与训练并发）。
- 工作量：读映射约半天量级的小工具（`tools/direct_read_q.py`，仅复现映射 + 批量直读）；运行量 396 根 × 16 动作 = 6,336 次 trunk 调用，可批量，GPU 上分钟级。
- 风险：若第 1 步发现教师每 arm 的 q 含搜索内 rollout 聚合、无法以单次 trunk 直读复现，则直读口径本身不成立——提案作废，回策略组另议，不许凑合换映射。
