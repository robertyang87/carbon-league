# k0_distill_u 三身份热启接口设计（工程组，2026-11-18）

- 依据：Codex 复核 `reports/DECISION_REVIEW_FOR_LOOP3_20261118.md`（3f28c966）§1；基底 `code/k0_distill_u.py` md5 `06826f59`（TRAINER_VERSION `k0-distill-u/1`），改后版本 `k0-distill-u/2`。
- 一句话：现行 `--parent` 一参三职（学生初值＋KL 锚＋教师身份闸的比对对象），步段链在构造上跑不通；本设计把三职拆成三个显式身份，数学与损失一字不动。

## 1. 耦合点（现行代码实证）

| # | 位置（06826f59) | 耦合 |
|---|---|---|
| C1 | `--parent` 参数（L1000–1002) | 唯一权重文件入口 |
| C2 | 身份闸（L1051–1056) | 数据代 producer 的 `teacher_weights_md5` 必须等于 `--parent` 文件 md5——教师身份被绑在初值文件上 |
| C3 | 双网同源加载（L1076–1084) | 学生 `net` 与锚 `parent` 由同一 `start_sd` 加载，初值≡锚 |
| C4 | 冻结快照（L1087/1096) | `init_snap`/`snap` 取自初值后的学生——今日因 C3 与教师同值，provenance 写 `bitwise_identical_to_parent` |
| C5 | 运行登记（L1118) | provenance 只有一个 `parent` 块，三职无从分辨 |

后果（复核原文）：段 2 传上一段学生撞 C2 教师闸；传环父网则 C3 使每段从父网重训。publisher（合并代）解决不了此耦合，必须改训练器接口。

## 2. 三身份接口

| 身份 | CLI | 语义 | 约束 |
|---|---|---|---|
| `label_teacher` | `--label-teacher`（必填） | 产生所消费标注代标签的教师权重文件。**只作身份锚**：读字节算 md5/sha256、解析验形，**不加载进任何网络**（priors/q/Q/终局标签都在数据代里，教师权重不参与本段计算） | 文件 md5 必须等于数据代 producer 的 `teacher_weights_md5`（原 C2 闸原样保留、只换比对对象）；全环固定——每段消费的源代教师身份都必须是它 |
| `student_init` | `--student-init`（必填） | 本段学生网络的初始权重（段 1＝环父网；段 ≥2＝上一段学生产物），独立登记 file/md5/sha256 | 加载进学生 `net`；架构 cfg 默认取此文件 meta 的 cfg |
| `KL_reference` | `--kl-reference`（选填） | KL 锚网络的权重，全环默认固定环父网 | 加载进独立锚网（eval、requires_grad=False)；省略时取 `--student-init` 同一文件（登记里明示 `relation_to_student_init`，不静默） |

- 三个文件各读一次字节，身份（md5/sha256）与解析共用同一读（沿现行纪律）；三个文件都过有限性闸（float 数组非有限即 die 点名文件与数组）。
- cfg 来源改口径：默认 `--student-init` 的 meta cfg（学生是被训对象，其架构决定一切）；`--cfg` 显式覆盖不变；锚网与学生同 cfg 构建，各自 migrate 自己的文件（缺失前缀容忍口径 `tattn.` 两边一致、各自登记 migrated_missing）。

## 3. 运行登记（provenance）

`eff["k0_distill_u_effective"]` 中原 `parent` 块替换为：

```
"identities": {
  "label_teacher": {"file", "md5", "sha256",
                    "matched_against": "annotation producer teacher_weights_md5",
                    "teacher_config_hash": <数据代记录>, "belief": <数据代记录>},
  "student_init":  {"file", "md5", "sha256",
                    "segment_role": "ring parent (segment 1) or previous segment student"},
  "kl_reference":  {"file", "md5", "sha256",
                    "relation_to_student_init": "same_file" | "independent_file"}
},
"anchor_continuity": {"status": "ring_anchor_baseline" | "held" | "moved",
                      "previous_kl_reference_md5": <或 null>},
"recipe_change": [ <机械检出条目> + <调用方 --recipe-change 声明原文> ],
"start_mode": "hot_start_weights",
recipe.optimizer = {"type": "AdamW", "mode": "hot_start_weights + fresh_optimizer",
                    "state_carried": false, "lr", "wd"}
```

产物 npz 的 meta 沿现行路径带同一 provenance（`k0_distill_u` 键），下一段以其为连续性证据（见 §5)。

## 4. 拒绝语义（逐条，全部在任何 update 之前）

1. 传 `--parent` → 立即 die，报错给迁移映射（见 §6)。
2. `--label-teacher` md5 ≠ 数据代 `teacher_weights_md5` → die（原闸文本口径不变，只换对象名）——教师错配必须拒绝，这是短链验收的负例。
3. 任一权重文件不可读/不可解析/float 数组非有限 → die 点名角色与文件（锚/初值另在 migrate 处对形状不合 die、源名分别为 `--student-init`/`--kl-reference`)。
4. 冻结前缀在学生上匹配为空 → die（现行闸保留）。
5. 数据代自身的全部 SS13-C 闸与 train_temp 三方同值闸不动（load_annotation 未改一字）。

## 5. 锚点连续性：机械登记，不靠调用方自觉

- 段 ≥2 的 student_init 是上一段本训练器的产物，其 meta 内嵌上一段登记的 `kl_reference` md5（/1 旧产物则取其 `parent` md5——/1 的锚就是它的 parent，口径自洽）。
- 本段启动时读出该值与本段实际锚 md5 比对：同 → `anchor_continuity.status = "held"`；不同 → `status = "moved"` 且 `recipe_change` 自动追加一条 `kl_reference_moved: <prev md5> -> <this md5>`（复核 §1「逐段移锚须另记为配方变化」的机械落地，不许藏在 init 参数里——此处锚与 init 本就是两个参数，移动无处可藏，登记再钉一层）。
- student_init 无本训练器 provenance（段 1 环父网）→ `status = "ring_anchor_baseline"`。
- 调用方另有配方变化可用 `--recipe-change "<文本>"`（可重复）声明，原文入登记，与机械条目分列来源（auto / declared 前缀区分）。

## 6. `--parent` 处置：硬切＋迁移 die（不留兼容别名）

- 理由：别名若把一个文件同时塞给三身份，正是 C1–C3 耦合的原样复刻；段 2 调用方误用别名只会在教师闸处得到一句难懂的错配报错。硬切＋die 文本直接给映射，旧运行手册在开跑前 loud fail、永不静默误训：
  `--parent F` → `--label-teacher F --student-init F --kl-reference F`（仅当三职确实同一文件时等价于旧行为）。
- `--parent` 在解析器中保留为可选参，只为触发这条迁移 die；新登记中不出现 parent 身份块。

## 7. 冻结头纪律再对齐（参照对象换人）

- 本训练器的冻结集是 `FROZEN_PREFIXES = ("aux.",)`（true-carbon 卷积前端；utility/wdl/margin 三头是训的，不在此列）。
- **参照对象＝student_init**：`snap` 取自学生加载 student_init 之后、任何 update 之前的自身状态（机械动作与现行一字不变），发布闸验的是「aux. 与本段起点（student_init 所载）逐字节同」。
- 为何不是 label_teacher：教师只经数据代供标签，其权重从不进学生；要求学生 aux. 等于教师 aux. 没有计算含义，且段 ≥2 会误杀合法的学生谱系。为何不是 KL_reference：锚是只读旁观网络，与学生的参数谱系无关。
- 谱系含义：aux. 的标定属于 init 谱系——段 1 继承环父网的 aux.、其后逐段由学生产物原样传递，哪一段的 aux. 变了，发布闸就在哪一段拒绝（check_frozen 不动）。provenance 字段随之改名 `bitwise_identical_to_parent` → `bitwise_identical_to_student_init`；`init_snap` 的全参快照（trainable_moved 与三头 unfreeze 对账的基线）同理改口径为「vs student_init（本段起点）」。

## 8. 优化器明示

每段启动都是 **hot_start_weights + fresh_optimizer**：只继承 student_init 的权重，新建 AdamW（无 optimizer state 携带、无 step 计数续接）。文档（文件头）与运行登记（§3 optimizer 块）都写这两个字面词；全文不以 resume 暗示完整续接（原 docstring「no resume…restart from the parent」一句改写为三身份口径）。

## 9. 单元级自证（本机无 torch，层级如实分列）

本机 VM 无 torch（全机探查无果），自证分两层，REPORT 里逐项标层级：

- **L1 纯函数/参数层（本轮实跑）**：身份闸函数错配 die/匹配过；`--parent` 迁移 die;`--label-teacher`/`--student-init` 缺失时 argparse 拒绝；锚连续性函数三态（baseline/held/moved＋recipe_change 条目生成）；上一段登记读取（/2 与 /1 两种 provenance 布局）。
- **L2 文件级加载分离构造证明（本轮实跑，numpy 层）**：main 的文件阶段被抽成纯函数 `_read_weights_file`（三身份共用），自证造两个内容不同的小权重 npz，经与 main 完全相同的调用路径分别取 student_init 与 kl_reference 的解析结果，断言两份 state 数组逐键来自各自文件、互不串扰——证明「两网各从其文件加载」的文件级一半；net 构建＋migrate 的 torch 一半与短链验收同批补（L20/座位访问恢复后），不在本轮冒称。
- 真训练路径（torch) 本轮未跑：短链配方见 §10，待主代理排期。

## 10. 短链验收配方（真实标签段 1→段 2，待访问恢复执行）

- 数据：环 1 joined 代（L20 `loop_pilot` 合并产物、format /2）或环 3 段 1/段 2 源代（取先可用者）；label_teacher＝该代记录的教师权重文件（md5 须与源代 `teacher_weights_md5` 合，环 1 代为 `06a03d9c…`)。
- 段 1:student_init＝环父网、kl_reference＝环父网（同文件），短程 `--max-updates 20 --bs 64`（冒烟量级）；段 2:student_init＝段 1 产物、kl_reference＝环父网（不动），同量级。
- 验项清单（逐项读回登记/产物判定）:
  1. 初值继承：段 2 首个 update 前的学生即段 1 产物——以段 2 产物登记的 student_init md5＝段 1 产物文件 md5、且段 2 `trainable_moved` 基线为段 1 产物字节（init_snap 口径）双证；
  2. 教师身份保持：两段登记的 label_teacher md5 都等于各自源代 `teacher_weights_md5`，且两段同值；
  3. 锚点：段 2 `anchor_continuity.status = "held"`、recipe_change 无 auto 条目；
  4. 有效 lr/step：两段 log 首行登记的 lr/max_updates 与命令一致、实际 updates_done 合；
  5. 冻结闸：两段产物 aux. 与各自 student_init 逐字节同（`bitwise_identical_to_student_init = true`)；
  6. 负例：段 2 换一个错误 label_teacher 文件 → 开跑前 die、无 update、无产物。
- 环境：L20（租约 2026-11-20 23:59 到期）或任一带 torch 的座位；本机 known_hosts 恢复后由主代理排期，本轮不跑、不可预支结论。

—— 工程组 2026-11-18
