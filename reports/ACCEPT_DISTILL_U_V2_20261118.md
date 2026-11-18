# 验收：k0_distill_u /2 三身份热启——文件级有条件 PASS（策略组，2026-11-18）

- 对象：code/k0_distill_u.py 升 k0-distill-u/2（958d13cf、cfbf227e）＋设计件 reports/DESIGN_HOTSTART_3ID_20261118.md（44340c12），自证 9/9（文件级）。
- 设计核对（策略组亲读）：现行 `--parent` 一参三职（初值＋KL 锚＋教师闸比对对象）使步段链构造上跑不通，拆三身份语义正确——label_teacher 只作身份锚不加载进网络（原教师闸保留、只换比对对象、全环固定）、student_init 为学生初值、KL_reference 独立锚（省略取 init 且登记明示关系）；锚连续性机械登记（moved 自动记 recipe_change）；`--parent` 硬切迁移 die 不留别名；冻结头参照改 student_init（谱系口径成立）；优化器明示 hot_start_weights + fresh_optimizer。数学与损失不动。**设计与文件级自证（L1/L2 分层如实）验收过。**
- **条件**：TRAINER_VERSION k0-distill-u/2 的入册生效，以设计件 §10 真实短链（L20 或带 torch 座位）六项验项全过并回执为前置——含负例（错误 label_teacher 开跑前 die、无 update、无产物）、锚连续性 held、冻结闸 vs student_init、教师身份两段同值。短链未过前 /2 只许文件级/冒烟用途，不进任何正式段训练。排期注记：L20 租约 2026-11-20 23:59 到期，短链须在租约内或另座执行，由执行线盯、不预支结论。
