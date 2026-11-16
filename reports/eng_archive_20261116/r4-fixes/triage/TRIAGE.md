# r4 复审定性报告（工程组定性线，只读）

基线核对（本趟 API 读回，5/5 逐字节 MATCH）：k0_train `4d5b3b67`、k0_annotate
`cdee7c03`、k0_distill `d5fc464f`、exit_merge `49c32b25`、k0_duel `215b18e0`。
交叉引用文件（同趟读回）：cnet2_train `80613485`（冻结器 W 轮修复后现行值）、
cnet2_build `8598f26d`、stage1_teacher_gate `e922d9d8`、cnet2_exit_train `23b92d70`。
结论先行：**8 条全部坐实，无一条与评审描述不符**；各条的落点、行号与归属如下。

---

## 1. W02 为何保留 —— 我的修复只盖住 k0_train 自家三件，数据集面仍是两段式

已单读绑定的部分（W02 修复范围，复核无误）：aux 束 `_read_aux_bundle`
（k0_train.py:246-263，校验/解析/身份同源，消费在:511-548 与 screen:1030+）；
init npz（:590-592）；resume 检查点（:607-617，marker 只作比对、受信值来自
已消费字节）；screen 检查点（:1051-1062）与 screen init（:1072 起同型）。

仍两段式的面，按嫌疑度排序：

1. **cnet2_train.load() —— k0_train 的全部训练特征都经它。**
k0_train.py:481 `D, is_val, meta = T.load(args.data)`。cnet2_train 的解析段
（cnet2_train.py:436-494）经 cnet2_build.load_generation 逐字段验 sha256＋
meta 封印，但 `load()` 本体（:628-631）随后把全部字段**按路径重新
`np.load(..., mmap_mode=mm)`**——验的是 A 的哈希、训练全程 mmap 读的是路径
上的现行字节，窗口覆盖整个训练期。这正是 V 轮已立案的 **N2 残留**
（cnet2_train.load 的 verify-then-reopen-by-path，当时以「代目录变可变/
共享可写」为重开条件立案）——r4 把 W02 保留，与 N2 落点完全吻合：
我的 W02 修复没有、也不可能从 k0_train 一侧盖住它。
2. **k0_train 自家残余：u_raw / label_status 按路径加载且从不自验哈希。**
k0_train.py:496/:499（main）、:1020/:1023（screen）。k0_train 内唯一被
哈希绑定的字段是 gid/step（row_key_sha256，:265-279）；U 监督列 u_raw 的
完整性完全寄托于 T.load 更早的按路径校验，之后在 k0_train 里被第二次
按路径打开——校验与消费分属两处两次打开。
3. 潜伏面（CLI 不触发）：check_aux_sidecar 的 prefetched=None 回退分支
（k0_train.py:355-361）按路径重哈希；两个 CLI 调用点都传 prefetched
（:512、:1031），仅库调用者可达。另：回放文件在 k0_train 里只验哈希、
不被加载（:340/:346），不构成消费缺口。

归属：工程组（cnet2_train、k0_train 均我线）。
建议修法一句话：load() 改为校验时单次读字节（或至少对 K0 消费的字段关
mmap、验后即驻留），k0_train 的 u_raw/label_status 改从 D 的已验副本取或
在本文件内单读绑定。

## 2. X02 —— train_temp 定义在 gate、记录在标注器、消费在蒸馏器处丢失

- 定义（认证语义）：stage1_teacher_gate.py:558-566（R23）——认证的目标分布
是 `q ** train_temp` 归一化，gate 自家软臂权重即按 tau=train_temp 计算
（:560-565）；:833 声明 certified_target 同此；:511 以
`_tempered_gain(cfg["train_temp"])` 计增益。默认 1.0（:808）。
- 记录：k0_annotate.py:178 把 train_temp 写进 teacher_config，随发布进
meta（publish:537 一带）；标注器存的是未 temper 的 q 与 priors，其
docstring（:43）明说 temper「可由产物导出」——即刻意留给下游。
- **丢失点：k0_distill.py 全文零处引用 train_temp。** construct_target
（:161-181）从 Q＋priors 以 q_temp=0.05 重建目标，得到的恰是
train_temp=1 的未 temper 分布；训练 target 在:407 由此构造，存档 q
（q_stored，:340）仅作一致性/填充检查、且其本身也未 temper。
- 后果：任何以 train_temp≠1 认证并标注的代（评审实例 =2），S 臂实际训练
分布是 q¹ 而非认证的 q² 归一化——更平、与认证分布系统性偏移；默认 1.0
时恒等，故全部 fixture（走默认）从未显形。

归属：工程组（k0_distill 消费侧）；定义/记录两侧无误。
建议修法一句话：蒸馏入口读 meta 的 teacher config，对构造目标（或存档 q）
施加 `q ** train_temp` 归一化，train_temp 缺失/≠记录值时 die 点名。

## 3. X03 —— k0_annotate 恢复分片：验了分片自身，没验回放字节与生产代码

恢复路径：main:771-795 读 progress.json → `_verify_shard`（k0_annotate.py:
382-424）逐片验后跳过。已验：分片文件 sha256 vs progress 记录（:391-396）、
内嵌 _meta 自洽、逐数组 sha（:405-408）、根数/根键 vs plan（:410-413）、
gid 列（:415-416）、run identity 全等（:417-420）、selected_sha256（:421-422）。

没验的两项正是 X03 所指：

- **回放内容**：run identity（build_plan:266-273）只含 games 列表的哈希
（`games_sha256` = 局名集合的哈希）与 selected 根集哈希；per_game 只记
`path`（:256），**回放文件字节的 sha256 从不记录、恢复时从不重算**。
plan 重建时 enumerate_roots（:192、:248）确实重读当前回放，但分片校验
只比根键集合——回放被换成「根集不变、内容不同」的字节（同局长/同终局
结构、不同着法）时分片照常验过跳过，旧内容算出的 Q/q 被静默沿用。
- **代码版本**：code_md5s()（:185-189）只在 publish 时算进代 manifest
（:563-567）；**run identity 与 _verify_shard 都不含任何代码 md5**——
换一版 k0_annotate/stage0_teacher/gate 代码（配置/权重/窗口/种子不变）
恢复，旧代码产出的分片全部验过跳过，新旧代码产物在同一代里混封。

归属：工程组（k0_annotate）。
建议修法一句话：plan identity 增 per-game replay sha256 与 code_md5s，
_verify_shard 逐项比对后再跳片。

## 4. X04 —— W02 范围之外「先 hash、后按路径重开」同型面（4 处坐实）

1. **k0_annotate.load_annotated（:724-731）**：verify_generation（:653-706）
已逐字段验哈希**并把数组载入内存**，load_annotated 却丢弃验过的数组、
对每个字段再次 `np.load(路径)`——验 A 消费 B 的教科书形态，且修复成本
最低（直接返回 verify 的数组即可
...[truncated 5016 chars]