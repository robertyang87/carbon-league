# 跨 config 混编可行性确认（工程组，2026-11-18）

- 问：环 3 步段草案（策略组 04bce102）的防遗忘混编——本段全量分片＋前序各段按根抽 20% 分片、各段窗口不同（config_sha256 不同）、同标注器同教师权重——现行 merge/build/join_u 链能否合法混编。
- 读件：`code/k0_join.py` fac55acb、`code/k0_join_u.py` 9054aadc、`code/k0_distill_u.py` 06826f59（均仓上现行版直读）。

## ① 结论：现行链不能，会 die 在两道具名闸

1. **记录闸（窗口容纳）**：join 与 join_u 的 `_validate_annot_records` 要求**每一根的 step 落在该代单一注册窗口内**（`step < lo | step >= hi` 即 die「step X outside the registered window」）。混编代无论按哪一段的窗口封印，其余段的根都会死在此闸；消费端经真实消费者（k0_distill.load_annotation）复验时同闸再拦一次。
2. **身份闸（标量 config)**：join/join_u 只接受**一个**标注代输入，产出的 joined 代 producer 只带**一个标量** `teacher_config_hash`（join_u L645 取自唯一标注代）；k0_distill_u 要求 producer.config_sha256、producer.teacher_config_hash、meta.teacher.config_sha256 **三处同值**（多于一个不同值即 SystemExit 拒训）。「一个 joined 数据集＝一个 config 哈希」是构造性的，无现行路径可绕。
3. 附带事实：现行任何 publisher 都封不出混编代——publisher 的 config 来自单次运行 args；环 1 三批合并之所以合法，是因 annot_a/b/c 三批 config 逐字同（3cc2e67e），不在此例。

注：窗口只是**选根参数**、不进任何根的计算——同根在不同窗口下产出逐字节同（batch-2 与双块探针均有窗口变体根对存档恰 0.0 的实证）。混编的障碍纯在身份/封印层，不在数值层。

## ② 最小合规改法：新增「合并 publisher」于标注代层（钉版与已验收件零改动）

新工具（工程组新件、独立身份）`merge_annot_generations`（暂名），输入 ≥2 个已封印生产代：

- **前置校验（逐项 die 点名）**：各源代 code_md5s 集全等、weights_md5 全等、config 字典逐字段比对**仅 window 一项可不同**（其余字段任一不同即拒）；各源代先经现行 loader 全套自验（封印/meta 一致/字段哈希/记录闸按各自窗口）。
- **合并**：games 表按名取并集、gid 统一重排；根按 (gid, step, seat) 拼接并去重——重叠根必须逐字节全等（窗口不进计算，不等即数据异常、die）；20% 抽样以确定性预注册采样器在计划层执行、逐源根数入册；合并窗口＝各源窗口并集（步段互不交叠时即分段并列）。
- **封印（诚实来源）**：新代 manifest/meta 记合并 config（并集窗口）＋派生 teacher_config_hash，并带 `merged_from` 来源清单（逐源：代 id、manifest sha、其 config_sha256、入选根数与步带）。**不许**把合并 config 伪装成单次运行的 config；合并身份是一个**新注册身份**，其语义（「同标注器同权重、仅窗口不同的源代之并集」）须由策略组裁定入册后方可用于生产代。
- **为何此层可行**：下游全部校验是自指的（meta↔manifest 一致、step∈本代窗口、三处哈希同值）——一致封印的合并代可零改动通过 join/join_u/distill_u；build 侧按局名匹配、gid 空间独立，不受影响；terminal-u 边车路径不经标注 config，不受影响。
- **被否备选**：(i) 在 joined 数据集层拼接——同一标量身份墙，且要改已验收的 k0_distill_u；(ii) 扩 ANNOT_FORMAT 记逐根 config——全部 loader 波及、爆破半径最大；(iii) 手工拼 out 目录按伪造的单次 config 发布——来源造假，纪律不许。

## ③ prescan 与 --verify：照常成立

- prescan（坏根预扫）在 joined 代上以训练器判据逐根扫，与标注来源无关——合并代经 join 后是普通同质 joined 代，照常成立。
- `k0_join --verify`（独立复核模式、经真实消费者 load）同理照常成立。
- 附加要求：合并 publisher 自身须带发布前复验——合并代先经现行 `load_annotation_source`＋记录闸全套加载通过才许封印（镜像 join_u「发布前经真实消费者验」的既有纪律）。

## 待裁

- 合并身份的语义与引用方式（策略组）：合并代的 teacher_config_hash 在训练 provenance 中如何引用其 `merged_from` 清单。
- 20% 抽样的确定性采样器口径（随环 3 步段草案定版时钉）。

—— 工程组 2026-11-18
