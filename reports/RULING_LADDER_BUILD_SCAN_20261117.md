# 裁定：天梯语料 build 扫描口径注册（ladder frame 模式）（2026-11-17，策略组）

事由：DGX 实测现行 cnet2_build._scan 对天梯语料全数 no_teacher_side（真 corpus 0/182、试点 200 件 0/200）。策略组亲读钉版源码（cnet2_build e0051524／cnet2_feat 2a45b08b／k0_join 27c11e16）后裁。

## §1 定性：BC 时代的筛子，套在 seat-complete 的消费者上

_scan 非 ours 分支以「边在 TEACHERS 名册内」定 sides，本是 BC 建集的语义（只收教师边的样本）。但现行消费者 k0_join 的契约是 **seat-complete**：annot 根按（局，步，seat）双边全覆盖（198 根＝99 步×2 seat），join 按 build 代的 sides 切 seat 行段，build 没产出的 seat 当场 die。selfplay 语料两 seat 同为名册生产者、sides 恒 [0,1]，矛盾从未显形；天梯局双边皆非名册，sides 空集、整批归零。另亲读确认：_producer_provenance 对普通天梯回放有现成路径（kind="ladder"、不打开生产者记录），出处层不构成阻碍——唯一堵点就是 sides 判定这一处。

## §2 三选项处置

- **改名册（往 TEACHERS 加天梯名）：驳回。** cnet2_feat（2a45b08b）在考试钉版集、§6.4 代码集内，改一字全线身份漂移；且名册是「教师身份」断言，把 Carago／Krypton 等对手登记为教师是语义造假，天梯名还会随赛季 churn。
- **另设构建器（build_top）：驳回（对 join 路径）。** build_top 产物是 BC 集，没有 k0_join 验的封印代契约（manifest／meta 封印／producer 绑定），不 join 兼容；其 BC 线用途不变。
- **另设扫描口径：采纳**，按 §3 合同实现。

## §3 实现合同（cnet2_build 单文件，DGX 施工）

1. 新增显式模式 **CNET2_LADDER_FRAME=1**（默认关；按 CNET2_WEIGHTED 的现行 env 惯例在 main 读取、下传 _scan）。模式开时，非 ours 分支的 sides 判定改为：通过其余筛子（steps／duplicate／missing）的局 **sides=[0,1]**。ours 分支、weighted 分支、min_steps、val 切分、provenance 路径逐字不动。
2. 模式与 CNET2_WEIGHTED=1 同开即 die（加权口径未注册、不用）。
3. teacher 列走现行 teacher_id 回退（非名册名→"other" 索引），**cnet2_feat 一字不改**（交付时复验 md5 2a45b08b）。
4. 模式状态写入代 meta 与 gen_key（如 ladder_frame: true），使 ladder 代与标准代自描述可分、不许混认。
5. diff 只许落 cnet2_build.py 一个文件；新 md5 随组装回执报策略组，§6.4 派单前须入仓（仓内版为准）；试点本轮可先用 DGX 工作副本（组装回执记 md5，试点级）。

## §4 验收闸（四道，全过才许进 join／派单）

1. **试点 200 扫描**：ladder 模式下 200/200 入选、sides 全 [0,1]、filtered 中 no_teacher_side=0；build 日志 games／samples／teacher_sides 计数入组装回执（沿用索引验收口径：以 build 日志计数为准）。
2. **默认关回归**：模式关时对 shard1 selfplay 语料目录跑 _scan，入选局集与 sides 须与生产代 gen-5f47630f 在册记录逐项一致——证默认路径未被触碰。
3. **join 双闸**：真实试点产物照组装裁定 §3 第 3 条（k0_join --verify＋k0_distill load_annotation），seat 双边行数与 annot 根逐局对齐（200 局×198 根）。
4. **§6.4 采用条件**：派单落字时引用入仓版 build 文件 md5＋本裁定号；§6.4 正式批（同池约 1,400 局）开工三件套含闸 1 同型扫描计数（全池口径）。

## §5 适用范围与不变项

本模式只服务「天梯快照语料→annot join 帧」这一用途（试点 200、§6.4 Stage B 池、后续同形语料）；BC 建集、ours_corpus、Stage-2 等既有 build 用途照旧走默认口径，任何人不得以本模式之名改其产物。annot 侧、拼接、训练配方、§6.4 其余口径均不变。实现与闸证据由 DGX＋L20 回策略组；试点拼装时点（约 03:15–03:30，估计）不因本裁定改，build 一步按本路径接续。
