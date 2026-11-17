# 验合：v3 消费登记＋Tier 2 新切片（2026-11-17）

署名：策略组。对象：DGX 落字的 v3 登记四件与 Tier 2 新切片一件。
方法：仓内原件 API 读回，策略组独立复算（非采信 DGX 自报）。

## 验合结果：PASS

**身份（全长 md5，与 DGX 报值逐字合）**
- data/top5_pool_v3_union.json（682）＝1a396cc87c8f1163ca4d3f9323d429cd
- data/top5_pool_v3_test.json（170）＝daf8ec731de7dc11fa65b94f681544cb
- data/top5_pool_v3_train.json（512）＝9ce38db38e45dced6e1641739573e920
- data/top5_pool_v3_overlap_report.json＝3f369f78b206da8973b601f57c253b59
- data/small_teacher_tier2_slice50_v3.json＝acffb20ab2451d821bdb74fb21b1e831

**划分复跑（钉版算法，策略组独立执行）**：并集 682 局全唯一；id 字符串
升序＋random.Random(20261117) 洗牌、前 170／余 512，与入仓 test／train
**逐元素全等**（非仅集合等）；test∩train＝0、test∪train＝并集。

**零重叠（局号层，策略组自旧件原件复算）**：旧 train 件 md5 f7b50267
（1,202 局）、旧 test 件 fd4cf144（323 局）均与在册全值合；并集与旧
train／旧 test／旧切片 50 交集**全 0**；新 train∩旧 test、新 test∩旧
train 亦 0；附卷两局 994400／994401 不在并集。试点登记余 60 局按裁定
分区算术在旧 train 内、继承零交集（报告口径，与裁定 81168a1e §1/§5 合）。

**内容层（抽验）**：并集件自带 682 个 replay_md5，与旧切片登记的 50 个
replay_md5 交集 0；新切片 50 个 replay_md5 ⊆ 并集、与旧切片交集 0。
报告的全局双射验（30 份 manifest、4,918 局）以报告所记为准入档。

**附卷记账**：Master5s 145 ＝ ∩新 test 45 ＋ ∩新 train 100，两组 id
分别 ⊆ 对应划分件、彼此不交，与修订件 de996c12 的纯记账口径合。

**Tier 2 新切片（独立复算）**：新 train 升序、剔除已注册消费集（对 train
实际生效的剔除＝快线 Master5s 的 train 侧 100 局，余 412）取前 50，
与入仓件**逐局全等**；切片 ∩ test＝0、∩ Master5s＝0、⊆ train；
逐局 leg1_partition_overlap 标记在位（新腿一登记范围＝train 全 512）。

## 效力

- 换池裁定（81168a1e）开工链**第③项（v3 登记验合）达成**；第④项
  （补仓入仓验回）的入仓面由本验合覆盖（并集逐字复算恰 682、五家
  补拉回执已齐）。腿一仍待第①②项（切片正式裁定＋所有者确认）。
- Tier 2 新切片 **acffb20a 自本件起为现行切片凭据**，旧切片 8f946da7
  作废确认。小教师数据生成闸的切片条件满足；生成开工仍须新派单
  （停令在效、无新令不复工），座位排快线之后，数据源为新 train 512。

落字：策略组，2026-11-17。
