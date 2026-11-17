# 验合：Tier-2 切片 v4（小教师凭据）（2026-11-17）

署名：策略组。对象：DGX 落字的 data/small_teacher_
tier2_slice50_v4.json 与标注注册表回填件。方法：仓内
原件 API 读回，策略组独立复算（非采信 DGX 自报）。

## 验合结果：PASS

**身份**：切片件 md5 476a0f442933d94418c5620d5a8f1727
（commit e0db1f68），与 DGX 报值逐字合。注册表回填
data/annot_registry.json md5 527b7007ec0ea698ebc2
cc30fde41588（commit 7fb7a6d7）：207 条＝fastline147
147＋leg1_pre60 60，status done 206／voided 1，与在册
事实全合（快线 146 的 product＝合并代 gen-1f1aa3c7；
pre-60 尾窗产物随 spot-48 回收丢失、仅存证，全窗重
标排后续波次——注册表逐条已注）。

**构造复跑（v3 同构造，策略组独立执行）**：源＝v4
train（696060d7，335 局）升序；剔已注册消费集（自
注册表原件提取）——快线 146 ∩ train＝**94**、
leg1_pre60 ∩ train＝**40**（旧 train／旧 test／K0
登记 ∩ train 均为 0，冻结验合 7a782bc7 已证）；候选
**201** 局，取前 50 与入仓切片**逐元素全等**（首
0086c374、尾 593d31a0，与 DGX 报值合）。切片 ⊆
train、∩ test＝0、∩ 快线＝0、∩ pre60＝0；
leg1_partition_overlap 标记 50/50 在位。

**replay_md5 抽验**：43 局有仓内家族语料者与四家
manifest（master5s／it720／late1／fablev8）逐局
全合、零分歧；7 局 file＝null（冻结增量、未入仓语
料），其 md5 出自 L20 交付清单（7dc459b4，L20 侧
335/335 md5sum -c OK）——待该清单入仓后随 train 集
确认一并终核（尾项、非阻塞）。

**备案核对**：v3 切片 ∩ v4 train＝19、∩ v4 切片
＝0，与 header 所记逐数合；v3 切片从未消费，其不作
排除集的理由（未消费即非消费集）成立。

## 效力

- 切片 v4（476a0f44）**自本件起为现行 Tier-2 凭据**，
  代替已作废的 acffb20a。小教师 50k 标定批的凭据条
  件满足；生成开工仍须作战室新派单（停令至派单为
  止在效），数据源＝v4 train 剔本切片 50 局（切片局
  不入蒸馏训练数据，两路皆然）。

落字：策略组，2026-11-17。
