# TOP4 冻结补报 + v4 切分落字（2026-11-17）

署名：DGX 线。依据：TOP4 裁定 8f49d758、作废局裁定（8759f821 所有者令）、
全窗修正 eb64d2e6（c82add8d）、策略组冻结补报要求（2026-11-17 15:17 / 15:34 锚定口径）。
冻结锚：快线 147 局落盘完成（2026-11-17 15:58）；四家清单于 16:31–16:32 在 DGX
经 117 端点直拉（ladder.py matches），本补报与 v4 四件为唯一验合依据。

## 1. 最终池 P

- 冻结清单条目：Master5s 153 / it720 83 / late1 159 / v8 163。
- 快照并集（事实包 13:17）456；冻结并集（互打去重、作废局硬排除后）**P = 505**。
- 增量 = 50 局（家族口径：Master5s +1、it720 +37、late1 +11、v8 +13，互打去重后 50）。
- 增量排除复核（逐项直读）：旧 train 1,202 → **0**；旧 test 323 → **0**；
  试点已分区 140（⊂ 旧 train）→ **0**；K0 消费登记 v2 200 → **0**。
- 增量完整性 SOP（audit_integrity 直读）：**50/50 满局**（300 条目、双 DONE、
  终局奖励在位），短局 0、异常 0。增量回放 50/50 已拉回 DGX
  （k0_dist/freeze_v4/incr_replays）。

## 2. 作废局

8759f8217d439cc5127f10aa2139935a-1 在冻结 Master5s 清单中在册，已按硬排除
移出池与切分输入；快线语料维持 146 局口径不变。

## 3. v4 切分

算法：game-id 字符串升序 → random.Random(20261117) 洗牌 → 前 170 为 test。
结果：**test 170 / train 335**。四件（md5）：
- data/top4_pool_v4_union.json a7e4fbaf（505）
- data/top4_pool_v4_test.json 6c5b7059（170）
- data/top4_pool_v4_train.json 696060d7（335）
- data/top4_pool_v4_overlap_report.json f9388577（增量/排除/残余计算原件）

## 4. 先行标注最终形状与残余（全窗修正后口径）

- spot-48 = 60 局：尾窗产物已落（config 9a2b2440），**留证排除、不入 join**，
  60 局按全窗重标（排入 49–52 波次，每台主批后 +15）。
- spot-49~52 = 248 局：部署完成后按全窗（config 3cc2e67e）开标。
  **四台尾窗 shard 确数 = 0**（从未部署、从未开标）。
- 残余 ① |v4 train ∩ 快线 146| = **94 局**：二态记账——收编（待 eb64d2e6 §3
  六闸：身份逐项比对、1 局试合并 RC、1 局重标抽验、provenance 入册）或全窗补标。
- 残余 ② it720 增量 37 局，其中 |增量 ∩ train| = **23 局**：全窗补标。
- leftover fe00511b3e01a0be3d8b89a87c27f0ae-1：**落 train**，全窗补标。
- leg-1 步数预期：按全窗 join 实测根数×5.17/128（修正 §2 估计约 7,000–8,000 步）。

## 5. 全窗 config 复核（DGX 复算，双证）

钉版 k0_annotate teacher_config 默认参数复算：全窗 [0,299) =
3cc2e67e66cff9a1d901b663a579abc2d62246d6bdf76017f00bcf998aea1ff3，
与快线产物 progress.json 内 config_sha256 逐字一致。尾窗复算 =
9a2b2440e9267680c9b27a1d51a50db9cacdbd04582ae101ef54abaa86d8d10c（备查）。
