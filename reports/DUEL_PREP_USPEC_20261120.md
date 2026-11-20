# 价值谱 U′ 决斗备件清单（2026-11-20，执行人备件——未开场）

本件只备开场条件，不开场。决斗场只在 NETEXP A 臂落地后才开（作战室派单明文）。

## 1. 决斗口径（设计 §7／预注册 §2–§3 钉死）

- 对阵：U′ 臂对 NETEXP A 臂（对照成立条件下；A 臂发车登记须读 train2＋fpw 1.0，否则借用即作废）。
- 200 局配对，种子块 **968600–968799**（预注册 §3 登记 registered-unused，与 968000–968399、968400–968519 及全部在册占用零重叠）。
- side＝seed%2（bigeval 内置：tasks＝(tag, a, b, s, s%2)）。
- 读数：bigeval 配对读数（paired ΔU＋在册不确定性口径误差棒）。
- 判读（事后检验，10:26 所有者改令）：胜出且超误差棒 → U′ 留任（已进 Gen-5 则坐实、未进则 Gen-6 采）；判空或为负 → 保留现行 U、如实报空，不许次级证据翻案。

## 2. U′ 臂（已备）

- 检查点：uspec300.npz，md5 8879aaef8a808f409d45f1de02175159（2,101,008 B；正式训 12:51:09 完训，rc=0）。
- 墙钟包：`/home/arm/valuespec_uspec/pack/uspec300_wall.py`，md5 **2ed4c1e6ec495ed85cb4656abbdc1de5**（2,867,315 B；打包器 pack_submission_uspec.py ad76e287＋底盘 apexcarbon_v1_fix3.py 4b3423a0＋U′ 桥）。
- §7 复验状态（本轮实测，四项全过）：① 前向 err 0.0000（容差 0.25）PASS、确定性 1 局×2 跑 IDENTICAL（终局现金 [8391.62, 7937.5]）PASS；步时 mean 0.783s／p95 1.666s／max 3.031s（0.75s 线）只报数、不参闸（0.75s 是天梯单步时限；决斗走线下 bigeval、不上天梯）；② 桥注入 verify PASS（f90af57d 口径，含 --packed 双线：packed twin 2.020e-09／1e-6、packed _u_of_margin 0.0／1e-12；退役点 0.9999999 精确复合误差 0.0）；③ perf_counter 0 处 PASS；④ 三档原值逐字合（_S_STAGES ((2,0.45),(5,0.22))、_S_D2 9、_S_ROWS 32，与底盘同行号）。
- 桥口径：包内注入 U′ 桥（bridge-def -2/+32 行）。

## 3. A 臂（未落地——开场唯一阻塞项）

- NETEXP A 臂检查点：不存在。DGX 镜像 `work0/netexp_ab/` 只有 seal_manifest.done＋seal_manifest.md5（封存集 3,575,143 行／2,788 片）；无 A 臂训练产物。
- A 臂墙钟包：不存在。待 A 臂训练（L20，Gen-4 后接档；现 L20 在跑 Gen-5 训练）完训后，以现行-U 打包器（pack_submission.py 57c75d84，--value-scale u）打包＋自验，方可列入。
- A 臂发车登记待核：须读 train2＋fpw 1.0（设计 §3 附条件），否则对照借用即作废、须另跑独立对照臂。

## 4. 开场命令形（待 A 臂落地后由作战室下令才执行）

- 工具：bigeval.py（DGX `/home/arm/junyang/carbon/code/bigeval.py`；用法 `bigeval.py A.py B.py SEED0 N TAG`，结果追加至 `../data/bigeval.jsonl`）。
- 形如：`python3 code/bigeval.py <U′ 包> <A 臂包> 968600 200 uspec_duel`（双包先后顺序与 ΔU 符号口径在开场时钉死并记入开场登记；本件不预设符号方向）。
- 座位：待作战室定（DGX 为自然候选：U′ 包与 bigeval 均已在位；L20 现跑 Gen-5 训练）。
- 开场前核对单：§7 四项全过（① 待 --verify 落定）∧ A 臂包存在且自验过 ∧ A 臂发车登记合附条件 ∧ 种子块未动用 ∧ 座位 GPU 空闲。

## 5. 本件不含

- 不开场、不跑任何一局决斗、不写 bigeval 记录。
- 次级读数（51 局归因）只作机制证据，不作判读、不得升格（预注册 §2）。
