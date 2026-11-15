# KataGo 线底座任务书（foundation_taskbook）

所有者 2026-11-13 05:30 定：先把底座搞好、工作流打通，再谈规模。
本任务书覆盖全链路七环：每环的输入/输出、现状、验收数字、执行机器与唯一
执行人、通道依赖。执行总纪律不变：单一驱动（作战室或其派出的执行代理；
分战区边聊是汇报线，spot 机的考试执行归其分战区边聊）、提交闸归所有者、
断言「在跑」必须三件套、建集＋开训必须同一条脚本链走完、三者计数断言。

本地已验证据（2026-11-13，foundation_stub/）：产局器定版在 SPGEN_STUB=1
下跑 6 局（seeds 991000–991005）→ corpus → 以 build_top 等价口径（本地
wrapper 复刻 CNET2_TOP_TEACHERS="m3s2r1@2" 覆写 TEACHERS）建出 3,588
样本（598/局＝299 步×2 边）、teacher_sides {m3s2r1: 12}、零报错。注意：
本地建集用的是 scratch/fix-c/test8 的 cnet2_build/cnet2_feat/fastsim/
eval_v53 快照副本（N03 之前的 scan 语义）；上机一律以各机 main 克隆为准。

---

## 环 1 · 产局器（selfplay_gen.py）

- 输入：权重 npz（裸网，SEARCH 关、13 键 CONF 逐字）、种子域、双边名字
  （须与建集时的 CNET2_TOP_TEACHERS 一致）。
- 输出：corpus 分片目录（index.json＋replays/*.json.gz），可多分片合并。
  每局带 raw margin＋U（定版新增，见下「现状」）。
- 现状：**定版完成**（hidden_files/selfplay_gen.py，md5
  b275e952da14d644e6fab41e4ce941c8；原稿 selfplay_gen.py.bak，
  md5 0767d73e…）。定版内容：index 条目新增 "margin"/"u"（按策略键，
  margin_b＝−margin_a，U＝0.5·W＋0.5·sigmoid(M/1000)）；replay 文件新增
  顶层 "result"（按玩家边存 scores/margin/u/winner_side/a_side）。
  steps 约定（entry t 的 obs.step==t、动作错位一格）与灭绝口径未动，
  本地一致性断言全过（margin 对消、U 公式逐位、index↔replay 对齐）。
  未验：真实权重路径（本地无 cnet2_search 真实现＋权重），即探针趟本身。
  ⚠️ 连带缺口：sp-probe-s29.sh / sp-probe-studio.sh 内嵌的是旧版 b64
  且硬校验旧 md5 0767d73e…（s29 第 28 行、studio 第 53 行）——定版后
  两脚本必 FAIL（script-md5-mismatch），上机前必须先本地重嵌＋改 md5
  期望（此步不需要通道，可先做）。
- 验收数字：SPGEN-RESULT errors＝0；index 条数＝replay 文件数；抽查
  ≥3 局 margin/u 存在且 margin_a＋margin_b＝0、U∈[0,1]；entry0 动作 {}
  且 obs.step＝0、entry1 obs.step＝1。
- 执行机器/执行人：产局工兵＝spot-29（C1 线执行其机）、Studio、DGX；
  每趟唯一执行人＝作战室派出的执行代理（脚本内嵌直发，不经汇报边聊）。
- 通道依赖：**是**（云助手通道；spot 机另有控制台网页云助手应急路，
  但那是浏览器线的窗口，非必要不动）。

## 环 2 · 数据格式 / 建集（cnet2_build ＋ build_top ＋ set_uw）

- 输入：一到多个 corpus 分片目录；CNET2_TOP_TEACHERS="名字@行"。
- 输出：数据集目录（.npy 字段阵列＋meta.json＋shards/），随后 set_uw
  1 1 1 写入 uw。
- 现状：格式已验（本地 6 局→3,588 样本，meta.games 与 index 条数
  一致）。**已知边界（本轮按令不改 build 主逻辑）**：dataset 分片里
  仍只有 value＝tanh(M/3000)，没有 raw margin 字段——scan() 只读
  index 的 {file, order, strategies, scores, steps}，build_game 只读
  replay 的 d["steps"]，margin 由 scores 现算后即压成 tanh。
  dataset 级 raw margin 的定点改动（三处，留待所有者/Stage-1 立项后动）：
  ① cnet2_build.py 的 FIELDS  dict 新增 "margin": (np.float32, ())
  （如需可同时加 "u"）；② build_game 里 r["value"].append(np.tanh
  (mg/3000.0)) 同处加 r["margin"].append(mg)（mg 已按边传入，零新计算）；
  ③ 消费端改读：stage0_smoke_heads 的 head_losses 以
  mg = 3000*atanh(b["value"]) 反推，改为直读 b["margin"]；正式 trainer
  的 E[U] 标签同理。注意 ① 会改变 _cfg_fingerprint（fields 在指纹内），
  全机 shard 缓存一次性失效——已知的计划内成本，须与 N03 一并排期。
  在此之前，E[U]/margin 的权威记录在 corpus 层（环 1 已保证可读回），
  存量数据集继续走 atanh 反推（|M|≳11400 饱和，尾部失真已知）。
- 验收数字：meta.games＝corpus index 条数（差 0）；total samples＞0；
  teacher_sides 与预期名字/行一致；val_games＞0（局数≥48 时）；三者
  计数断言：replay 文件数＝index 条数＝meta.games。
- 执行机器/执行人：DGX（全队建集枢纽），唯一执行人＝作战室执行代理。
- 通道依赖：是（DGX 只能经云助手通道到达）。

## 环 3 · 三头模型头 ＋ 旧权重加载补丁（stage0_heads_patch.py ＋ stage0_smoke_heads.py）

- 输入：机器上 main 克隆的 code/cnet2_model.py、cnet2_train.py；旧
  权重 npz（如 cnet2_m3s2r1L.npz）；存量数据集 ds_m3s2。
- 输出：模型新增 utility（E[U] 主头，sigmoid∈[0,1]）/ wdl（3 logits）/
  margin（E[M]/1000）三头；trainer 的 BC_INIT 缺键放行名单扩到三头；
  smoke 报告（loss 曲线、val 指标）。
- 现状：两件草稿就绪、py_compile 通过。补丁是精确锚点替换＋幂等标记
  ＋锚点计数断言（≠1 即非零退出，不留半成品）。**唯一未验项**：锚点
  字符串与 main tip 现行 cnet2_model.py / cnet2_train.py 是否仍逐字
  一致——本地无 main tip 的这两文件副本，上机第一步先 grep 锚点再跑
  补丁。smoke 必须在补丁之后跑（依赖 forward_heads）；其标签走 atanh
  反推（存量 ds 口径，设计内）。
- 验收数字：补丁输出 PATCH-OK（utility∈[0,1]、wdl [B,3]、margin [B]）；
  旧权重加载的 missing keys 恰为三头 6 键（weight/bias×3），无其他缺键；
  smoke：finite＝true、l_u 后 50 步均值＜前 50 步（decreasing＝true）、
  val wdl_acc 明显高于 0.333 随机线、val margin MAE 有读数、U 范围⊆[0,1]。
- 执行机器/执行人：DGX 首选（cuda），Studio 备选（须 BC_DEVICE=mps）；
  唯一执行人＝作战室执行代理。
- 通道依赖：是。

## 环 4 · 根候选教师 v1（stage0_teacher.py ＋ stage0_measure_teacher.py）

- 输入：裸网权重 npz；真实对局位置（测量脚本自产 2 局自对弈取 12 位）。
- 输出：每位置的候选臂、先验 P_A、Q（U 尺度）、软目标 q、熵、与裸网
  greedy 的一致率；成本汇总（ms/步、折算每局教师秒数）。
- 现状：草稿就绪、py_compile 通过。已自带文档的 v1 简化：场景＝对手
  动作采样＋续局 RNG（belief.py 未接）；分支 Feat 历史清零；终局叶
  用现金差近似（未加 F01 灭绝罚分）；H 的口径＝根步之后再走 H 步
  （总深度 1＋H）。成本测量足够，语义简化留 Stage-1 处理（见缺口清单）。
- 验收数字：≥12 个位置的 ms 均值/p95/max 有读数；每局教师成本
  ＝ms 均值×（环 1/吞吐实测的步/局）折算成秒并写进报告；熵与
  agree_greedy 分布有读数（质量闸阈值是 Stage-1 的预注册事项，
  底座阶段只要求数字存在、可复现）。
- 执行机器/执行人：DGX（CPU 为主），唯一执行人＝作战室执行代理。
- 通道依赖：是。

## 环 5 · 训练单链脚本（建集＋开训同链、三计数断言）

- 输入：corpus 分片（环 1 产出或语料拉取）、训练配置（BC 环：BC_RAM=1
  BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1
  BC_LR=3e-4；注意力版 cfg 加 ZI:1,TA:4,TNH:4 与 BC_INIT）。
- 输出：训完的双臂权重（或 Stage-1 的单臂＋三头版）＋链内自证块。
- 现状：**本环是七环里唯一还没有脚本本体的**。模板在 main：
  code/replica_chain.sh（md5 07fa5e91…，三条防复发规矩已固化：
  单链走完、三计数断言、完工＝三件套＋自证块）。KataGo 环的链与
  复刻链只差第一步（产局/拉语料不同），本地可先写：
  kata
...[truncated 5011 chars]