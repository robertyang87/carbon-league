<!-- 落仓头（工程组，2026-11-20）：执行人报告原件逐字转录（本地 md5 0b21367c0cb4f508d395e23281175a74，DGX /home/arm/junyang/vec2_ablation/）；主代理独立核验见 reports/VERIFY_ABLATION_VEC2_20261120.md （187 s、D1、vec2 跨跑确定性均经 DGX 直读复核；判读：旧线 FAIL／拟线 0.095 PASS，待策略组核准）。 -->

# vec2 闭环消融报告（策略组裁定 a33b73c6 第一步）＋ G4 零基线与 mean_margin 线重定提案（裁定二，重开件 c4ecdf98）

执行人：工程组（DGX，spark-29db）。座位 DGX。执行人不落仓；主代理独立核验后落仓。
口径变更登记：**闸程序修订、非改线**（见 §2）。

## 0. 三件套

- 开工：2026-11-20 10:51（侦察）／三跑驱动 10:58:43 启动。
- 预期完成时间（估计）：约 2.5 小时（侦察＋harness 约 40 分钟；三臂实跑约 10 分钟；两闸＋核对约 25 分钟；诊断约 30 分钟；报告＋本地副本约 30 分钟）。
- 实际完成：2026-11-20 12:02（D1 收尾 11:53:38，报告定稿 12:02）。三臂实跑因争用慢于预估
  （A1 447 s、Z2 921 s vs 干净口径 187 s），诊断因 GPU OOM 改 CPU 口径多花约 20 分钟；
  总耗时仍在 2.5 小时估计内。
- 进度口径：检查点 ① 侦察完成 → ② 三臂分片落地 → ③ 两闸跑完＋重定推导 → ④ 诊断小结 → ⑤ 报告＋本地副本。失败从最近检查点续跑。

## 1. 结论先行

- 消融闸（匹配对手口径，旧线 0.05）：**gate_pass=False**，唯一不过腿 mean_margin **0.057357＞0.05**（B 段 0.085346）。其余腿全过。
- G4 零基线实测 m0 = **0.025156**（vec1 对 vec1，gen4＋hof，基底 968400／968460），
  分层局层 bootstrap（20,000 次）零分布 q95 = **0.094909** ⇒ **拟重定线 0.095**
  （3 位小数，KS 线先例口径；解析正态交叉核对 0.070——重尾下低估，仅作参考）。
  消融 mean_margin 0.057357 在拟重定线下**过**（闸以 --mean-shift-line 0.095 重跑，
  gate_pass=True，rc=0）。**最终判读待策略组核准拟重定线后定；核准前旧线判读（FAIL）仍为在案判读。**
- 对手序列已证匹配（§3）：两臂逐局对手指派表与预抽表逐字合；且 B 段两臂同样已匹配——**hof 抽样口径差假说在代码层＋分片指纹层双重证伪**，均值移是两实现固有的分布差，非对手混合差。
- 最终判读待策略组核准（§5）；**线只在策略组核准后才变，测量本身不自动改线**。
- Gen-5 生成维持 vec1 不动（裁定 a33b73c6）。

## 2. 口径（闸程序修订、非改线）

B 段生产口径逐项同：gen4 ckpt md5 42888eabe142d688de3e0110dcda48e2＋hof gen0/gen1r/gen2、
n=120、种子块 968400–968519、逐局种子 = 基底 + 1000003×gid、deep-frac 0、value-scale u、
threads 4、dtype bf16、device cuda、vec2 pool-games 24＋precision fp32＋batch-cap 2048、
vec1 workers 24。闸 gate_vec2.py md5 93557d15（commit cc07cece）。

**匹配对手口径**（本次新增，登记为闸程序修订）：同局索引用同一 hof 对手序列——
预先抽取一次（opp_table.py，见 §3），两臂同种子抽样，逐局核对指派一致。
修订只改“对手序列的产生与核对方式”，不改任何判读线、不改闸代码。

三跑：
| 臂 | 种子基底 | 生成器 | 用途 |
|---|---|---|---|
| A1 vec1_b968400 | 968400 | vec1（e59f0700） | 消融旧臂；兼零基线一臂（同基底、同对手序列、同口径——复用关系明写于此） |
| A2 vec2_b968400 | 968400 | vec2（3abdfc54） | 消融新臂 |
| Z2 vec1_b968460 | 968460 | vec1（e59f0700） | 零基线二臂（gen2 零基线先例：两臂基底 968400/968460） |

## 3. 对手序列：预抽、核对、假说证伪

### 3.1 预抽表

opp_table.py（md5 f170987fa1a099eb4997a1eef4ccee9c）按两实现逐字同的抽头复算：
`seed = base + 1000003*gid mod 2**32；rng = RandomState(seed)；r = rng.rand()；
r<0.5 mirror（598 行）／r<0.8 m3s2（299 行）／否则 hof[rng.randint(3)]（299 行），opp_seat = gid%2`。
本地 numpy 1.26.4 与 DGX numpy 2.5.4 复算 md5 逐字同。

- 基底 968400：mirror 66／m3s2 33／hof 21（hof0 6／hof1 9／hof2 6），预期总行 55,614。
  表 md5 d69736d32ec11dced2fd47d85feb9b92。
- 基底 968460：mirror 60／m3s2 38／hof 22（hof0 9／hof1 8／hof2 5），预期总行 53,820。
  表 md5 5ffbfaccd8098d79eaff14e54f40e6fb。

### 3.2 代码层：抽头逐字同（机械 diff）

`diff <(vec1 L433-456) <(vec2 L374-397)` 仅 def 行／docstring 行不同；
seed wrap、RandomState 创建、`layout.new_game_state`（只动全局 RNG）、
`r = rng.rand()` 及整段 if/elif/else 逐字同。**两实现在对手抽头之前不消耗任何 RNG**，
且抽头是每局 fresh RandomState 的第 1 次（r）／第 2 次（randint）抽取。
因此同（基底，gid）⇒ 同对手（类别＋hof 成员＋座位），与实现无关。

### 3.3 分片指纹核对（verify_opp.py，md5 6f2f8ae7d6a34e386945a8617568a47a）

原理：行内 sc[2] = step/300，sc[2]==0.0 标每 299 行一段的段首；一局 = 1 段
（m3s2/hof，只记 cur 座）或连续 2 段（mirror，两座各 299 行）；写盘以整局为单位、
buf≥1024 才切分片，局不跨分片。vec1 按 worker 轮询（wid 取 gid = wid, wid+24, …），
故每 worker 分片尺寸序列是表的确定函数——逐 worker 逐分片比对（强检查）；
vec2 按池完成序落盘，只核对总量＋段数＋每段恰 299 行（多重集检查）。

核对结果：
| 臂 | 结果 |
|---|---|
| A1 vec1_b968400 | VERIFY_OPP PASS（24/24 worker 分片序列＋段模式全合，总行 55,614） |
| A2 vec2_b968400 | VERIFY_OPP PASS（43 分片，总行 55,614，段 186，全 299 行） |
| Z2 vec1_b968460 | VERIFY_OPP PASS（24/24 worker 分片序列＋段模式全合，总行 53,820） |
| B 段 vec1_g4（复核） | VERIFY_OPP PASS（24/24 worker 全合，总行 55,614） |
| B 段 vec2_g4（复核） | VERIFY_OPP PASS（43 分片，总行 55,614，段 186，全 299 行） |
| B 段 armA（gen2、无 hof，复核） | VERIFY_OPP PASS（行数层同——无 hof 时 r≥0.8 落 m3s2，仍 299 行；指纹只验 mirror／非 mirror 分界，不验 m3s2／hof 成员） |

**假说证伪**：B 段两臂在分片指纹层已逐局匹配（vec1 逐 worker 逐分片、vec2 总量＋段数），
且代码层抽头逐字同——“hof 抽样口径差致均值移”不成立。B 段 mean_margin 0.0853 与
消融 0.0574 之差，是 vec1 臂自身跨跑分布抖动（§5 零基线正是在量这个），不是对手混合差。
指纹的能力边界如实记：行数指纹只直接验 mirror／非 mirror 分界；hof 成员身份由
“同种子＋逐字同代码＋指纹已验首抽”三段链锁定（首抽同 ⇒ 同 RandomState 状态 ⇒ 同代码下
第二抽必然同）。

### 3.4 逐局对手指派表

完整 120 局 × 两基底见附表文件 opp_table_968400.json（md5 d69736d3…）与
opp_table_968460.json（md5 5ffbfacc…）。摘要：968400 基底 mirror 66 局、m3s2 33 局、
hof 21 局；968460 基底 mirror 60 局、m3s2 38 局、hof 22 局。

## 4. 消融闸（A1 vs A2，旧线 0.05）

闸报告 gate_ablation.json（gate_md5 93557d15，rc=1 即闸不过）。全部腿：

| 腿 | 实测 | 线 | 判 | B 段对照 |
|---|---|---|---|---|
| margin KS D | 0.053763 | ≤0.176 | ✓ | 0.064516 ✓ |
| **mean_margin** | **0.057357** | ≤0.05 | ✗ | 0.085346 ✗ |
| win_rate shift | 0.010753 | ≤0.02 | ✓ | 0.010753 ✓ |
| searched_frac shift | 0.0 | ≤0.01 | ✓ | 0.0 ✓ |
| aux_delta KS h1/h2/h3 | 0.0111/0.0103/0.0119 | ≤0.05 | ✓ | 0.0099/0.0161/0.0173 ✓ |
| rows-per-game 相对差 | 0.0（463.45/463.45） | 0.01 REPORT-ONLY | 报数 | 0.0 报数 |

- mean_old −63.43（消融 vec1）vs B 段 −82.02；mean_new −25.19 与 B 段**逐字同**
  （−25.18693578884166，sd_new 654.4230393719249、win_new 0.4838709677419355、q_new 亦逐字同）。
- schema：old 49 件／55,614 行、new 43 件／55,614 行，problems=[]。
- 墙钟（争用态，见 §6）：A1 447.0 s（0.27 局/秒）、A2 167.1 s（0.72 局/秒）。
  闸腿是正确性比对，不受争用影响。

## 5. G4 零基线（A1 vs Z2）与 mean_margin 线重定提案

### 5.1 零基线闸（旧线 0.05）

闸报告 gate_zerobase.json（gate_md5 93557d15，rc=0，**gate_pass=True**）：

| 腿 | 实测 | 线 | 判 | gen2 零基线对照 |
|---|---|---|---|---|
| margin KS D | 0.056631 | ≤0.176 | ✓ | 0.062724 |
| **mean_margin（m0）** | **0.025156** | ≤0.05 | ✓ | 0.0149 |
| win_rate shift | 0.010215 | ≤0.02 | ✓ | — |
| aux_delta KS h1/h2/h3 | 0.0113/0.0121/0.0123 | ≤0.05 | ✓ | 0.0063/0.0066/0.0086 |
| rows-per-game | 463.45 vs 448.50（行数 55,614 vs 53,820） | REPORT-ONLY | 报数 | — |

- mean_old −63.43（A1）、mean_new −38.63（Z2）；sd_old 678.75、sd_new 1217.61、
  pooled SD（闸口径，见 §5.2）985.72。两臂行 SD 差大是局终 margin 重尾所致（见 §5.2），
  不是口径错——两臂口径逐字同（config 除 seed 外逐项同），行数差（55,614 vs 53,820）
  是对手混合差（mirror 66 vs 60 局）所致，属种子决定。
- schema：old 49 件／55,614 行、new 50 件／53,820 行，problems=[]。

### 5.2 重定推导（recalibrate.py，md5 bce3175a…；输出 recalib_g4.json，md5 014cbf54…）

统计量 T = |mean_new − mean_old|／pooled SD（行层）。零假设：两臂为同分布的独立抽样。
局内行完全相关（同一局同一座的每行 margin 皆为该局终局 margin），故独立单位是**局**；
mirror 局两座贡献 +m 与 −m、合计恰为 0，只 54（A1）／60（Z2）局非 mirror 局推动分子。
方法（与 margin-KS 线先例 5fbbad47 同类：线取零分布的 α=0.05 临界值，不是实测基线值本身）：
分层局层 bootstrap（mirror／非 mirror 层按 counts 保持，20,000 次，种子 20261120），
逐次重抽两臂的局、重算 T* = |Δmean*|／SD_pooled。

- 实测零基线 m0 = **0.025156**（与闸报告逐字合——推导与闸交叉验证通过），
  在 bootstrap 零分布中居 **38.2 百分位**（典型零抽样，非极端）。
- bootstrap T*：q90 = 0.080789、**q95 = 0.094909**、q99 = 0.122123。
- 解析交叉核对（正态近似）：局 margin SD 730.84／711.39，SE(Δmean) = 42.08，
  线 = 1.64485×SE／SD_pooled = **0.070221**。正态近似在重尾局 margin 下低估尾部
  （q95／解析值 ≈ 1.35），故不取解析值，只作参考。
- **拟重定线 = 0.095**（bootstrap q95 0.094909 按 3 位小数取整，KS 线 0.176 的取整先例；
  精确值 0.094909 随推导同报）。状态：**PROPOSED——线只在策略组核准后才变**（重开件 c4ecdf98）。

### 5.3 两种判读（消融 mean_margin 0.057357）

| 线 | 判读 |
|---|---|
| 旧线 0.05（现行） | **FAIL**——0.057357＞0.05，gate_pass=False（gate_ablation.json） |
| 拟重定线 0.095（待核准） | **PASS**——0.057357≤0.095，闸以 --mean-shift-line 0.095 重跑 gate_pass=True、rc=0（gate_ablation_recal.json，md5 750be3f4…） |

最终判读待策略组核准拟重定线后定。**若核准 0.095，则消融结论为：匹配对手口径下
vec2 与 vec1 等价（全部腿过），提速成立（§6），闭环按“等价过＋提速”采纳——但采纳本身
仍由策略组／作战室裁定，本报告只给测量与两种判读。**若不核准或另定线，旧线判读生效、
闭环以不采纳收口（裁定 a33b73c6）。

## 6. 墙钟与争用

- 争用态：三跑期间 DGX 同驻价值谱正式训（PID 1898309，GPU 训练步进行中，GPU 利用率峰值约 96%）
  与 ③ 批 2 打包（PID 2625484，pack_submission --verify，CPU 约 400%，A2 结束前已退）。
  A1 447.0 s（0.27 局/秒）、A2 167.1 s（0.72 局/秒）、Z2 920.8 s（0.13 局/秒）
  均为争用读数，不得与 B 段干净读数（vec1 187 s、vec2 134 s）直接比快慢；闸判读腿不受争用影响。
  墙钟取日志 birth→config mtime（文件系统跨度）；驱动 epoch 跨度为 447.4／169.0／921.1 s，
  差值 ≤2 s 为驱动自身快照开销。Z2 慢的主因是价值谱正式训在其全程占满 GPU（利用率约 88–96%），
  与之互为争用——价值谱线速率亦受本线拖累（主代理 11:16 记 load 4.76→12.56），两线均未停。
- 187 s 独立实测核查（附带项，结论）：**两阶段 vec1 墙钟同为 187 s 是两次独立实测，非誊抄**。
  文件系统时间戳为证（日志文件 birth＝shell 重定向建文件时刻≈开跑，config mtime＝跑完写盘时刻）：
  阶段 1 vec1：日志 birth 10:22:12.804 → config 10:25:19.718 ＝ **186.91 s**；
  阶段 2 vec1：日志 birth 10:29:05.273 → config 10:32:12.840 ＝ **187.57 s**。
  两值同取整为 187 s（截断），差 0.66 s 在同口径重跑抖动内；同工作量同机器的两跑墙钟相近是预期，
  不是巧合疑点。对照：阶段 1 vec2 134.26 s、阶段 2 vec2 104.88 s（收据记 106 s，约 1 s 级差，
  估计为计时起点口径差——收据可能含启动前准备，不影响 187 s 项的结论）。
  消融跑以同样方法重测：A1 447.0 s、A2 167.1 s、Z2 920.8 s（均为日志 birth→config；
  驱动 epoch 跨度 447.4／169.0／921.1 s，差 ≤2 s）。三者皆为争用读数，见上段。

## 7. vec1 非确定性有界诊断

### 7.1 代码层已定位的机制（侦察结论）

- vec1 推理服务器 `_server_main` 的成批规则：阻塞取首请求，随后**非阻塞排空**已到请求
  （`req_q.get()` 后 `while len(batch) < 256: get_nowait()`），按 tag 分组、concat 后走
  bf16 autocast 前向。成批组成直接取决于 24 worker 的到达时序——逐跑必变。
  不同 batch 形状 ⇒ bf16 GEMM 舍入末位不同 ⇒ 近 tie argmax 翻转 ⇒ 轨迹发散 ⇒ 分片逐字不同
  （bmask/searched 与 logits 无关故逐字同；x/carbon/margin/win 不同）。
- 排除项：对局内全部 RNG 均为每局 fresh RandomState（种子确定）或 fresh 定种子 RandomState
  （deep_move 内 seed0+7／+38 等）；`layout.new_game_state` 内 `np.random.seed(seed)` 每局重钉
  全局 RNG；`sample`／`fast_action` 只用传入的每局 rng；`_TEMP` 模块全局量在 try/finally 内
  置／复原；worker 为独立 spawn 进程，无跨局状态泄漏通道。代码 grep 未见未播种的
  np.random.*／random.*／time／hash 调用（除上述已钉死的两处）。
- **vec2 对照：确定性**。B 段 vec2_g4 与消融 vec2_b968400 的 43 件分片 md5 逐件同
  （排序后 md5 集 dc0b2bef…）；vec2 锁步池（固定池序、前向成批顺序固定）不引入时序抖动。
  vec1 B 段 vs 消融分片 md5 集不同（c4ed5daf… vs ac3ef425…），与诊断一致。

### 7.2 D1 单 worker 双跑（diag_vec1_w1_r1／r2）

口径：gen4 ckpt＋hof×3、games 8、seed 968400、deep-frac 0、value-scale u、
dtype bf16、threads 4、workers 1。**device 由 cuda 改为 cpu**（见 §7.4；成批逻辑与设备无关）。

| 跑 | 墙钟 | 结果 | 行数 |
|---|---|---|---|
| r1 | 573.2 s（CPU 争用态） | rc=0，3 分片＋config | 3,887（与表定 gids 0–7 预期逐字合） |
| r2 | 546.1 s（CPU 争用态） | rc=0，3 分片＋config | 3,887 |

**分片逐字节同：D1 IDENTICAL**——排序后 md5 集同为 efaaabc6250f6497425fb1e9001b3460；
逐件 md5 diff 为空（diag_d1.diff 为空文件，md5 d41d8cd9…）；总行数两跑皆 3,887。

### 7.3 定性：已知良性属性

单 worker 下 vec1 逐字节确定；24 worker 下逐字节不同；vec2（锁步池）逐字节确定；
零基线分布闸全过（§5.1）。故非确定性来源 = **推理服务器成批组成的并行时序抖动**
（§7.1），且**分布稳定**。按裁定 a33b73c6 记为**已知良性属性**，不另立生产缺陷件。
诊断不堵口径切换与 Gen-5 生成（Gen-5 仍用 vec1）。

### 7.4 附带观察（非本次诊断的两个分支，不立缺陷件，请主代理酌定是否跟进）

- D1 首发（GPU）因 DGX 统一内存耗尽而失败：推理服务器启动时 `net.eval().to(device)`
  报 `torch.AcceleratorError: CUDA error: out of memory`
  （`cuDevicePrimaryCtxRetain` 返回 CUDA_ERROR_OUT_OF_MEMORY）。
  当时 DGX 内存 102/121 GB 已用、空闲 3 GB、swap 15/15 GB 满——价值谱正式训的
  dataloader 以共享内存持全量训练集（约 85 GB）。服务器死后 worker 在 `resp_q.get()` 上
  永久阻塞、主进程在 `join()` 上永久阻塞——该路径无任何超时或 server 存活检查，
  失败表现为静默挂死而非报错退出。首发已 kill（主进程 3691578、驱动 3691533；
  孤儿 worker 3692237 随后亦 kill），D1 改 CPU 口径重跑（§7.2）。
  这是鲁棒性缺口（无超时／无健康检查），不是未播种 RNG、也不是状态泄漏——
  生产 Gen 跑在内存充裕座位上不触发；是否修、由主代理／策略组酌定。
- 另：D1 期间 DGX 路由曾瞬断约 3–4 分钟（11:55–11:58，egress proxy CONNECT 断，
  L20 同代理通——DGX 路由单点抖动）；DGX 侧驱动未受影响、r2 自行跑完。
  11:44 左右本 VM 服务亦重启一次（exec 中断，已重查确认无影响）。

## 8. 交付物与 md5

DGX 原件：/home/arm/junyang/vec2_ablation/；本地副本：~/workspace/genexp/ablation/。

| 件 | DGX 路径 | md5 |
|---|---|---|
| 对手表 968400 | opp_table_968400.json | d69736d32ec11dced2fd47d85feb9b92 |
| 对手表 968460 | opp_table_968460.json | 5ffbfaccd8098d79eaff14e54f40e6fb |
| 预抽脚本 | opp_table.py | f170987fa1a099eb4997a1eef4ccee9c |
| 指纹核对脚本 | verify_opp.py | 6f2f8ae7d6a34e386945a8617568a47a |
| 三跑驱动 | run_ablation.sh | 397d535290e4df1ce3fca2c8aed22888 |
| 重定推导脚本 | recalibrate.py | bce3175af82f3c70e0946c45a5ea0fc6 |
| 诊断驱动 | run_diag.sh | dd7774fe26b4758f17460a8a0cbfb9d0（CPU 版；GPU 首版已被 CPU 版覆盖，md5 未留存） |
| 消融闸报告 | gate_ablation.json | b43ac87b5e1235ad6d7830d0f5e6d2ad |
| 消融闸日志 | gate_ablation.log | 1e8715cc97e2bd9828a5304250c824cf |
| 拟重定线闸报告 | gate_ablation_recal.json | 750be3f4c3969b69734185ee3680d816 |
| 拟重定线闸日志 | gate_ablation_recal.log | 5b2df16165a40771ff6e68c624ec7475 |
| 零基线闸报告 | gate_zerobase.json | 9ce236938dba34e0db8e4cae65a02ce7 |
| 零基线闸日志 | gate_zerobase.log | 9fb906c42306b0277e781f27428fa3ee |
| 重定推导 | recalib_g4.json | 014cbf54e7b164da3815c342ab72e6bc |
| 三跑驱动日志 | run_ablation.log | 7f9211622d4b41a94716a2b8d7612ada |
| 诊断日志 | run_diag.log | 32072a483675427c8f16023e113c3277 |
| D1 逐件 diff | diag_d1.diff | d41d8cd98f00b204e9800998ecf8427e（空文件＝逐件同） |
| 本报告 | REPORT_ABLATION_VEC2_20261120.md | （自哈希不嵌正文；定稿 md5 见执行人终报，落仓时以读回为准） |

现行件只读，未改 vec1／vec2／闸代码。新写 harness 如上表列 md5；是否落仓由主代理定。

## 附：用户问“vec1 现在多久出 8000 局”

DGX 干净口径（B 段）：vec1 0.64 局/秒 ⇒ 8000 局约 12,500 秒 ≈ **3.5 小时**。
消融争用口径（A1）：0.27 局/秒 ⇒ 约 8.2 小时（争用读数，不作定额依据）。
L20 口径（Gen-3 实测 2.26 局/秒，24 worker）：8000 局约 59 分钟——Gen-4 在 L20 的实测为准。
vec2 若采纳（未采纳）：DGX 0.90 局/秒 ⇒ 约 2.5 小时。
