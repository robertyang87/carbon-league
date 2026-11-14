# PREREG — Master2 基线批次不可复现定位探针（诊断场，无闸效力，不耗 R3）

状态：**定稿 v3（2026-11-14 22:2x）**——输入全钉死，定稿后开跑；本文件同存 L20 探针目录。
登记问题（策略组 2026-11-14 20:09）：同种子块 996000–996199 的两日 Master2 基线批次，
逐局 margin 0/200 全同、胜负翻转 26 局、批次间 |ΔU| = 0.0174。
（L20 线自归档原件复核 2026-11-14 22:1x：0/200 全同、26 翻转、ΔU(95−85)=+0.0174、max|Δmargin|=2621.19，逐项合。）

## 跨日差异的事实层（DGX 扫描，2026-11-14）
两日批次并非「同包同 harness」——三处候选变量：
| 场 | actor 包 ah | harness hh | 考官 bh |
|---|---|---|---|
| it85 日 Master2 基线 | 73542cdd…（早期 m3s2r1 打包件） | dceb1ebe…（早期 bigeval 构建） | 2aebfbc1… |
| it85 日 Echo2 基线 | 73542cdd… | dceb1ebe… | （Echo2 件，另行） |
| it95 日 Master2 基线 | 11d80059…（m3s2r1_base.py） | 354bcd1b…（现行钉死） | 2aebfbc1… |
| it95 日 Echo2 基线 | 73542cdd… | 354bcd1b… | （Echo2 件，另行） |
- 考官 bh 四场全同 2aebfbc1008d3475ece8e178696e292d＝bigeval 实际加载的 Master2 考官件
  （W04 台账冻结真考官；与真身 master2_src.py 的 404114a0 分开记，DGX 双验在册）。
  考官件本体：exp/opp_master2r.py（693 B 包装，载入同目录 master2_src.py、_SEED=20261111 固定）；
  真身 exp/master2_src.py md5 404114a0b5e422516322f62e9ddb6870。两件 L20 本地已有、已落探针目录验 md5。
- 两打包件对 Echo2 的产出逐局全同（配对 ΔU 恰 0，DGX 扫描）——包字节差对 Echo2 无效；
  对 Master2 不可先验等同，正是本探针要测的。注：两打包件 CONF 均为 SEARCH=False（纯网），
  73542cdd 件（DGX s26stage/code/exp/m3s2r1.py）为包装形态（载 weights/cnet2_m3s2r1L.npz），11d80059 件为自含整包。
- 归档逐局线原件在本 VM ~/workspace/it74_paired/：h29_it85base_m2.jsonl（792,418 B）、
  h29_it95base_m2.jsonl（881,696 B）；原件另在 sync/h29 的 bigeval_live/。

## 假设（可证伪，一次一变量）
- H1 线程/调度非确定性：BLAS 线程数或进程调度改变浮点归约顺序／搜索时序 → 逐局分岔。
  预测：全钉死（四线程变量=1、worker 数固定）下同配置两批复制逐局全同。
- H2 墙钟搜索预算（考官侧）：两基线包均 SEARCH=False（纯网确定性，Echo2 场跨日跨 harness
  200/200 全同为旁证）；非确定性候选在考官 Master2 真身内部搜索——若其按墙钟截断，
  机器负载/速度改变有效搜索深度 → 基线侧同招、考官落子不同 → 逐局分岔。
  预测：钉死两批在低负载下或可全同、高负载下分岔；或钉死下仍分岔（预算极紧时）。
- H3 包/harness 字节差：跨日两批的 actor 包与 harness 均不同版，任一均可解释分岔。
  预测：现行配置自复现成立时，单换包（或单换 harness）即复现出与另一日批次同型的分岔。
- H4 跨机环境差（CPU 型号/numpy 构建）：仅经与归档批次的次比较间接观察，不在本轮单独操纵。

## 固定输入（钉死项）
- harness（第一步）：bigeval.py md5 354bcd1bc2dc8bc188c813f53d28c893；eval_v53.py md5 45792dfbea3101812836fe77e6752706。
- 基线包（第一步）：m3s2r1_base.py md5 11d8005930b298290c2e150949349647（it95 日 Master2 场同件，已确认）。
- 对手：Master2 考官件 exp/opp_master2r.py md5 2aebfbc1008d3475ece8e178696e292d
  ＋同目录真身 master2_src.py md5 404114a0b5e422516322f62e9ddb6870（探针目录内已落位验过）。
- 种子：996000–996199，N=200，座位按种子奇偶（harness 固有）。
- 环境：OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1；
  EVO_WORKERS=**24**（定稿固定值；考试日值 C1/C2 协查未回，若日后到且不同，
  以修订版预注册另行登记单变量对照批，不改本定稿）。
- 机器：L20（gpu-l20-6，32 vCPU），开跑前验全机空闲（无训练、无其他 bigeval），各批顺序跑、不并行。

## 批次（分步，每步跑前确认上一步判读）
- 第一步 P1、P2：现行钉死配置（包 11d80059＋harness 354bcd1b＋线程全 1），调用完全一致仅 tag 不同
  （probe_m2_p1／probe_m2_p2），各自全新 live 文件（bigeval 为追加模式，禁止复用旧 tag/旧文件）。
- 第二步（条件触发，需 73542cdd 原件到手）：单换包——包 73542cdd＋harness 354bcd1b＋余皆同，一批 P3，
  与 P1（或 P1/P2 共识批）配对比较。
- 第三步（条件触发，需 dceb1ebe 原件到手）：单换 harness——包 11d80059＋harness dceb1ebe＋余皆同，一批 P4，
  与 P1 配对比较。
- 任一批出现 err 行或 berr>0：该批作废，先诊断再重跑，作废批次也入报告。

## 度量（判读口径，跑前锁定）
- 主比较（每步）：逐种子 margin 全同数 /200、胜负翻转数、ΔU（U 按 harness _u_of 自 raw margin 计）、max|Δmargin|。
- 次比较：各批 vs 归档两批（A85/A95）：同上四项；只作定位参考（跨包、跨 harness、跨日），不作成败判据。
- 批次 U 均值、胜负数一并报。

## 判读线
- P1 ≡ P2（200/200 全同、0 翻转、ΔU=0）→ 现行配置自复现成立；跨日分岔落在 {包字节，harness 字节，日环境}
  → 进第二步（单换包）；第二步无分岔再进第三步（单换 harness）；两步皆无分岔 → 指向日环境/负载（H4/H2 跨日形态），
  以修订版预注册另行登记。
- P1 ≠ P2（钉死下仍分岔）→ H1 不成立、H2 优先（当前配置内生非确定性）；本注册内不临时加变量，
  以修订版预注册另行登记零负载复跑＋逐局墙钟记录；第二、三步暂缓（先有自复现基线才谈跨配置比较）。

## 报告
- 开工三件套（进程在册＋负载＋首行落盘）报作战室；每步结果表报作战室＋策略组。
- 本探针为诊断场：结论不产生任何闸效力，不消耗 R3 确认名额。

## 证据增补（2026-11-14 22:36，C2 线 spot-31 直读经作战室转交；不改已定稿设计与已跑批次结论）
- 两批 Echo2 基线场在 spot-31 跑（it85 日 07:11–07:19／it95 日 15:28–15:33），启动脚本原文
  env：**EVO_WORKERS=62＋四线程变量全 =1**。线程钉死与本版一致；worker 数当日真值（E2 半边）为 62，非本版 24。
- 73542cdd＝spot-31 /root/carbon/code/exp/m3s2r1.py 薄包装＋权重 cnet2_m3s2r1L.npz（b6db08c9，与新线起点同一件）；
  11d80059＝A4 搜索包——**两「包」是同一 ep19 权重的两种考试形态（薄包装＋原权重 vs 搜索包），actor 变量收敛为「包装形态」**。
- 旧 harness **dceb1ebe 原件已定位**：spot-31 code_backup_pre_bigevaltrio_20261114/bigeval.py（12,735 B）；
  时间线：it85 场跑于其下，08:06 换 354bcd1b，it95 场跑于新 harness——第三步单换 harness 至此可执行（原件待转交我线考试目录）。
- Master2 半边（两批 M2 基线场的机器/env）仍缺，C1 协查中；其真值到后，若 worker 数非 24，
  按本版既定修订条款另登记 probe_m2_p1b/p2b 按真值复跑，不改本版结论。

## 证据增补二＋修订批登记（2026-11-14 22:38，第 3 问闭合，DGX 汇总 C1/C2 双直读互证）
- 机器：Master2 两批在 spot-29（脚本 it85-exam.sh／it95-exam.sh）；Echo2 两批在 spot-31。
  **env 四批逐字相同：EVO_WORKERS=62＋四线程变量全 =1**；机器规格两日一致。
- 配对矩阵（行内实证）：it85 日两场＝73542cdd＋dceb1ebe；it95 日 M2＝11d80059＋354bcd1b；
  it95 日 E2＝73542cdd＋354bcd1b。跨日变量全因子化为两个二值变量：**actor 形态×harness 代次**。
  旁证：Echo2 两批仅 harness 代次不同而输出 200/200 逐字同——harness 代次单独对 Echo2 无效（对 Master2 待 P4 测）。
- **修订批登记（本版修订条款触发）**：probe_m2_p1b／p2b——包 11d80059＋harness 354bcd1b＋线程全 1，
  仅 EVO_WORKERS 改为当日真值 **62**，种子同 996000–996199；排 P3 后机器空闲自动接跑。
  判读：p1b≡p2b 看 62 下自复现；p1b vs p1 单变量看 worker 数（24 vs 62）敏感性。v3 结论不动。
- 2×2 因子表进度：(包装，旧h)=A85／(包，新h)=A95·P1·P2／(包装，新h)=P3（已排）／(包，旧h)=P4（待 dceb1ebe 转交）。

## 第三步解锁（2026-11-14 22:4x，策略组协查结果）
- dceb1ebe＝仓内 code/bigeval.py @ commit **6eb9f596**（12,735 B，md5 逐字合）——我线已自 GitHub API
  取回验讫（blob c411ecab），落 L20 /root/carbon/probe_m2/code_oldh/（code 全量副本仅替换 bigeval.py；
  eval_v53 维持 45792dfb、包维持 11d80059）。P4（tag probe_m2_p4，EVO_WORKERS=24 余皆同第一步）
  已链式排在修订批之后自动接跑。
