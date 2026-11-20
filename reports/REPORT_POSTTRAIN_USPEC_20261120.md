<!-- 落仓头（工程组，2026-11-20）：执行人终报原件逐字转录（本地 md5 62aca8f9）；主代理独立核验见 reports/VERIFY_POSTTRAIN_USPEC_20261120.md（包／§7 日志 DGX 直读、次级读数本地重算全合；A 臂状态以核验时刻 13:22–13:24 直读为准，执行人 13:19 备件快照已过期）。 -->

# 价值谱 U′ 刻度 300 正式训完训后处理 — 执行人终报（2026-11-20）

执行人：工程组（DGX，接续执行人；前轮 8f3eca99 已完成正式训发车与监护）。
执行人不落仓；主代理独立核验后落仓。

## 0. 开工三件套

- 开工 2026-11-20 12:53（DGX 定位＋变体件 md5 核验先行）；预期完成约 15:00（估计，约 2 小时）；
  实际定稿约 13:25（提前，pack --verify 与次级读数并行推进所致）。
- 进度口径：① 打包＋自验 → ② §7 四项复验 → ③ 51 局次级读数 → ④ 决斗备件 → ⑤ 报告。
- Gen-5 训练（L20，12:41 发车）全程未动；L20 镜像未动（只读）。

## 1. 完训核实（接手时直读，与作战室派单逐字合）

- 正式训 12:51:09 完训，run_formal_train.py 退出 rc=0，wall 6735.7 s，峰值 RSS 86.78 GB、
  torch peak alloc 1.13 GB（formal_train.log 尾行）。
- 产物 `/home/arm/valuespec_uspec/train/`：uspec300.npz（2,101,008 B，
  md5 8879aaef8a808f409d45f1de02175159）＋uspec300.pt（4,515,195 B，
  md5 b885712e910e9c04b12aa0b4caab9791）。
- 变体件 DGX 直读全合：train2_uspec.py aa1f8596、bridge_uspec.py 4619fbb9、
  pack_submission_uspec.py ad76e287、bridge.py f31b07dc、pack_submission.py 57c75d84、
  layout.py 6877bc73、model.py c102c297、verify_bridge.py 7577c41e、train2.py 06e3c87c。

## 2. 打包（任务 1）——完成

- 命令（DGX）：`pack_submission_uspec.py --chassis /home/arm/l20_katago_mirror/src/apexcarbon_v1_fix3.py
  --ckpt /home/arm/valuespec_uspec/train/uspec300.npz --value-scale u
  --out /home/arm/valuespec_uspec/pack/uspec300_wall.py --verify`。
- 底盘 apexcarbon_v1_fix3.py md5 4b3423a0（已核）。
- 首跑 12:56 失败（非包错）：`--verify` 前向检查 `import extract_m3s2` 找不到模块
  （extract_m3s2.py 在镜像 tools/katago_m3s2/ 下，不在 src/ 的 sys.path 内；批 2 建包时
  在碳仓树语境下跑，未撞此问题）。处置：包文件本身已落（md5 2ed4c1e6），
  以 `PYTHONPATH=/home/arm/l20_katago_mirror/tools/katago_m3s2` 12:57 重跑整令；
  重建包 md5 仍为 2ed4c1e6（重建逐字节同），随后 --verify 走完。失败经过如实记，不计入包证据。
- 包：`/home/arm/valuespec_uspec/pack/uspec300_wall.py`，md5 **2ed4c1e6ec495ed85cb4656abbdc1de5**，
  2,867,315 B；桥 patch bridge-def -2/+32 行（U′ 孪生较原桥 +30 行多 2 行，刻度代入所致），
  clock-removal 补丁全上，post-condition「no perf_counter in output」通过。
- 自验（pack --verify，13:13 落定，PID 258068）：
  - 前向 err **0.0000**（容差 0.25）→ PASS（§7① 前半）。
  - 确定性 1 局×2 跑 **IDENTICAL**（终局现金 [8391.62, 7937.5]）→ PASS（§7① 后半）。
  - 步时 1196 次 agent 调用：mean 0.783s／p95 1.666s／max 3.031s（0.75s 线）→ FAIL——
    只报数、不参闸：0.75s 是天梯单步时限（裁定 1a6e45f0 同款口径），决斗走线下 bigeval、不上天梯；
    且本轮步时为 DGX 争用下墙钟读数（同期次级读数 8 线程 CPU 全量评估在跑）。§7 四项不含步时项。

## 3. §7 复验（任务 2）——四项全过，决斗开场前置齐备（只差 A 臂）

| 项 | 口径 | 实测 | 结论 |
|---|---|---|---|
| ① 前向 err | 打包件前向对 U′ 检查点参考网，容差 0.25；＋确定性 | err 0.0000；1 局×2 跑 IDENTICAL | PASS |
| ② 桥注入 verify | verify_bridge --packed 全项（f90af57d 口径） | 网格 2.626e-09／1e-3；孪生 2.020e-09／1e-6；packed _u_of_margin 0.0／1e-12；逆残差 2.220e-16／1e-9；单调双向、钳区饱和、零点与跳变全过；退役点 0.9999999 精确复合误差 0.0；**packed twin 2.020e-09／1e-6、packed _u_of_margin 0.0／1e-12**；饱和探针 B(_U_AT_CLAMP)＝B(±1)＝±1.5 精确 | PASS（rc=0；原版「saturation values」行对退役点报 FAIL 系预期内，已由新口径探针取代） |
| ③ perf_counter | 墙钟壳计时调用计数与在册形态合 | 包内 `perf_counter` 0 处 | PASS |
| ④ 三档原值 | 壳内注册常数逐字合 | _S_STAGES ((2,0.45),(5,0.22))、_S_D2 9、_S_ROWS 32，与底盘同行号（972/973/976）逐字合 | PASS |

② 的运行件：`gate/run_bridge_verify_packed.py`（新口径 runner＋--packed，桥变体绑为 bridge；
首次执行因 sed 换 argv 未生效而漏了 --packed 行，已当场发现并以 python 精确改写重跑，
两次输出除 --packed 两行外逐值同——漏跑的一次不计入证据）。
② 各值与闸阶段（GATE_FINAL）逐值同（网格 2.626e-09、孪生 2.020e-09、逆残差 2.220e-16）。

## 4. 次级读数（任务 3）——完成，只报数，不作判读

- 路径：与 ①／复判同解码同归因路径。评估件 `eval_uspec.py` 由 `eval_rejudge.py` 改写，
  diff 仅六处（路径、检查点、输出名）＋文首说明；解码（master3s2.py，md5 57f9a594 钉版源）、
  encode、fp32 CPU 前向（8 线程）与复判同口径。评判者：uspec300.npz（md5 8879aaef）。
  DGX 镜像语料 echo9/echo10 逐件 sha256 与 ① 同一 loss_list 验身；model.py 用镜像件
  （与 L20 复判件同 md5 f9e6fe7d；与仓内 model.py c102c297 的 diff 仅 1 行注释）。
- 评估：51/51 margin_ok（现金误差 0、碳误差 0），300 状态／局。单局冒烟（1aa40c1d）先行通过
  （final -493.12＝platform，v0 -0.0343→v299 -0.6332；U′(-493.12)＝-0.6356，终值吻合）。
  终值合理性：51 局 |v299 − U′(终局分差)| 中位 0.0411、p90 0.0866、max 0.1100（训练价值损失量级，无异常）。
- 统计件 `analyze_uspec.py` 由 `analyze_rejudge.py` 改写，diff 仅路径／标签（m_→u_）／文首，
  公式、判据（ADV_EPS＝5.0、噪声带 B＝本评判者 p99 |Δw|、双条件）一字未动。
- 噪声带：U′ 评判者 B＝**0.1286**（51 局×299 转移 |Δw| p99；中位 0.0200、p90 0.0659、max 0.4185）。
- **归因三分（各按本评判者塌方步）**：

| 归因 | gen3（①） | 复判 m3s2（除 step-0，判读读） | U′ 原始 | U′ 除 step-0 |
|---|---|---|---|---|
| 对手得分主导 | 21（41.2%） | 20（39.2%） | 25（49.0%） | 25（49.0%） |
| 局面性重估 | 20（39.2%） | 14（27.5%） | 16（31.4%） | 15（29.4%） |
| 自身受损主导 | 8（15.7%） | 14（27.5%） | 8（15.7%） | 9（17.6%） |
| 混合 | 2（3.9%） | 3（5.9%） | 2（3.9%） | 2（3.9%） |

  逐局标签与 gen3 相同：31/51（原始与除 step-0 同）。
  锚点说明：复判 27.5% 是其**除 step-0**的判读读（原始列 51.0% 被 step-0 伪影灌水，不作判读；
  复判报告 §1／§4）。U′ 无 step-0 伪影：原始 argmax 落 step-0 仅 1/51 局，step-0 跌幅中位 0.0140、
  仅 1 局越带——故原始列与除 step-0 列几乎同一读数。
- **塌方步重合**：与 gen3 同一步 20/51（39.2%），±5 步 24/51，±20 步 31/51；
  |步差| 中位 7（复判除 step-0：17/51 同一步、|步差| 中位 23）。
- **集中度形态**：塌方占毛跌落中位 0.037［0.031, 0.044］（gen3 0.035［0.028, 0.044］）；
  半幅步数中位 9（gen3 9、复判除 step-0 9）；最大／次大中位 1.16（gen3 1.18、复判 1.14）——
  「无单步失误签名、坏步成簇」在 U′ 评判者下同样成立。
- **2 局全程未察觉（预注册双条件：最大单步跌幅＞B 且净跌落 D＞B）**：

| 局 | 平台分差 | gen3 D | U′ w0→w299 | U′ 净跌落 D | U′ 最大跌幅 | 判定 |
|---|---|---|---|---|---|---|
| 82cbda80… | −24.57 | +0.0040 | −0.0295→−0.1266 | +0.0971（＜B） | 0.1804 @177（＞B） | 未看见 |
| 1f765a0d… | −58.33 | +0.0090 | −0.0286→−0.0261 | −0.0025（净升） | 0.2368 @251（＞B） | 未看见 |

  两局在 U′ 下均未被看见（与 gen3 同；与复判 m3s2 两局均看见不同）。U′ 下净跌落 ≤0 的局 1/51
  （1f765a0d；复判 0/51、gen3 2/51）。两局塌方步归因均为局面性重估（除 step-0 同步同归因）。
- 判读界线重申：以上只作机制证据，不作判读、不得升格（预注册 §2）。U′ 未能复现复判的
  自身受损上移（15.7%→27.5%），其自身受损占比仍与 gen3 同为 15.7%（除 step-0：17.6%）——
  此为次级读数事实，不改变任何主判读（主判读只由决斗定）。

## 5. 决斗备件（任务 4）——备件完成，未开场

- 见 `DUEL_PREP_USPEC_20261120.md`（本地＋DGX `valuespec_uspec/duel_prep/`）。
- U′ 臂已备：包 md5 2ed4c1e6、§7 四项全过。
- **开场唯一阻塞项：NETEXP A 臂未落地**——无 A 臂检查点、无 A 臂包（镜像 netexp_ab/ 只有
  seal_manifest.done＋seal_manifest.md5）；A 臂发车登记待核（须读 train2＋fpw 1.0）。
- 开场命令形：`bigeval.py <U′ 包> <A 臂包> 968600 200 <tag>`（side＝seed%2 内置；双包先后顺序与
  ΔU 符号口径在开场时钉死）；座位待作战室定（DGX 为自然候选）。
- 本单未跑任何一局决斗、未写 bigeval 记录。

## 6. 交付物

- DGX `/home/arm/valuespec_uspec/pack/`：uspec300_wall.py（2ed4c1e6）、uspec300_verify.log、
  bridge_verify_packed.log；`gate/run_bridge_verify_packed.py`。
- DGX `/home/arm/junyang/valuespec_secondary/`：eval_uspec.py、trajectories_uspec.json（51 局）、
  eval_uspec_full.log、eval_uspec_smoke.log、loss_list.json、per_game.json。
- DGX `/home/arm/valuespec_uspec/duel_prep/`：DUEL_PREP_USPEC_20261120.md。
- 本地 `~/workspace/valuespec/run/`：本报告、DUEL_PREP_USPEC_20261120.md、trajectories_uspec.json
  （md5 b63bee31363ceee39324878e81d783bb）、per_game_uspec.csv（ff808fa5）、per_game_uspec.json
  （cc859cf9）、analyze_uspec.py（b5fb780e）、analyze_uspec.log、uspec300_verify.log、
  bridge_verify_packed.log、eval_uspec_full.log。

## 7. 未做与如实记

- 首跑 pack --verify 因 extract_m3s2 导入失败中断（见 §2），PYTHONPATH 补齐后重跑通过；包 md5 两次一致。
- §7② 首次执行漏了 --packed（sed 换 argv 未生效），当场发现并重跑通过；漏跑的一次不计入证据。
- analyze_uspec.py 输出尾行仍印「wrote per_game_rejudge.csv/json」（原脚本写死的文案未改），
  实际写出的是 per_game_uspec.csv/json（已核）；另两处 stdout 标签残留（「new-judge pooled」「n_half_drops
  median: m3s2」）为文案未改，数字为 U′ 评判者实测——不影响文件内容。
- 未在 GPU 上跑评估（CPU fp32 与复判同口径；DGX GPU 空闲但评估路径与 ①／复判同为 CPU）。
- 步时项 FAIL 只报数（见 §2），不参闸。
