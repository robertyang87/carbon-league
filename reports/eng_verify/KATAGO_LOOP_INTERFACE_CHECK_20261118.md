# KataGo 全环接口核对回执（2026-11-18，工程组）

派单：作战室 2026-11-18 派单＋`reports/PLAN_KATAGO_LOOP_20261118.md` §3（文件 md5 `5d5110b8`）；设计口径 `reports/DESIGN_KATAGO_FULL_LOOP_20261118.md`（md5 `562f5f3c`）§2/§3/§4.1。
口径更新（作战室 07:18）：产局已解耦先行（L20 复算身份先产），本回执卡产局之后的训练步；产局侧证据优先取真实落盘样本。

## 0. 判定（先行）

**不同构。** 产局输出与 k0_distill 输入在**容器**与**内容**两层都不同构：

- 容器：产局产物是 cnet2_build 回放语料（`index.json`＋`replays/*.json.gz`，每步 record）；消费器只读封印代 `cnet2-k0-annotation/1`（`current.json` 指针＋特征行＋两册标签 npz）。
- 内容：设计 §2 要求的每步五元组 (x, aux, π_search, 终局 U, 步带）中，**π_search 在产局侧根本不存在**（产局是裸网自博，见 §1）；**终局 U 在产局侧存在（每局一份）但消费器无槽可放**（见 §4 停报项）；x/aux 产局侧不落盘、由下游 build 从 observation 重建。

处置（只动接口层）：新文件 **`tools/selfplay_loop_adapter.py`**（blob `f22d9289`，md5 `7d5bd94a`，11,642 B）串起钉版接缝（build → annotate → join → 预扫）并把终局 U 落为旁路 sidecar。冒烟一局全链在本机（muse VM）跑通四环（§5）。**policy/蒸馏半环今日即可用现行钉版件闭合；E[U] 价值半环卡在训练核心（k0_distill 无终局 U 输入与可训价值头），按纪律停报（§4），未动核心。**

证据来源（按 07:18 口径更新注明）：产局侧字段证据以 **L20 真实落盘样本**为准——`katago_pilot_loop_selfplay_150`（产物册登记，L20 2026-11-18 07:55 完工，150 局 seeds 300002–300151），本线自 L20 直读其中一局 `selfplay-300002.json.gz`（md5 `afc6d9a7`）＋`index.json`（md5 `e3b26aa6`）逐字段验；仓内产局代码盘点为辅；本线自产一局（§5 环 1）为第三重互证。三处 schema 逐字段一致。

## 1. 产局侧盘点

产局代码：`code/producer/selfplay_gen.py`，md5 `bd62ead9`（42,717 B），仓内唯一现行 KataGo 环产局器（`code/gen_value_selfplay.py` 是旧 value 网生产器、nnfeat 栈，非本环产局器；cnet2_search 的 `st["rec"]` 只是单步搜索的内存访问记录，从不落盘）。

- **裸网自博，无搜索**：包配置 `CONF`（L176）`SEARCH=False`；文件头（L19、L24–28）明示产出的是 cnet2_build 回放语料。Agent 统计只读错误计数，不落任何搜索量。
- **每步 record**（play() L381–419 构造，payload L441–452）：`steps[t][side] = {observation（含仅监督用的 full_carbon 全量局面）, action（实际打出的联合动作）, reward（现金；灭绝时终局分）, status, info}`，共 300 条（平台移位约定：条目 t 含 step t 的 observation 与自 step t−1 observation 选出的 action）。文件顶层 `{id: "selfplay", steps, identity, result}`。
- **终局块**（每局一份，在 `result` 与 index 条目各一份）：`scores / margin / u`（均按 side）、`winner_side`、`terminal`、`end_reason`。U 口径 `_u_of_margin`（L202–206）：`U = 0.5·W + 0.5·sigmoid(M/1000)`，与设计 §2 的 U 定义逐字一致；平局 W=0.5。
- **π_search：不存在。** 对真实样本全文计数：`"visits"`、`"priors"`、`"arms"`、`"q"`、`"Q"`、`pi_search` 均为 **0**（L20 样本 afc6d9a7 与本线自产样本 6e24b180 两验一致）。
- **x/aux：不落盘。** 产局只存 observation；特征由下游 cnet2_build 以同一套 `cnet2_feat.Feat/Prefix` 从 observation 重建（§3）。
- **步带支撑**：每步 record 在 `steps` 序列中的位置即 step；index 条目带 `steps` 总数；build 样本与标注/消费键均带显式 `step` 字段——step≥200 计数在每一环都可算（L20 登记口径：150 局 step≥200 共 15,000 步，与 150×100 一致）。
- L20 样本实测（300002 局）：300 条 step 记录、键集 `{action, info, observation, reward, status}`；`result.u = [0.833488, 0.166512]`、`margin = [+694.54, −694.54]`；index 同步镜像 margin/u；producer identity weights_md5 `06a03d9c`、config_hash `27b08baa`（与产物册登记一致）。

## 2. 消费侧盘点（k0_distill.py，md5 `2728fee6`，58,243 B）

- **唯一入口格式**：数据集根 → `current.json` → 封印代，manifest 格式必须 `cnet2-k0-annotation/1`（`verify_annotation_generation` L276 起、格式核 L326–328），verify-or-refuse，无第二路径。
- **特征字段**（`FEATURE_FIELDS` L151）：`maps, scal, tok_pos, tok_kind, tok_mask, tok_cargo, hist_cell, hist_cargo, hist_move, uf, mf, legal, carbon, teacher, gid, step`（即 x/aux 的消费形态，由 cnet2_build 产出）。
- **标签两册**（`load_annotation` L384 起，键集核 L430–441）：`policy_labels.npz = {game, step, seat, arms[R,K,OWN] int16, n_arms, priors[R,K] f64 (P_A), q, pi0, pi1, pi}`；`value_labels.npz = {game, step, seat, Q[R,K] f64}`——value 侧输入只有**每根每臂 Q（教师根评估的每候选平均 U）**，无任何终局 U 数组。
- **目标量构造**（`construct_target` L185–199，文件头 §13-C L12–21）：`q_k ∝ sqrt(P_A,k)·exp((Q_k − maxQ)/0.05)`，由消费器自 (Q, priors) 构造；落盘 `q` 只作交叉校验（差 > 2e-6 即 die，L521–527）。
- **缺失即 die 的硬闸**（逐条）：两册键集必须一致（L471–477）；pi 资格（L479–490）；填充候选必须惰性（arms −1、priors 0、q 0、Q NaN，L492–496）；priors 逐根 >0 且和为 1（L499–503）；q ≥ 0 且和为 1（L505–507）；臂值域与臂对 tok_mask 的合法性（L509–518）；producer 身份字段（teacher_config_hash、teacher_weights_md5、belief）缺失即拒（L447–456）；main() 另核 producer 的 teacher_weights_md5 必须等于 `--parent` 文件 md5（L945–950）。
- **训练目标**：policy CE ＋ anchor KL（对 parent 的每臂分布），仅此两项（文件头 L42–47）。`FROZEN_PREFIXES`（L137）＝ `utility. / wdl. / margin. / aux.`——**全部 value 侧头冻结**。
- **坏根预扫判据在本文件内**：`expand_candidate_batch`（L679 起）经 `_root_cash_set`/`_f16q`（L619 起）在候选前缀现金 f16 不可判定跨合法阈值时 die——适配器预扫即调用此函数本体（§5 环 3）。

## 3. 逐字段对照表

| 设计 §2 字段 | 产局侧（selfplay_gen 产物） | k0_distill 需要 | 判定 |
|---|---|---|---|
| x（每步特征行） | 不落盘；仅存 observation（每步 record） | FEATURE_FIELDS 15 字段/根（build 产出） | **可平凡映射（既有钉版路径）**：cnet2_build（build_top.py）自 observation 以同一 cnet2_feat 栈重建，按 (gid, step) 键对齐 |
| aux | 不落盘（同上） | 含于特征行（uf/mf/legal/carbon 等）＋标注侧 n_unknown/belief | 同上，经 build＋annotate 两件既有工具补齐 |
| π_search（搜索改进分布） | **缺失**（SEARCH=False；全文字段计数 0） | priors (P_A)＋Q（每臂）→ 消费器 construct_target 构造 q 并校验落盘 q | **缺失→接口层补**：钉版 k0_annotate 以 `stage0_teacher.evaluate_root`（K16/M8/H2 注册默认档）对窗口内每根产 (arms, priors, Q, q)；环内口径＝教师取产局方 parent 本人（自标注），与计划 §2「产局方即标注方」一致 |
| 终局 U | **存在**：每局一份（result＋index 双写），U = 0.5W＋0.5·sigmoid(M/1000)，口径与设计一致 | **无槽**：value_labels 只有每根 Q；value 头全冻结；loss 无 U 项 | **缺失（消费侧）→核心缺口停报（§4）**；接口层仅以 sidecar 保存（适配器 `terminal_u.json`，逐字抄自 index，标注「不被 k0_distill 消费」） |
| 步带标记（step≥200） | step 由 steps 序列位置承载；index 有 steps 总数 | keys 带 step；窗口由 annotate 注册参数 [--lo, --hi) 决定（k0_annotate L19–21：0 ≤ lo < hi ≤ 299 之外拒） | **同构（可算）**：每环均可按 step 计数；正式环窗口按注册取 [200,299) 或全窗 [0,299) |
| 实际动作 action | 每步 record 有（联合动作 dict） | 不直接消费（教师标注自回放重演得根） | 多余（无害）：留在语料里供 build/annotate 重演 |
| reward（每步现金/终局分） | 每步 record 有 | 不消费（U 由 margin 在产局侧算毕） | 多余（无害） |
| producer 身份（weights/config hash） | identity 块＋index/identity.json 齐 | 硬闸：teacher_weights_md5 须等于 --parent md5 | **同构**：环内 parent 即产局权重，天然满足 |

## 4. 停报项（核心缺口，只定位不动手）

**终局 U 的 E[U] 价值训练在现行 k0_distill 内无路可走**，缺口定位：

- 输入层：`code/k0_distill.py` `load_annotation` 的两册键集（L430–441）无终局 U 字段；封印格式 `cnet2-k0-annotation/1` 的 schema（文件头 L76–100）无此字段定义——加字段即改封印格式与 join/annotate 两钉版件的产物 schema。
- 目标/损失层：训练目标只有 policy CE＋anchor KL（文件头 L42–47）；`FROZEN_PREFIXES`（L137）冻结 utility/wdl/margin/aux 全部 value 侧头，无 E[U] 回归项。
- 结论：让终局 U 成为训练目标，必须改 k0_distill 本体（新输入字段＋解冻/新设价值头＋新损失项），属训练核心变更，超出接口层边界。**本线停报至此，请策略组裁定**（改核心另立项，或环 v1 只跑 policy 半环、U 仅以 sidecar 积累待用）。适配器 sidecar 已保证 U 数据不丢：每局 margin/u/终局状态随语料同行落盘，裁定后可直接接。

另两处口径存疑（不 invent、列明待裁）：
1. 设计 §4.1 的 value 头三目标（E[U] 主头直回归、WDL、margin(σ=1000) 辅助）与现行教师网络的 utility/wdl/margin 头命名相近，但现行 k0_distill 语境里它们是被蒸馏网络的冻结头、且其监督来自教师根评估而非终局标签——是否同物，请策略组在改核心立项时一并定。
2. 本冒烟标注用 cnet2_search 为仓内 main 版 `c3222eea`（本机栈），生产钉版为 `94f7fb64`；产物格式由版本戳自证且已过真消费器全闸，但正式环开跑前须以钉版六件套复跑一局对账（本线建议列入开产清单）。

## 5. 冒烟四环证据（本机 muse VM，2026-11-18）

环境：venv torch 2.15.0+cpu、numpy 2.5.5（机队标准 2.5.4，差异在此注明）；代码栈 md5：k0_annotate `7f567a2e`、k0_join `fac55acb`（main 版；生产钉版 `27c11e16`）、k0_distill `2728fee6`、build_top `08d2ab33`、cnet2_build `5c2dd58e`、stage0_teacher `c6ade5ce`、gate `e922d9d8`、cnet2_feat `2a45b08b`、cnet2_np `d8c15ad4`、fastsim `aefe6c83`、eval_v53 `45792dfb`、selfplay_gen `bd62ead9`、cnet2_search `c3222eea`（main，见 §4 存疑 2）。

- **环 1 产局**：`selfplay_gen.py --weights 06a03d9c…npz --seed0/--seed1 987001 --workers 1 --warmup 0 --open-t 1.0 --open-steps 50 --open-seed 987001`，RC=0，墙时 23.2 s。产物 `selfplay-987001.json.gz` md5 `6e24b180`。**局完整性**：299 步满步、terminal=true、end_reason=horizon、终局 rewards 在位（result.scores a 9290.22 / b 8819.97，margin a +470.25，u a 0.807721 / b 0.192279），errors=[]。（搜索档口径：产局路径 SEARCH=False 无搜索档可配——计划 §2 的 K16/M8/H2 是标注档，已在环 3 以 annotate 注册参数落实，特此如实记录。）
- **环 2 建集（适配器）**：`tools/selfplay_loop_adapter.py --corpus corpus1 --weights <06a03d9c> --window-lo 200 --window-hi 230 --workers 3`，总墙时 1,484.5 s：build RC=0（代 `gen-4f6b7ce57c46bbfa`，598 样本＝299 步×2 座，exact 598 / approx 0 / unusable 0）；annotate RC=0（代 `gen-3bd8b0585fce9f1f`，60 根，配置 K16/M8/H2、belief sample、teacher=产局权重 06a03d9c、config_sha256 与代内记录一致）；join RC=0（代 `gen-fc347a6811bb972e`，60 根 / 960 候选，**join 自带真消费器预检 PASS**，臂填充哨兵翻译 0 条——60 根候选全满 16）。终局 U sidecar `terminal_u.json` 已落（源 index md5 `c51e9033`）。
  - 标注单价（本机实测）：探针窗 [200,206) 12 根 284 s；正式窗 60 根 1,479.8 s ⇒ **约 24.7 s/根**（k0_annotate 进程池按局 map，1 局只忙 1 worker，与 workers 数无关）。全窗 [0,299) 598 根在本机约 4.1 小时（估计）——正式生产不应在本机跑，此数只作接口核对的速率证据。
- **环 3 预扫**：适配器以 k0_distill 本体 `expand_candidate_batch` 对 joined 全 60 根逐根（二分定位法）执行训练器自身拒训判据：**全根 60、问题根 0、问题局 0**（0.1 s）。训练环内消费器自报 candidate_features：cash_recovery_roots 5（前缀现金按 r4 口径重算恢复、非拒训）、cash_ulp_roots 0。
- **环 4 训练**：`k0_distill.py --data <joined> --parent <06a03d9c npz> --max-updates 10 --bs 8 --lr 5e-4 --seed 7 --device cpu`，RC=0，10/10 步完，elapsed 159.4 s。loss total：update 1 = **3.12047**（CE 3.12047、anchor KL 0.0——学生自 parent 初始化故首步 KL 为 0），update 10 = **7.91241**（CE 5.02116、KL 加权 2.89124）；中间步 loss 随批次构成波动（update 2 曾达 13.17976），10 步冒烟只证链路与数值有限性、不作收敛判读。**落权重**：`student_smoke.npz`（本机 `~/workspace/katago-loop-smoke/work/`），**md5 `63878dd8`，13,994,948 B**，sha256 `e78dd4f3…`，已登记产物册（条目 `katago_loop_smoke_student_10upd`，册 blob `0b9a0fff`，读回验合）。
  - 内存记录（本机特有，非链路问题）：bs 32 首跑在本机 6 GB 级内存上限下于 update 1 前被 OOM 杀；bs 16 跑到 update 5 后同因被杀（实测 RSS 3.9 GB）；bs 8 完跑（RSS 约 2.8 GB）。正式环训练在 L20/GPU 座执行，不受此限；本机只作接口冒烟面。

## 6. 结论与交接

- 接口判定：**不同构 → 接口层适配已交付并冒烟验通**（适配器 `tools/selfplay_loop_adapter.py`，blob `f22d9289`，md5 `7d5bd94a`，提交后 API 读回逐字节验合）。
- 训练步解卡范围：本回执后，**policy 半环**（产局→build→annotate→join→k0_distill）可按计划 §4 环参数开跑（开跑前先以钉版六件套复跑一局对账，见 §4 存疑 2）；**E[U] 价值半环**等策略组对 §4 核心缺口的裁定，裁定前 U 只积累不消费。
- 本线未动任何现有钉版/核心文件；未占 spot-64 / L20 / DGX 执行面（L20 仅只读拉取一份样本）。
