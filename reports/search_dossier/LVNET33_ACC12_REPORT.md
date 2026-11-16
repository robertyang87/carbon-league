# L-vnet33 验收一/二执行报告（本地工件，未入仓——按补正口径先报策略组）

run_id: lvnet33-acc12-20261116
协议： LVNET33_ACCEPTANCE_PROTOCOL_DRAFT.md 定版 md5 7030c663d24710a9324cdb138f9e91c9
主体叶： L-vnet33 ＝（检查点 33，源 state md5 e3df8f0203ce11b7a858c08f23d87929）vnet 段，
npz md5 bb9c5b00980c8f36b939c635bfba39f5（逐张量 sha256 对 manifest 186/186 已验）
主体包： Echo4 包 md5 819250bfc82632786063dabd46a616bd
样本： 诊断块种子 986201–986223 共 23 局 Echo4 自博弈（见口径声明 §3）
判读： 只对照定版线，不放宽。

## 1. vnet 前向语义（代码证据，先行关键步）

- 训练器 `code/rl2_ppo_train.py`：`--sep-value` 默认 1（L108），vnet 为独立
  CNet2：`vnet = ad.make_model(w0)`（L543）；critic 读数
  `ad.evaluate(vnet, bt, args.temp)["value"]`（L859）回归至 returns（clamp
  [−1,1]，终局-only 奖励 `2·(U−0.5)`，`game_rewards` L288–294）；启动器
  `tools/ppou2/launch_ppou2_v5.py` 传 `--reward u --gamma 1.0 --sep-value 1
  --value-head tanh`。
- 适配器 `code/rl2_ppo_cnet2.py:258-266`：value-head=tanh → `model(b)`；
  `code/cnet2_model.py` 前向 `value = torch.tanh(self.value(ctx))`
  （L219/285/311），ctx 为 encode 输出。
- 生产 numpy 镜像 `proto/vendor/cnet2_np.py` Policy.act L183：
  `value = tanh(ctx @ value.weightT[:,0] + value.bias[0])`。
- **结论：z ＝ vnet `value` 头在 ctx 上的预激活；U ＝ (tanh(z)+1)/2；
  有限性在 tanh 前查 z（SI06）。** 后端复用生产 cnet2_np.Policy 装 fp32
  vnet 权重（npz 无 meta → 注入 actor 包 cfg：C=128 NB=8 D=128 NH=4 L=2
  E=32 G=8 TA=4 TNH=4）。
- 交叉验证（探针局 stdout）：以同一 z 提取路径作用于 actor 本体，
  tanh(z) 与包 Policy.act 的 value 在 step 0–2 双席 6/6 逐位相符
  （0.543226/0.541606/0.556809）。

真值口径：
- 教师 U：`code/stage0_teacher.py:29` `U = 0.5·W + 0.5·sigmoid(M/1000)`，
  W∈{0,0.5,1} 按终局现金差符号（平局 0.5）；M ＝逐席终局现金差。
- **RULING_R2 第 5 条执行**：终局真值一律走内嵌 leaf 的 F01/F04 权威逐席
  口径（`proto/vendor/cnet2_search.py:666-677`：灭绝方
  pen = step_count − episodeSteps − 1、存活方取入步前现金；时间终局取
  真实终局现金）。**未引用 s1_verify3 的 run_arm 或其结算函数**（grep
  自查 harvest.py / vnet_backend.py 零命中）。为此修正初版 harvest 三处：
  ① 灭绝后继真值原写 prev_cash 差 → 改直接调内嵌 leaf_fn 取权威结算分，
  pen 口径状态不伪造 margin/U（记 None）；② 主循环界 step_count<300 →
  <299（SI01，最后可行动 obs 为 step 298）；③ 验收二局末真值改权威逐
  席 rew（灭绝局按 F01 逐席构造、end_kind 入档）。

## 2. R2-01 自查（steer 2）

harvest 唯一 rollout 调用为 eval_root 内 `rollout(root.full, cand, 1, 0)`，
horizon 硬编码 1；H≥2 续弈循环（guard 推进路径）在 H=1 下不执行——
**R2-01（对手 guard 复制己方历史，位于 H≥2 续弈推进）未触发**；
verify3 未使用（R2-03 不适用）。

## 3. 口径声明（执行方裁定两项，显著声明待追认）

1. **开局多样性**：部署 CONF（OPEN_T=0）双席同包纯贪心自博弈在对称图
   上镜像锁死——实测种子 986200（探针局，OPEN_T=0）：终局现金精确打平
   4192.84＝4192.84、双席 step-298 根逐候选全等、校准真值全塌 0.5
   （vnet 预测 0.568–0.639）。该续弈下两验测的是口径伪影。故采集局启
   用 agent 自带数据生成机制 OPEN_T=1.0（OPEN_STEPS=50 包内默认、双席
   orng 种子本异），第 50 步起全程纯部署贪心；状态/根全出自 Echo4
   (it=20) 自博弈。探针局（OPEN_T=0）不计入样本。
2. **同局状态相关**：验收二每局 36 状态共享该局同一终局（双席状态的
   真值 U 互补、和恒为 1，池均值恒 0.5 为构造成立非异常）；按定版线
   原样判读，相关性在此声明。
3. VM 口径：离线机制测量、不涉 deadline；步成本（单局墙时约 9–17 分
   钟）为本机共享 CPU 伪影，不外推。

## 4. 验收一（行为匹配）

全根集（报告项）：n=594 根；vnet33 与叶① argmax 相符率 **0.4579**；
逐根 Kendall τ 均值 0.2708（τ 可定义 383 根）。

可判定子集（n=28 根，step-298 战术根、全候选后继终局；无效计数全 0）：

| 项 | 数值 | 定版线 | 判读 |
|---|---|---|---|
| 叶① chosen 相符率 | 1.0000（28/28） | —（基准方） | 身份核验：叶①值与权威结算 max 偏差 0.0 |
| vnet33 chosen 相符率 | 0.4286（12/28） | — | — |
| **配对差（vnet33 − 叶①）** | **−57.14pp，bootstrap 95%CI [−75.00, −39.29]pp** | 点 ≥ −2pp 且 CI 下界 ≥ −5pp | **FAIL** |
| Kendall τ（vnet 叶值 vs 真值结算） | 均值 0.530，95%CI [0.295, 0.707]（τ 可定义 8 根） | CI 下界 > 0 | PASS |

描述性拆解（不改变判读）：28 根中 20 根全候选结算同分（该根 chosen 按
约定取 A0；vnet 在其中相符 6/20）；8 根结算有别，vnet 相符 6/8。
**验收一判读：FAIL**（配对差线未过；τ 子项过）。

## 5. 验收二（校准）

状态集 n=828（23 局 × 双席 × 18 步位），vnet 误差 0；10 等宽箱：

| 箱 | n | 预测均值 | 真值均值 |
|---|---|---|---|
| 0.0–0.1 | 4 | 0.088 | 0.095 |
| 0.1–0.2 | 74 | 0.154 | 0.130 |
| 0.2–0.3 | 85 | 0.255 | 0.211 |
| 0.3–0.4 | 96 | 0.354 | 0.293 |
| 0.4–0.5 | 85 | 0.443 | 0.333 |
| 0.5–0.6 | 81 | 0.553 | 0.464 |
| 0.6–0.7 | 89 | 0.652 | 0.604 |
| 0.7–0.8 | 151 | 0.753 | 0.675 |
| 0.8–0.9 | 143 | 0.842 | 0.829 |
| 0.9–1.0 | 20 | 0.913 | 0.896 |

| 项 | 数值 | 定版线 | 判读 |
|---|---|---|---|
| **ECE** | **0.0561** | ≤ 0.05 | **FAIL** |
| 单调性 | 最大相邻回落 0.000 | 容忍 ≤ 0.02 | PASS |
| Brier | 0.0522 | （报告项） | — |
| 池 Kendall τ（描述） | 0.592 | — | — |

曲线形态：全程单调，但中段系统性高估（0.4–0.5 箱预测 0.443 vs 真值
0.333；0.5–0.6 箱 0.553 vs 0.464；0.7–0.8 箱 0.753 vs 0.675），ECE
主要由这三箱贡献。**验收二判读：FAIL**（ECE 未过线，差 0.0061）。

## 6. 总判读

- 验收一：**FAIL**；验收二：**FAIL**。按协议四项须全过，L-vnet33 在本
  轮不满足进 S3 独立臂的前置（验收三/四未跑，不在本任务范围）。
- 无效/异常计数：eval_error 0、degenerate 0、vnet_err 0、
  dec_not_all_terminal 0、dec_vnet_err 0——无样本被静默丢弃。
- 执行 provenance 注记：采集期一批次后台进程被 SIGTERM 回收、另有一
  份与磁盘矛盾的完成回执已按损坏丢弃；全部局数以逐局 JSON 落盘直读
  核验（23/23 在盘）。

## 7. 本地工件清单（未入仓）

见回执 md5 清单：harvest.py、vnet_backend.py、analyze.py、
run_g986201…986223.json（23）、probe_g986200.json（探针，口径外）、
lvnet33_acc12_evidence.json、本报告。

## 8. 策略组裁定与入档（2026-11-16 后补，本节外正文与先报版逐字一致）

- 裁定：`reports/search_dossier/RULING_LVNET33_ACC12_20261116.md`（commit c013419e，blob 08f9014e，md5 b8cda8f2）。
- 两项执行口径（§3）均追认：OPEN_T 采集记为采集口径澄清，判读线一字未动；同局相关追认为声明事项。**验收一 FAIL、验收二 FAIL 全额成立。**
- 线处置：**L-vnet33 关闭**——不进 S3 独立臂，MS04 不开；验收三/四记「不跑·前置未过」（非 FAIL）。校准映射等改造件属新叶身份，须新协议注册从验收一重走。
- 本报告与证据 JSON 按裁定④入档于 reports/search_dossier/；证据 JSON 与先报版逐字一致（md5 3cc6ff727034b245860f1be34f204dca），策略组将自证据抽算头条数复核。
