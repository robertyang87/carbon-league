# 开工备案 — PPO-U 新线（m3s2r1 本体起点）

提交：L20 线（单一驱动）→ 策略组（记录性，非二次批准）
状态：**主体经策略组 2026-11-14 22:34 对签确认在册**；考官 md5 附件已补（见文末）；
余锚 state md5 一项，暖机落定后补入。策略组附加：首轮 achieved KL 随开工三件套同报（0.20 外推校准点）；
池面口径裁定——以声明表＋--pool-authoritative 为准，m3w1L／m3w2L 系 league 代码遗留名、不带入定案。

## 线与编号
- 主臂 tag `ppou2_m`（λ=0.95），out `/root/carbon/data/rl3/ppou2_m`，日志 `/root/carbon/data/rl3/ppou2_m.log`
- 对照臂 tag `ppou2_c`（λ=1.00），out `/root/carbon/data/rl3/ppou2_c`，日志 `/root/carbon/data/rl3/ppou2_c.log`
- 轮次用各 out 目录本地计数，与 snipe_n1 的 it 号无关。

## 代码基线（L20 在位，md5 读回验过）
- rl2_ppo_train.py 2c8262fbf8763f8b792ce10e2e9ba9c2（2026-11-14 刷新，旧版备份 *.pre_refresh_20261114）
- cnet2_build.py e005152499b2ba30a56f92c7096e2707（同上）
- rl2_ppo_cnet2.py 62a9adc79069ca6dfa6f344a439a8310（§19，未动）；fastsim.py aefe6c83826efae4de83bef8c643c80b
- 考试 harness：bigeval.py 354bcd1bc2dc8bc188c813f53d28c893、eval_v53.py 45792dfbea3101812836fe77e6752706
  （置于独立考试目录 /root/carbon/probe_m2/code，不动生产 code）

## 起点与锚
- --init ＝ /root/carbon/data/weights/cnet2_m3s2r1L.npz，md5 b6db08c9b076257c54817377bb93130d，
  13,968,972 B；与 DGX carbon/data/weights/cnet2_m3s2r1_ep19.npz 双机直读逐字同（m3s2r1 本体）。
  KL 参考即 init（不设 --ref）。
- 锚 state 定义：鲜启 --vwarm 3 暖机完成、首个正式轮之前的 state_last.pt 快照，
  另存 out/state_anchor_frozen.pt；其 md5 暖机落定后补入。随附验证：锚 actor 与 init 逐张量全等
  （暖机只动 critic，it80 口径三件套）。

## kl_coef 初值
- 双臂同为 **0.20**（鲜启 --kl 显式给，R03 语义下落入 state）。
- 推导（估计）：snipe 三段实测 coef→achieved：0.15→~0.085、0.12→~0.094、0.10→~0.108；
  线性外推 achieved 0.06 ≈ coef 0.20，对本令目标带 0.05–0.07 取中。
- 纪律照令：achieved ≥0.10 先报；单轮 ≥0.13 先停改系数再续；系数改动只经 state
  （resume 时显式 --kl，按 R03 压 state 值），每次改动入轮次记录。

## critic 出身（更正一，策略组裁定 2026-11-14 23:16——记录更正，非设计变更）
- --value-head tanh；**不带 --utility-source**（该参数只与 utility 头配伍）。--sep-value 1：
  独立 vnet ＝ init 全量拷贝，读数走 tanh（v̂=(tanh+1)/2）。
- 本体 npz 内唯一真受训的标量值头是 `value.*`（value.weight (1,128)＋value.bias (1,)；
  同工件 792 根实测对终局 U 的 Pearson 0.849，策略组引数）。`U.*` 是解码器动作嵌入投影
  （策略 logits 的组成），不是价值头；utility 头是 it79 才起用的，本工件 epoch 19 本无此头。
- 原备案句「utility 头取自本体 npz 的 U.*（适配器映射为 utility.*）」撤回——AD.load_npz
  只回原始键字典、无键名映射；原 argv 鲜启下 utility_source_plan 缺键即 SystemExit
  （第 0 分钟必被拒），已由本更正堵住。先例：snipe it74–78 同线同读法（Terminal-U 配 tanh）。
  F6 干净（不涉 gamma/rscale）；KL 记账是 actor 的 drift，critic 读法不影响。
- load 路径：trainer 经 AD.load_npz 直读 --init npz → make_model；不走 cnet2_train.load()
  代际消费路径，r4 过渡控制不适用本线；无任何手写 state。

## 池（鲜启 league，--pool-authoritative 钉死为下表全体；合计 prior 4.95）
停点声明池 11 人减 m9xj320r1（本次开池执行移除）；起点留锚 1.10，其余权重不动。
league 遗留成员 m3w1L／m3w2L 从未在声明池内，不带入（如口径不同请指正）。

| 成员文件（/root/carbon/data/weights/） | prior | md5 |
|---|---|---|
| cnet2_m3s2r1L.npz（起点本体） | 1.10 | b6db08c9b076257c54817377bb93130d |
| cnet2_m3s2g1L.npz | 0.55 | 975c518c10198723fec7bfa57744d9c3 |
| cnet2_m7bsr1L.npz | 0.55 | 0835830f792840ed66eef6d5be84b417 |
| cnet2_m9xj260wr1_fromstudio.npz | 0.50 | f0f81835f2920f639b3fca4b2dcb8bd0 |
| cnet2_m9xj260r1.npz | 0.45 | afc7aedc6fb4c838343eafe141ad8297 |
| cnet2_m6r1L.npz | 0.45 | 0cdb80cc578ff362217ab3927cc3e4fe |
| cnet2_m7bsv2r1.npz | 0.425 | 84254f2aaaf643491897f3889e70b47e |
| cnet2_m9xj260g1.npz | 0.35 | c9f106377a9614b4f613fcab1ba9998d |
| cnet2_m7bsv2g1.npz | 0.30 | 304a5041c4a74dc0dc48898badc0f3a4 |
| cnet2_caragor1_ep19.npz | 0.275 | 82027e53f3b397307b2dfb41c003a7da |

- SELF：--self-prior 0.5；自快照：--snap-every 10 --snap-prior 1 --max-snaps 2（照 snipe 生产值）。

## 双臂 argv（共同部分；差异仅 --lam／--out）
--adapter cnet2 --device cuda --server 1 --reward u --value-head tanh
--init （本体，见上） --pool-authoritative --pool （上表 JSON）
--self-prior 0.5 --snap-prior 1 --snap-every 10 --max-snaps 2
--vwarm 3 --workers 14 --lockstep 16 --dec-frac 0.25 --epochs 1 --mb 768 --pre-mb 4096
--games 4000 --lr 1e-5 --kl 0.20 --gamma 1.0 --rscale 1.0
--ckpt-every 1 --eval-every 5 --eval-pairs 32 --eval-opps exp/opp_master2r.py,exp/opp_m3echo.py
--seed0 3000000（双臂同种子）；主臂 --lam 0.95／对照臂 --lam 1.0
env：CNET2_TEACHER=2、RL2_SNAP_TEMP=0.5、CNET2_TEMP_KIND=0.15,0.3,0.3、RL2_SERVERS=1、
RL2_SERVER_MAXB=1024、PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（照 snipe 生产 env）
F6 合同自检（更正后）：gamma=1、rscale=1、c_dep=c_lost=0、tanh 头配 --reward u —— u_mode_check 只拦 utility 头配非 u 奖励，无问题项，不带 override。

## 并发与节奏
- 双臂同机并发（单 GPU 时间盒算术：顺序跑则 20 轮检查点在盒内不可达，估计）；
  每臂 workers 14（32 vCPU：14×2 worker＋2 server＋2 learner），开工三件套验 GPU 占用后定格；
  若首轮节奏显示检查点不可达，立即报作战室。
- 检查点：主臂每 20 轮、对照臂每 40 轮；检查点考试在本线考试目录跑（harness 同上），
  候选包由 L20 code/packs/cnet2_search_pack.py（48392e98…）以 m3s2r1_base 同 CONF 打包、md5 入记录。
- 累计 KL(π‖π_起点）：每检查点在 rootset32（Stage 1 根集，/root/ppo_attrib）上量，均值 >0.15 停线。
- 存活/停线、终局确认（保留块 984000–984999，确认考前任何检查点不碰）、硬停 2026-11-20 12:00 CST：照令。

## 第 0 步基线场（开训前，考试目录内，线程四变量钉死 =1，EVO_WORKERS=24）
- ppou2_base_m2_a：exp/m3s2r1_base.py（11d8005930b298290c2e150949349647）vs Master2（见附件），
  种子 980000–981999，N=2000
- ppou2_base_m2_b：同上第二批（独立批次；检查点读数对两批各配、均值定格）
- ppou2_base_e2_a：vs Echo2（见附件），种子 982000–983999，N=2000

## 附件：考官 md5（2026-11-14 22:34 补入备案）
- Master2 考官：exp/opp_master2r.py md5 2aebfbc1008d3475ece8e178696e292d；
  真身 master2_src.py md5 404114a0b5e422516322f62e9ddb6870（同目录载入，_SEED=20261111）
- Echo2 考官：exp/echo2.py md5 15b6d754e6e21de33870074d3983a16f；
  权重 data/weights/cnet2_m3w5L.npz md5 82fde46b2633950d3cbb29e8cacbfdc2；
  参考权重 cnet2_s64.npz md5 c84eef933d1d71272e89de3742081451（make_search_agent 的 RW 件）
- harness：bigeval.py 354bcd1bc2dc8bc188c813f53d28c893、eval_v53.py 45792dfbea3101812836fe77e6752706

## 附件二：考官形态声明（2026-11-14 22:40 补入）
第 0 步基线场与 C 探针一律用考官件形态（生产校准口径）：Master2＝exp/opp_master2r.py（载同目录
真身），Echo2＝exp/echo2.py（载 cnet2_m3w5L.npz）；真身直加载形态本线不用。
行内实证：第 0 步 M2 首行 bh=2aebfbc1、Echo2 烟测 bh=15b6d754、C 探针 P1/P2 全行 bh=2aebfbc1。


## 执行记录（2026-11-14 晚）

- **第 0 步定格（23:27:01）**：Master2 两批（980000–981999）逐局全同，1830 胜、均分 +1154.4、meanU **0.8242**＝冻结起点 Master2 定格；Echo2 一批（982000–983999）1461 胜、+433.3、meanU **0.6624**＝Echo2 定格。三批 rc=0、err=0。
- **C 探针因子表收口**：包装形态主效应 ΔU≈+0.0147（薄包装 0.8198／搜索包 0.8345）；harness 代次两行效应恰 0（P3≡A85、P4≡P1 逐局全等）；worker 数 24/62 效应恰 0；日残差 −0.0027。
- **双臂开训（23:36:42）**：主臂 ppou2_m λ0.95 PID/PGID 89312、对照臂 ppou2_c λ1.00 PID/PGID 89314，启动器 /root/carbon/launch_ppou2_v2.py（md5 0e5ec073…，tanh 版）。两臂 effective_params 直读：kl_coef 0.20、value_head tanh、reward u、gamma/rscale 1.0；league.json 池表 10 成员与先验合备案。锚 state 待暖机落盘捕获后补 md5；首轮 achieved KL 同报。


## 更正二：池表编码（2026-11-15 00:0x 执行中发现并修正）

- 首发（2026-11-14 23:36:42，启动器 v2）双臂均在首个 rollout 批次崩溃（约 23:38）：`--pool` 我按原始 npz 路径传参，rollout 的 `make_opponent` 只认 `SELF`／`SNAP:` 两种形态，其余走 `load_bot` 按 .py 加载，npz 路径直接 AttributeError。失败目录已归档为 `ppou2_{m,c}.failed1`（含日志）。
- 生产口径实证（snipe 停点 league.json 直读）：池成员一律以 **`SNAP:cnet2:<绝对路径>#T2`** 形态登记（#T2＝style row 2），先验与备案池表逐项一致；m9xj320r1 在 snipe 表内为 0.55、新线按备案移除；m3w1L/m3w2L 为 legacy、不在声明池内，不带入。
- 修正：启动器 v3（md5 f395bf3c04ec79182060043ec8198ac9）仅改池表编码，其余 argv 与 v2 逐字同；2026-11-15 00:06:26 重发（主臂 PID 91223／对照臂 PID 91224），00:09 直读已过崩溃点：双进程在册、GPU 94%、league 11 键全 SNAP 形态、spawn_main 30＝28 worker＋2 tracker。备案的成员与先验未动——此为编码更正，非池表变更。


## 常设规矩（所有者 2026-11-15 00:25 定，作战室传达）

任何新任务的开工公告（三件套）必须带预期完成时间：按当趟实测速率给、标「估计」；无实测依据的标「未校准」。无 ETA 的开工公告视为不合格。本线自此执行；ppou2 双臂线在首轮落地前，整线完成 ETA 记「未校准」，首轮出数后按实测轮长给检查点 ETA。


## 常设规矩补条（所有者 2026-11-15 00:25 定，00:26 传达）

ETA 必须开工前预估：以开工前的实测单价／同类任务速率为据；不许开工后再补估、不许拿开工后的速率回填开工公告。运行中可另报修正估计，但须与开工前预估分开标注。注：ppou2 双臂开工（23:36 首发／00:06 重发）均在本规发布之前，开工公告无预估 ETA 属规前情形、不回填；本线此后新任务一律先估后开，运行中估计一律标「运行中修正估计」。


## 更正三：主臂 OOM 死＋监视盲（2026-11-15 01:1x 直读核实）

- 主臂 ppou2_m（PID 91223）死于 00:41:18，torch.OutOfMemoryError（Tried to allocate 170 MiB；卡总 44.39 GiB 仅剩 111.38 MiB；m 占 31.91 GiB、c 占 10.29 GiB、server 各 1.03 GiB）。死前零产出（train/warmup 0 行、无 state）。双臂并发只核过端口不撞，未核显存峰值重叠——缺口在我。
- 对照臂 ppou2_c 活着：train.jsonl 两行均为 "warm": true（it=1 00:45:06、it=2 01:07:51，暖机迭代，actor 冻结故 kl/gnorm 恰 0.0，非异常）；it=0 EVAL（n=64）：Master2r 61 胜 +1286.7、m3echo 58 胜 +751.6。轮长实测约 23 分钟（it=2−it=1）。
- 监视盲：锚捕获与首轮监视进程一直在册，但 ssh -c 多行脚本在远端 shell 解析失败，轮询全空——同步趟「监视未在跑」措辞不准，应为「在跑但盲」。已改 Python 监视＋L20 侧检查脚本（remote_anchor_check.py ccfecc2b…／remote_round_check.py 22cade9f…，双向 md5 合），现读真实值。
- 锚口径修正：捕获触发是 train.jsonl 出现 it=3 且 warm:true（vwarm=3 完），不是 state_last.pt 首次出现（现为 it=2 暖机态，md5 8b873df8…；state_previous 39b45efd… 为 it=1）。主臂重发决策等作战室裁，本线不擅自重发。

## 校正二：kl_coef 0.20 → 0.05，双臂同值（2026-11-15 02:26，策略组裁）

- 对照臂首轮正式 it=4 达成 KL 0.0161，低于 0.05–0.07 目标带——kl_coef 0.20 的外推点如实交账，确为 overshoot（新线 coef→achieved 斜率远陡于 snipe 三段）。
- kl_coef 改为 0.05，双臂同值，单变量改动；λ、种子、步数、检查点口径、时间盒全不动；两臂只差 λ 一项的对比结构不变。
- 执行：主臂已重发 fresh（0.20 暖机 it=1 作废，整目录归档 ppou2_m.aborted_kl020_20261115/；02:30:43 新启 PID/PGID 104993，--kl 0.05，resume null，验收四项全过）；对照臂下次 resume --kl 0.05 生效；对照臂 it=4（0.20）保留为校准点，不改写。
- 下轮达成 KL 为再校准点：仍在带外（<0.03 或 >0.10）即报、等裁，不再盲步；单轮停线（≥0.13）未触发。
- games 定义（策略组附问，trainer 代码核实）：train 行 "games" = 本轮更新实际消费的局数（rl2_ppo_train.py:906 len(games)）；--games 4000 是下限目标（"at least args.games"，:691），在途并行批次（lockstep 16 × 14 worker）＋跨轮窗口（overlap 1-iter stale）使其落在 4338–4457；非累计（it=1 4372、it=2 4338、it=4 4457 非单调）、与备案 argv 无冲突。

## 校正二补记（2026-11-15 02:38，策略组签收）

- 主臂 (A) fresh 重发按等效接受，不返工（执行与裁定交错约 1 分钟；暖机未耗 KL、0.20 未产生训练影响）。余验收：主臂 it=4 行 kl_coef 读 0.05 落地为准。
- 对照臂落法钉死：下次 resume 必须先改 state 内 kl_coef 为 0.05，再 --kl 0.05 resume；it=5 行 kl_coef 读 0.05 为验收。已提前执行（c 停机态，state 稳定）：2026-11-15 02:40 编辑 /root/carbon/data/rl3/ppou2_c/state_last.pt——kl_coef 0.2→0.05（it 4 不变），读回 0.05；其余全部顶层项与备份逐项深比全等（脚本 remote_edit_kl.py，首版比对遇 dict 内张量 == 歧义崩过一次、已改递归深比后重验通过，幂等重跑 md5 不变）；备份 state_last.kl020.bak md5 5a36129e0ea6924447bbcee80b248cac（原 0.20 态），改后 state md5 8c3ae29c07810b45fe6c1e2fc477c13d。resume 时启动参数 --kl 0.05 与 state 值一致，两路同值。

## 主臂锚＋首轮正式 it=4（2026-11-15 04:0x）

- 主臂锚：/root/carbon/data/rl3/ppou2_m/state_anchor_frozen.pt，54,613,777 B，md5 26631aef5678fa7bf74f55e80a9e2934，03:40 捕获；actor≡init 已验（180 键，max abs diff 0.0，6 extra 辅助头同 c 锚）。
- 首轮正式 it=4（kl_coef 0.05）：达成 KL 0.0208，clipfrac 0.166，gnorm 11.019，games 4427，winrate 0.558；it=1–3 暖机行 kl_coef 全 0.05、kl 0.0。
- 校正 2 再校准点：0.0208 ＜ 0.03，带外（下侧）——已报策略组，等裁，不盲步；本线按 0.05 继续跑。c it=4 0.0161@0.20 对照：系数降 4 倍，达成 KL 只升约 29%。
- 本 VM 约 04:01 又被替换：driver 死（末心跳 04:01:07），看门狗 04:03:53 重拉（新 PID 1987）；两 watcher 同死，但活已干完（锚已捕、it=4 直读捕获），不重拉。

## 校正 2 再校准裁定（2026-11-15 04:09，策略组裁）

- 主臂锚签收（md5 26631aef），开工备案全齐；it=4 行 kl_coef 0.05 验收通过。
- 达成 KL 0.0208 带下，**kl_coef 维持 0.05，不再下调**：① 系数降 4 倍、达成只升 29%（约 0.18 幂），外推触带须系数约 0.0004 量级，该带在此配置下不是系数可达的，再切是盲步；② 达成带是监视目标、不是门，绑定线是检查点 ΔU 生存线与累计 KL 停线（0.15），两项均未触发；③ 跑到首个检查点 it=20（约 11:30 估计）以 ΔU 判读，若平坦届时再议配方问题，检查点前不预判改配方。lr 等其余 pin 不动。
- 本线按 0.05 继续。余验收：对照臂 it=5 行 kl_coef 读 0.05。

## 系数裁定（2026-11-15 10:49，策略组裁）

- it=20 达成 KL 0.1288 未触发 ≥0.13 单轮停线（差 0.0012），但已在噪声内贴线。
- 裁定：下一轮起（主臂 it=21、对照臂 it=5）**kl_coef = 0.15，双臂同值**——在检查点边界换，不等 it=21 触发。机制：resume 时显式 --kl 0.15（R03 语义，显式胜出）；对照臂 state 同步改 0.05→0.15 作双保险。
- 换后每臂首轮为新校准点，逐值必报；仍 ≥0.10 当轮即报、等策略组裁下一档，不许自行连步。
- 重拉时机照作战室已裁（检查点考试完＋本裁到齐）。累计 KL 一线随检查点结果判读。
- 10:15 触发预承诺对后续轮次照旧有效（≥0.13 先停、已 0.15 则 HOLD 等裁）。
- 执行回执（本线，2026-11-15 约 11:05）：kl_state.json 已置 current 0.15、calib_pending true、last_seen_it {m:20, c:4}；对照臂 state_last.pt 已改 0.05→0.15（改后 md5 7f9005b47cd6babe58bfc880b989f592，读回 kl_coef 0.15、it 4）；0.05 备份 state_last.pt.kl0.05.bak md5 f38c4a5b62ef34a8dc94902801e5571f（内容级重建，仅 kl_coef 一键不同；原 0.05 文件 md5 8c3ae29c07810b45fe6c1e2fc477c13d）。教训：remote_edit_kl.py 不自建备份，调用方须先 cp——本次首调因备份路径不存在致其深比对崩溃，改动本身已读回验合，备份已补建。主臂 state 待其 resume 时由 driver 的 ensure_state_kl 改 0.15（双保险＋--kl 0.15 双路径）。
- 策略组验收（2026-11-15 约 11:06）：三项实质验合——① kl_state.json；② 对照臂 state 改后读回；③ 触发机制入 driver＋FILING 补记。重拉仍等检查点回执到齐；重拉后两臂首轮（对照 it=5、主 it=21）为新校准点，当轮必报。
- fix-on-touch 已修（2026-11-15 约 11:08）：remote_edit_kl.py 改为编辑前自动备份（不存在时字节拷，不覆盖已有备份），张量安全深比对保留；已部署 L20（md5 4cb91aa72570069d5b93938e5370da1b，本机与机上一致），试件实测：备份与改前文件 md5 逐字同（7f9005b4…）、改后 other-keys identical、EDIT_OK、试件已清。0.05 备份（f38c4a5b）只作档案、不作可恢复字节件——系数只往前走，无实际影响，照此登记。

## it=20 检查点判读（2026-11-15，主臂 λ0.95 首检查点）

- 被测方：ppou2 主臂 it=20 检查点（起点 m3s2r1 本体）；考官：Master2 真身、Echo2 真身；口径：同种子配对 ΔU、种子聚类 bootstrap B=20000。
- 打包：ckpt_20.npz → ppou2_m_it20.py，md5 f4fff1638e51efd9e2c07720831627a4（source_check ok、模块哈希全合、CONF 与 m3s2r1_base.py 逐字同）。两场各 2000 局、err 0、候选 ah f4fff163／hh 354bcd1b、基线 ah 11d80059。
- Master2 场（980000–981999）：候选 U 0.8641 vs 冻结 0.8242；ΔU +0.0399，CI [0.0296, 0.0504]（A/B 两批点估计逐字同）。
- Echo2 场（982000–983999）：候选 U 0.7158 vs 冻结 0.6624；ΔU +0.0534，CI [0.0359, 0.0708]。
- 合并（4000 簇）：ΔU +0.0466，CI [0.0367, 0.0568]。生存线通过。
- 累计 KL（rootset32，K0 口径）：mean 0.0241、p99 0.0981、max 0.2721（自检 KL(ref‖ref)=0）；未触 0.15 停线。单轮 KL 0.1288 <0.13 未触发。
- 早期确认触发条件满足（合并 ≥+0.03 且 CI 下界 >0）；策略组 2026-11-15 11:19 裁定现在行使早期确认考（主体 f4fff163 原包、保留块 984000–984999 基线新鲜产、合并 1000 簇 CI 下界 >0 为成立判据、轮边界开考、结果无论正负训练照旧跑满时间盒、不涉天梯提交）。
- 重拉验证（2026-11-15 11:17–11:18）：补丁 driver 重启（START 11:16:47），对照臂 resume pid/pgid 221998，launch 记录 kl "0.15"、argv --kl 0.15，verify OK（alive、GPU 426 MiB、league_keys 11、spawn_main 15）；看门狗 cron 已恢复 enabled。

## 早期确认考判读（2026-11-15，保留块 984000–984999）

- 对照臂 0.15 首轮校准（it=5，落地 11:43）：kl_coef 0.15，达成 KL 0.029，games 4469，clipfrac 0.144，winrate 0.541；<0.10 不触待裁线，已当轮报策略组（0.15 站稳）。勘误在册：此前回执参考点「it=4 0.0161@0.05」系数标错——0.0161 是 kl_coef 0.20 下测得（校正 2 校准点），对照臂从未在 0.05 下跑过正式轮；FILING 正文（校正 2 节）本就正确，错只在回执消息。
- 开考：对照臂 it=5 边界停臂（state_match 真），driver＋看门狗 cron 暂停，11:47 五场开跑，12:11 全完。主体 ppou2_m_it20 包 f4fff163 原样；基线冻结起点包 11d80059 保留块新鲜产（Master2 两批＋Echo2 一批）。
- 完整性：五场各 500 局、err 0；候选 ah f4fff163／hh 354bcd1b，基线 ah 11d80059。
- Master2 块（984000–984499）：ΔU +0.0356，CI [0.0169, 0.0551]（对 A/B 两批点估计逐字同；候选 U 0.8707 vs 基线两批均值 0.8350）。
- Echo2 块（984500–984999）：ΔU +0.0683，CI [0.0349, 0.1023]（候选 U 0.7275 vs 基线 0.6592）。
- 合并（1000 簇）：ΔU +0.0520，CI [0.0328, 0.0713]。判据（合并 CI 下界 >0）满足——预注册最终确认判据通过，本线主张成立。策略组 2026-11-15 12:14 复核在册（合并点＝两分场均值 0.05195、U 差逐项合、CI 宽按簇数 √ 缩放合；新鲜种子复现主块读数、无胜者诅咒缩水；PPO-U 全线第一个确认级正结果）。
- 训练恢复（2026-11-15 12:12）：driver 重启，对照臂 resume pid/pgid 231481（launch 记录 kl "0.15"、resume state_last.pt），verify OK（alive、GPU 426 MiB、league_keys 11、spawn_main 15）；看门狗 cron 已恢复 enabled。照旧跑满时间盒至 2026-11-20 12:00；后续检查点（主臂 it=40 起、对照臂 it=40）按主块口径出 ΔU＋累计 KL 回执。本考不涉天梯提交（所有者闸）。

## 残树＋未卖载货侧车（策略组 2026-11-15 13:12／13:59）

- 残树：it=40 回执标 GAP（现有归档无终局观测输入，不代用近似场）；侧车在检查点场后补跑（每场前 100 种子升序、候选＋基线各一遍、身份闸逐局对 margin/win，任一失配即作废上报）。口径：终局 full_observation 里 players[p][3] 树字典全量计数，不按 owner 前缀过滤（26 局语料复算：胜均 2.941／负均 5.0／负 ≥7 且对手 ≤1 计 4 局）。
- 未卖载货（13:59 扩展，并列记）：终局 full_observation 里 players[p][2] 工人字典各工人携带量（条目下标 1）之和；players[p][1] 是静态招募中心属性、无存货。口径复算证据：51 局 Echo4 天梯语料（echo4_pull51，按创建时间升序编号）逐值合——#46 361.79、#49 143.54、#42 115.64、哈基米 v2 pro 279.14、哈基米 v3 399.64（Echo4 按名认、席位按 order 对齐）。
- 实现：remote_sidecar_trees.py（本地＝L20 md5 580cdd1439ba4c9f695996f94bd90587），逐局记 cand/opp trees 与 cargo（cargo_p0/p1 全量）；身份闸不变。验证（2026-11-15）：it=20 M2 场前 10 种子候选包重跑，gate 10/10，cargo 值域 4.73–281.99 正常。首跑在 it=40 场后；回执同报候选 vs 基线残树均值与未卖载货均值（同样本、胜/负分列）。

## 分臂并行（2026-11-15，所有者令：抢占机分臂＋2 分钟同步）

- 背景：所有者令租杭州 J 区抢占式 L20 试跑（carbon-l20-spot-1，实例 i-bp1dr7ogkb6kz5qav903，tailnet 100.92.60.120，实付 ¥6.5485/时），主臂留 gpu-l20-6、对照臂迁抢占机，结束串行排队（作战室 14:08 令、本线 14:11 方案）。
- 目标机处置：裸镜像 Ubuntu 24.04.5／Python 3.12.3；DNS 被 tailscale MagicDNS 接管且无可用上行——持久修复＝resolved drop-in（DNS 223.5.5.5/223.6.6.6）＋tailscale accept-dns=false；apt 内网镜像 100.100.2.148 不可达，源改公网 mirrors.aliyun.com；unattended-upgrades 占 dpkg 锁，已停并 mask（防其后自动升级扰动）。驱动 nvidia-driver-580-server 装成（580.178.04、CUDA 13.0、MODPROBE_OK、nvidia-smi 验 L20 49140 MiB；与 gpu-l20-6 的 580.126.09 同分支不同小版，登记在案）。
- 复制：gpu-l20-6 直连 rsync（spot_sync 钥）。首趟 venv rc=0 后 code/weights 两趟撞上本 VM 替换期的 tailnet 中断（broken pipe），run_copy.sh 的 rc 记录失真（rsync 报错仍记 rc=0）——以机上直读为准：code 1250/1388 文件、weights 0。补传后全量对账合：code 1388 文件、weights 36 文件、trainer 2c8262fb、本体 b6db08c9、launcher v5 539474bf、c 锚 573fb3af、state_last.pt 587ae5305cb7b497120c9a84704ce526、train.jsonl f0e724d5、league.json c9113cf5（两机逐字同）。教训：批量复制回执只认目标机直读计数＋md5，不认源端脚本自报 rc。
- 切换：先拆串行 driver＋其看门狗 cron（ppou2-serial-driver-watchdog 已 remove），再停对照臂于 it=11 边界（state_match 真、sha fdbe7005；整组验收：组进程 0、spawn_main 0、GPU 0 MiB），终同步后 resume。
- 2 分钟同步环（所有者 14:13 令）：拉式环跑 gpu-l20-6（PID/PGID 269975，落点 /root/carbon/data/rl3/ppou2_c_mirror/，环日志 /root/carbon/data/rl3/c_mirror_sync.log）。开训前验活通过：首趟镜像 state md5 与源逐字同、探针文件 sync_probe.txt（probe-20261115-split-v1）一个周期内到达且内容合；其后周期 rc=0/0/0。被回收恢复口径：以镜像 state 在 gpu-l20-6 续跑，损失 ≤约 2 分钟＋在飞半轮。
- 开工三件套：对照臂（spot，PID/PGID 24804，launcher v5，kl 0.15，resume it=11 态，15:01 spot 机时）verify alive／GPU 426 MiB（rollout 相）／spawn_main 15／league 12 键与源机逐键同（SELF＋snap_10＋备案池 10；校验脚本「11」为首快照入盟前旧期望，已正）。主臂（gpu-l20-6，PID/PGID 270207，launcher v5，kl 0.15，resume it=20 态）verify alive／GPU 426 MiB／spawn_main 15／league 13 键（SELF＋snap_10＋snap_20＋池 10）。
- 进度口径：两机各自 train.jsonl 轮次计数＋state mtime（外部可读）；检查点位置：主臂 /root/carbon/data/rl3/ppou2_m（gpu-l20-6）、对照臂 /root/carbon/data/rl3/ppou2_c（spot）＋镜像 ppou2_c_mirror（gpu-l20-6）。串行排程作废，两臂各自连续跑向 it=40 检查点。
- KL 看守：serial driver 退役；kl_watcher.py（本 VM，每 60 秒双机直读 remote_kl_check）接棒——主臂 it=21 为其 0.15 首轮校准点（calib_done m=false），对照臂已校准（it=5，0.029）；≥0.10 当轮报、≥0.13 先停该臂＋kl_hold.flag 待裁（两臂均已 0.15，不自步进）。看门狗 cron ppou2-kl-watchdog（每 10 分钟）保 kl_watcher 与镜像环存活。时间盒不变：2026-11-20 12:00 硬停。

## 主臂 it=21 0.15 首轮校准＋策略组裁定（2026-11-15 15:43–15:47）

- 校准值：主臂 it=21 落地 15:43:11，达成 KL **0.1128** @ kl_coef 0.15（games 4324、winrate 0.607、clipfrac 0.174，直读 train.jsonl 末行；看守 15:43:36 记 KL15_CALIBRATION_HIGH）。对照 it=20：0.1288 @ 0.05——系数 ×3 换来达成 KL 约 12% 下降，响应弱（与约 0.18 幂响应一致），0.05–0.07 带仍不可由系数到达、从来不是闸。
- 上报：按 10:49 裁定当轮报策略组（15:44 送出）；裁定 15:47：**HOLD 0.15，继续训**——it=22 在飞不拦，不动系数；单轮 ≥0.13、累计 KL 均值 >0.15 任一触发即停线后纠；已在 0.15 下若 ≥0.13 再现，按 10:49 裁 HOLD 待裁、不步进，当轮报策略组。
- 节奏勘误（策略组接受）：it=17→20 间隔 24.2/23.7/24.0 分钟，ETA 口径 ~24 分钟/轮不变（主臂 it=40 约 23:20，估计）；it=21 的 41 分钟是 resume 首轮开销。进度汇报趟「it=18–20 约 40 分钟/轮」为误读，作废。

## 主臂 it=22/23 续报＋节奏减速（2026-11-15 16:21–17:02，直读）

- it=22 落地 16:20:55，KL 0.1086 @ 0.15（games 4339、winrate 0.615）；it=23 落地 17:01:50，KL **0.1006** @ 0.15（games 4367、winrate 0.611）。三连 ≥0.10（0.1128→0.1086→0.1006），均 <0.13。it=22 策略组裁「仅备案，无新裁」；it=23 已当轮上报待复。
- 节奏：rollout 吞吐 resume 后逐轮下滑——games_per_min 184.9（it=20）→141.6→114.2→107.1；roll_s 1438→1833→2280→2447 秒；upd_s ~600、pre_s ~305 平稳。排除：机上负载 3.59/32 核（未饱和）、resume 未改配置（workers 14、argv 一致）、局数/决策数/units 平稳（局没变长，dec_per_s 920→532）。机制未定位，需 rollout 期 worker 状态采样＋server 队列探针。
- ETA 修正：按 ~40 分钟/轮，主臂 it=40 约 11-16 04:20（估计）；~24 分钟口径的 23:20 作废。对照臂未见同款减速。

## ★ KL 语义更正（2026-11-15 17:18，策略组亲核钉死代码 2c8262fb 成立，本节覆盖此前一切 KL 读法）

- **日志 `kl` = KL(π_ref ‖ π) 的逐决策均值 = 当前策略离冻结参考（π_ref = --init 本体）的距离，不是单轮步幅。** 代码据：docstring 第 12 行「pi_ref = frozen BC init」；--init 帮助（第 60 行）「also the KL reference unless --ref」；第 538–541 行 wref = load(args.ref) if args.ref else w0、ref 冻结；第 846–847 行 klu = (pr_·(lr_ − lpz)).sum(1)（pr_/lr_ 为参考策略概率/对数概率，lpz 为当前策略）；第 911 行落盘。ppou2 两臂启动 argv 均无 --ref（launch 记录直读确认），参考即 m3s2r1 本体、永冻。
- **it=21–23 重读**：0.05 段距离单调上行 0.0208（it=4）→0.1288（it=20）；抬系数到 0.15 后距离被往回拉：0.1128→0.1086→0.1006。**10:49 边界裁与 15:47「响应弱」读法建立在步幅误读上，作废**；+0.0466 是在距离 0.02→0.129 的行程里拿到的；距离回退 ≠ 强度回退，强度以检查点考为准。
- **两条停线的实际所指**：「单轮 ≥0.13」实为总距离封顶 0.13（比 rootset32 累计 KL 0.15 线先绑）；rootset32 累计 KL 是另一把距离尺。停线按策略组修正案重写（每锚距离上限 0.15 到界停考；单轮距离跳升 >+0.03 报告；检查点考另报离原始起点距离），修正案待策略组发来落笔。
- **即时令（17:18）**：主臂 it=25 边界停臂待裁；考 ckpt_25（标准规程，与 it=20 +0.0466 直接可比）＋报其落盘 kl；对照臂不动、其 it=20 落地加考同规程；裁决触发线三分支预先钉死（<+0.0367 → 从 ckpt_20 重锚 --ref=ckpt_20＋--kl 0.05 续训；≥+0.0466 且 CI 下界 >0 → 从 ckpt_25 续跑，点估 >0.0568 留 0.15 否则回 0.05、it=30 再考；介于两者 → ckpt_25 以 0.05 续跑、it=30 再考）。
- **--ref 预研（已验，无撞）**：training_contract 只含 reward/value_head/gamma/rscale/sep_value（第 387–390 行）；utility-source 合同校验在 --value-head tanh 下直接返回 None、不生效；--ref 在启动期第 538 行独立消费、与 checkpoint 无校验耦合——resume 改 --ref 不撞 contract。
- **次要项采纳**：检查点改每 5 轮一考（考试机上考、不停臂，本轮待裁停一次为例外）；新确认种子块 988000–988999（前半 Master2／后半 Echo2）注册为盒末确认考用；候选检查点提交讨论前加一场对 Echo4 包直打。
- **rollout 探针结论（17:08–17:19 采样 12 次）**：瓶颈定位在单个 server 进程（PID 270216，占第二 GPU 上下文 1092 MiB）恒钉 102% CPU（单核跑满）；14 个 rollout worker 恒 ~12–16% CPU（Sl 等待）；整机负载 ~3.5/32 核、GPU 利用率 19–33%——rollout 被 server 单核串行卡住，worker 与 GPU 均喂不满。upd/pre 平稳与此一致。

## 《ppou2 预注册修正案一 · KL 语义与锚规》（策略组，2026-11-15 17:19，原文存档）

1. **语义更正**：日志 `kl`＝KL(pi_ref‖pi) 逐决策均值（trainer 2c8262fb 第 12/60/538/846/911 行），pi_ref 为冻结参考（未传 --ref 时即 --init 本体）。此量是**离锚距离**，不是单轮步幅。此前一切按步幅的读法（预注册停线、10:49 边界裁、15:47 读法、it=20 判读的安心口径）作废。
2. **停线重写**（旧两条 KL 停线退役；ΔU 生存/停止线不变）：
   - 每锚距离上限 **0.15**：落盘 kl 达 0.15 → 轮界限停臂、开检查点考；其后锚移或线停，二选一，不许越界续跑。
   - 单轮距离跳升 **>+0.03** → 当轮报告（无真步幅日志下的代理警报）。
   - 每次检查点考另报**离原始起点距离**（对 m3s2r1 的 rootset32 KL），锚移后总行程保持可见。
3. **锚规**：锚只许落在**已确认检查点**上（标准考合并 ΔU 对原始冻结起点 CI 下界 >0）。移锚动作＝从该检查点 state resume、--ref 设为该检查点权重、--kl 0.05；移锚后落盘 kl 读作离新锚距离。声明基线永不变（一切 ΔU 永远对 m3s2r1 本体）。
4. **ckpt_25 裁决触发线**（本修正案首个适用件，照 17:15 令）：点估 <+0.0367 → 从 ckpt_20 重锚续训（21–25 只留证据）；≥+0.0466 且 CI 下界 >0 → 从 ckpt_25 续跑（点估 >0.0568 留 0.15、否则回 0.05）；介于两者 → ckpt_25 以 0.05 续跑、it=30 再考。
5. **考频**：主臂检查点考改每 5 轮一考（考试机上考、训练不停；本次 it=25 停臂待裁为一次性例外）。对照臂 it=20 加考（已令）、it=40 照原注册。
6. **臂差口径**：两臂系数史不同（主 0.05→0.15／对照 0.20→0.15、从未跑 0.05），it=40 臂差**不得读作 λ 效应**；对照臂角色重注册为距离对照。
7. **确认种子**：984000–984999 已花销结案；新确认块 **988000–988999**（988000–988499 Master2／988500–988999 Echo2）注册为盒末确认考保留块。
8. **候选直打**：拟提交检查点在提交讨论前加一场对 Echo4 包直打（诊断口径）；上榜闸仍归所有者。
9. **不变项**：时间盒硬停 2026-11-20 12:00；ΔU 生存线（点估 ≥+0.01 且 CI 下界 >−0.01）与停止线（连续两考合并 CI 上界 <0）照原注册。

## 效率处置（2026-11-15 17:29 策略组裁：server 数是服务基础设施、非备案训练口径，无须另裁）

- 机制坐实：rollout 推理 server 为单进程（PID 270216）恒 102% CPU 单核串行，14 worker 恒 ~14% 等待、GPU 19–33%；塌方与 snap_20 入盟、对手转发量 +97% 同刻。
- 执行：ckpt_25 考后 resume（任一分支）以启动器 v6（md5 2274b7c9…，RL2_SERVERS 1→2＋--ref 透传）拉起。注脚：① 首轮 gpm＋kl 当轮报；kl 跳升 >+0.03 先按修正案报告线报再续。② 变量口径：非重锚分支 --ref 透传不生效、实际只动 server 一个变量；重锚分支两件同动、轨迹自 ckpt_20 分段新读，不伤比较。仍饱和逐级 3、4，一次一级。
- it=40 ETA 待 v6 首轮 pace 出后按新速率重估，塌方期 ~40 分/轮口径作废。

## ckpt_25 检查点裁决（2026-11-15 20:0x，修正案一 §4 首应用）

- 考：受试包 ppou2_m_it25.py md5 7f61d2c4（ckpt_25.npz 8ff0f6d5，18:22 落盘），CONF 与基线逐字同；harness 354bcd1b；M2 种子 980000–981999、E2 982000–983999，各 N=2000；配对按 seed 对冻结起点基线（M2 两批取均值），bootstrap B=20000、rng 20261115；err 0。
- ΔU：M2 +0.0340 CI [0.0231, 0.0451]（候选 U 0.8582 vs 基线 0.8242；1906 胜、+1323.2）；E2 +0.0635 CI [0.0465, 0.0807]（候选 U 0.7260 vs 基线 0.6624；1616 胜、+678.7）；**合并 +0.0488 CI [0.0386, 0.0590]**（4000 簇）。
- §4 分支：点值 0.0488 ≥ 0.0466 且 CI 下界 > 0 → 0.15 段无害，从 ckpt_25 复训；点值 < 0.0568 → kl_coef 回 0.05。生存线通过。
- 距离：落盘 kl 0.0878（@0.15）；离原始起点累积 KL（rootset32 N=6336，自检 0.0）均值 0.0188、p99 0.0660、max 0.1758，较 it=20（0.0241/0.0981/0.2721）回落。
- Echo4 诊断场（附录，非 §4 输入）：ckpt_25 vs echo4_int8.py（819250bf），1000 局、种子 20291101–20292100：519 胜/481 负，场均 +72.5，均值 U 0.5159，基本持平。
- 残树＋载货 sidecar（每场前 100 种子升序，候选与基线各跑一次，身份门 100/100；样本读数）：M2 候选残树 4.10 vs 基线 4.00，载货 99.47 vs 93.97；E2 候选残树 3.87 vs 基线 3.61，载货 63.24 vs 46.48。如实读法：终局干净程度相对基线基本持平，ΔU 增益来自胜局更多；残树仍高于真人参考 2.1。
- --ref 预研：无冲突（training_contract 不含 ref；--value-head tanh 下 utility-source 检查直接返回 None；--ref 启动时独立消费 line 538；F6 只碰 gamma/rscale）。本次复训分支不传 --ref。
- 探针根因：RL2_SERVER_MODELS 默认 12 vs 联盟 13 → LRU 换入；单次载入价实测均值 0.055 秒（min 0.031/max 0.226）。三配置吞吐对照（1/12、1/20、2/20）待补——归因项非闸，首轮复训 pace 干净测完后补。
- 执行：主臂 20:06:36 以启动器 **v7**（md5 22d0f6bd…，SERVERS 2、MODELS 20；效率补文所写 v6 已被 v7 取代）从 ckpt_25 状态（sha 133ed992…）复训，kl 0.05，PID/PGID 313195，λ0.95 不变，--ref 不传。首轮 gpm＋kl 当轮报，验收线 ≥150 gpm；kl 跳升 >+0.03 先报再续。下考 it=30（M2 场改 Echo4 场，修正案二随裁决注册）。
- 对照臂 c it=20 边界重启已执行（18:39 停、18:40 v7c 重拉，SERVERS 1、MODELS 20；c_it20_restarted.flag 在位），兜底 cron ppou2-c-it20-restart-backstop 已于 20:0x 移除。c it=20 加考排在主臂首轮 pace 干净测完之后。
- 对照臂 pace 口径注：c it=21 单轮 258.2 gpm 为快轮离群值，it=20/22/23 稳定 ~185 gpm；看守 pace_ref 钉单轮最大值致正常轮误报 PACE_ALERT，已请作战室改钉（看守归作战室所有）。

## 《修正案二 · 考官退役 Master2》（2026-11-15，策略组随 ckpt_25 裁决注册，原文）

① 自 it=30 检查点考起，Master2 场退役；标准考双场＝Echo2（982000–983999 沿用）＋Echo4（考官包 819250bf，种子域 20292101–20294100；it=35 20294101–20296100、it=40 20296101–20298100，预注册序列）；② it=30 触发线（锚定 it=25）：点值 <0.0386 → 回锚 ckpt_25（--ref＝ckpt_25，--kl 0.05），26–30 轮只作证据；点值 ≥0.0488 且 CI 下界 >0 → 从 ckpt_30 复训；其间 → 从 ckpt_30 复训、it=35 再考；系数在各分支均维持 0.05，不再抬回 0.15（无新修正案不重启）；③ 时限终确认考（988 块）双场改为 Echo4（988000–988499）＋Echo2（988500–988999），口径不变、合并 1,000 簇、CI 下界 >0 立主张；④ 每检查点保留 1,000 局候选对 Echo4 包诊断，固定种子集 20291101–20292100（同局集、候选更替，系列可比）；⑤ 可比性说明：合并 ΔU 跨检查点不再考官全同，Echo2 场逐场 ΔU 为可比线索（it=20 +0.0534、it=25 +0.0635），Echo4 场逐场系列自 it=30 起记。

## 修正案二执行形态附则（2026-11-15 21:29 所有者定规，策略组附则，原文）

自 it=30 起，检查点考由 C1／C2 执行（it=30 分工：C2 跑 Echo2 场，C1 跑 Echo4 场＋Echo4 诊断），gpu-l20-6 不再做考试、只跑训练并负责交包与汇总裁决；考试设计（种子域／考官包／触发线／合并口径）不变。你线在跑的 c it=20 加考 E2 场原地跑完，属例外尾巴。

## 对照臂 it=20 检查点考裁决（2026-11-15 21:46；修正案一第 5 条新增考）

- 考题：对照臂 ckpt_20（/root/carbon/data/rl3/ppou2_c_mirror/ckpt_20.npz，13,972,532 B，md5 f043b1f0ef9dc73b871f96d46fb9a9d7）→ 包 /root/carbon/probe_m2/code/exp/ppou2_c_it20.py（md5 7a9a3e97，source_check ok）。
- 考场：gpu-l20-6 原地跑完（2026-11-15 21:29 所有者定规之例外尾巴；M2 场 20:48–21:09，E2 场 21:09–21:45，均为 2000 局、err 0、ah 7a9a3e97、hh 354bcd1b）。
- ΔU（B=20000，rng 20261115，vs 冻结起点）：M2 批均 **+0.0204，CI [0.0089, 0.0317]**（A/B 两批同值）；E2 **+0.0276，CI [0.0099, 0.0456]**；**合并 4000 簇 +0.0240，CI [0.0135, 0.0347]**。候选均值 U：M2 0.8446（起点 0.8242）、E2 0.6901（起点 0.6624）。
- 生存线：点估计 ≥ +0.01 且 CI 下界 > −0.01 → **通过**。
- 累积 KL（vs 原始起点，N=6336，自检 0.0）：**kl_mean 0.0109，kl_p99 0.0386，kl_max 0.0879**。落盘距离 kl（it=20）：0.0605 @ kl_coef 0.15。
- 读数：对照臂 it=20 相对冻结起点真实前进（合并 CI 不含 0），但幅度约为主臂 it=20（+0.0466）的一半、it=25（+0.0488）的一半；其累积距离亦远小于主臂 it=20（0.0109 vs 0.0241）。按修正案一第 6 条，双臂系数史不同，it=40 臂差不读作 λ 效应。
