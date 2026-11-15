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
