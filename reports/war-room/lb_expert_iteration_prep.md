# 线 B 专家迭代预置备忘（m3s2r2）

作战室预置代理 2026-11-12 晚。状态：**预置完成、未立项**——立项闸门仍是 K4 探针
（h24，预注册过线＝SEARCH=True 均分 ≥−330（收回 ≥60% 的 −824 缺口）且 worst_ms ≤200，
见 reports/live/k4_probe.md）。探针未过线前不批量重标、不开训；过线后照本备忘直接派开工，
不再重新设计。方案母本：reports/live/replica_gap_plan.md 线 B（本备忘是其落地细化，
冲突时以本备忘的细化口径为准并回填母本）。

## 0. 缺口与路线（一行版）
m3s2r1 终局残树 6.8 vs 真身 2.1（ep10→ep19 反变差，BC 修不了）；专家迭代＝
用 clone+search 在终局状态上重标动作、混训出 m3s2r2，单变量只加「搜索重标终局样本」。

## 1. 终局状态来源（双源，均为 step≥270、每局最多 30 行）
### 源① 教师局终局段（语料重演）
- 原料：`data/m3s2_corpus/`（仓内 128 个回放 .json.gz＋`data/m3s2_matches.json` 118 条；
  **仓内没有 index.json**——它由 ladder_corpus.py 在建集时于机上生成，只存在于机上语料目录
  （如 DGX 本地 data/m3s2_corpus）。脚本 corpus 模式读 index.json，缺失时报错并指明此点，
  不许静默降级）。
  回放格式含逐步 observation 原文与下一步 action（cnet2_build.build_game 同款消费方式），
  因此 obs 流可逐字重演——Feat 与搜索代理都从 step 0 喂真 obs 预热（见 §2 机制），
  不存在「冷启动到 270 步」的问题。
- 取数口径：index.json 中 strategies 含教师名 `claude-AlphaCarbon-Master3s2` 的局，
  只取教师所在 side。教师行 row＝2（build_top 口径，Master3s 恒在 row 2，
  也是 Echo 包与 m3s2r1 wrapper 在推理时用的行）。
- 样本量：ds_m3s2 全集 35,282 行（训 31,993 / 验 3,289）；其中 step≥270 的训练行
  3,103（全教师合计，Studio 加权试点实测 9.7%）。教师本人（teacher==2）的终局行数
  开工第一步在机上实测（ds 目录一行命令：
  `python3 -c "import numpy as np,sys; d=sys.argv[1]; s=np.load(d+'/step.npy'); t=np.load(d+'/teacher.npy'); print(int(((s>=270)&(t==2)).sum()))" <DS_DIR>`），
  预计 ~1–1.5k 行量级（以实测为准，不许拿估计派工）。
- ds 本体位置：DGX `/home/arm/junyang/carbon/data/ds_m3s2`、Studio `~/junyang/carbon/data/ds_m3s2`
  （21 文件/368M，2026-11-12 17:59 直读在位）；仓内只有语料没有 ds 本体，开工机二选一或先同步。

### 源② 替身自对弈终局段（主力增量）
- 玩法照 dagger_gen 骨架：FastCarbon 局，clone 纯网（cnet2_agent.make_agent，
  weights=cnet2_m3s2r1L.npz、teacher=2）驱动我方，对手只许训练池网
  （脚本 POOL 写死：v6t5g/v5t5g/v7s57/v9s57/style_t0，**绝不用考官**，与 dagger_gen 同规矩）。
- 状态由 clone 的真实动作推进（DAgger 口径：分布是 clone 自己会到的分布），
  标签由同局并行的 search 代理给出（见 §2）。
- 规模 v1：100 局 × 30 行 ≈ 3,000 行；种子段 510000+（与考试种子 99xxxx、
  dagger 3000 段、probegen 777000 段均不重叠）。局数可按源① 实测行数调，
  目标两源合计 ~4–4.5k 行。

## 2. 搜索重标管线（脚本已备，dry-run 已验）
- 脚本：`code/lb_relabel.py`（与本备忘同批入仓）。双源双模式：
  `--src corpus CORPUS_DIR TEACHER_NAME ROW CLONE_W` / `--src selfplay CLONE_W ROW SEED0 N [WORKERS]`，
  另有 `--dry-run` 单局计时模式。
- 搜索口径与 K4 探针严格一致（同一把尺）：cnet2_search.make_search_agent，
  conf＝fleet DEFAULTS（MAX_BR=16、BUDGET=0.38、MIN_DEPTH=3、MODE=sweep），
  只改 weights＝clone 权重、TEACHER＝2。ROLL 仍用 s64
  （依赖 `data/weights/cnet2_s64.npz` 在开工机存在——spot-25 冒烟教训：先查这个文件再开跑）。
- 预热机制（关键设计）：make_search_agent 的 cfg 挂在 agent.cfg 上可写。
  代理以 SEARCH=False 建号、从 step 0 逐 obs 调用（纯网成本、内部 Feat/信念与实战同路预热），
  到 step 270 把 cfg["SEARCH"] 翻 True——只有终局 30 步付搜索成本，且标签与行特征同源于一条 obs 流。
- 行规则：y＝Feat.labels(search 动作)；Prefix 的 uf/mf/legal 按新 y 做 teacher forcing
  （与 build_game 对原 y 的做法镜像）；uw＝与 clone 默认动作不同的自家槽 3.0、相同槽 1.0、
  空槽 0（dagger_gen 的 DG_UW_CHANGED 约定）；teacher＝2；value＝tanh(终局分差/3000)；
  碳图 full_carbon 口径同 build_game。产物：每局一个 shard（FIELDS＋uw）＋games.jsonl
  （含 changed 数、searched 数、单步 ms，成本与改标率全程可审计）。
- **dry-run 实测（2026-11-12 本机沙箱，b128_q8 代 clone，语料模式，1 局真实对局）**：
  局＝`0efcfa12…-1`（Master3s2 座 0 对 Godslayer-CV，真身终分差 +2250.7）。
  结果：预热 270 步 676.6s（≈2.5s/步，含 clone＋search 体两次解码）；翻 SEARCH 后终局
  29 步**搜索触发 29/29、errors=0**，单状态耗时**均值 2.85s / 最大 4.99s**（本机口径）；
  产出 29 行、监督槽 345，shard 与 cnet2_build.FIELDS 的 dtype/shape 逐项校验通过（schema_ok）。
  自对弈模式另跑一局（对 pass 对手）：全程零报错走完 300 步，但该局无争夺点、搜索未触发，
  只证明 cfg 翻转与管线稳定性，不计入耗时口径。
  **警示（必须带进开工）**：本局搜索对 clone 默认动作**推翻数＝0**（stats changed=0），
  而重标标签与真身实际动作在 63/345 槽（18%）不同——本局 +2250 已定盘、无争议终局，
  推翻为零有合理解释，但也意味着现口径下重标可能退化为 clone 自蒸馏。开工先测聚合
  推翻率（见 §5 第 3 步），低于阈值不许跑全量。
  注：本机为共享 1.5 核 VM 且当晚负载高，绝对值只作量级参考（考场机净解码步约 40–190ms，
  预计整体快约一个数量级）；开工机以语料模式单局复测为准。
  外推口径：全量 ≈（源① 局数 × (预热 ~270 步纯网 + ~30 步搜索)）＋（源② 局数 × 整局），
  考场级 CPU 估计语料模式约 35–90s/局、118 局 7 worker 约 10–25 分钟（估计值，开工回执给实测外推）。

## 3. 混训配方（m3s2r2 草稿，单变量）
- 合并：`merge_dagger.py ds_m3s2 <lb_out> ds_m3s2_lb`（DAGGER_W 默认 1.0）。
  **必做修正**：merge_dagger 会按 crc 给 ~10% 的 dagger 局标 val——lb 局一律改回 train
  （合并后对 meta.json 中 src=="dagger" 的 games 置 val=false 并读回验证），
  保证 m3s2r2 的验证集与 m3s2r1 逐字相同、val 可比。
- 混比 v1：base 35,282 行不动＋重标行 ~4–4.5k（约 11%）；终局切片内重标行与原终局行
  约 1.2:1。不叠线 A 的 ×3 复制（那是线 A 的变量，叠了归因即死）——r2 与 r1 的唯一差异
  就是这批重标行。
- 训练：与 m3s2r1 完全同配方——BC_INIT=cnet2_arch_n1_last.npz，
  cfg {C:128, NB:8, D:128, ZI:1, TA:4, TNH:4}，env BC_RAM=1 BC_USE_W=1 BC_USE_UW=1
  BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4，20 epochs，单臂（注意力版），
  产物 cnet2_m3s2r2.npz。开训走单链脚本口径（建集→合并→开训一气呵成，三件套自证）。

## 4. 验收（预注册草稿，过线立项时冻结，不许事后改）
- 考官与 r1 终标同口径同种子配对：vs Master2r 200 局（seed 996000）、vs E0 200 局
  （seed 996200），bigeval＋behavior_stats 同步出行为统计。
- 主指标：终局残树 6.8 → **≤4.5 为过线**（约收回一半的 4.7 缺口）、≤3.5 为强过线；
  副指标：两场均分任一场回退超 r1 的 1 SE（r1 对 Master2r：+1197/se 63；对 E0 的终值
  与 se 以 reports/live/exams.md 落盘为准）即判不过；种树量只监视（应在 ~39–40 带内，
  掉出带说明重标伤了基本盘，同样判不过）。
- 终局 30 步得分曲线：harness 字段**已落地**（eval_v53.py commit 5e5afd51 逐局线含
  cash270＋终局 cash/cargo，bigeval.py seqc 回填 e27200da）——曲线自 r2 定标起可直算，
  **升为主指标**：终局曲线向真身 +2093 靠拢（与残树并列判定，口径＝逐局
  (终局分差 − cash270 分差) 的均值，r1 同种子同法算基线）。部署前提：r2 两场开考前
  先同步仓内 eval_v53.py 并核 md5 **6d24da0e**，不碰在跑的场。
- 不过线处置：r2 不进池、不叠第二轮重标（先复盘 changed 率与搜索在终局的实际改标模式再议）。

## 5. 探针过线后开工清单（应只剩执行，不再设计）
1. 作战室指派开工机（重标是 CPU 活：DGX 或空闲考机；训练再按 first-free）＋
   核 clone 权重在位（cnet2_m3s2r1L.npz，md5 b6db08c9b076257c54817377bb93130d）与
   data/weights/cnet2_s64.npz 在位。
2. 开工机先以语料模式跑 1 局复测单步耗时 → 外推全量工时回作战室（再开批量）。
   （本机 dry-run 已验通路与格式，此步只复测速度并验机上 trace/index 衔接。）
3. **聚合推翻率断言（新增，源于本机 dry-run 的零推翻警示）**：先跑约 10 局语料，
   统计 changed 槽占监督槽比；**≥2% 才放行全量**；低于 2% 立即停手回作战室复盘
   （候选处置：降 MARGIN、或状态来源改以势均力敌局为主），不许先跑完再说。
   过闸后批量：源① 全教师局＋源② 100 局，4–8 worker。
4. 合并＋val 修正＋计数断言（总行数＝base＋shard 行数；重标行数、changed 率入回执）。
5. 单链开训 cnet2_m3s2r2 → 定标＋行为统计 → 按 §4 裁决，结论回填 replica_gap_plan.md。
