# K0 数据源 A 路生产规格（新产专用 ≤128 局）

- 状态：**草案待排产**（工程组起草 2026-11-14；同日按策略组批二新 producer/build 修订；本规格只起草，未执行生产）
- 裁定依据：策略组已裁数据源选 A——用已修复 producer 专产 ≤128 个独立局，建成认证 generation 后由 `k0_aux_labels.py` 打标，供 `k0_train.py` 的 A0/A1 对照消费。gen1r shard0 保留给终局 pilot、Stage B 池不碰。
- 起草基线：main HEAD `d72e1de9`（2026-11-14 批二落地后修订读回；初稿基线 `4d5ab97a`）。代码钉（生产预检须逐件复核，任一不符先停后报）：
  - `code/producer/selfplay_gen.py` md5 `bd62ead949c93d70afaa7dc10b9d5e8e`（批二版：`_run_identity` 的 code_md5 已补入 cnet2_feat/cnet2_np，第 1 节）
  - `code/cnet2_build.py` md5 `8598f26de678450a2dd24e308107416e`（批二同批落地的 B 包版；本规格第 5 节 CLI 形态已按此版复核：`--resolve` 在位（:1500）、u_raw 按类计算（:394-400）、CNET2_ALLOW_MIRROR 纪律不变。若排产前 build 再变，仍须先重钉并复核第 5 节再产）
  - `code/build_top.py` md5 `08d2ab337b8584b2907f1fc7f499041d`
  - `code/k0_aux_labels.py` md5 `f3b40b3b789e7c3116fd235534df42b1`（与定版契约值一致，起草时读回核对 ✓）
  - `code/k0_train.py` md5 `f16888849cfa7061ef1bb6bb81eba423`（与定版契约值一致，起草时读回核对 ✓）
  - `code/cnet2_search.py` md5 `94f7fb6438bb7577b141e388f068e122`（producer identity 的 code_md5 成员）

---

## 1. Producer

- 入口：`python3 code/producer/selfplay_gen.py`（argv 定义 :663-683）。产出布局：`<out>/index.json` + `<out>/replays/*.json.gz`（文件头 :4），另写 `<out>/identity.json`（:812-815）。
- 关键 argv（生产命令模板见第 5 节，此处为口径）：
  - `--weights`（必填）：**单一文件、双座同权重**——producer 没有双文件接口，两座用同一 npz 对弈（自博弈如实申报，见第 2 节）。
  - `--name m3s2r1`（缺省即此值，:669）：双座同名；`--name-a/--name-b` 本次不用。名字决定建集时教师行归属（第 5 节），不许改名。
  - `--open-t 1.0 --open-steps 50`：开局采样，多样性的唯一来源。取值取 gen-1 重造定版先例（spec §16「多样性：--open-t 1.0、--open-steps 50」）。机制：前 open-steps 步以温度 open-t 采样，每局派生开局种子 `(open_seed_base × 1000003 + game_seed) mod 2^31`（文件头 :84-88）。码内 CONF（:176-179）无 OPEN_T 键、缺省 0.0——**不传 --open-t 即全贪心镜像局**，故此二参数为生产命令的必填项（形式上是可选 argv、口径上必填）。
  - `--open-seed 988`：开局种子基（显式传，不用缺省 0；与种子域同号便于回溯，见第 3 节待裁点 3）。
  - `--allow-mirror`：**禁用**（不传）。闸链：无开局采样且不传此旗，producer 在任何一局开打前以 exit 2 拒绝（:695-700）；带开局采样时若成品 mirror share > `MIRROR_FAIL_SHARE = 0.5`（:220），以 errors 形式汇入结果并 exit 1（:818-823、:840-842）。本规格靠这两道原生闸，不另设人工检查替代。
  - `--max-games 128` 与 `--max-seconds 86400`：均为**全程预算**（argv :672-674；调度集合 :722）。缺省值（20 / 180.0）会把本批截断，必须显式放大；局数的硬约束不在这里，在种子域宽度（第 4 节）。
  - `--workers 56`：取 producer 文件头用法示例值（:9，DGX 级机器）；实际值以排产机器为准并记入回执。注：预热局用种子 `seed0 − 1000 − i`（i < workers，:733-735），种子域清算已把此段计入（第 3 节）。
  - 其余 CONF 不动：以 producer 码内 CONF（:176-179，GUARD=True、GUARD_NARROW=True、TEACHER=2、TEACHER_LATE=2、LATE_STEP=240 等）为准，生产不改码内 CONF；§16 定版的生产 CONF（GUARD_CASH=40 等搜索侧项）走 cnet2_search 缺省，与 gen1r 同口径。
- 多样性配置如何进产出 manifest：`diversity_of()`（:228-239）生成的块 `{mode, open_t, open_steps, open_kinds, open_seed_base, open_seed_rule, allow_mirror}` 是 run identity 的组成部分，随 identity 写入 `<out>/identity.json`、**每个 index 条目与每个 replay**（文件头 :88-91）；终局汇总行 `SPGEN-RESULT`（:839 打印，result 字典 :825-838）另带 `mirror_signature_games` 与 `mirror_signature_share`（:835-836），且新版 result 自带 `wall_s`/`total_wall_s`/`games_per_min`（:825-826，墙钟口径见第 6 节）。建集后 generation 的 meta.json 还有 mirror_gate share 一道（验收口径见第 7 节）。
- **identity 覆盖（原缺口已由批二关闭）**：初稿基线（producer `523963c2…`）的 identity `code_md5` 未含实际打局依赖的 `cnet2_feat`/`cnet2_np`（由 cnet2_search 导入、决定信念特征与 NP 推理），当时策略组裁定带标注＋带外钉生产。**批二已把两件补入 `_run_identity`**（:242 起，feat/np 循环 :249-258，经 sys.modules 取实际加载文件的 hash），identity 自此自描述——**初稿的「缺口标注＋产前产后带外钉、任一变化整批作废」条款作废**，改为验收口径：identity.json 的 `code_md5` 必须含 `cnet2_feat.py`/`cnet2_np.py` 两键且与生产机实物 md5 一致（现行值 feat `2a45b08b6f97508541e62867e408748e`、np `d8c15ad4fdd8ee079363cf215d52daf3`；验收见第 7 节第 2 项）。产局窗口内的代码冻结仍由第 0 节代码钉兜底。

## 2. 权重（双座来源与 K0 起点同源）

- **单一源文件定版**：producer 的 `--weights` 与 k0_train 的 `--init` 用**同一个文件**——`cnet2_m3s2r1L.npz`，md5 **`b6db08c9b076257c54817377bb93130d`**。仓内两处记录互证：`reports/live/packs_ledger.md`（Echo3 行，源权重）与 `reports/live/exams.md`（m3s2r1 定标判定行，「权重 cnet2_m3s2r1L.npz（md5 b6db08c9…）可进 snipe 池」）。该文件不在 git 内（机上工件，正典副本在 DGX/L20）——生产预检必须在生产机上对实物算 md5 与此值核对，不符即停，不许以「同名文件」放行。
- 接口差异与取舍：
  - producer 只收一个 `--weights`，由搜索代理（cnet2_search）执子；k0_train 的 `--init` 收的是「cnet2_train export .npz」（`k0_train.py:270` 的 help 原文），且 A0/A1 协议要求显式声明的共同起点（:293，无 --init/--resume 即拒启）。`cnet2_m3s2r1L.npz` 本就是复刻链的 cnet2_train 导出件，一件同满足两边接口、零转换——这是选它作单一源的根本原因：identity 里的 `weights_md5` 与 K0 回执里的 init md5 必须逐字相等，不留「同源不同件」的解释空间。
  - 与 gen1r 先例的偏离及理由：gen1r 双座用的是 `cnet2_m3s2r1_heads_v2.npz`（spec §16，md5 仓内仅记前缀 `06a03d9c…`）。K0 不沿用，因为 (a) K0 的监督（终局 U、死亡事件）全部由对局轨迹事后实现，不读任何 stage0 头；(b) 打局路径不消费 stage0 头——`cnet2_search.py` 全文无 utility/wdl/margin 引用、`selfplay_gen.py` 与之皆无 `want_u=True` 调用（cnet2_np 的 utility 读取仅在 want_u=True 时发生、且缺键即 KeyError，`cnet2_np.py:179-187`）——heads_v2 与 L 的差异键（stage0 三头）不在打局因果链上；(c) 反向哨兵：若打局路径真需要 utility 头，用不带该头的权重会在第一局 KeyError 响亮失败，不会静默改道——预检的计速切片（第 6 节）即覆盖此哨兵。
  - 不选 `cnet2_m3n1L.npz`（md5 `ad5af484…`，181 键）：它是 L20 在跑的 snipe_n1 线的祖先 init（活线谱系中段），且其定标站位是 m3s2r1 之前的旧代；K0 的起点应是已冻结、已定标的正典裸网（m3s2r1 定标：校准 +412、对 Echo2 +472，exams.md）。

## 3. 种子域

新域定版：**game_seed ∈ [988000, 988128)**，即 `--seed0 988000 --seed1 988128`（seed1 为开区间上界，调度集合 = `range(seed0, seed1)` 去掉已产局，producer :722）。

已实查的占用表（game-seed 命名空间；查证范围＝STAGE1 spec、reports/live/exams.md、packs_ledger.md、gate 与 bigeval 源码）：

| 域 | 用途 | 出处 |
|---|---|---|
| 991000–991019 | producer 文件头用法示例（原 gen 时代示意），按已占处理 | selfplay_gen.py:9 |
| 994000–994199 | n1r 定标考试场（h23） | exams.md |
| 994400–994799 | n1r 考试扩跑场；gen1r shard1 的一局 selfplay-994577 数值落在此段内——994xxx 整段按争议/已占处理 | exams.md；spec §18 附记二 |
| 995300 / 995310 | 教师闸 --seed-scene / --seed-anchor 的 RNG 基（pilot「同根 995300」）——RNG 命名空间，非 game-seed，但数值邻域避让 | stage1_teacher_gate.py:686-689 |
| 996000–996199 | Master2 定标考试场（全队标准场） | exams.md 多处 |
| 996100 | 教师闸 --seed-cont RNG 基（数值上落在考试场内，既存重叠、非本规格所能改） | stage1_teacher_gate.py:688 |
| 996200–996399 | E0 考试场（seed 996200 × 200） | exams.md |
| 999000–999199 | Echo2 定标考试场 | exams.md 多处 |
| 999200–999399 | E0 基线场（seed 999200 × 200） | exams.md |
| 999901 | 冒烟单局 | exams.md |
| EVO_SEEDS | bigeval 的 env 指定种子集，取值即上面各考试场 | bigeval.py:36 |
| Stage B 池 | 1,525 局存档局以局 key 标识（DGX `data/stageB_pool.json`），非 producer 种子域；其根集抽样走 seed-scene 995300 | 舰队记录 |

清算结论对新域：`988xxx`、`986xxx`、`987xxx` 在上述全部查证文档中**零命中**；新域上界距最近的已占块（991000）2,872 个种子；producer 预热种子段 `seed0−1000−i`（workers ≤ 56 时为 986944–986999）同样零占用。残留风险一条：gen1r shard0 的确切种子段仓内无记录（仅 shard1 的 994577 被钉住），若 shard0 曾用 988xxx 本表会有洞——概率判断与处置见待裁点 1。

## 4. 局数

- 定版 **128 局整**（种子域宽度即 128）。
- 理由：(a) 这是作战室上限（≤128）与消费侧硬闸的交点——`k0_train --max-games` 缺省 128 且其语义是「取前 max_games 个局」（k0_train.py:281、:357），128 局不多不少全量消费，不产生「产了不用」的尾巴；(b) A0/A1 的推断单位是局（按局聚类），独立局数取满上限即统计分辨率取满，辅助监督的效应量级预期不大（ΔU 0.02 级），没有理由在上限内留白；(c) 成本侧：128 局是 gen1r 已执行过的 400 局批的 32%，且后续打标/建集都随局数线性，量级已验可承受。
- 预期行数量级（供回执对账、非闸）：按 gen-1 密度 598 位/局（spec §16），约 76,500 行；死亡监督的有效槽位与正例数以打标 sidecar 的实测计数为准（第 7 节），不预设。
- 硬闸在生产命令里的体现（三重，缺一不可）：
  1. **种子域宽度**：`--seed1 − --seed0 = 128`，producer 的调度集合被 range 锁死（:722），域外种子不可能被产出——这是真正的硬顶；
  2. `--max-games 128`：全程软预算（其 help 明示批界执行、可 overshoot 至 workers−1 局）——注意它只能提前停、不能突破种子域，写它是为了让回执里的预算字段与域宽一致，不许把局数控制寄托在它身上；
  3. 消费侧 `k0_train --max-games 128`（显式传，尽管是缺省值）：A0/A1 任何一臂的 games_used 都不得超过 128，eff 行可验。

## 5. 建集与打标路径（确切步骤）

生产机预检（任一不过即停）：
1. 代码钉复核：对第 0 节六件 md5 逐件算并记录；weights 实物 md5 = `b6db08c9…` 复核。（批二起 feat/np 已入 identity，产前不再带外钉；产后由 identity.json 的 code_md5 覆盖验收，第 7 节第 2 项。）
2. 计速切片先行（第 6 节），切片局即正式局（种子 988000–988007 在域内，producer 的 resume 语义保证正式跑只补缺局）。

生产命令模板（路径以排产机为准，占位符 `<>`；本规格起草方不执行）：

```
# (a) 产局（DGX，war room 排窗）
python3 code/producer/selfplay_gen.py \
  --weights <weights>/cnet2_m3s2r1L.npz \
  --seed0 988000 --seed1 988128 \
  --out <corpus>/k0_a1 \
  --workers 56 \
  --name m3s2r1 \
  --open-t 1.0 --open-steps 50 --open-seed 988 \
  --max-games 128 --max-seconds 86400
# 注：不传 --allow-mirror。SPGEN-RESULT 行与 identity.json 留存。
# 产后核 identity.json：code_md5 须含 cnet2_feat.py/cnet2_np.py 两键（第 7 节第 2 项）。

# (b) 建集（教师行：双座名 m3s2r1 须落在 row 2——Echo 包所用行，
#     build_top.py 以 CNET2_TOP_TEACHERS 覆写 F.TEACHERS 后调 cnet2_build.main）
CNET2_TOP_TEACHERS="m3s2r1@2" \
python3 code/build_top.py <ds>/k0_a1_ds <corpus>/k0_a1
# 注：不许带 CNET2_ALLOW_MIRROR（§16 回执纪律）。
python3 code/cnet2_build.py --resolve <ds>/k0_a1_ds
# --resolve = load_generation(verify=True)：打印并验封代 generation id；
# 批二新版对带 meta_sha256 封印的代连 meta.json 字节一并验（发布时 manifest
# 写入 meta_sha256 = sha256(meta.json)，build :1349；读侧 :974-978）。记入回执。

# (c) 打标 + 验签（k0_aux_labels.py CLI 形态见其文件头 Usage）
python3 code/k0_aux_labels.py <ds>/k0_a1_ds <workers>
python3 code/k0_aux_labels.py --verify <ds>/k0_a1_ds
# (c) 的 stdout JSON（含逐局 first_terminal / terminal_cause /
#     physical_deaths / clearing_deaths 与 summary）原样进数据回执；
# --verify 须返回 {"ok": true, ...} 且 exit 0。

# (d) 入口启动校验（证据形式，非训练；updates = 0）
python3 code/k0_train.py --data <ds>/k0_a1_ds \
  --init <weights>/cnet2_m3s2r1L.npz \
  --aux-dir <ds>/k0_a1_ds --selftest-routing --device cpu
# 须 exit 0；k0_effective 与 selftest_routing 两行 JSON 进数据回执。
# 此步走完 generation 钉 + sidecar 校验 + 有限性闸全链（校验在训练
# 循环之前，k0_train.py:314-357 → eff :426 → selftest :501-523）。
```

口径注记：
- set_uw **不在**本路径内：k0_train 的 anchor 是纯净联合 CE、不带 w/uw 乘子（其文件头口径），U 监督来自 generation 的 `u_raw`（build 由 producer index 的 realized margin/u 按类计算，`cnet2_build.py:86-91、:394-400`）；k0_train 对无 u_raw 的数据集拒启（:314-323），故建集成功且 (d) 过，即证明 u_raw 在位。将来别的消费者要 w/uw，按派生层注册纪律另行，不在本规格。
- first-terminal 与死亡分类：打标器（闸①②已验收版）逐局给出 `first_terminal`、`terminal_cause`（status / extinction）与 physical/clearing 两类计数，sidecar 逐局 lineage 携带同字段——数据回执直接引用 (c) 的 stdout 与 sidecar，不许二次转述数字。

## 6. 预计墙钟

- **无实测依据，先声明**：仓内没有任何一条 producer 吞吐的干净实测记录（gen1r 的 DGX 回执只记局数/位数/验收量，不记产局耗时）。以下只有两条弱锚，性质标注清楚：
  - 配置缺省锚（非实测）：producer 缺省 `--max-games 20 --max-seconds 180` 暗示作者预期量级 ≲9 秒/局（参考机、56 workers 示例口径下）；
  - 事件窗锚（上界非速率）：gen1r 400 局在 2026-11-13 晚 §16 验收（23:10）至 shard1 建集裁定（2026-11-14 00:12）之间于 DGX 产完，但同期 DGX 多线并行，只能 bound「400 局 ≤ 一晚」，不能反推单局速率。
- 因此本规格把计速写成生产的第一步而非估计的替代品：先以与正式跑完全相同的 weights/CONF/workers 跑 8 局计速切片（988000–988007），实测墙钟 T₈；外推全批 ≈ T₈ × 16（切片局计入 128，不重产）。回执记 T₈ 与外推值；**若外推超过作战室给的产局窗，跑完切片即停、回执上报后再续**（producer 的 resume 语义使停续无损）。注：批二 producer 的 SPGEN-RESULT 已自带 `wall_s`/`total_wall_s`/`games_per_min`，T₈ 直接读切片跑的 result 行，不必另设测速。
- 量级预估（估计、待切片证伪）：若单局落在 5–30 秒带、56 workers 并行，128 局的纯产局墙钟在数分钟到数十分钟量级；建集与打标为 CPU 线性处理，同量级或更低。此数只用于排窗直觉，不作验收依据。

## 7. 验收清单（数据回执必含项，缺项不验收）

1. **局数与种子**：index 条目恰 128，game_seed 集合恰为 988000–988127（无域外、无缺号）；SPGEN-RESULT 行原文。
2. **身份链**：identity.json 的 `weights_md5` = `b6db08c9…`，且与 K0 侧 init md5 一致；生产时点的代码钉六件；且 identity.json 的 `code_md5` 须含 `cnet2_feat.py`/`cnet2_np.py` 两键、与生产机实物 md5 一致（批二起 identity 原生覆盖打局依赖，初稿的缺口标注与带外钉条款已作废）。
3. **多样性**：identity.diversity 块齐全且 mode = open-sampling、open_t = 1.0、open_steps = 50、open_seed_base = 988、allow_mirror = false；SPGEN-RESULT 的 mirror_signature_share 与建集 meta 的 mirror gate share 均 < 0.5（预期 ≈ 0）。
4. **建集**：`--resolve` 输出的 generation id、总行数、manifest `cfg_fp`（即 sidecar lineage 的锚）与 `meta_sha256` 在位（批二新版封印，--resolve 验过即证明 meta.json 未被动过）；建集命令的环境（CNET2_TOP_TEACHERS 原文、无 CNET2_ALLOW_MIRROR 的声明）。
5. **exact / APPROX 口径（策略组裁定落点）**：generation 内 `label_status` 分布——exact(0) / APPROX(1) / UNUSABLE(2) 的行数与占比。预期全 exact（producer index 每局带 realized margin/u，build 按类算 u_raw，`cnet2_build.py:394-400`）。判定：UNUSABLE > 0 即异常、先查因后验收；**APPROX 占比 > 0 时，回执必须追加 exact-only 敏感性复算**（仅 exact 行的 mean u_raw、行数、死亡监督有效槽/正例数，与全量口径并列）。闸豁免不扩围——k0_train 现行口径（仅 UNUSABLE 的 NaN 可豁免）不因本批数据改动。
6. **U 非退化**：u_raw distinct 数与均值（gen1r shard0 参考点：distinct 384、mean 0.54——参考非闸）；margin 为 0 的局数单列（与 mirror share 对照）。
7. **打标与验签**：`k0_aux.json` 在位、format `cnet2-k0-aux/1`；`--verify` 输出 `{"ok": true}` 原文；逐局表（gi / first_terminal / terminal_cause / physical_deaths / clearing_deaths）原文，及两类死亡的全局总计、first_terminal 分布一句话摘要（有多少局打满、多少局早终）。
8. **入口启动校验**：(d) 的 exit 0 + `k0_effective` JSON（其中 games_used ≤ 128、data_fingerprint.generation = 第 4 项的 generation id、lineage 与 row_key_sha256 在位）+ `selftest_routing` JSON 原文。
9. **墙钟**：计速切片 T₈、外推值、正式跑实际总墙钟、workers 实值（第 6 节口径）。

---

## 待裁点（起草方拿不准、须作战室/策略组裁）

1. **gen1r shard0 种子段无仓内记录**：占用表只能钉住 shard1 的 994577；若 DGX 侧留有 shard0 的 seed0/seed1 记录，请回填本表复核 988 段无冲突后再排产（或直接确认放行并把残留风险记入回执）。
2. **排产机器与 workers**：本规格按 DGX + 56 workers 写（producer 示例口径）；若改派其他机器，计速切片的外推与窗口须按该机重估，回执记实值即可，但请作战室在派单时点名机器。
3. **--open-seed 基值 988**：gen1r 定版段落未记其 open_seed_base 先例值，此为本规格自选（与种子域同号、便于回溯）；若策略组有既定基值口径，以其为准、改此一处即可。
4. **计速切片 8 局计入 128**：本规格按「在域内、计入总数」写；若作战室希望切片独立于正式批（域外另设），种子域需相应扩裁。
