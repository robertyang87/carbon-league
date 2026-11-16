# r4 标注器修复线回执（X03 / X04 两半 / X05 / X09 / join 镜像）

对象：`code/k0_annotate.py`（基线 cdee7c03c5be68001eeef95eab356d95）、`code/cnet2_exit_train.py`（基线 23b92d70889038c527ad89ce12019d9d）、`code/k0_join.py`（基线 78363299a7da04e09724c7327f0d1c86）。gate 全程未碰（落仓前后均读回 e922d9d890c3e1eff6643e947289acb0）。证据在本目录 `evidence/`、探针在 `probes/`、版本链在 `versions/`。

## X03 · k0_annotate 恢复身份绑定（含 P3(a) 消费半边）

**复现（基线探针 probe_x03_plan）**：同一回放文件换字节（gzip 重序列化／JSON 加顶层键）后，根集不变（8 根）、run identity **逐字不变**——内容替换静默过片坐实；代码 md5 只进 publish manifest、不进 identity（亲读基线）。

**修法（v1）**：`build_plan` 开工即冻结身份——每局回放单次读字节，sha256 与根枚举同源（`_enumerate_roots_bytes`），identity 增 `replay_sha256{局:sha}`；`code_md5s` 与 publish manifest 同源（同一 `code_md5s()`），且 **publish 改用 plan 冻结的 code_md5s，缺失即 die**（不再 publish 时现读）。恢复双层核：progress 层比对（内容变更 die 点名局并列双 sha；代码变更 die 列 differing keys 含 code_md5s）＋分片层 `_verify_shard` 同口径复核（P3(a)：按 id 解路径读回放前先验内容哈希，不符 die 点名局）。旧 progress/shard 无绑定记录：**显式迁移旗口径**（W01 先例）——默认 die「UNBOUND」点名缺失键；`--accept-unbound-identity` 采纳时以当前字节补绑并在 progress 落 `identity_migration{missing,flag,time,adopted_code_md5s,adopted_replay_sha256}`、代 meta 记 `identity_migration: progress|shard`；内容 sha 已记录而矛盾者与旗无关、一律 die。

**验收**：
- plan 级：两种 doctored 回放均使 identity 变化，`replay_sha256` 与 doctored 字节逐字合（probe_x03_plan fixed）。
- 续跑·内容替换：v5 成品 out 副本＋994001 回放 regzip → die 点名 selfplay-994001 并列 recorded/current 双 sha（评估前）。
- 续跑·代码替换：annotator 加一字注释后续跑 → die，differing keys = ['code_md5s']。
- 续跑·旧身份：基线产出 out（无绑定）＋v5 续跑：默认 die UNBOUND；加旗后两分片全 skip-verified（零重算）、发布完成、progress 落 migration 记录。
- kill-9 续跑回归：见「回归」节。
- 消费侧 verify 与 resume identity 分开：gen-1dcc9611575bfbe3 的 `--verify` 在 v5 下照常通过（见回归节），X03 新绑定不误杀既有真实产物。

**新 md5**：v1 = ce2e597a59aaaadb6cd5d35330ccea87（commit 733a75a52787）。

## X04（报告本体）· 教师权重加载单读绑定

**复现（probe_x04w 基线）**：plan 按路径记 weights_md5=A（06a03d9c…）；把权重文件一处 conv 权重放大 1.5 倍换盘（md5 变 6aa3e41e…）后，按路径加载的评估 Q 由 0.9068351649458328 变 0.9068266110710121——记 A 跑 B（评审实证同型，其换法 Δ 更大：0.982→0.018）。

**修法（v2）**：`build_plan` 单次读权重字节，`weights_md5` 取该字节 md5，plan 冻结 `weights_bytes`；worker 初始化改吃字节（`io.BytesIO`），hash 与解析同源，评估全程不再按路径重开权重文件。

**验收**：同探针固定版——换盘后 Q = 0.9068351649458328，与教师 A 参考逐位同。

**新 md5**：v2 = 9e7871fd86428c21a67ce0825c425f06（commit 4fffc0a54f9f）。

## X04 同类增补（我线自查、报告未点名，与报告本体分开计数）

### (a) load_annotated 单读

**复现（probe_x04a 基线）**：对 gen-1dcc 每个字段文件 np.load 恰 2 次（verify 一次＋按路径重读一次）；verify 后换盘 q.npy，消费端拿到换后值 0.123456789 ≠ 验讫 0.43112821247700395。
**修法（v3）**：`verify_generation(..., want_product=True)` 返回验讫的 meta＋数组原件；`load_annotated` 直接返回，零二次打开。
**验收**：固定版每字段恰 load 1 次；换盘后消费仍得验讫值 0.43112821247700395（received_verified=True）。

### (c) publish 组装单读

**复现（probe_x04c 基线）**：分片经 `_verify_shard` 验讫后，把一片的 Q_std[0,0] 改 123.456 并自洽重封其内嵌哈希（文件 sha 与记录 7695a8bb… 不符 → 591c0481…/464cd297…）：基线 publish **静默发布**，成品代含 doctored 值 123.456。
**修法（v3）**：组装逐片单次读字节、先与 shard meta 记录的 `shard_sha256` 复核，不符 die 点名局并列双 sha、无记录 fail-closed；解析走 `np.load(BytesIO)` 同批字节。
**验收**：固定版 publish DIED 点名 selfplay-994000 与双 sha。

### (d) cnet2_exit_train 消费单读

**复现（probe_x04d 基线）**：d1-seal 封存 fixture（`~/workspace/d1-seal/fixture/ds`）因哑元 q 数据先于 V01 语义门、原样字节即被现行 loader 拒（基线/修复同拒、非本趟问题），故另造合规手工封存代 gen-probe-x04d-0001（N=6、25 字段、全行 unsearched）。基线实证：封存路径 verify 后换 value.npy，消费 value[0]=1000.0 ≠ 认证 0.0，且 load 打印 identity 里 value sha 63348d3d… ≠ manifest 认证 a2416c19…（记 A 跑 B）；legacy 平面路径 identity 记原 sha、消费换后字节 1000.0。
**修法（exit v1）**：`_verify_exit_generation` 每文件单次读字节、校验全过该批字节并返回 bundle；`load_exit` 只消费 bundle 验讫字节（memmap 逐字段 BytesIO 解析、identity 哈希自消费字节算）；legacy 分支改为单读字节同口径。**mmap 取舍**：弃用——exit 代为有界快照（D1 形态约 MB 级），loader 的有限扫描本就触遍全部浮点页、训练跨全集 gather 行，驻留与实际用量同阶，mmap 的惰性收益在此消费形态下不成立；单读绑定优先。
**验收**：固定版封存路径换盘后消费 = 认证值 0.0（consumed_certified=True）；legacy 路径 consumed == hashed（原 vulnerable 的 hash-then-mmap 两步已不存在）。

**新 md5**：annotator v3 = 60a0536cb446ac77175c92ce0416db30（commit 00eb08a3650f）；exit v1 = 883d0c6f321b6bff9481f7d89a2ad667（commit c2abe25efde6）。

## X05（报告本体，独立计数）· publish 复用漏验 meta.json seal

**复现（probe_x05 基线）**：发布 A→发布 B（current 指 B）→把 A 的 meta.json 改一字段（seal 破）→重发 A：基线返回 reused=True 且**指针翻回坏 A**（ptr_now = gen-cca3d1fa89946a7d = 坏 A）。
**修法（v4，照 W05）**：复用分支在 core 等值后、翻指针前，对已落地代 meta.json 实际字节 sha256 与旧 manifest 的 `meta_sha256` 复核，缺失/不符 die 点名 meta.json，指针不动。
**验收（probe_x05 固定版）**：「坏 A／健康 B／重发 A」→ DIED 点名 meta.json seal，current 仍指 B（ptr still B: True）。

**新 md5**：v4 = fcd83b39cc39027ea66fbd5fc744b5fc（commit 32e7512a0dd8）。

## X09（原 P3(b)，报告精度校正后）· q/priors 有限门

**复现（probe_p3b 基线，标注器与 join 镜像双侧同病）**：clean 过；q_nan、priors_nan、priors_inf、priors_sum_half **四坏例全 PASS**（静默放行）。
**修法（v5）**：`validate_records` 逐根在 Q 有限门后、任何比较门之前加 q 实臂 isfinite、priors 实臂 isfinite、priors 行和≈1（容差 1e-9、与 q 归一门同口径）三门，均 die 点名字段。封存前（publish 自检）与消费复验（verify/load）共用此函数，一处即两处同门。**Q_std 按校正件不加有限门**（诊断字段、可空语义保持现状）；Q 与 π 的 NaN 现行已正确拒绝、本趟未动。
**验收（probe_p3b 固定版）**：四坏例在标注器与 join 镜像双侧全 DIE 并点名字段（q/priors）；clean 过；真实产物口径下 priors 行和最大偏差实测 2.2e-16（gen-1dcc），1e-9 容差不误杀。

**新 md5**：v5 = 7f567a2e8544b78e51d16aa7660bc47b（commit f0ae52ddea7a）。

## 镜像对账说明（join 镜像同步，常设项）

提交：`code/k0_join.py` 78363299 → 27c11e168741a4e9eb3c168d377cdf90（commit 06d8bc6168b7），仅动镜像校验段 `_validate_annot_records`（＋docstring 锁步注记），join 其余逻辑零改动。

**逐项对齐点**：
- X09：join 镜像新增与标注器 validate_records **逐字同口径**三项——q 实臂 isfinite、priors 实臂 isfinite、priors 行和≈1（1e-9），die 文案同式点名字段；标注器侧函数是封存＋复验共用门，join 侧是其封存代消费复验的本地重写，两侧探针（probe_p3b）同组五例结果逐行一致（clean 过、四坏例双侧同 die）。
- X03（resume 身份）：**不触及 join**——join 无 progress/shard/resume 机制，只经 current 指针消费已封存代，且其 producer 已双绑两源代 id＋manifest sha；标注器 resume 身份是生产侧续跑概念，join 镜像面无对应物。
- X04(a)（load 返回路径）：**不触及 join**——join 从不调 load_annotated；其自有 load_annotation_source 每字段单次读字节校验 sha 后解析，本就单读，无同型二次打开。
- X04(c)（publish 组装）：**不触及 join**——join 无分片概念，组装在内存完成并经真消费者 preflight 后才发布，无「已验分片按路径重读组装」形态。

**真产物实证**：gen-1dcc9611575bfbe3 经新版 join 镜像复验仍过——v1 join 以其为标注源完成全链（见回归节），镜像门零误杀；且新版标注器复标同 12 根的产物（gen-f00ff3c5d3f38b66）经 join 亦全链通过。

## 回归（标注器原五项＋join 全链）

- 真教师重算数组全等：recompute_check 经新 load_annotated 路径对 gen-1dcc 抽根独立重算，**all roots bitwise identical**。
- 新旧产物等价：v5 全新标注同 12 根 → 14 字段与基线产物 gen-1dcc **逐字段全等**（字节管道改造未扰动评估）。
- kill-9 续跑：phase 1 在首分片落盘后 kill -9；phase 2 续跑完成 3 局 12 根（gen-32dbef197d35081e），shard1 sha256（be2693d9…）与 mtime 均未变（未重算）、日志确认 verified skip、终 --verify 通过。
- 缺 π die：verify_gate_check missing-pi → DIED 点名 missing fields ['pi']。
- 窗口拒收：verify_gate_check window → DIED 点名 root #0 step 150 outside window [200, 299)。
- gen-1dcc 消费 verify：v5 `--verify` 通过（14 文件、12 根）。
- join 全链：v1 join × gen-1dcc ＋ build gen-75de8ffb → joined gen-b98219474ba7b0ab，16 个 bundle 字段与 round-2 验收产物 gen-f18f59064c91e5cd **逐位全等**（gen id 不同仅因 join 本件 md5 入 producer 身份）；v1 join × v5 新标注 gen-f00ff3c5 → joined gen-d9d6c1a99189a02c，字段同与验收产物全等；两链均过真消费者（k0_distill.load_annotation）preflight。
- 蒸馏短训：S/C 两臂各 2 updates（bs 4、seed 7）均 exit 0 且检查点落盘；S 臂 update 1 loss 2.6606968109857347、batch sha 99067fd4e49bb14f…与 round-2 验收逐字同，C 臂 update 1 loss 2.69299552661902，两臂 update 2 loss 亦与验收日志逐位全等（5.730261899922348 / 5.8223334896137935）。本趟均跑满 2 updates（与验收同口径，非缩减）。

## 提交清单

| # | 文件 | 项 | commit | 新 md5 |
|---|---|---|---|---|
| 1 | code/k0_annotate.py | X03 | 733a75a52787 | ce2e597a59aaaadb6cd5d35330ccea87 |
| 2 | code/k0_annotate.py | X04 报告本体（权重） | 4fffc0a54f9f | 9e7871fd86428c21a67ce0825c425f06 |
| 3 | code/k0_annotate.py | X04 同类增补 (a)(c) | 00eb08a3650f | 60a0536cb446ac77175c92ce0416db30 |
| 4 | code/k0_annotate.py | X05 | 32e7512a0dd8 | fcd83b39cc39027ea66fbd5fc744b5fc |
| 5 | code/k0_annotate.py | X09 | f0ae52ddea7a | 7f567a2e8544b78e51d16aa7660bc47b |
| 6 | code/cnet2_exit_train.py | X04 同类增补 (d) | c2abe25efde6 | 883d0c6f321b6bff9481f7d89a2ad667 |
| 7 | code/k0_join.py | X09 镜像同步 | 06d8bc6168b7 | 27c11e168741a4e9eb3c168d377cdf90 |

每笔提交均单文件、提交前 API 读回钉 sha、提交后读回逐字节验 md5 一致。

落仓后 gate 读回：e922d9d890c3e1eff6643e947289acb0（未动）。
