# 打包 → 上天梯标准流程（runbook）v2

立规：v1 2026-11-13（Echo3 打包链复盘后作战室制定）；**v2 同日由所有者令作战室梳理定版：DGX 为打包枢纽**。本版取代 v1 的角色表，其余技术口径沿用并补齐新断点。
适用：任何把权重打成单文件 pack、经验证后提交天梯的全过程。台账：`reports/live/packs_ledger.md`。

## 断点清单（本流程逐条堵死）

v1 四断点（Echo3）：① 参数无存档（CONF 只在所有者个人 Mac）；② 打包器用错（cnet2_pack.py 废包）；③ 交付定义缺失（暂存区落盘冒充交付）；④ 执行人漂移（同机双考险情）。
v2 新断点（2026-11-13 下午 it320／Lite-V7／哈基米三轮实证）：⑤ **角色漂移**——runbook 写 L20 建包，实际三轮都由工程组临时建包、作战室逐件验包，文档与现实脱节；⑥ **快照环部署丢可执行位**（it320 首趟 Permission denied）；⑦ **VM 替换吃掉 /root/.ssh/known_hosts**（临时根盘，C1 环被杀后 ssh 假报对方换钥）；⑧ **Contents API 对 >1MB 文件不返内容**，验包必须走 git blob API（全 40 位 blob sha）或 raw 字节；⑨ **DGX 的 GitHub key 是只读 deploy key**，DGX 产出无法直推，入仓必须走作战室 API 通道——流程必须按此设计，不许按「DGX 直推」设计。

## 角色（每步唯一执行人，禁止双执行）

| 步骤 | 执行人 | 备注 |
|---|---|---|
| 发打包单 | 作战室 | 缺项不开工 |
| 权重归集 | DGX | 权重先落 DGX、两跳 md5 验讫才算备料完成 |
| 建包＋验包 | DGX | **唯一建包方**；DGX 可调度其他线供料，每步仍只许一个执行人 |
| 入仓 | DGX 备字节、作战室 API 提交 | DGX key 只读（断点⑨）；入仓完成＝作战室从 main 读回 md5 一致 |
| 备考送达 | DGX | 追到考试机落盘＋执行人读回 md5 为止 |
| 考试（等价性／计时／定标） | 打包单点名的考试线 | 一机一执行人；计时须空载 |
| 裁决 | 作战室 | 考试线只报数，不裁决 |
| 提交天梯 | 作战室 | 唯一提交人；提交闸归所有者逐次开口 |
| 台账维护 | DGX | 每过一关更新一行，经同一 API 通道入仓 |

## 飞行前检查（一台凑齐全套，缺一不开工）

1. 源权重已在 DGX 落盘，文件名＋完整 md5 已核；
2. 打包器与 CONF 模板以 main HEAD 的 `code/packs/` 为准（不用任何机器上的旧拷贝／工作目录副本）；
3. 考试机与执行人已在打包单点名；
4. 入仓通道确认（作战室 API 可用、目标路径无冲突）；
5. 提交路线已注明（ladder-submit 工作流＋所有者开口），本单是否走到提交先写明。
另：新考试机接手先验出站密钥对（v1 条款沿用：spot-31 首拉三件 Permission denied 的教训）。

## 步骤与验收口径

**0. 打包单（作战室发出）**：源权重文件名＋md5、目标 pack 名、打包器、CONF 出处、预期字节带、验证任务书（考试机、执行人、基线、种子）。缺一不开打。

**1. 建包（DGX）**：打包器只认 main 的 `code/packs/cnet2_search_pack.py`；CONF 只认 `code/packs/PACK_CONF_TEMPLATE.txt` 的 13 键逐字（SEARCH=False、ROLL="self"、GUARD=True、GUARD_NARROW=True、MODE="sweep"、MIX=False、TEACHER=2、TEACHER_LATE=2、LATE_STEP=240、BUDGET=0.7、HARD=0.85、ABORT_DL=0.95、START_DL=None），只许换权重，不许改键值；确需改键先报所有者。（N01）`code/packs/cnet2_search.py` 与权威源 `code/cnet2_search.py` 逐字节同步，改 search 必须两处同改并跑一致性检查；旧 v12 副本冻结在 `code/packs/historical/cnet2_search_v12_frozen.py`，只许复现 n1r/Echo3 历史字节，禁止新生产；已入仓历史产物只读，永不改写／重打包。packer 构建时把内嵌模块（cnet2_feat／cnet2_np／fastsim／cnet2_search）的 md5 写进包内 `_BUILD_HASHES` 与构建 JSON 并与 `code/` 权威源逐一比对，不一致且不在 `ALLOW_DIFF`（默认空）内即 exit 2；无 `code/` 树的机器降级为只记录 hash＋警告，此时交付前必须在仓库环境跑 `--check-source`（exit 0）。`--check-source` 只验源码、不验制品：standalone 包发布前必须在干净 checkout 通过 `--verify-pack <包> --expect-commit <权威 SHA>`（exit 0），输出的 pack_md5／pack_bytes 记入台账为发布身份。环境 `CNET2_NO_ROLL=1`。生成前复验源权重 md5；生成后报 pack md5＋实测字节；字节偏离先例带（n1r 4,579,266／Echo3 4,576,790）超 1% 先停手查因。1 局对 Master2 冒烟 err 为空（只验通路）。**注**：pack 自含内嵌代码，内嵌版本可与考试机上的代码副本不同（如哈基米包内嵌 cnet2_search b81be6fe、机上副本 10a0674e）——以包内 `_BUILD_HASHES` 为准，不追机上副本。

**2. 入仓（DGX 备字节 → 作战室提交 → 读回）**：pack 提交 main 的 `code/exp/<名>.py`（考试用）或 `code/packs/<名>_int8.py`（提交天梯用），台账同步补行。>1MB 文件读回走 git blob API 全文算 md5（断点⑧），不许只看提交回执。读回一致才算入仓完成，DGX 收到作战室读回数后才可进入送达。

**3. 送达（定义改死）**：**pack 在考试机上落盘、且验证执行人回执 md5 一致，才算交付完成。** 暂存区只是中转，不是交付。路线：DGX→考试机走 tailnet／公网直连；跨机传权重走 DGX→Studio 局域网→目标机中转。**文件永不走 ZeroTier**（ZT 只做心跳）。送达方主动追回执，10 分钟无回执即升级作战室。

**4. 考试（点名执行人连做，中间不停手换人）**：
- md5 闸：机上 pack 的 md5／字节与打包单不符直接拒收；台账标 VOID 的包永不进考场。
- 快照环：脚本部署必须带可执行位（断点⑥），开考前验活（有一次成功落盘＋行数增长回执），开考后每 2 分钟全量快照到机外；ssh 的 known_hosts 一律指持久化路径 `/home/hatch/.ssh/known_hosts`（断点⑦）。
- 等价性：pack 对 Master2 真身 200 局、seed 996000、SEARCH 关；对照基线写进任务书，落在合并噪声带内算通过，明显掉带立即停手报告。
- 计时／资源认证（N04／F04）：只认 main 的 `code/certify_pack.py`（子进程外层测量）：cgroup v2 1.5 核／512MiB 下测 init 墙钟、逐步墙钟（含 IPC）、p95、慢步计数与超额池、整局墙钟、子进程峰值 RSS、内部错误数；硬闸：init ≤20s、单步软 1s／硬 3s、慢步（>1s）≤3 次且额外耗时合计 ≤5s、整局 ≤1500s、内部错误 0——任一不过 exit 非零并输出逐项 PASS/FAIL 的 JSON。`code/cnet2_search_timing.py` 只是子测试，可作冒烟，不能单独作认证结论。certify 必须在完整 `code/` 树下运行。
- 定标（裸网包）：对 Master2 真身（seed 996000）＋Echo2（seed 999000）各 200 局，校准分＝对 Master2 均分 −785；闸口径由打包单写明（现行 +150）。
- 计时与定标期间考试机必须空载：同机不得并行任何考试或训练，否则计时作废重测。
- 结果双路核验：launcher JSON 与独立重算（逐局线）一致才算数；裁决由作战室出。

**5. 提交天梯（作战室）**：提交前逐项核——入仓读回 ✓、等价性 ✓、计时 ✓、所有者已对本次提交开口 ✓、当日同作者提交数 <2（站点每日只留 Top 2）✓。只走 ladder-submit 工作流（secrets 登录），不走其他通道。提交后当场从榜单读回确认（名次／局数），立即报所有者，并回填台账。

**6. 异常纪律**：任何一步失败＝停手＋报 exact 卡点（run 号／错误行／rc／实测值），不许换路线自行重试；重试由作战室重新派单。一个 pack 任务在任何时刻只有一个推进者；交接以消费方直读回执为准，生产方自报不算数。作废包台账标 VOID 并注明 md5，各线见之拒收。

## 附则

- 一切 md5 期望值写完整 32 位（s30push-0205 假警报教训：29 位截断＋grep 多空格）。
- 本流程的变更经所有者或作战室写入本文件并提交 main，口头改动不算数。

## 登记·同名文件处置（2026-11-14 作战室裁定，工程组删前引用扫结论）

- `code/packs/cnet2_pack.py`（5,048 B，无戳版）：**活的生产依赖，永不删**。权威打包器 `code/packs/cnet2_search_pack.py` 第 30–31 行以同目录 import 它，全部权重经其 `P.quantize()` 量化；其 quantize/dequantize 与根目录现行版（md5 119e9bb9bf8413685b87721cff8cfe18）逐字节同、从未漂移。审计的「陈旧变体」标签作废——形态旧、职能活；按此标签清理此文件＝DGX 打包线 ModuleNotFoundError、无回落路径。
- `code/cnet2_search_pack.py`（根目录，6,411 B）：**复现／证据工具文件，保留不删**。无现行生产引用（生产建包只认 `code/packs/cnet2_search_pack.py`），但 mkref.sh、build_v17.sh、四件评审证据 harness、data/s64*_eval.sh 仍引用它实跑，删之断复现与证据链。同名混淆的防护靠本登记与打包单口径，不靠删除。
