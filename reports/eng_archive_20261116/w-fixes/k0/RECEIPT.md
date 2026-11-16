# W01 + W02 回执 — k0_train.py（Carbon League 工程组，W 轮）

基线：main `code/k0_train.py` ＝ md5 `0253817b26a9a2e9d8cc04177dda7acc`（59,055 B，blob `d2cfbba0…`），动手前 API 读回核对一致。两项各一个 commit、sha 钉基线、落仓后读回逐字节验。只改此一文件，未动活进程、未碰 DGX。

**终值：k0_train ＝ md5 `4d5b3b67b55ac5df376e0b224e3f2cb7`（65,216 B）**——DGX 冻结版的下轮拾取基线随本修更新为此值（原 0253817b 作废）。

---

## W01 · 抽样算法变化后旧 checkpoint 恢复静默换训练集

**机制（与派单描述一致，代码坐实）**：P7 协议断言只比 seed＋max_games 两个标量与指纹类字段，`games_used_sha256` 虽自 B2 起写进 effective，resume 时从不比对。前缀时代（e49dcf33 及更早、含 DGX 4ca79f2f 期）的 checkpoint 根本没有这个字段，在现行 seeded 算法下同 seed/同 cap 恢复，子集已换、无任何信号。

**复现**（`evidence/w01_base/`，探针 `probes/repro_w01.py`）：用 e49dcf33 本体在 ds128（128 局）上真训 2 步产 checkpoint——eff 无 `games_used_sha256`、子集为前缀集（sha `7a464492…`）；基线版同参 resume **exit 0 静默通过**，恢复运行的子集 sha ＝ seeded 集 `a29764e6…`（与 B2 回执记录值逐字相同）≠ 前缀集。训练集在无信号下被更换，坐实。

**修复**（commit `ac3a2cc51118c5751bb2ed5e887936224d80cf42`，md5 `61ad3e2ea046520e66ce8bdae0a873ad`，63,084 B）：
- resume 断言新增子集身份一项：用现行算法重算的 `games_used_sha256` 与 checkpoint 记录逐值比对，不一致 die 点名两侧 sha 与两侧 sampling 代际。
- 记录缺失（B2 之前）默认 die 点名（说明其子集口径不可证＋给出迁移路径）；仅 `--accept-unrecorded-subset` 显式放行，放行事实写入新运行 effective 的 `subset_migration`（含两侧 sampling 与本次子集 sha），随 checkpoint 本体落盘。
- 头部文档与 B2 注释同步改口径（断言钉的是重算后的子集 sha 本身，不是 seed＋cap）。

**回归**（`evidence/w01_w01fix/`、`w01_w01accept/`、`w01regress_w01/`）：
- 旧 checkpoint 默认 resume：exit 1，die 点名缺失记录与本次子集 sha。
- 加旗 resume：exit 0，新 checkpoint 的 effective 同时带 `games_used_sha256` 与 `subset_migration.accept_unrecorded_subset=true`。
- B2 后 checkpoint 正常路径：基线版与修复版各做「产出＋恢复」一对，fresh eff 逐键全同、4 步 loss 全精度逐位相同、两对 checkpoint 载荷深度比对全等（恢复对仅差 `.effective.start_sha256`——那是输入 checkpoint 文件的 sha，而本仓 checkpoint 文件字节跨同码同参运行本就不确定：B2 自家 cap64A/B.pt 同码同 seed 文件 sha 亦不同，属既有序列化性质、非本修引入）。
- 篡改记录：把 checkpoint 的 `effective.max_games` 改 64→32（子集 sha 留 cap-64 值）、以 `--max-games 32` 恢复（标量关放行）→ W01 子集断言开火，die 点名 `games_used_sha256` 与两侧 sha（`a29764e6…` ≠ `fe2cb037…`）。未篡改的纯 CLI 改 cap 则由既有标量关先 die 点名 max_games。
- 注记：`probe_w01_regress.py` 在 A 组跑完后无回执死掉（结果 json 未写）；A 组结论由其 stdout 工件＋载荷深度比对独立核实，B/C 两步由收尾以同命令手工补跑，工件同目录（`regress_summary.json`）。

## W02 · aux 先加载后重算 hash，可训 A 记 B

**机制（含对派单描述的一处细化）**：同一运行内 aux 标签文件实际被读**三次**——`check_aux_sidecar` 按路径 hash 校验（读 1）、`np.load` 解析（读 2）、V07 identity 又 `_sha256_file` 按路径重读（读 3）。训练消费读 2 的数组、身份记录来自读 3：两读之间换包即训 A 记 B（派单所指为读 2→读 3 这一段，细化点仅在读 1 的存在）。

**复现**（`evidence/w02_base/`，探针 `probes/probe_w02.py`＋`driver_swap.py`）：np.load 钩子在两 aux 数组都被消费后把磁盘文件换成合法异字节（die 翻转、mask 滚动）。基线版 swap 运行：训练 loss 与无干扰运行**逐位相同**（训的是 A），而记录的 `files_sha256` ＝ 换后 B（`4f254ae0…/ceb4e12f…`），与消费字节 A（`3fed048b…/e2029d41…`）不符。坐实。

**修复**（commit `d3a3f79b51680bb11133161f6b50833a50d4b116`，md5 `4d5b3b67b55ac5df376e0b224e3f2cb7`，65,216 B）：
- 新增 `_read_aux_bundle`：sidecar＋两标签文件各只读一次字节。
- `check_aux_sidecar` 增 `prefetched` 参数：sidecar 解析与标签 hash 校验改自预取字节（缺省 None 时旧路径行为不变）。
- main() 与 `--screen` 两处调用点统一：验封/校验、BytesIO 解析、identity 哈希全部派生自同一份字节；训练路径与 screen 路径均不再有第二次按路径读取 aux 内容。identity 值对未受干扰 bundle 与改前逐值相同（哈希对象同一份字节）。

**回归**（`evidence/w02_w02fix/` 及下列各处）：
- swap 场景复跑：记录 identity 恒 ＝ 消费字节 A、loss 仍与无干扰运行逐位相同、记录 ≠ B。
- V07 原三路 die 在终版复跑（`~/workspace/v07v08-fix/evidence/v07_w02fin/`）：翻转标签 die 点名 `aux.sidecar_sha256`、sidecar 重排同字段、伪造 sidecar 哈希后逐文件关 die 点名 `aux.files_sha256[aux_die.npy]`（两侧哈希与前任证据逐字相同）。
- battery3（P5/P7 全套，`~/workspace/k0-screen/evidence/w02fin/`）：**ALL OK**（21 项：15 项独立检查＋协议六字段逐一 die，与 V07/V08 回执的 21/21 同口径）。
- screen 不回退（repro_v08，tag w02fin）：1e38 臂 exit 1 点名 non-finite 计数、无 NaN 字面量；正常臂 stdout sha256 `12a3872c…` 与基线逐字相同——screen 的单次读改造未扰动其输出。
- W01 在终版抽查（`evidence/w01_w02fin/`）：旧 checkpoint 默认 resume 仍 exit 1 点名缺失记录，两修互不干扰。

## 证据索引

- 探针：`~/workspace/w-fixes/k0/probes/`（repro_w01.py、probe_w01_regress.py、probe_w02.py、driver_swap.py）
- 工件：`~/workspace/w-fixes/k0/evidence/`（w01_base / w01_w01fix / w01_w01accept / w01regress_w01 / w01_w02fin / w02_base / w02_w02fix / commit_w01.json / commit_w02.json）
- 版本链：`0253817b`（基线）→ W01 `61ad3e2e`（commit ac3a2cc5）→ W02 `4d5b3b67`（commit d3a3f79b，**现行终值**）
