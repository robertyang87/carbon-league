# W 轮 certify 三项修复（W06/W07/W08）— 工程组回执

## 落仓
- 仅改 `code/certify_pack.py`：commit **`1ed4c9628f3ddb6f7ed2270bcbb2f53d86000df1`**
- 基线 md5 `1f4811b450d0c93417f6465c6835f215`（47,298 B，blob `bfe9086b…`，动手前读回核对一致）
  → 新 md5 **`fc33be6cc0255d87d69c8eef8b9a2875`**（52,416 B，blob `d118d27a…`）
- 提交时 sha 钉基线 blob；提交后 API 读回与本地修复版**逐字节一致**；落地本体复跑布局全套 ALL OK、全路径抽跑 PASS。
- diff 在 `out/change.diff`（348 行）：`_read_quota` docstring、`_walk_v2`、`_walk_v1`、解析收尾段、main 的 quota_missing 与 check note。判定逻辑其余部分、证书结构未动。

## it95 证书重签时点（应作战室/策略组注记单列）
- **新版 md5：`fc33be6cc0255d87d69c8eef8b9a2875`**，已落 main 并读回验字节（commit `1ed4c962`，2026-11-14 本趟回执时点）——此即「修复版就位、可供重签」的时点。
- 按策略组裁定：**重签须待本修复经策略组验收后进行**；验收前以旧版（`1f4811b4…`）签发的 it95 证书一律标临时、不得消费，消费前按本版重签。验收与重签排期归作战室/策略组，本回执只钉版本与就位时点。

## W06·P2 测试覆盖产未标注正式 quota 证据 —— 已修（标注口径）
- **复现**：基线版在 `CERTIFY_PACK_TEST_CGROUP_FILE/MOUNTINFO_FILE` 指向 L1 合成 FS 时全路径跑 `--require-quota`：overall **PASS**、quota check pass=True、missing=[]、source 记 `cgroup_v2`——证据全来自注入文件却与机器证据无从区分（`out/cert_base_L1_req.json`）。
- **修法（选标注而非 die）**：任一覆盖变量**存在于环境**（不论其值）即在 `_read_quota` 全部出口强制 `source="test_override"`；main 把 `"test_override"` 追加进 `quota_missing`，`--require-quota` 的 pass 逻辑（`not quota_missing`）不变、自动不满足且点名。理由：die 会使合成布局测试方法本身无法走全路径、且任何误设变量的环境直接崩；标注保留值可见与可测性、只剥夺其正式证据资格——source 本就是证书的来源标注字段，拒认它不需要新机制。无覆盖的默认路径行为逐字不变。
- **验证**：L1 覆盖＋`--require-quota` → overall FAIL、check pass=False、missing=`["test_override"]`、值仍记录（`out/cert_fix_L1_req.json`）；覆盖＋无旗 → PASS、check pass=None、source=test_override（纯记录语义保留）；LH 半缺覆盖＋旗 → missing=`["memory","test_override"]`（memory 仍单独点名）。

## W07·P2 最深挂载偏好遗漏可见祖先更紧约束 —— 已修
- **复现**：新布局 L9（同进程路径 /a/b/leaf 由两 cgroup2 挂载暴露：深挂载 root=/a/b 链上只有宽值 8 CPU/8 GiB；根挂载链经过的 /a 有紧值 1.5 CPU/512 MiB）。基线版只走深挂载 → 记 `800000 100000`/8589934592，紧约束被漏（`out/layouts_baseline.txt`）。
- **修法**：`_walk_v2`/`_walk_v1` 改为遍历**全部**候选挂载逐链合并：有效值＝所有挂载祖先链上的最小值（同层级多来源取最紧）；原排序只决定记录链的条目次序。链条目增记 `mount`（所经挂载点）与 `hier`（v2/v1）以便区分来源。
- **验证**：L9 记 `150000 100000`/536870912（紧值），链含两挂载共 6 级证据（`out/layouts_fixed.txt`）。

## W08·P2 hybrid 提前返回 v2 漏读 v1 —— 已修
- **复现**：新布局 L10（hybrid，v2 在 /sub 为 2 CPU/1 GiB、v1 为 1.0 CPU/256 MiB）：基线版 v2 一有读数即返回 → 记 `200000 100000`/1073741824、source=cgroup_v2，v1 紧约束全漏。
- **修法**：v1/v2 两侧都走完再解析：双侧都有读数时逐控制器取更紧者、以胜方层级的原生表示记录（v2 原文 / v1 数值比；内存同为字节），平局取 v2 表示；source 记 `cgroup_v1+v2`，祖先链两侧拼接（条目 hier 可辨）。单侧有读数时走原纯 v2/纯 v1 分支、行为不变；都无读数时 fallback 链合并双侧已走链（值仍全 null）。合并分支 process_path 保留 v2 路径，v1 路径见记录链。
- **验证**：L10 → cpu=1.0（v1 数值）、mem=268435456、source=`cgroup_v1+v2`；L11（v2 更紧）→ cpu=`150000 100000`、mem=536870912、source=`cgroup_v1+v2` 且 v1 链入证。源标签经「无环境变量、open 重定向」模式 B 实测（覆盖模式下 source 按 W06 记 test_override，层级标签以模式 B 为准）。

## 回归
- **八布局＋LH 解析级**（`evidence/layouts_w.py`，模式 A）：L1–L8 的 cpu_max/memory_max/process_path/链路径与 v09 期望**逐项一致**；差异仅 W06 的 source 标注（覆盖下记 test_override）与链条目新增 mount/hier 两键（证据范围扩展，path/value 投影与基线全等）。
- **S3 全套全路径**（tiny fixture）：本机无旗 PASS（quota 全 null、source None，与基线同型）；本机带旗 exit 1、missing=[cpu, memory] 双点名——与基线/S3 记录一致；覆盖场景按 W06 新口径（见上）。
- **证书其余字段对照**（基线 vs 新版，同 fixture 本机无旗）：identity.pack 逐字一致；params 删 quota 后逐字一致；全部 checks 的 pass 值逐项同值；games（去计时与 self-md5 后）逐项一致；quota 键集不变（6 键）；唯一文案差异＝quota_self_proof 的 note 增加 test_override 一句（W06 口径补充）。harness self-md5 随本体变更（应然）。
- **本机实读**：仍如实 null（cpu/memory/source 全 null、affinity=4），与改前同结论。

## 证据（~/workspace/w-fixes/certify/）
- `evidence/layouts_w.py`（L1–L8 复用 v09 构建＋L9/L10/L11/LH 新布局，模式 A/B＋本机实读断言）
- `out/layouts_baseline.txt` / `out/layouts_fixed.txt`（基线复现与修复后全套输出）
- `out/cert_base_L1_req.json`（W06 基线假 PASS 证书）、`out/cert_fix_*.json`（新版五份全路径证书）、`out/cert_landed_noflag.json`（落地本体抽跑）、`out/fullpath_summary.txt`、`out/change.diff`
- `baseline_certify_pack.py`（基线原件）、`work-fixed/`（修复版树）、`landed/`（main 读回落地本体树）、`layouts/`（十二布局合成 FS）
