# 阶段 4D：换 trunk 重跑搜索比对——执行报告（座位不可用，停报）

- 时间：2026-11-18 01:32–01:45 CST
- 结论：**停报，未开工。** 主座位 spot-36 与备选 spot-37/38 均不在 tailnet 上，无可用路由；未写 harness、未跑对照闸、未跑全量、未提交任何文件。

## 1. 批复与提案验合（均合）

| 件 | 读数 |
|---|---|
| 批复 `reports/APPROVAL_SMALL_TEACHER_STAGE4D_RETRY_EXEC_20261118.md` | commit `41cd950f`，md5 `dbe84d45` 合（1,947 B） |
| 执行提案 `reports/eng_verify/STAGE4D_RETRY_TRUNK_PROPOSAL.md` | commit `063aed1d`，md5 `dd0c4b0c` 合（12,992 B） |
| 裁定 `reports/RULING_SMALL_TEACHER_STAGE4D_RETRY_TRUNK_20261118.md` | md5 `c52d44cd` 合（2,267 B） |
| 教师件 `code/stage0_teacher.py` | md5 `c6ade5ce` 合（25,186 B） |
| `code/k0_annotate.py` | md5 `7f567a2e` 合（49,151 B） |
| 学生 plug-in `code/stage0_teacher_small.py` | md5 `5c0dc757` 合（26,412 B） |
| 比对器 `reports/eng_verify/shard_compare_v4.py` | md5 `1e9ad085` 合 |

## 2. 座位核查——spot-36/37/38 不在网（停表触发）

- 三方独立 tailnet 视图一致（8 节点，无 36/37/38）：DGX `tailscale status`、L20 `tailscale status`、本机 `/opt/hatch/bin/tailscale status`。在网节点：muse 100.92.70.87、dgx-spark 100.84.148.73、gpu-l20-6 100.79.251.87、mac-studio 100.78.149.20、radxa-orion-o6 100.104.105.100、carbon-spot-29 100.65.8.27（offline 2d）、carbon-shanghai-spot-63 100.85.41.0（offline）、carbon-shanghai-spot-64 100.72.73.119（在线）。
- 本地可达源逐一查无 36/37/38 地址：`~/.ssh/carbon_config` 仅有 dgx／spot29／spot31／spot34；DGX `~/.ssh/config` 无 spot 条目、`known_hosts` 全哈希不可读；DGX `fleet_deploy_fleet*.log` 为 11-15 旧件、无 36/37/38；DGX `k0_dist/annot_registry.json` 含 36/37/38 记录 0 条；DGX `k0_dist/artifact_registry.json` 含 0 条；仓 `data/artifact_registry.json` 含 0 条；作战室同步态与巡检日志仅有 11-17 日间历史提及、无地址。
- 背景：2026-11-17 11:38 所有者释放令覆盖 36/37/38（快线座位释放）；此后 11-17 20:50 曾有 spot-38 新购记录（i-uf6bqjh1t9gouuiazhd3，公网 47.117.95.30），但无证据表明该实例仍在、且未探——新地址首联需用户审批，01:40 探一个存疑旧地址不值得打扰所有者；37 无任何地址线索。
- 按提案 §7 停表「座位冲突／不可用即停报」，执行至此停止。**未擅自改席**：spot-64 在线、spot-63 离线仅作观测事实记录（未探）；改席须策略组另批，批复只授权 36（备选 37/38），排除 L20／DGX／Studio／spot-34。

## 3. L20 源材料先行核验（已做，与座位无关的部分）

| 件 | 读数 |
|---|---|
| 学生 npz `stage4/student.npz`（L20） | md5 `b22020ff` 合 |
| `stage4/k0_annotate.py`（L20） | md5 `7f567a2e` 合 |
| `stage4/shard_compare_v4.py`（L20） | md5 `1e9ad085` 合 |
| 回放 994400 `stage4/appendix/selfplay-994400.json.gz` | sha256 `6ed2623e` 合（226,125 B） |
| 回放 994401 `stage4/appendix/selfplay-994401.json.gz` | sha256 `6607fe5e` 合（229,043 B） |
| 权重 `weights/cnet2_m3s2r1_heads_v2.npz`（L20） | md5 `06a03d9c` 合 |
| 教师 generation `annot_teacher/generations/gen-3f2c17f8b7beb5b6`（L20） | `current.json` 指明该 generation，total 396；`manifest.json` 在位 |
| 教师分片 `annot_teacher/shards/selfplay-994400.npz` | md5 `2aad09a5`（待与 manifest 对账，座位定后） |
| 教师分片 `annot_teacher/shards/selfplay-994401.npz` | md5 `ce7f84fc`（待与 manifest 对账，座位定后） |
| L20 本机状态 | 32 核、247 GB 内存、load 0.11，空闲；但 L20 为排除座位，不跑 |

注：教师 generation 目录以逐字段 `.npy`（arms／Q／Q_mean 等，396 行）＋`manifest.json`／`meta.json` 存放，另有 `shards/` 下逐局 `.npz`；harness 取臂集／先验时按提案 §1.1 以教师分片字段为准，座位定后先对 manifest 逐件对账。

## 4. 未做事项

- 未写 harness；未跑对照闸；未跑全量重跑；未跑比对；未提交任何结果件。
- 直读四项实测值：无（未执行）。可比根数：无。Tier 落点：不判。

## 5. 待决（请主线定）

- 座位：请作战室／C3 确认 spot-36/37/38 当前是否在世、给出现行地址；或请策略组另批改席（观测：spot-64 在线 100.72.73.119、spot-63 离线，均未探）。座位一定，执行按提案 §2–§5 照序推进（身份机上核验→harness→对照闸→全量→比对入仓）。

---

## 6. v2 批复与准备执行（2026-11-18 02:53–03:25 CST，隧道未批，准备分支）

### 6.1 批复验合与隧道状态

- v2 批复 `reports/APPROVAL_SMALL_TEACHER_STAGE4D_RETRY_EXEC_V2_20261118.md`：commit `530c70e3`，md5 `0d124b21`，API 读回验过 ✓。v2 提案 `reports/eng_verify/STAGE4D_RETRY_TRUNK_PROPOSAL_V2.md`：commit `8d5fc52f`，md5 `24986e48`，API 读回验过 ✓。
- 隧道审批 `14400c86-1ac6-444b-9ee3-02274cf55af5`（通往 spot-64 100.72.73.119，requested 2026-11-17T17:45:44Z）：**2026-11-18 ~03:25 再查，仍 pending**。按派单只做与座位无关的准备，不循环重试 SSH——**本轮未对 spot-64 发起任何 SSH 尝试**。无关待批 `df9d713b`（139.196.109.84，Dgx Status Report）未动。

### 6.2 harness 已写＋入仓＋本地冒烟验过

- 文件：`reports/eng_verify/retry_trunk_harness.py`，commit `46cf3202`，blob `6a248f62`，35,320 B，md5 `53df4f2e5e2ea3cff438a6129f5ca0ef`（API 读回验过 ✓）。
- 机制（按提案 §1）：`evaluate_root` 拆两阶段——Phase 1 教师 trunk（信念场景＋对手根动作），Phase 2 参数化 trunk（续演＋聚合）；臂集／先验取教师分片透传，不调 `gen_candidates`；信念强制 sample（否则 fail closed）；输出与 k0_annotate 同 schema，identity 三件 verbatim＋caliber「换 trunk 重跑」等附加键；`--code-dir` 在钉版 import 之前解析进 sys.path（harness 文件本身可置于 code dir 之外）。
- 钉版源读取验合（全合）：stage0_teacher `c6ade5ce`、k0_annotate `7f567a2e`、stage0_teacher_small `5c0dc757`、shard_compare_v4 `1e9ad085`、stage1_teacher_gate `e922d9d8`、cnet2_np `d8c15ad4`、cnet2_feat `2a45b08b`、fastsim `aefe6c83`。注意：main HEAD 的 cnet2_search 是 `c3222eea`（非钉版），钉版 `94f7fb64` 取自 L20 staged code dir；钉版 cnet2_search **不 import cnet2_comparator**（仅 main HEAD 有），钉版栈自包含。
- `gen_candidates` 安全分析（源读验证，非假设）：`gen_candidates` 经 `guard.apply` 只改 root guard 的 `lostw`／`prevw`；`evaluate_root` 恢复的快照里 `prevw` 已是根步 worker uid 集、且 `gen_candidates` 对同一根观测 apply，`cur == prevw`，lostw 增量恰为 0；`guard_base` 在每次 guard 使用前由 `reset_step()` 清空，快照里的 `guard_base` 无关。对照闸实证仲裁；harness 另记每根恢复的 `me_snap` lostw 进 diagnostics.jsonl。
- 本地 1 根对照冒烟（本机，教师 trunk，994400 / step 200 / seat 0）：RC=0，墙时约 5.0 分钟（本机 1.5 共享核，**非座位速率**）；身份核验全过；act 调用数 **648 恰与预测一致**（680 − 32 gen_candidates），诊断 `gen_candidates` 32 亦恰与预测一致（1 + 15 + 16）；`me_snap_lostw`=31；`n_unknown_cells`=0。
- 同机确定性复跑：第二遍与第一遍 q／Q／Q_std／priors／arms **全部逐位一致**，分片 md5 相同（`7df7ab0f…`），generation id 相同（`gen-retry-trunk-1b24175869f5`）。本地证据目录 `~/workspace/small-teacher-distill/retry_trunk/smoke/` 与 `smoke2/`。

### 6.3 对照闸口径的重要发现——跨机浮点噪声（须策略组定夺）

- 本地 1 根对照 vs L20 教师分片：arms 逐位一致；priors 透传逐位一致（max|dp| = 0.0）；诊断 `gen_candidates` arms 一致、priors 最大差 **1.357e-06**（两次运行值完全相同——稳定的机器间差异，非随机噪声）；q 最大差 **2.20e-07**，Q 最大差 **7.03e-08**，argmax 相同（index 8）；Q_std 不一致。
- 判定：**harness 确定、结构忠实**（调用数精确、臂集精确）；残差 ~1e-7 是跨机浮点噪声（L20 vs 本机，BLAS／指令集不同），不是机制错误。诊断 (a) 已排除（同机两次逐位一致）。
- 但批复的对照闸原文要求「q 与 Q_mean 须逐位一致」，(b) 条的容忍上限是「偏差均匀 ≤1e-12」——1e-7 量级的跨机噪声既不满足逐位一致、也不在 (b) 的 1e-12 内。**spot-64 与 L20 是不同机器，座位上跑对照闸极大概率同样触发 (b) 停表**（报数字、停、等裁定），白耗一轮座位往返。
- 建议（请主线转策略组定夺，执行方不擅自改闸）：**首选同机对照**——座位上另跑生产 `evaluate_root` 直接调用（教师 trunk，8 根对照根），与 harness 对照输出逐位比对，隔离 harness 忠实度与机器噪声；草稿脚本已备好（本地 `~/workspace/small-teacher-distill/retry_trunk/direct_control_roots.py`，md5 `09d2705e…`，py_compile 过，**未入仓、未获批前不用**）。备选：把 (b) 的容忍放宽到 max|dq| ≤1e-6 且 argmax 相同、arms 一致，记为「跨机噪声内一致」。任一方案都须策略组批复后才执行。

### 6.4 源材料本地转运（L20 → 本机 staging，全合）

- 目录 `~/workspace/small-teacher-distill/retry_trunk/stage/`（18M，SCP_RC=0）；逐件 md5 与 L20 直读值对账全合：code_teacher/ 10 件、code_student/（stage0_teacher `5c0dc757`、其余同）、student.npz `b22020ff`、shard_compare.py `17b4ec07`、shard_compare_v4.py `1e9ad085`、回放 md5、权重 `06a03d9c`、分片 `2aad09a5`／`ce7f84fc`、generation meta.json `3df9455d`、manifest.json `36c67887`。
- 教师分片 `_meta` 身份块已全文读过：config_sha256 `9a2b2440…`、games_sha256 `c3649caa…`、replay_sha256、seed 12345、pi0/pi1=1.0、selected_sha256 `9117c25c…`、weights_md5 `06a03d9c…`、window [200,299]；字段 arms (198,16,12)、priors/q/Q (198,16)、n_arms 全 16、step 200–298（99 步 × 2 seat = 198 根／局——harness 按分片自带 step/seat 取根集，恰复现教师根集）。
- **回放哈希标注纠正**：本报告 §3 及提案 v2 §4、4D 停报里写的回放「sha256 `6ed2623e…`／`6607fe5e…`」实际是文件的 **MD5**；真 sha256（sha256sum 实测，与教师 generation 身份块记录一致）：994400 → `55d713d64c002d6c204988280804da47b96c979464f4ae9740b2676338581436`；994401 → `866a8ce7cc7125af8962e51f6db7b1a41686c3c13dbab35b2cec0c653a66baf5`。生成身份自洽、只错 paperwork 标签。harness 双重核验（md5 == staging 记录 且 sha256 == generation 身份）。

### 6.5 mesh 钥对已生成（本地）

- 私钥 `/home/hatch/.ssh/spot64_mesh`（600，仅存本机，不写入报告、不贴进消息）；公钥 `/home/hatch/.ssh/spot64_mesh.pub`（644）。
- 指纹：`SHA256:sLYABsIs9NxzauiUoehOW/sygVzvb3dcP2V2OhpKN0E`（ed25519 256，comment spot-64-mesh）。
- 公钥分发＋逐一验通须等隧道批复后（提案 §3.3，座位直读闸通过后）。

### 6.6 未做事项（缺隧道审批）

- 未对 spot-64 做任何直读（座位直读闸未跑：CPU／内存／负载实测规格未知）；未分发 mesh 公钥、未验通；未做机上身份三件套核验；未跑座位对照闸；未跑全量；未跑比对；未提交结果件。
- 墙时预算：提案的约 1 小时（估计，未校准）暂维持；本地冒烟 5.0 分钟／根是本机慢速，仅作 harness 正确性证据，不作座位速率——首分片落盘后按实测单价重钉 ETA。

### 6.7 待决（请主线定）

1. 隧道审批 `14400c86`：请所有者批准（2026-11-18 02:51 已请过一次，~03:25 再查仍 pending）。
2. 对照闸口径：建议请策略组先行裁定同机对照方案（或放宽 (b) 容忍），避免座位上必触发 (b) 停表。草稿脚本已备好，批复后即用；未批复前执行方按原文闸跑、触发即停报。
3. 审批一到，执行按提案 §7 照序：① 座位直读闸 → ② mesh 分发验通 → ③ 身份机上核验 → ④ 对照闸 → 全量 396 根 → v4 比对 → 结果入仓 → Tier 四项实测值报策略组（只报数）。开工四件套先报作战室。

---

## 7. v2 执行（2026-11-18 07:04–07:22 CST，闸 4 停报：同机直跑脚本异常）

- **结论：停报，全量未开。** 闸 1–3 全过；闸 4(a) harness 对照跑完赛，且与 L20 教师分片逐位一致（见下）；闸 4(b) 同机直跑脚本 `direct_control_roots.py`（已批 md5 `09d2705e`）在 worker 内以 TypeError 异常退出、未产生分片，同机逐位比对无从执行。按纪律：不改写已批脚本、不以跨机读数替代同机硬闸（裁定 ef932ea3）放行。待策略组裁定（修脚本另批，或以本次跨机逐位一致读数改判放行口径）。

### 7.1 闸 1 座位直读（通过）

- 07:04 CST 直读：hostname `carbon-shanghai-spot-64`；64 vCPU；内存 121 GB（空闲 119）；load 0.00／0.00／0.00；磁盘 99G、余 91G；up 6:38；进程仅系统与云助手守护（aegis／aliyun-assist／tailscaled），无他人活。
- 系统 python3.12.3 无 numpy；部署 venv `/root/carbon/.venv` 可用（Python 3.12.3、numpy 2.5.4、scipy 1.18.1），本趟全程用该 venv。
- 首联曾遇一次 banner exchange 超时（ConnectTimeout 30s，exit 255），放宽至 60s 后连通——通道慢，非座位故障。

### 7.2 闸 2 mesh（通过，验通范围如实）

- 本机 `spot64_mesh.pub` 已入 spot-64 `~/.ssh/authorized_keys`；本机以该私钥免密登录验通（`MESH_KEY_LOGIN_OK carbon-shanghai-spot-64`）。
- spot-64 本机 ed25519 钥对已生成（原无）：指纹 `SHA256:ABNuhNn3S2n5a9Hi/0INEPZiC7dbU+qpe2q3xFzfXmg`（comment spot-64）。
- spot-64 公钥入 DGX `authorized_keys`（共 15 行，spot-64 恰 1 条）与 L20（共 13 行，恰 1 条）；从 spot-64 直连验通：DGX 回读主机名 `spark-29db`、L20 回读 `gpu-l20-6`，RC 均 0。
- spot-29／spot-63 离线，未验（如实记录，不算通过项）。

### 7.3 闸 3 身份机上核验（通过）

- 转运 35 文件至座上 `/root/small_teacher_50k_retry/`（code／replays／teacher_shards／teacher_generation 等分目录），逐件 md5 直读全合：harness `53df4f2e`、direct `09d2705e`、比对器 v4 `1e9ad085`、学生 npz `b22020ff`、权重 `06a03d9c`、代码栈 stage0_teacher `c6ade5ce`／cnet2_np `d8c15ad4`／cnet2_search `94f7fb64`／cnet2_feat `2a45b08b`／fastsim `aefe6c83`／gate `e922d9d8`／k0_annotate `7f567a2e`／cnet2_model `3bb5857f`／eval_v53 `45792dfb`、carbon.json `cd55531b`、教师分片 `2aad09a5`／`ce7f84fc`。
- 回放双验全合：md5 `6ed2623e`／`6607fe5e`，真 sha256 `55d713d64c002d6c204988280804da47b96c979464f4ae9740b2676338581436`／`866a8ce7cc7125af8962e51f6db7b1a41686c3c13dbab35b2cec0c653a66baf5`。
- harness 启动时 `_verify_materials` 再验全过；config 重算 `config_sha256 = 9a2b2440e9267680c9b27a1d51a50db9cacdbd04582ae101ef54abaa86d8d10c`。

### 7.4 闸 4(a) harness 对照跑（完赛）＋(d) 跨机参考读数

- 对照跑 8／8 根完赛（教师 trunk，2 worker），墙时 1.09 分钟；每根 act 调用 648（min＝max＝mean，恰合预测 680−32）；generation `gen-retry-trunk-1b24175869f5`（与本地冒烟同 id）；`me_snap_lostw_zero_roots = 0`。
- 诊断 gen_candidates（8 根全）：`arms_match = True`、`priors_max_abs_diff = 0.0`、调用数 32／根。
- **跨机参考（harness 对照输出 vs L20 教师分片，8 根逐根对位）**：994400 max|dq| = 0.0、max|dQ| = 0.0；994401 max|dq| = 0.0、max|dQ| = 0.0；argmax 全同、arms 全同、priors 全同——**逐位一致**，强于裁定参考线（≤1e-6）。本机（VM）同根读数曾为 2.20e-07，系 VM 与 L20 的浮点环境差异；spot-64 与 L20 浮点环境一致、无此噪声。但裁定明示跨机比对仅参考、不作放行依据，执行方不据此放行。

### 7.5 闸 4(b) 同机直跑异常（停表触发，原文照录）

- 首次启动曾因本人命令 cwd 拼接失误（脚本相对路径在 /root 下未找到）未跑起来，已即刻以绝对工作目录重起；重起后脚本身份打印全合（七件代码 md5＋权重 `06a03d9c` 与座上直读一致），随后 worker 抛 TypeError 退出，`out_direct/shards` 未产生。异常与 harness 无关、非数值不等。
- `direct.log` 关键段原文：

```
[direct] stage0_teacher.py c6ade5ced42e2c54a77c1499163aa151
[direct] k0_annotate.py 7f567a2e8544b78e51d16aa7660bc47b
[direct] cnet2_np.py d8c15ad4fdd8ee079363cf215d52daf3
[direct] cnet2_search.py 94f7fb6438bb7577b141e388f068e122
[direct] cnet2_feat.py 2a45b08b6f97508541e62867e408748e
[direct] fastsim.py aefe6c83826efae4de83bef8c643c80b
[direct] stage1_teacher_gate.py e922d9d890c3e1eff6643e947289acb0
[direct] weights 06a03d9cfeec239af3065a49fefec382
concurrent.futures.process._RemoteTraceback:
"""
Traceback (most recent call last):
  File "/usr/lib/python3.12/concurrent/futures/process.py", line 263, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "/usr/lib/python3.12/concurrent/futures/process.py", line 212, in _process_chunk
    return [fn(*args) for args in chunk]
  File "/root/small_teacher_50k_retry/direct_control_roots.py", line 81, in _direct_root
    feats = [F.Feat.observe(scene, 2, 7000)]
TypeError: Feat.observe() takes 2 positional arguments but 3 were given
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/root/small_teacher_50k_retry/direct_control_roots.py", line 145, in <module>
    main()
  File "/root/small_teacher_50k_retry/direct_control_roots.py", line 130, in main
    results = list(ex.map(_direct_root, tasks))
  ...
TypeError: Feat.observe() takes 2 positional arguments but 3 were given
```

- 定性：`direct_control_roots.py` L81 把钉版 `cnet2_feat.Feat.observe`（实例方法，签名 `(self, obs)`，共 2 个位置参数）当作类级工厂以 3 参调用——脚本与钉版栈 API 不兼容，自始不能运行。脚本本身即批复件（md5 `09d2705e`），执行方不自行改写。

### 7.6 未做事项与收尾空载状态

- 未做：全量 396 根未开；学生重跑确定性抽查未做；shard_compare v4 未跑；`compare_stage4_retry_trunk_396.json` 未产出；Tier 四项无读数、不判。
- 收尾（所有者 07:03 接档裁定）：停报时刻 2026-11-18 07:21:57 CST spot-64 直读——本线进程零残留（`ps` 按 retry_trunk／direct_control／k0_annotate 扫为 0；harness 对照跑已完赛退出、direct 已异常退出），load average 0.01／0.15／0.09，座位空载，可供搜索组 §4 测速接档。§4 测速与探针 (b) 本线未起，等作战室通知。
- 座上留存：`/root/small_teacher_50k_retry/`（备料＋out_control 对照产物＋control.log／direct.log），未删，供复核与续跑复用。

## 8. v2 续跑（2026-11-18 07:39–07:54 CST，接档令＋裁定 346a7e47 放行：全量完赛、确定性复核过、比对入仓）

- **接档与裁定引用**：作战室接档令（2026-11-18 07:39 CST）：§4 测速完结、搜索组进程已清、spot-64 空载确认，4D 续跑开始。放行依据：裁定 `reports/RULING_SMALL_TEACHER_STAGE4D_GATE4_PASS_20261118.md`（commit `346a7e47`，md5 `8ac6b84d`）选项②——闸 4(d) 跨机逐位一致（0.0）替代同机硬闸；`direct_control_roots.py` 本次作废不用；附条件＝学生抽 2 根自身确定性复核、比对经 v4 出 Tier 四项＋compared_roots 报策略组（只报数）。
- **第 0 步座位直读（07:40:44 CST，开跑前）**：hostname `carbon-shanghai-spot-64`；load 0.02／0.11／0.12；`ps` 定向扫仅系统 python3 两件（networkd-dispatcher／tuned），无他人计算进程，空载开跑。座上备料 md5 复读全合：harness `53df4f2e`、比对器 v4 `1e9ad085`、学生 npz `b22020ff`、权重 `06a03d9c`、教师分片 `2aad09a5`／`ce7f84fc`。
- **全量（完赛）**：07:41:03 CST 启动（PID 11491，2 worker，解释器 `/root/carbon/.venv/bin/python`，日志 `student.log`），harness 启动自验身份全过（identities verified）。396／396 根完赛，`run_summary` 墙时 **9.18 分钟**（约 07:50 完赛）；实测单价 396 根／9.18 分钟 ≈ 43.1 根／分钟，即墙时 1.39 秒／根（2 worker 合计）、约 2.78 秒／根／worker。generation `gen-retry-trunk-829a171e5fce`；分片 sha256：994400 `a4c798a352aa77dd834f42f6cb1e1ebd1b11d662204ff94122c8343cc7c3e095`、994401 `4046ff28e3ae3a351ff8955678155cbfa64ba9e19a21fd5f8f3c6644d8f640ec`（各 198 根）；`n_act_calls` min 8／max 648／mean 636.36（min 8 为个别根 rollout 即刻终局所致）；`me_snap_lostw_zero_roots = 0`；产物 identity（config `9a2b2440…`、games `c3649caa…`、replay 真 sha256、code_md5s、weights_md5）与教师件逐字同（progress.json 直读）。
- **确定性复核（裁定附条件，通过）**：抽 994400（step 200, seat 0）与 994401（step 250, seat 1）以 `--only` 重跑至 `out_spotcheck`（07:53:02 CST，2／2 根，两根各 648 调用，generation id 与全量同为 `gen-retry-trunk-829a171e5fce`），与 `out_student` 对应根逐字段 `np.array_equal`：q／Q／Q_std／priors／arms／n_arms／belief／n_unknown 八字段、两根全 True。
- **比对（shard_compare v4 `1e9ad085`；base＝教师分片 gen-3f2c17f8b7beb5b6，cand＝重跑分片，口径「换 trunk 重跑」）**：结果件 `reports/eng_verify/compare_stage4_retry_trunk_396.json`（6,328 B，md5 `dd6b3e9d2a6f44a381b8b246e0723377`，本节同趟入仓）。identity 前检查全过（config／games true、replay_sha_match true、arms_set_diff_roots 0、roots_exact_arm_set 两局各 198）。**compared_roots 396／396**（994400 198、994401 198），可比动作条目 **6,336**（各局 3,168）；comparability floor（100）未触发，保底条款不启用。Tier 四项实测值（只报数，落点归策略组）：总——flips **101（25.51%）**、|signed mean bias| **5.42e-20**、max|Δq| **0.21810**、max|Δdu| **0.11090**；994400——flips 48（24.24%）、|bias| 2.79e-19、max|Δq| 0.17591、max|Δdu| 0.07987、du_mean_signed_delta −0.00368；994401——flips 53（26.77%）、|bias| 1.71e-19、max|Δq| 0.21810、max|Δdu| 0.11090、du_mean_signed_delta −0.00608。附带读数：q |Δ| 分位 p50 0.0262／0.0309、p90 0.0745／0.0960（两局序同上）。注：q 为同一钉死臂集上的分布、两侧逐根和均为 1，signed mean bias 构造上归零（构造事实，非判读）。
- **收尾空载直读（07:53:51 CST）**：本线进程零残留（`ps` 按 retry_trunk／shard_compare／bench_／k0_annotate／direct_control 扫为 0），load 0.05／0.83／0.77（1 分钟值已落至 0.05），座位空载；探针 (b) 接档由作战室排。座上留存 `/root/small_teacher_50k_retry/`（备料＋out_control＋out_student＋out_spotcheck＋compare 件＋各日志），未删。
