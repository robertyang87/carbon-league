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
