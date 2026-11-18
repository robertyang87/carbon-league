# 钉版六件套对账一局（987001 冒烟局，钉版↔main 两臂量化）——正式对照记录

> **文件头注**：依策略组裁定 `fa2fcef7` 及补注 `3b02527e`，本件为**正式对照记录**入仓存档；同时依同一裁定，本件**非正式环前置**——正式环「钉版六件套对账」前置已由 Studio 件顶替销项（Studio 线以钉版六件套＋适配器对 300002 局跑完同窗四环全过，接口回执 §4 存疑 2 已由作战室宣布关闭；其差异根源定位在平台层——macOS APFS 大小写不敏感致 Q.npy/q.npy 撞名——不在 search 语义）。与 Studio 件的关系：Studio 件为 300002 局、钉版单臂、四环全过；本件为 987001 冒烟局、main↔钉版两臂逐字段量化，口径不完全相同，独立存档。

- 原任务依据：策略组裁定 `reports/RULING_KATAGO_CORE_U_HEAD_20261118.md` §3（commit `12b4a792`）；前情回执 `reports/eng_verify/KATAGO_LOOP_INTERFACE_CHECK_20261118.md`（md5 `51bf0606`）§5 冒烟（其 annotate 用 main 版 cnet2_search `c3222eea`，本对账即隔离量化这一处版本差）。
- 执行线：工程组（本机 muse VM 单机执行，未占 spot-64/L20/DGX 执行面；环境同冒烟：venv numpy 2.5.5、torch 2.15.0+cpu）。

## 1. 对账协议

- 同一局：冒烟局 seed 987001 产局回放；同一窗：[200,230)，60 根；同一 build 产物、同一教师（parent 权重 `06a03d9c`）、K16/M8/H2、belief sample、seed_teacher 7000、pi0/pi1 1.0、同一 k0_annotate `7f567a2e` 与其余共用件。
- 唯一变量：cnet2_search——A 臂＝main 版 `c3222eea76ea9d42c160626c2e239ca0`（冒烟原口径，产物直接取用未重跑）；B 臂＝钉版 `94f7fb6438bb7577b141e388f068e122`（codeB 栈：15 件中仅此一件换钉版，其余 14 件与冒烟栈逐字节相同，落盘前逐件 md5 验合）。
- 比对：两臂 60 根按 (gid, step, seat) 对齐、按臂身份（12 元组）逐字段比——arms 集合一致根数、priors max|Δ|、q max|Δ| 与 signed 均差、Q max|Δ|、argmax flips；config_sha256 两臂各自以本体 `teacher_config` 重算。
- 判读口径（协议定，只报数）：差异为 0 或浮点噪声量级（≤1e-6）记「对账一致」；flips>0 或 max|Δq|>1e-4 记待决并列分布。

## 2. 身份（两臂均工具自记＋直读双验）

| 件 | A 臂（main） | B 臂（钉版） |
|---|---|---|
| annotate 代 | `gen-3bd8b0585fce9f1f`（冒烟代） | `gen-7f2d08e42a7cbd97` |
| cnet2_search.py（producer.code_md5 自记） | `c3222eea76ea9d42c160626c2e239ca0` | `94f7fb6438bb7577b141e388f068e122` |
| 其余六件 code_md5（自记，两臂逐字相同） | stage0_teacher `c6ade5ced42e2c54a77c1499163aa151`、gate `e922d9d890c3e1eff6643e947289acb0`、cnet2_np `d8c15ad4fdd8ee079363cf215d52daf3`、cnet2_feat `2a45b08b6f97508541e62867e408748e`、fastsim `aefe6c83826efae4de83bef8c643c80b`、k0_annotate `7f567a2e8544b78e51d16aa7660bc47b` | 同左 |
| config_sha256（工具自记） | `79c44c1f0f83fdb9971de3413365dbb29d62540d9669bf016654b7a172373942` | 同左 |
| config_sha256（本件重算，两臂各自本体） | `79c44c1f0f83fdb9971de3413365dbb29d62540d9669bf016654b7a172373942` | 同左 |
| weights_md5 | `06a03d9cfeec239af3065a49fefec382` | 同左 |
| 代完整性 | manifest 14 字段 sha256 全合、meta_sha256 合（VM 重启后复验） | manifest 14 字段 sha256 全合、meta_sha256 合、发布时工具自验 14 文件过 |

- 局与窗：回放 `selfplay-987001.json.gz` md5 `6e24b18051bf8a3f0ea3fdff1b7144bc`（与回执 §5 所记合，重启后复验不变）；build 代 `gen-4f6b7ce57c46bbfa`（两臂共用）；窗 [200,230)，两臂各 60 根、根键集完全相同（only-A／only-B 均为空）。
- config 注：`teacher_config` 入参不含 cnet2_search（其身份走 producer.code_md5），故 config 两臂相同是构造使然，非判读。

## 3. 逐字段差异（B − A，60 根／960 臂条目）

| 字段 | 结果 |
|---|---|
| arms 集合一致根数 | **60/60**（且顺序一致根数 60/60） |
| priors max\|Δ\| | **0.0** |
| q max\|Δ\| | **0.0** |
| q signed 均差 | **0.0** |
| Q max\|Δ\| | **0.0** |
| argmax flips | **0** |
| 附带核（非协议项）：Q_std／belief／n_unknown／n_arms／pi | 逐根 `array_equal` 全 True |

即两臂全部 60 根、全部字段**逐位一致**（非噪声量级，是恰 0）。

## 4. 执行记录（含中断与恢复，如实）

- A 臂：冒烟原产物直接取用；VM 重启后按 manifest 逐件 sha256 复验完整无损，未重跑。
- B 臂共三次启动：第 1 次（约 08:09 CST）被本机 VM 约 08:13 CST 重启打断，0 根落盘（死点目录本机 `recon/annotB_attempt1_dead/`，其 progress.json 自记身份正确）；第 2 次（约 08:26 CST）依首轮停跑令于 08:30 CST 停掉，0 根落盘（`recon/annotB_attempt2_stopped/`，停后 pgrep 无残留）；其后依修正令（正式件口径）第 3 次启动，同命令同口径，完赛落盘代 `gen-7f2d08e42a7cbd97`（60 根，墙时约 23 分钟，与冒烟单价 1,479.8 秒量级一致）。
- VM 重启影响范围：仅 B 臂第 1 次启动；A 臂产物、回放、权重、两套 code 栈重启后逐件 md5/sha256 复验均不变。

## 5. 结论

**对账一致**：在本局（seed 987001）、本窗（[200,230)，60 根）、本机环境下，cnet2_search main 版 `c3222eea` 与钉版 `94f7fb64` 的 annotate 产出逐位相同（全部差异恰 0，flips 0）——接口回执 §4 存疑 2 所指的版本差，在本量化范围内对产出无影响。本结论仅覆盖此一局一窗一机，不外推其他局/窗/平台（Studio 件的平台层撞名问题属 macOS 特有，不在本件范围）。
