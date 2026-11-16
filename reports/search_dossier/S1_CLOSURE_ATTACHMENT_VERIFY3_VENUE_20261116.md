# S1 关账附件 · verify3 考位重跑（C1 spot-34，2026-11-16）

本附件随 S1 关账整包重提（不另开包）。结果 JSON：`s1_verify3_results_20261116_venue.json`（md5 ae01fd10fc1548abcd9b3c01dbfa0f9a，139,338 B）。

## §0 标的标签更正（入档前纠正，数字不受影响）

**RUN_BUNDLE 与任务书的「运行包 be96d457」标签是错的。** 冻结代码即权威：

- `real_env.py:112`：`EXPECTED_PACK_MD5 = "819250bfc82632786063dabd46a616bd"`（基线包），`load_pack` 硬校验、不合即停；
- `s1_verify3.py:598`：无参 `fetch_pack()`，经 `real_env.load_pack` 加载的是**基线包 echo4_int8.py**；把 be96d457 交给它会直接报 md5 不合停机——**冻结代码不可能以搜索包为标的跑 verify3**。

**本趟实际标的＝基线包 819250bf（agent 来源）＋冻结原型比较器（real_env.decide_at_root）。** 搜索包 be96d457（9,030,752 B）机上已验在位，但未被本脚本消费；它是 S2 的标的。VM 首趟（02:12）同一代码、同一基线包标的，标签错误自运行包起即存在，本趟在考位执行时由执行方发现、经我按冻结代码核对后纠正。改脚本去啃 be96d457 等于改冻件、须重验收，不可擅动。

## §1 开跑三件套（实际值）

- 启动时刻：2026-11-16 09:32:11 CST（第二启；第一启见 §5）
- 进程：PID 65180，`/root/carbon/.venv/bin/python -u s1_verify3.py`，scope `verify3-echo4search.scope`，启动后查验 ActiveState=active；完赛后进程已退、scope inactive
- 口径读回：scope `CPUQuotaPerSecUSec=1.500000s`（150%）、`MemoryMax=536870912`（512 MiB）；BLAS 四线程变量全钉 1；agent 局内持久、跨局重建；decide 预算生产 0.7 s（冻结 SI02）
- 机上 md5 复验（开跑前逐件，全合）：s1_verify3.py 17310cc1 / real_env.py f9bfacb7 / comparator.py 53e5c1a1 / smoke_real_s1.py 2425f340 / telemetry.py 91a7121f / leaf_adapters.py a5852c73 / pack_bridge.py ad522153；vendor：feat 2a45b08b / np d8c15ad4 / fastsim aefe6c83 / search 代码 md5（去 3 行头注释）94f7fb64；基线包 819250bf（4,601,242 B）、搜索包 be96d457（9,030,752 B）均在位

## §2 完赛

- 总墙钟 **54.879 s**；退出码 0；无 traceback；有效验证 3
- scan_games：局 0（seed 986400）：到步 269、记录 58（步 240–268 全覆盖）、离线 decide 成功 58、**changed 24**（12 步 × 双席）；局 1（seed 986401）：跳过——「第一局 changed 候选已足（24 个），留墙钟给验证」
- 选根 3 个：(267, seat 0, idx 2)、(267, seat 1, idx 2)、(266, seat 0, idx 1)；**有效验证 3/3，restore_reproduction_ok 全 True**；三根无一灭绝，final_basis 全为 F04_terminal_cash

## §3 配对双臂 ΔU（搜索组按 RULING_R2 §R2-08 自逐根逐臂终局分数对换算）

口径：M_arm ＝ 根席终局分数 − 对手席终局分数；WDL 按分差符号（胜 1／平 0.5／负 0）；U_arm ＝ 0.5·WDL ＋ 0.5·sigmoid(M_arm/1000)；ΔU_root ＝ U_chosen − U_A0。

| 根 | A0 臂终局分数 [席0, 席1] | chosen 臂终局分数 [席0, 席1] | M_A0 → U_A0 | M_chosen → U_chosen | ΔU |
|---|---|---|---|---|---|
| 267 seat 0（idx 2） | [5652.48, 5652.48] F04 | [5997.49, 5814.35] F04 | 0 → 0.500000 | +183.14 → 0.772829 | **+0.272829** |
| 267 seat 1（idx 2） | [5652.48, 5652.48] F04 | [5814.35, 5997.49] F04 | 0 → 0.500000 | +183.14 → 0.772829 | **+0.272829** |
| 266 seat 0（idx 1） | [5652.48, 5652.48] F04 | [6090.32, 6090.46] F04 | 0 → 0.500000 | −0.14 → 0.249983 | **−0.250017** |

判读域：按 R2-08，本表判读限晚局根集；中局「改子即更优」仍开放、归 S3/S4。

**两点如实声明（不改数字，只防误读）：**
1. 267 的双席两根是**同一独立步的镜像对**（同一步、双席各一根，ΔU 逐字同值）——判读域计数时按 1 个独立正例步计，不按 2 个。
2. 266 根的 ΔU 为负是定版 U 形的断点效应：根席现金 chosen 臂 6090.32 vs A0 臂 5652.48（＋437.84），但分差 M＝6090.32−6090.46＝−0.14，WDL 按符号翻为负 → U 0.249983、ΔU −0.250017。现金分差不得推翻 U 结论（R2-11）；该根按定版口径记为负例。

## §4 tie_check（旧 3420.32 平局未复现）

- `seen: true, evaluable_arms: 6`；3 条触发全为 `within_arm_exact_tie`，均在 A0 臂、终局分数 [5652.48, 5652.48]（根 267 seat 0／267 seat 1／266 seat 0）
- 三根 details：`cross_arm_seq_identical_frac` 全 0.0（两臂续弈逐步不同轨）、`within_arm_mirror_frac` 全 1.0（平局臂内双席镜像推进）
- 结论：旧 3420.32 平局**未在本趟出现**；本趟平局是 A0 臂内双席精确打平 5652.48＝5652.48，属基线自对弈对称镜像——**不是两臂退化同轨**（cross_arm 全 0.0 为证）。R2-03 后的镜像核查行如实交出，未复现即未复现。

## §5 第一启失败披露（启动即失败、已修复，不并进成功回执）

- 09:31:28 第一启在 vendor 核验通过后、`fetch_pack()` 处抛 `RuntimeError: gh_api 不可用且缓存/checkout 均无钉版包`、进程秒退，未产生任何证据。
- 根因：`fetch_pack()` 的 checkout 候选路径是 `../../packs/echo4_int8.py`（相对 search_proto 两级上跳），在部署树里落在不存在的 `/root/carbon/packs/`；且 `/tmp/echo4_int8.py` 缓存当时不存在、C1 无 gh_api 可走在线通道。
- 修复：把机上已验 md5 的基线包按脚本自带口径拷为 `/tmp/echo4_int8.py` 缓存（复验 md5 819250bf），冻件**一行未动**；第二启 09:32:11 正常开跑。失败日志已留存（`verify3_run_attempt1_failed.log`，md5 a44f6489a23f2c7888c56a69c55a76e8）。

## §6 文件

- C1 机上：`/root/carbon/code/search_proto/s1_verify3_results.json`（md5 ae01fd10fc1548abcd9b3c01dbfa0f9a）、`/root/carbon/verify3_run.log`（47a0ec3056751eb7c0676be139072190）、`/root/carbon/verify3_run_attempt1_failed.log`（a44f6489a23f2c7888c56a69c55a76e8）
- 本地副本：`~/workspace/search-group/verify3_rerun/` 同名三件（md5 同上）
