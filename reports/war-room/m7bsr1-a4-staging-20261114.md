# m7bsr1 A4 选模预置回执 — 2026-11-14 13:22 CST（Studio 线）

派单：作战室 2026-11-14 13:15，Studio 为唯一执行人；V06 已清账。
状态：**预置全部完成（all_ok: True），选模未启动**。v3 pilot 仍在跑（13:22 直读 launcher 89589 / gate 89592，已跑 1:12:04，12 worker 在册，未动）；选模等 pilot 实际完工后再开。

## 代码（Studio 盘上，读回验合）

| 件 | 位置 | md5 | 结论 |
|---|---|---|---|
| cnet2_train.py（选模模块） | /Users/arm/junyang/carbon/code/cnet2_train.py | a3ad86d4a74424134ecc650275a5161e | ✓ V06 清账版；py_compile 过；旧版 b6a1fc7d 已备份至 /Users/arm/junyang/carbon/code_backup_pre_a4m7bsr1_20261114/cnet2_train.py（读回 md5 b6a1fc7d…，与 g1/r1 两臂所用版一致） |
| opp_master2r.py（考官 Master2 真身） | /Users/arm/junyang/carbon/code/exp/opp_master2r.py | 2aebfbc1008d3475ece8e178696e292d | ✓ 与派单一致 |
| cnet2_pack.py（对决打包引擎） | /Users/arm/junyang/a4_pack_stage/cnet2_pack.py | 119e9bb9bf8413685b87721cff8cfe18 | ✓ 钉死值未动 |
| cnet2_search_pack.py | /Users/arm/junyang/a4_pack_stage/cnet2_search_pack.py | 48392e98c03f25a3a6f616561c8a072e | 记录（两臂同款落盘，未改） |

**来源说明（与派单措辞有一处出入，已核实）：** 派单写“rl2_ppo_train.py”，实测仓库 main 的 `code/rl2_ppo_train.py` 是另一个小文件（md5 689989b2…，51,052 B），选模模块实际是 `code/cnet2_train.py`（118,712 B）。取回值 a3ad86d4 在 main 与 commit 10907a81（"fix(selection): V06 freezer enumerates __file__-anchored dynamic loads"）下逐字节一致。本地副本：~/workspace/a4_g1/cnet2_train_a3ad86d4.py。

**API 核对：** a3ad86d4 含驱动所用的全部 _selection_* 入口（_selection_config / _selection_policy_file / _selection_eval / _selection_pick / _selection_paired_se / _selection_md5）；相对 b6a1fc7d 只新增 _selection_freeze_examiner 与 _selection_snapshot_deps（V06 冻结器），无删减。

## 权重（Studio /Users/arm/junyang/carbon/data/weights/，DGX 局域网拉取，逐件 md5 对账，21/21 合）

DGX 源：/home/arm/junyang/carbon/data/weights/，2026-11-14 ~13:19 直读；合计 279,382,403 B。训练日志 cnet2_m7bsr1_log.jsonl：20 行全为 val 行，最后 20 个 ep 恰为 0–19（驱动 agreements() 断言可过）；ep10 val all = 0.8629（与派单 val best 一致），ep19 val all = 0.8604。

| 文件 | md5 | 大小 (B) |
|---|---|---|
| cnet2_m7bsr1_ep0.npz | b1f34a5b9a920859dd756db9d7813315 | 13,968,960 |
| cnet2_m7bsr1_ep1.npz | ac46d578b1923cb9659f5b8ac4f8f0f3 | 13,968,964 |
| cnet2_m7bsr1_ep2.npz | 7daa47818985e3c1532a23a270a54893 | 13,968,964 |
| cnet2_m7bsr1_ep3.npz | 9a4309ffefd3ca6b60ef3762aa83e7fd | 13,968,960 |
| cnet2_m7bsr1_ep4.npz | 3c10abee985203497cf3c568e7206c24 | 13,968,964 |
| cnet2_m7bsr1_ep5.npz | bfdfca6ea66a07fa06019b318ef9c024 | 13,968,968 |
| cnet2_m7bsr1_ep6.npz | 34309a78f816811838b2452313360c77 | 13,968,960 |
| cnet2_m7bsr1_ep7.npz | 7cb813c99d0a4184622552ff50630531 | 13,968,968 |
| cnet2_m7bsr1_ep8.npz | 69e573ad0372084d8ee50bf4263a3712 | 13,968,968 |
| cnet2_m7bsr1_ep9.npz | 1648472084d1ae14e5cd760aa50d2b42 | 13,968,960 |
| cnet2_m7bsr1_ep10.npz | 13fa6dfbac809b9d8671b47f83c4258d | 13,968,968 |
| cnet2_m7bsr1_ep11.npz | d928fdf475244be6f82c887beccf3abe | 13,968,956 |
| cnet2_m7bsr1_ep12.npz | b0f66ac31dbe22dd215e3d98ac23c303 | 13,968,964 |
| cnet2_m7bsr1_ep13.npz | 98d5257a49bec7ea7306357c54a647ea | 13,968,972 |
| cnet2_m7bsr1_ep14.npz | 1f524c70f7c02731ec152121b317c71e | 13,968,972 |
| cnet2_m7bsr1_ep15.npz | 0ecd7bd594d05fa681a0ae05f7455d2b | 13,968,972 |
| cnet2_m7bsr1_ep16.npz | 1825769ad6f57044066897f41926ba24 | 13,968,972 |
| cnet2_m7bsr1_ep17.npz | 14537836e8a51bd8db3c235ab59f0fe2 | 13,968,972 |
| cnet2_m7bsr1_ep18.npz | 24538549307c3752c01af37cc8fc5954 | 13,968,968 |
| cnet2_m7bsr1_ep19.npz | 0835830f792840ed66eef6d5be84b417 | 13,968,972 |
| cnet2_m7bsr1_log.jsonl | f21842dab9638935a3fe135d93c1c76f | 3,079 |

未拉取：*.stale1320 件（旧版残留）、cnet2_m7bsr1.npz / cnet2_m7bsr1_last.npz（选模驱动不用）。

## 选模驱动（已落盘 Studio，py_compile 过，未运行）

- 驱动：/Users/arm/junyang/a4_select_driver_m7bsr1.py（由 g1 臂驱动逐字改 arm/权重前缀而来，diff 仅 6 处：docstring、log 路径、权重路径、arm 元组、完工打印；其余含平局带对决逻辑与 g1 臂逐字一致）
- 启动器：/Users/arm/junyang/a4_launch_m7bsr1.py（PGID＋主 PID＋预期 worker 16 落盘，与 g1/r1 启动器同款）
- 本地副本：~/workspace/a4_g1/a4_select_driver_m7bsr1.py、a4_launch_m7bsr1.py、pull_m7bsr1.py、dgx_m7bsr1_read.sh、cnet2_train_a3ad86d4.py
- 待 pilot 完工后执行的命令（在 Studio 上）：
  `/Users/arm/junyang/carbon/.venv/bin/python /Users/arm/junyang/a4_launch_m7bsr1.py`
- 选模口径：种子 997000–997063 共 64 对、每 epoch 同一套；平局带前二部署形态对决、种子自 997064 续；selection_form = naked-proxy；meta 落 /Users/arm/junyang/a4_selection_20261114/m7bsr1_selection.json（与 g1/r1 同目录）

## 仍待办事项

1. 等 v3 pilot 实际完工（watcher 报 done；估计 15:31，估计），再执行上述启动命令开选模。
2. 选模完工回执须含：全 epoch mean U／对数／种子范围、每 epoch 墙时、winner、winner 与 ep19 同异结论、selection_form 声明、meta 文件路径。
3. winner 若 ≠ ep19，作战室再派 DGX 打包＋C1 配对场；winner 若是 ep19，本臂即关闭。

## 过程备注

- 13:16–13:19 期间本机出口代理两次断开（nc: proxy read: Broken pipe），60 秒后自愈；期间未做任何写操作，属传输层瞬断，无影响。
- 预置全程未抢 pilot CPU：传输为局域网 scp（约 280 MB），刷盘为文件拷贝＋编译检查；13:22 复核 pilot 进程与 worker 数无变化。
