# A4 选模 m7bsr1 臂完工回执（Studio 线，2026-11-14）

执行人：Studio 线（唯一执行人）。被选臂：m7bsr1（带注意力臂，权重 cnet2_m7bsr1_ep0–ep19）。
选模模块：cnet2_train a3ad86d4a74424134ecc650275a5161e（V06 清账版，开跑前与完工后两次直读 md5 均合）。
干净解释器起跑：2026-11-14 15:39:43 CST；driver PID/PGID 42374，16 worker；meta 落盘 15:57，总历时约 17.5 分钟。
考官：Master2 真身 code/exp/opp_master2r.py，md5 2aebfbc1008d3475ece8e178696e292d。
口径：selection_form = naked-proxy；排名块种子 997000–997063（64 对、每 epoch 128 局）；errors 全 0、failed_epochs 无。

## 全 epoch（排名块 mean U / se / 墙时秒 / agreement）
ep0 0.3325/0.0247/43.8/0.8470；ep1 0.1341/0.0169/43.4/0.8522；ep2 0.5384/0.0289/43.7/0.8545；ep3 0.4025/0.0293/43.6/0.8561；ep4 0.4701/0.0271/43.7/0.8554；ep5 0.5921/0.0285/43.7/0.8572；ep6 0.5297/0.0316/43.6/0.8587；ep7 0.6001/0.0266/43.7/0.8593；ep8 0.5432/0.0263/43.6/0.8582；ep9 0.6497/0.0267/43.7/0.8590；ep10 0.7102/0.0232/43.7/0.8629；ep11 0.6902/0.0257/43.7/0.8610；ep12 0.6443/0.0286/43.7/0.8612；ep13 0.6511/0.0250/43.6/0.8597；ep14 0.7633/0.0208/43.8/0.8613；ep15 0.7071/0.0243/43.7/0.8596；ep16 0.7187/0.0232/43.9/0.8606；ep17 0.7198/0.0235/43.7/0.8605；ep18 0.7362/0.0207/43.8/0.8602；ep19 0.7569/0.0204/43.7/0.8604。

## 双 pick（分歧点名）
- 平局带：ep14（0.7633）对 ep19（0.7569），d=0.00637，se_diff=0.02228，分不出。
- pick_code（代码原生，排名块部署形态重打）：ep14 0.7595（se 0.0202）对 ep19 0.7113（se 0.0269），d=0.0482 > se_diff 0.0312 → winner ep14，rule=duel_deployed。
- pick_operative（作战室口径，新鲜块 997064–997127 部署对决）：ep14 0.7453（se 0.0225）对 ep19 0.7247（se 0.0208），d=0.0206 < se_diff 0.0258，分不出取晚 epoch → winner ep19，rule=duel_tie_later_epoch。
- winner_is_ep19 = true。

## 结论（按预注册口径）
operative winner 就是现役 ep19 本人 → 本臂即关闭：无新提名、无打包、无 C1 配对场。pick_code 与 pick_operative 的分歧已点名如上，形态裁定归策略组/作战室。
A4 三臂事实汇总：g1 operative ep17、C1 ΔU −0.0236 未采纳；r1 operative ep18、C1 ΔU +0.0058 未采纳；m7bsr1 operative winner 为 ep19 本人。

## deps 清单（策略组条件③对锁用）
- 本线驱动走 _selection_eval 直调（不经 _selection_open），运行中未在 sel 目录写 examiner_snapshot 文件、meta 内无 deps 字段——如实点名。
- 完工后用同一已清账枚举器（a3ad86d4 的 _selection_snapshot_deps）对考官原件复导：deps = [{file: master2_src.py, md5: 404114a0b5e422516322f62e9ddb6870}]，与策略组备案的真身源 md5 合。
- 考官本体 opp_master2r.py md5 2aebfbc1（config 时一次读入计 md5、完工后直读复核不变）；两文件 mtime 均为 2026-11-11（考官 16:16、dep 14:21），远早于运行窗 15:39–15:57，运行窗内无可变实项。
- 打包链：cnet2_search_pack 48392e98、引擎 cnet2_pack 119e9bb9bf8413685b87721cff8cfe18（日志首行 pack_chain 在案）；对决包 ep14_pack.py 4,597,730 B、ep19_pack.py 4,597,410 B，在 Studio a4_selection_20261114/m7bsr1_sel/。

## 工件与收尾
- meta：/Users/arm/junyang/a4_selection_20261114/m7bsr1_selection.json（54,763 B，md5 0da6491344a3ff54bdb9fcba8695f8b8）。
- 零残留已验：driver 进程组不在册、pgrep -f a4_select_driver_m7bsr1 = 0（watcher 与完工后直读双验）。
- Studio 现为正常空闲（任务完工、meta 终版在册）。请作战室派下一活；Studio 可接 MPS 训练/选模/gate 类任务。
