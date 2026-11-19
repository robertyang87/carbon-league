# 原版考位交接

训练第 04 批完全封存后，第 05 批会在启动首场源局前等待。开发方占用 CPU 0、1，运行冻结的 Sapling 候选和 v1 父代，对同两张独立地图各自换席，合计八场。训练后续批次仍通过既有的顺序监督器启动。[预约与候选协议身份](resources/veto_dev_slot/reservation.json)

开发方先签发 `planter_survival/native_dev/gate.json`、`slot_request.json`，运行 `python resource_slot.py await-grant`。原版比赛的所有阶段放在 `with resource_slot.native_session():` 内。该上下文会持续持有排他文件锁。中断后从同一目录重启即可继续尚未签收的席位。

两臂的完整比赛、逐局动作及官方复算齐备，并发布 `comparison.json` 后，运行 `python resource_slot.py release`。签收器检查两个种子、物理席、权重和包身份，对每局逐步重放动作，核对双方分数、终局步与胜负，最后原子发布 `resources/veto_dev_slot/release.json`。训练进程会自行恢复，不根据开发赛胜负决定是否让出考位。

若部署近似的离线核验失败，开发负责人在独立注册目录发布 `decision=do_not_run_native` 的失败门结果。主动撤回写 `decision=owner_withdrawn` 与 `owner=paired_hybrid_parent`。两者均需要 `started_native_games=0` 和可读、哈希匹配的 `evidence_files_sha256`。运行 `python resource_slot.py cancel --evidence planter_survival/native_dev/gate.json --reason '具体失败或撤回原因'` 可在开场前取消预约。已有对局进程或实际成绩时，将保留已开始的开发批次供恢复。所有签收文件都可重复核验。
