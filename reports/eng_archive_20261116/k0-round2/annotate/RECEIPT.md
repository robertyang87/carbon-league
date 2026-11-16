# K0 第二轮 §10 件 1：标注器 —— 实现回执

- 交付物：`code/k0_annotate.py`（新建，唯一改动文件）
  - md5：`cdee7c03c5be68001eeef95eab356d95`，37,672 字节，828 行
  - 落仓 commit：`cf8d6b8ef7b7`（main），提交后 API 读回逐字节一致（BYTE-IDENTICAL: True）
- 纪律核对：
  - 闸门 `code/stage1_teacher_gate.py` md5 动手前/后均为 `e922d9d890c3e1eff6643e947289acb0`（本地副本与仓内读回双验）；教师本体、exit 三件、k0_train 均未改动。
  - 候选与 Q/q 全部经闸门器械路径产生：`G.extract_scene` → `T.evaluate_root`（与 `stage1_teacher_gate._q_scene` 同一调用形态，含 prefix feat、guard 快照、seed_base=seed_teacher+step），`G._expect_teacher` 身份断言在列；本文件内无第二份 q 公式（q 只从 evaluate_root 返回值落盘）。
  - 未启动任何生产标注，未碰任何机器进程。fixture 权重为 heads v2（`06a03d9cfeec239af3065a49fefec382`，本 VM 在手权重），回放为 gen1r_smoke 的 994000 段自博弈局。

## 产物形态（§13-C 落地）

- 格式 `k0-annotation-generation/1`，封存机制照 exit 侧 `cnet2-exit-generation/1` 适配：staging → meta.json＋manifest.json（self-seal＝插入 seal 前的 canonical 哈希）→ gen id 内容寻址 → 已发布同代验封复用、绝不删除（V02 后口径）→ current.json 指针最后翻转；消费走 pointer-only（`load_annotated`/`--verify`），直指 generation 目录被拒。
- 每根一条记录，字段分文件（.npy 各一）：策略标签 arms/priors/q；价值标签 Q/Q_std；另 gid/step/seat/n_arms/pi0/pi1/pi/belief/n_unknown。记录组装说明在 meta 的 `record_schema`。
  - priors 一并落盘的理由（§13-C 未禁、组装需要）：q 的 P_A 因子在 train_temp≠1 时不可由 q 反推，第二轮 C 臂 q₀∝P_A^0.5 需要 P_A 本身。
  - 教师配置哈希（config_sha256，canonical JSON）与权重 md5 是代级身份，写在 meta 的 `teacher` 块并由 manifest 的 producer 绑定封印；代码 md5 集＝闸门 header 口径六文件＋k0_annotate.py 自身。
- 稀疏采样：两趟（A：整局以 π0 抽中取全根；B：未抽中局逐根以 π1 抽），边际包含概率恒等 π＝π0＋(1−π0)·π1 并按此式逐根落盘；单 RNG 流由显式 seed 派生、消费顺序冻结，选中集 sha256 全局＋逐局落账（k0_train B2 同式）。窗口为登记参数，默认 [200, 299)（裁定条件 I）；隔离清单为显式可重复输入。

## 五项验收（fixture 实证）

1. **身份绑定读回 —— 通过。** fixture 代 `gen-1dcc9611575bfbe3`（2 局、窗口 [206,209)、12 根、降格 K4/M2/H1）：发布内 verify 与独立 `--verify` 均通过（14 文件）；manifest producer 的 code_md5 与仓内锚点逐项一致（gate e922d9d8、teacher c6ade5ce、feat 2a45b08b、np d8c15ad4、search 94f7fb64、fastsim aefe6c83），weights_md5＝heads v2 实测 md5，config_sha256 与 meta 记录互洽；同参重跑两分片 `shard_skip_verified`、代 `reused: true`（复用不删）；直指 generation 目录消费被拒。
2. **抽样重算 q 逐位一致 —— 通过。** 独立脚本 `recompute_check.py`（不经标注器求值封装、直调 extract_scene＋evaluate_root，配置取产物清单记录值）对 selfplay-994000 的 3 根（206/0、206/1、207/0）重算：n_arms、arms、Q、Q_std、q、priors 六项全部 `np.array_equal` 逐位全等。教师为真实教师（v3/belief=sample），本 VM 可跑，仅按单根约 15–28 s 的实测速度降格了根数（12 根）与候选规模（K4/M2/H1，非规范 K16/M8/H2），未用 stub。
3. **窗口/隔离硬校验 —— 通过。** 进料侧：隔离文件含 selfplay-994001 → die 点名该局；`--window-hi 300` → 登记窗口越界 die。产物侧：`validate_records` 对 step=199 与 step=299（窗口 [200,299)）均 die 点名；封印全真但 step=150 的手工代被 `--verify` 以窗口违规点名拒绝（非封印错误）。
4. **断点续跑 —— 通过。** `resume_check.sh`：3 局串行跑，首分片（994000，4 根）落盘后 kill -9；progress.json 完整；重跑日志 `shard_skip_verified: selfplay-994000`，该分片续跑后 sha256（0060354f…）与 mtime 均未变（未重算实证），余两局补齐，发布 `gen-87d4ccb63b1cf6eb`（12 根）且 `--verify` 通过。
5. **缺 π 拒收 —— 通过。** 记录级：pi 字段缺失、pi=NaN、pi 与 π0/π1 恒等式不符（想要 0.75 实为 1.0）均在封存前自检 die 点名；代级：pi 字段整体缺失的手工封存代被 `--verify` 点名拒绝。自验中另修正一处过严口径：π0/π1 作为两趟速率允许为 0（该趟关闭），仅 π 本身必须 ∈(0,1]（IPW 要除它）；pi0=0/pi1=0.5 的稀疏代端到端落盘验证 π＝0.5 如实记录（fixture `gen-9387b77852cd913a`，2/4 根）。

## 证据位置

`~/workspace/k0-round2/annotate/`：本回执、`recompute_check.py`、`verify_gate_check.py`、`resume_check.sh`、`fixture/`（out1 主代、out2 续跑代＋phase1/2 日志、out3 稀疏代、bad_window/bad_pi 拒绝样本）、`run/`（可运行副本与依赖）、`src/`（参考源）。
