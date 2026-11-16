# 逐局仪表化 · 逐 hunk 源码审验（策略组，2026-11-16）

对象：`code/instr_20261116/`（commit 555e2ee0）。基线：trainer
`code/rl2_ppo_train.py` md5 `2c8262fb`（blob ba40f397）；rollout
`code/rl2_ppo_rollout.py` md5 `f91b13c3`（现行生产版）。审验方逐 hunk
直读 diff（trainer 3 hunk、rollout 6 hunk），非凭回执。

## 1. Trainer（md5 `8696c4ef`）：PASS

三 hunk 全在日志路径：
- `--game-log` 旗标（默认 1，0 仅供等价性闸）；
- 任务字典透传 `game_log` 键；
- 主进程写盘块：单份 `games.jsonl`、逐局 append＋flush、字段集逐字对
  注册（own_cargo/opp_cargo 显名、own_score/opp_score 现金双向）、
  `written != len(games)` 即抛错、写失败在 train.jsonl 留
  `game_log_write_failed` 标记行并 raise（非零退出）。
- `not resume_warm` 守卫经源码核对成立：resume 预热批
  `resume_warmup_counts_as_iteration=False`、轮行走 wlog 不进
  train.jsonl、league 亦不更新——预热局不入日志与完整性闸口径一致。

## 2. Rollout（md5 `84fd90a7`）：FAIL——非日志 hunk 回退生产防护

与基线 `f91b13c3` 的 6 个 hunk 中，3 个日志 hunk 合格，但另 3 处在
日志路径之外、回退了现行生产防护（该文件系从旧版 rollout 分叉，
非从钉版基线分叉）：

- **H-a**：`_snap_temp()` 校验函数被删（RL2_SNAP_TEMP 的 [0,2] 范围
  校验＋import 期 fail-fast 一并删除），两处调用点改裸
  `float(os.environ.get(...))`。后果：环境值写错不再响亮失败，
  快照温度被静默改变——正是该校验要防的事。
- **H-b**：`play_batch` 的 `role=task.get("primary_role", "learner")`
  被写死 `role="learner"`，V04 修复回退（独立评估器以
  primary_role="eval" 调用时将静默按 learner 温度打）。

注册口径：改动只许在日志路径。以上两处不合，**本版 rollout 不予
验收**。

日志 hunk 本身（单独判）：合格且只读——死亡计数为步前/步后工人集
差（pre/post 只读、不动状态不耗 RNG，终局步计入）；终局树/货物
取终局状态只读计数；SELF 镜像记录字段对位互换。三 hunk 原样保留。

## 3. 修复令（机械、范围钉死）

以基线 rollout `f91b13c3` 为底，仅重贴 3 个日志 hunk，重新入仓钉
新 md5；trainer `8696c4ef` 不动。改完策略组复审 diff（应只剩日志
hunk）后即销此 FAIL。

## 4. 等价性包络腿定义（按确认件 526515e9 (c) 钉死）

- U1/U2（定包络）＝**生产构建**：trainer `2c8262fb`＋rollout
  `f91b13c3`；P1＝**修正后仪表化构建**：trainer `8696c4ef`＋新
  rollout md5。同冻结态、同 argv、同种子流、普通钟。
- 已入仓的 `gate_run.sh`/`gate_pin_time.py` 是已撤的钉时钟方案
  工件（两腿都跑 code_instr 分叉版＋钉钟），**不作验收跑器**；
  若在跑的 U1 是分叉版 --game-log 0，该腿不算包络腿，须以生产
  构建重跑。验收证据须逐腿写明构建 md5 对。
- cargo 的 round(…,2)：字段自检时以同局 sidecar 口径对读，若
  碳量原生 2 位小数则精确相合，自检数字说话；不合则去 round 重贴。

验收状态：trainer 过、rollout 待修复复审、包络＋字段自检未到——
**仪表化尚未验收**，du05 与主臂边界生效点不变，均等本件销 FAIL
＋包络证据齐。
