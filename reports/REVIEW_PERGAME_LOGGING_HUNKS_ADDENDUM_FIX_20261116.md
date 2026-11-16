# 逐局仪表化审验补件：rollout 修复复审 PASS（策略组，2026-11-16）

对象：`code/instr_20261116/rl2_ppo_rollout.py` 修复版，commit 326cc299，
md5 `97f4fde5`（blob 15481611）。审验方直读与钉版基线 rollout
`f91b13c3` 的全量 diff 并逐点核对原文。

## 复审结论：源码腿 PASS，REVIEW_PERGAME_LOGGING_HUNKS 的 FAIL 销

- diff 恰 3 个 hunk，全在日志路径，与原审验认可的日志 hunk 逐字同
  （game_log 旗标读取／步后 pre/post 工人集差死亡计数／终局树·货物·
  terminal_kind＋SELF 镜像对位互换）。
- 被回退的两处防护已恢复且逐字在位：`_snap_temp()` 定义（:42）、
  import 期 fail-fast（:57）、两处调用点（:135/:206）；
  `role=task.get("primary_role", "learner")` 透传（:192）及 V04
  注释块完整。
- trainer 维持 `8696c4ef`（已 PASS，不动）。

## 仪表化验收剩余项（仅剩证据腿）

1. 等价性包络：U1/U2＝生产构建（trainer `2c8262fb`＋rollout
   `f91b13c3`）、P1＝仪表化构建（trainer `8696c4ef`＋rollout
   `97f4fde5`），同冻结态、同 argv、普通钟；P1 每项落 U1/U2 跨度内
   ＋完整性独立验＋pace ±3%。证据逐腿写明构建 md5 对。
2. 字段自检：含 cargo round 2 位与 sidecar 同局对读。

两项齐并经策略组核对后出验收件；验收件出前 du05 不开跑、主臂边界
不折入。
