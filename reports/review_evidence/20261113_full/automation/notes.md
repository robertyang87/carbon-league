# 第 5 轮：自动化、CI 与 cloud 队列审查

基线：`98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c`。证据：`probe.py`、`results.json`、`syntax.json`。仅从真实源码抽取 shell helper/状态分支在临时目录执行，bigeval 使用假 executor/假对局结果，ladder HTTP 使用假 opener；没有执行原始 cloud 队列、真实对局/训练、workflow、云 API 或平台提交。134 个 `code/cloud/*.sh` 经 `bash -n` 均通过，语法通过不代表业务正确。

## 已通过与边界

- N02 原始 `EVO_SEEDS` 数量与 N 不匹配问题已修：h14d/h15 固定列表分别 57/30 个，且各自唯一；h14 动态列表冻结一次后计数。抽取的真实 `collect_group` 对失败子进程返回非零，不能再说仍用裸 `wait` 吞失败。
- `dgx_full.sh` 新无确认前缀在 `cd`、等待、复制、训练之前 exit 1；确认前缀可通过。这里只证明 opt-in 护栏有效，未验证真实机器 `.d0dd809` 的内容或存在性。
- 新 `ladder-submit.yml` 仅 `workflow_dispatch`，权限 `contents: read`；当前 Echo3 包大小 `4,576,790`、MD5 `7b7620499411a088cf4dc17c0e929bd7` 与 workflow 固定值相符。没有实际运行 workflow、登录或提交，也未核验线上状态。不要把“没有触发本轮提交”写成“历史 Echo3 未提交”。

## A01 — P2：续跑漏掉失败 seed，且只验增量数量，不能证明完整考试健康

位置：`code/cloud/q_h14.sh:6–9,20–47,68–97`；相同健康 helper 在 `q_h14d.sh:14–43` / `q_h15.sh:18–47`。

`miss()` 将 live 中任何有 seed 的行都当成已完成，不检查 `err`、`serr`、有效 margin、actor/对手/配置身份。真实 bigeval 会先把错误行写进 live。因此旧记录中 seed 994000 失败时，恢复列表只剩 99 个且不重试这个失败 seed；这 99 局的新摘要 `FAIL=false,n=99` 可被现有 helper 接受，原期望 100 个健康 seed 没有被全量核对。探针只执行 helper，没有执行整个队列；若其余组正常，源码后续即可走到 SX_DONE。

补充负例：同 tag、`n=2,FAIL=false`，但 actor/对手错误且 games 是两个相同 seed 的构造摘要，helper 也 exit 0；没有摘要且没有输出时仅警告后 exit 0。后者是健康 helper 的 fail-open 边界，**未证明正常当前 bigeval 会在 exit 0 时遗漏摘要**。

空列表应分情况：同配置的全部预期 seed 都健康且唯一，才是 `ALREADY_COMPLETE`；输入缺失、错误、身份未核或缺 seed 都应 fail/重试，不能一律成功。目前 100 条健康记录也返回空列表并被硬编码 `SX_RC=1` 拒绝，属于保守阻塞；不是新的假成功。当前 live 格式不含完整配置 hash，仅有 seed 不能承担身份核验。应建立带 actor/对手/权重/config/规则 hash 的 run manifest，并对累计结果验全集、去重和错误；不能只把空列表分支直接改为 success。

证据键：`failed_old_seed_omitted_then_resumed_batch_accepted`、`complete_healthy_missing_list`、`wrong_identity_duplicate_seed_summary_accepted`、`no_summary_no_output_accepted`。

## A02 — P2：失败后旧 DONE 仍存在，下游仅看文件存在会跨轮放行

位置：`q_h14.sh:92–97`、`q_h14d.sh:4,81–86`、`q_h15.sh:95–117`；消费者 `q_ex1_h15.sh:2`。

新逻辑不会在本轮失败时写新的 DONE，但没有在启动/失败时撤销旧文件，也没有 run/config 身份。探针先放一个旧 `sx.done`，再执行真实失败分支：exit 1、新 `sx.fail` 出现，但旧 `sx.done` 仍在。h14d 只等 `s50.done`，ex1_h15 只等 cb/fkcar.done；因此旧完成文件可被下游认作本次成功。应以 run_id + manifest 的原子状态发布与消费为准，同时处理旧标记及失败停止。

另外未修的历史 `q_h14b.sh:5–6,12` / `q_h14c.sh:9,19` 仍会按进程消失或裸 wait 写同名标记。属于历史队列剩余风险，没有证据它们在当前云机运行；不能说三份修复覆盖了所有 marker 生产者。

证据键：`stale_done_after_failure`、`per_pid_failure_propagates`。

## A03 — P2：bigeval 接受重复显式 seed，独立样本数可虚增

位置：`code/bigeval.py:12–25,84–99`。

当前仅检查列表长度与 N 一致，未检查唯一性。真实 `main()` 配合假 executor 输入 `EVO_SEEDS=17,17,N=2`，返回 exit 0、`FAIL=false,n=2,se=0`，games 中实际只有 1 个唯一 seed。文件头明确将每 seed 定义为独立样本；重复项会把同种子/同席位重复计入统计和成功局数。应在 worker 创建前拒绝重复，或显式将有重复设计与独立 seed 统计分开。

**限定：** 本次 h14d/h15 的 57/30 列表均无重复；不声称当前固定列表已污染。此处是假对局证据，只验证任务构建与统计契约，未跑游戏。

证据键：`bigeval_duplicate_explicit_seeds`。

## A04 — P2：提交所用客户端会自动重放结果未知的 POST

位置：`code/ladder.py:66–81,99–102,174–178`；新入口 `.github/workflows/ladder-submit.yml:33`。

`request()` 的外层只给非 GET 一次机会，但 `_open_retry()` 内层对所有方法都最多尝试 6 次。服务器已经接收提交、响应头返回前连接断开时，客户端无法判断是否接受，却再次发送同一 POST。替身模拟第一次“已接收但断开”，观察到两次 POST。可能造成重复尝试、额外提交额度消耗，或第一次已成功但重试被拒后 CI 显示失败；具体后果受服务端幂等/同名约束影响。

建议突变接口不自动重放结果未知请求：服务端提供 idempotency key 时使用；否则先按回执/策略身份核实原请求状态，再由显式恢复流程决定。此为客户端已存在问题被新提交 workflow 接入，**不是已经观察到 Echo3 重复提交**，本轮零网络请求。

证据键：`ladder_post_ambiguous_retry`、`ladder_submit_static`。

## A05 — P3：cloud-assistant 明文 command 在 runner 提前做 shell 展开

位置：`.github/workflows/cloud-assistant.yml:38–41`，历史 workflow。

`${{ inputs.command }}` 直接插进 `run` 的双引号源码。输入 `echo "$REVIEW_REMOTE_ONLY"` 时，替身 Python 收到 `echo LOCAL_RUNNER_VALUE`，远端变量语义已被 runner 提前解释。命令替换/嵌入引号也会在 runner shell 层产生副作用。当前 b64 路径可绕开普通命令文本的问题；建议所有动态输入通过 env 传值后以 `"$VAR"` 作为实参，并验证 instance/region。未触发云 API，未读取凭证；不扩大为已被攻击。

证据键：`cloud_command_local_expansion`。

## A06 — P3：DGX opt-in 是运行前提醒，确认后的失败仍可生成 FULL_DONE

位置：`code/cloud/dgx_full.sh:25–33`。

确认后的复制和四次训练均未检查退出码，末尾无条件 `FULL_DONE`。提取真实尾段放进临时目录，缺少 `.d0dd809`，且 env 替身让每次“训练命令”直接返回 7（没有启动训练程序），仍 exit 0、写 FULL_DONE，原 trainer 保持原内容。这是启用历史路径时的完成语义错误，**不否定本轮新增无确认即拒绝的护栏**。建议复制来源 hash/存在性与成功状态校验、训练每步失败即停止、完成记录含实际源身份；未确认路径按现状仍安全停止。

证据键：`dgx_unconfirmed_prefix`、`dgx_confirmed_prefix_only`、`dgx_confirmed_tail_failures_still_done`。

## 复现

在仓库根目录：`CARBON_REPO="$PWD" python3 <证据目录>/automation/probe.py`。探针输出保存在自身目录的 `results.json`。除证据输出外仅使用临时目录；不对业务源码作修改。
