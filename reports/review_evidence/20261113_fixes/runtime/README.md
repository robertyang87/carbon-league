# Runtime 第三轮复核证据

被审查代码：`14d53320b20d58bdb82676f7fa866b8e11a55062`。
修复前比较基线：`a8eaff21`。覆盖 F03、S5b 和 F09 的运行时部分。

本目录只收录复现脚本和已观察的 JSON/JSONL 结果，不含生成策略包、权重、旧版完整源码或 Python 缓存。
脚本整理后仅做静态语法、路径和 JSON 校验，没有为整理材料重跑计时或评测。

## 依赖和配置

- 复核环境为 Linux、Python 3.12、NumPy 1.26.4；需要 Git，不需要 PyTorch、GPU、线上账号或网络。
- 仓库需检出 `code/`（含 `env_src`）及已跟踪的 `code/exp/cnet2_search_final_packed.py`。
- `CARBON_REPO` 指定仓库根目录；不设时从脚本目录逐层向上查找 `code/fastsim.py`。
- `CARBON_REVIEW_OUT` 指定所有临时产物和重跑结果的目录；默认 `/tmp/carbon-rereview-3/runtime`。
- `CARBON_REVIEW_BEFORE` 可覆盖采样回归的比较提交，默认 `a8eaff21`；本地 Git 必须已有该提交对象，脚本不自动联网补历史。
- BLAS 线程固定为 1。业务源码只读，当前 packer 生成的包、提取权重和考试日志只写入输出目录。

`common.py` 由上轮已入库的 `reports/review_evidence/20261112_fixes/runtime/common.py` 复制并调整默认输出路径；本目录包含完整副本，运行不依赖上轮证据目录。

## 运行顺序

在本目录执行，`python` 应为满足上述版本的解释器：

```bash
export CARBON_REPO=/path/to/carbon-league
export CARBON_REVIEW_OUT=/tmp/carbon-rereview-3/runtime
python prepare.py
python cache_checks.py
python sampling_before_after.py
python gates.py
python timing_short.py
python extra_gates.py
python first_error_order.py
# 同一规则故障包通过 bigeval 的对照（预期退出 3）：
python gates.py --worker rule_invalid 970500 2
```

各脚本的作用：

| 脚本 | 验证内容 |
|---|---|
| `prepare.py` | 从历史包解析权重字面量，再用当前 packer 生成合法包、坏 teacher 包及两个对照策略 |
| `cache_checks.py` | 原坐标 TTA 缓存恢复；两个正常 Feat 输入的货量签名碰撞 |
| `sampling_before_after.py` | 同权重、同观测、同采样配置下，对比修复前后 guard 行为 |
| `gates.py` | 每局两步，检查合法包、根推理错误和 2 成功+1 外层异常的退出码与持久化摘要 |
| `timing_short.py` | 验证计时工具对合法、根异常和分类规则异常的错误门，另记录规则包真实统计 |
| `extra_gates.py` | 0/1 有效局、单局根错误、EVO_SEEDS 数量与 n 不一致的边界摘要 |
| `first_error_order.py` | 使最低 seed 故意后完成，确认外层 first_error 按最低 seed 选择 |

`timing_short.py` **只在内存 AST 中将计时工具唯一的 `range(299)` 改为 `range(2)`**，并断言恰好修改一处；其余真实打包、推理、统计和退出逻辑不变。它只测试错误门，输出中的毫秒数不得作为性能认证。不会修改仓库中的计时源码。

采样比较通过 `git show` 在内存读取旧版 `cnet2_search.py`，不保存旧版完整源码。该项复用当前其余模块，验证范围是第 0 步 guard 回归，不是两版本完整对局强度比较。

采集脚本自身退出 0 仅表示证据采集完成，应阅读记录中的被测子进程 `exit_code`。部分探针故意注入错误，最后一条单独运行的 worker 命令预期非零。

## 已观察结果

`observed/` 保留实际复核输出，绝对路径是采集时的语境；重跑使用上述环境变量。

- `cache_checks.jsonl`：TTA 原坐标恢复差为 0；原货量碰撞变为 `same_sig=false`，错误编码差为 0。
- `sampling_before_after.jsonl`：修复前 `KeyError('T_now')`；修复后内部错误为 0，恢复双基地招募采集员。
- `gates.jsonl`：合法包退出 0；根异常退出 3 且 FAIL=true；2 成功+1 外层异常退出 2 且 FAIL=true。
- `extra_gates.jsonl`：0/1 有效局都有 FAIL 记录并退出 2，但没有 first_error/内部错误计数；EVO_SEEDS 仅两个而 n=3 时仍出现 FAIL=false、退出 2。
- `timing_short.jsonl`：合法包退出 0，根异常退出 1；规则故障包产生 tg_errors=2，却被计时工具报告 errors=0/root_errors=0 并退出 0。
- `rule_invalid_bigeval.json`：同一规则故障包在 bigeval 中正确累计内部错误、FAIL=true 并退出 3。
- `first_error_order.json`：完成顺序 970501、970503、970502、970500，first_error 正确来自最低 seed 970500。
- `fixture_provenance.json`：历史权重所在文件的 SHA-256 和原始 metadata。

## 证据边界

权重来自已入仓历史包 `code/exp/cnet2_search_final_packed.py` 的主网：C=128、NB=8、D=128，2,946,610 参数，epoch 13，metadata 数据集 `../data/cnet2_ds_v2`，按行 int8 量化后解量化。提取时只解析 `_Q1/_M1/_B1` 字面量，不执行历史包。它不是最新 Echo2/n1r 线上权重，测试提交包均由复核时当前代码重新生成。

坏 teacher、TG_MAX 类型错误及外层异常均为受控故障注入，不证明线上策略存在相同配置错误。采样配置 OPEN_T=3/OPEN_SEED=8 用来复现确定分支，不代表线上贪心版本。缓存测试检查输入契约，不作策略强度推断。

本轮没有重复 299 步计时，也没有施加平台 1.5 CPU/512 MiB cgroup；全部耗时输出仅供确认探针完成。
