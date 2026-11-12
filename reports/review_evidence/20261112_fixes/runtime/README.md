# Runtime 修复复核证据

复核对象：`c64bbeda8a00b2f4d79ace589f201bab3cca5163`；修复前基线：`5f338eb`。
覆盖 R05（内部异常失败门）、R11（纯网计时）、R12（TTA 缓存及采样回归）。
这里只保存复现脚本和文本结果，不保存生成的提交包、权重、缓存或旧版完整源码。

## 依赖和路径

- Python 3.12、NumPy 1.26.4、Git。此次复现不需要 PyTorch、网络、线上账号或 GPU。
- 完整检出本仓 `code/`（含 `env_src` 以及已跟踪的 `code/exp/cnet2_search_final_packed.py`）。
- `CARBON_REPO` 指向仓库根目录；不设时从本脚本目录向上寻找 `code/fastsim.py`。
- `CARBON_REVIEW_OUT` 指向临时产物目录，默认 `/tmp/carbon-fix-review/runtime`。
- 改前/改后比较需要本地存在 Git 对象 `5f338eb`；浅克隆缺失时该项会明确失败。可用 `CARBON_REVIEW_BEFORE` 指定已有的修复前提交。
- 所有 BLAS 线程变量固定为 1。不会修改业务源码、运行训练或调用线上服务；生成产物和考试结果写入临时目录。

在本目录执行，`python` 应为满足上述版本的解释器：

```bash
export CARBON_REPO=/path/to/carbon-league
export CARBON_REVIEW_OUT=/tmp/carbon-fix-review/runtime
python prepare.py
python cache_checks.py
python sampling_before_after.py
python gates.py
# 可选，包含两局完整 299 步；不要为整理文档重复运行：
python timing_checks.py
```

`prepare.py` 从现有历史包提取 `_Q1/_M1/_B1` 字面量，以当前 packer 重新打包；不会执行历史包。
它生成 `baseline.npz`、合法包、坏 teacher=999 故障包、两个对照策略以及 `fixture_provenance.json`。
后者记录历史包 SHA-256 和权重 metadata。故障包不用于真实比赛。
`gates.py` 每局只执行两步，分别检验合法、根推理错误和外层异常，记录被测子进程的退出码。
脚本自身成功结束仅表示证据采集完成，判定须看 JSON 中的 `exit_code` 与字段。

## 已观察结果

`observed/` 内文件是实际复核时的原始输出；为可移植性整理脚本后只检查语法和路径，未重复执行计时。
原始输出中的绝对路径保留采集时语境，重跑的产物将使用上述环境变量。

- `gates.out`：合法包退出 0；根异常包退出 3，持久化 `FAIL=true`、4 次根异常；2 成功+1 外层异常退出 2，但持久化 `FAIL=false`。
- `timing_checks.out`：合法纯网 299 步成功，`searched=0`、搜索分位数 null，p99 38 ms/max 86 ms；故障包显示 299 次根错误但工具仍退出 0。
- `sampling_before_after.out`：相同权重、观测与采样配置，修复前 guard 正常招募采集员；修复后 `KeyError('T_now')` 导致 guard 回退。
- `cache_observations.out`：两个由 `Feat.observe` 生成的输入仅 `tok_cargo/hist_cargo` 不同，签名相同，返回编码与正确编码最大差 0.061544。
- TTA 原始缓存偏差 0.587379 修复后为 0 的实测记录见 `tta_restore.json`；可由 `cache_checks.py` 首项重现。

## 证据局限

权重来自已入仓的 `code/exp/cnet2_search_final_packed.py` 主网：C=128、NB=8、D=128，2,946,610 参数，epoch 13，metadata 数据集 `../data/cnet2_ds_v2`，已按行 int8 量化并解量化。
这是历史 b128 固定测试样本，**不是**最新 Echo2/n1r 线上权重。合法/故障提交包均以复核时当前代码重新生成。

计时只验证该样本在本容器正常执行，没有施加线上 1.5 CPU/512 MiB cgroup，不是线上预算认证。
坏 teacher 与外层异常是受控故障注入，不证明真实线上包存在这些配置错误。
OPEN_T=3/OPEN_SEED=8 的一调用比较证明采样分支回归，不证明线上贪心配置受影响。
货量对照证明缓存签名契约不完整，尚未证明该碰撞会在当前标准线上调用序列中自然发生。
