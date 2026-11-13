# S1 修复第二轮复核证据

对应 [R2 报告](../../REVIEW_STAGE1_FIXES_R2_20261113.md)。冻结业务源码 `7b025ae58f83250265e75a5cccf274f63ca7d0a3`；原反例见 [首轮证据](../20261113_stage1/README.md)。本目录6个便携脚本及结果；源码身份为 `source-manifest.json`，证据身份为 `evidence-manifest.json`。

所有业务源码只读，无训练、远端任务、比赛提交或部署。网络 fixture 是确定性未训练小模型，或为隔离控制流而固定的推理输出。gate 探针替换昂贵 workers/终局回报表，实际执行源码中的主流程、候选锚点汇总、统计、工件与I/O。因此正式 PASS/FAIL 的反例是程序判定证据，不是真实教师收益测量。

## 复跑

本轮解释器 `/workspace/carbon-review-venv/bin/python`，Python3.12、NumPy1.26.4、Torch2.5.1+cpu、SciPy已安装，BLAS单线程。需要仓内原版环境 `code/env_src/carbon_challenge-master` 及对应依赖。保留 Git 的 `bc1a84d4`、`a5c63648`、`7b025ae5` 对象，供旧实现和源码身份对照。无需正典训练权重。

`CARBON_REPO` 指向冻结版 checkout，`S1R2_EVIDENCE` 指向本目录所在的新报告版本。两者可以不同：不要为了取新脚本而把被测源码切到未来修复版。

```bash
export CARBON_REPO=/path/to/checkout-at-7b025ae5
export CARBON_REVIEW_OUT=/tmp/carbon-s1-r2
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S1R2_EVIDENCE=/path/to/reports/review_evidence/20261113_stage1_r2
python "$S1R2_EVIDENCE/terminal/teacher_terminal_probe.py"
python "$S1R2_EVIDENCE/runtime/probe_runtime_r2.py"
python "$S1R2_EVIDENCE/candidates/check_candidate_priors.py"
python "$S1R2_EVIDENCE/method/probe_method_r2.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/method"
python "$S1R2_EVIDENCE/inputs/probe_inputs.py"
python "$S1R2_EVIDENCE/docs/probe_docs_r2.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/docs"
```

runtime 探针会先运行冻结版已有的 `reports/review_evidence/20261113_stage1/runtime/guard_probe.py`，复用它的未训练 fixture，再检查新快照入口与实际短续弈。该文件已经入仓，不依赖审查者私有目录。输出中的临时权重、复制的旧实现、日志不应当作正典制品或提交到业务目录。

脚本 exit0 表示预期观测完成；部分断言用于确认缺陷仍存在，不能将其写成“S1 全过”。修复后应保留反例输入并更新正确预期。源路径、run_id、对象id、wall time等因运行不同而变化，跨机器比较语义字段，不要求所有 JSON 的整体 hash 相同。

## 证据索引

| 目录/结果 | 已通过或反例范围 |
|---|---|
| terminal/results.json | S01：16个合法root终局边界通过；另有非终局与已到时输入；两臂排序、双席隔离、灭绝root拒绝。P3到时root仍多推进 |
| runtime/results.json | S04历史快照、gate/guided_match双席短续弈、restore alias、style2的Actor/NP等价；无真实战力结论 |
| candidates/results.json | S05先验误差0、5个最终候选合法；原7臂中的非法后缀剔除 |
| method/results.json | τ1原例FAIL、聚类du原例FAIL、10%配额；R21锚点root/game异号PASS；R23变温分布与训练q对象差异；正负控制例 |
| inputs/results.compact.json | 21项实际CLI/采样/工件观测；R22诊断pilot、训练污染pilot、零方差旧sidecar路径；旧隔离/重复/缺工件拒绝与I/O通过 |
| docs/results.json | packs及NP不变；原空/同错误/NaN拒绝；guard-on NaN仍PASS等验收断言缺口；fixture归档边界 |

inputs 的精简 JSON 明确标为 derived，含完整原始结果 SHA256，删除冗余输出而保留全部检查、资格/身份及判定。`probe_inputs.py` 每次生成完整与精简结果，可复建控制流证据。没有将作者 Studio 的正典 heads/replay 验收冒充本轮独立复跑。
