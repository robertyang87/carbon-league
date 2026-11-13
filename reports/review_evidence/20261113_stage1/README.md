# S1 独立复核证据

对应 [复核报告](../../REVIEW_STAGE1_TEACHER_GATES_20261113.md)，源码冻结 `a5c63648e037e342b154949c1b4eea6d11eef139`，比较基线 `bc1a84d4`。8 个脚本及结果只验证列明的接口、控制流、数学与规则边界，不代表真实教师收益或比赛战力测量。

`source-manifest.json` 记录审查源码 SHA256，并核对这些文件在提交前检查的 main `5aa7b544` 上仍相同。`evidence-manifest.json` 为本目录其余文件的 SHA256。结果中的原始绝对路径是运行来源，不是复跑依赖。

## 环境与复跑

本轮 Python 3.12.14、NumPy 1.26.4、Torch 2.5.1+cpu，Torch/BLAS 单线程。需保留仓库 Git 历史（guard/NP 比较从 `git show bc1a84d4:...` 读取旧实现）及原版游戏源码 `code/env_src/carbon_challenge-master` 的运行依赖。完整仓库使用已有依赖环境；本轮解释器为 `/workspace/carbon-review-venv/bin/python`。无需下载真实权重，网络探针构造确定性未训练小模型，没有 optimizer。

请让 `CARBON_REPO` 指向审查 SHA 的独立 checkout；脚本可放在包含本报告的新版本仓库中。源版本与证据脚本分开指定，避免把未来修复后的结果误认为本次证据。

```bash
export CARBON_REPO=/path/to/checkout-at-a5c63648
export CARBON_REVIEW_OUT=/tmp/carbon-s1-review
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S1_EVIDENCE=/path/to/reports/review_evidence/20261113_stage1
python "$S1_EVIDENCE/network/check_want_u_cache.py"
python "$S1_EVIDENCE/runtime/guard_probe.py"
python "$S1_EVIDENCE/runtime/teacher_config_probe.py"
python "$S1_EVIDENCE/simulator/teacher_terminal_probe.py"
python "$S1_EVIDENCE/candidates/check_candidate_priors.py"
python "$S1_EVIDENCE/method/probe_method.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/method"
python "$S1_EVIDENCE/data/recheck_gate_inputs.py"
python "$S1_EVIDENCE/docs/probe_docs.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/docs"
```

`teacher_config_probe.py` 依赖先运行 `guard_probe.py` 生成同一个 tiny 权重。输出目录是临时工件目录，不要指向仓内源码/已归档证据。simulator 每次创建新结果子目录。method/data 探针替换耗时 workers，但执行实际主流程/采样/统计函数；docs 使用 AST 隔离断言测试。

部分脚本的 exit 0 表示**成功复现并写出证据**，其中断言可能要求确认缺陷存在。不能将“8 个脚本运行成功”写成“S1 全部通过”。修复后应调整预期并保留原失败输入复测。

## 文件映射

| 目录 | 结果及用途 |
|---|---|
| network | `results.json`：20 个位置、5 种加载入口、旧三元组/RNG、cache/teacher/权重切换，独立通过 |
| runtime | `guard-probe-results.json`：新旧 guard 等价通过、fresh root 历史反例；`teacher-config-results.json`：style 的输入语义；`acceptance-surrogate.json`：原 A3 synth 在替代 fixture 上的结果 |
| simulator | `results.json`：13 个终局/H 边界、最后展开灭绝反例、错误两臂排序、双席位分支隔离 |
| candidates | `results.json`：guard 后 7 个候选的先验/合法掩码，2 个非法后缀 |
| method | `evidence.json`：负软收益仍 PASS、成组反例、10% 层配额、功效与选优偏差数学例 |
| data | `results.compact.json`：15 项 CLI/采样观测，含实际 terminal anchor 无 learned leaf 验证 |
| docs | `results.json`：packs 同步、来源检查、空/同错误/NaN acceptance 漏检 |

data 精简结果明确标记 derived，去掉原始输出里重复的完整 JSON，将长 rows 改成数量/唯一性/root sequence 和 set hash；记录原始完整结果 SHA256 `ac90c922050dbf2711399aaf538ec7aefe0c12f6f1d1d68434f4f59c0bed2505`。`recheck_gate_inputs.py` 可重新生成完整观测，临时路径及 wall time 等运行字段可能变化，不能要求跨机器整体 JSON hash 恒定。

作者的正典 `heads_v2.npz`、20 个真实 replay fixture、Studio 原始输出未包含在审查 SHA 的可重建材料内。本目录的小模型和替代 synth 测试明确不是作者报告的正典验收复跑。临时权重、复制旧源码、pyc、重复 stdout 未入库；无训练数据、密钥或线上部署操作。
