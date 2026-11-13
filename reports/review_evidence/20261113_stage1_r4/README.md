# S1 v1.3 / R4 确认复核证据

对应 [确认报告](../../REVIEW_STAGE1_V13_R4_20261113.md)。业务源码冻结 `8fc10d075e28599359e3f10dbbca292642478537`，比较 v1.2 `4538fddf`；上一轮 [R3 证据](../20261113_stage1_r3/README.md) 保留原失败状态。

本目录3个便携脚本及3个结果，关闭 R21 训练目标软场景聚合、P3-a 指定数量/身份检查、P3-b 无清单 du 诊断入口，并验证原有正负路径保持。`source-manifest.json` 绑定源文件和未变路径，`evidence-manifest.json` 绑定证据文件。

## 复跑

本轮环境：Python3.12.14、NumPy1.26.4、Torch2.5.1+cpu、SciPy1.17.0，解释器 `/workspace/carbon-review-venv/bin/python`。保留仓库 Git 历史及原版游戏环境依赖。脚本无需真实训练权重；不会训练模型、运行云端任务、部署或修改业务源码。

```bash
export CARBON_REPO=/path/to/checkout-at-8fc10d07
export CARBON_REVIEW_OUT=/tmp/carbon-s1-r4
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S1R4_EVIDENCE=/path/to/reports/review_evidence/20261113_stage1_r4
python "$S1R4_EVIDENCE/method/probe_method_r4.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/method"
python "$S1R4_EVIDENCE/pilot/probe_pilot_r4.py"
python "$S1R4_EVIDENCE/acceptance/probe_acceptance_r4.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/acceptance"
```

被测 checkout 与新报告所在 checkout 可以不同。请明确指定冻结源码；未来版本重新运行得到的是新版本结果。临时路径、时间、run_id 等字段会变化，不要求跨机器整体 JSON 字节一致。

## 结果解释

| 文件 | 实际执行与控制输入 | 结论 |
|---|---|---|
| method/results.json | 原始 main、sampler、场景/全臂锚点汇总、聚类与功效函数；教师/终局为确定性表，回放只提供长度 | 长短局混合例正式FAIL，裁决soft字段与game独立派生值逐值一致；锚点、温度、正负控制及功效回归保持 |
| pilot/results.compact.json | 真实main/采样/工件/I/O；昂贵workers替为明确记录 | 28项观测符合预期；无ds q/du正常诊断、pilot诊断级，合格A→B正路PASS，旧四拒绝路径保持 |
| acceptance/results.json | 原验收函数；实际drive_agent配no-op policy；实际roots配真实FastCarbon/Feat及合成无动作回放 | 63唯一compare与50唯一teacher正路通过，缩减/重复/非有限等拒绝；实际生成结构与JSON身份哈希兼容 |

这是软件契约确认，**不是正式教师或模型战力实验**。gate探针的PASS表示控制输入满足代码条件；FAIL/INDETERMINATE表示对应反例被正确处理。不能把探针的exit0或正向PASS写成实际教师收益已通过。

pilot精简结果明确标derived，保留全部28项检查、关键工件/运行身份及原始完整结果SHA256；脚本能重建完整和精简输出。未归档冗余stdout、临时权重、复制源码、pyc或完整重复日志。

作者Studio正典权重/回放的验收声明单独记账。本轮的合成fixture与实际函数执行没有冒充那组制品；也没有因未变源码而无差别重跑全部旧测试。关账范围以确认报告为准。
