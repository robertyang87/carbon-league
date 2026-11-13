# S1 v1.2 / R3 独立确认复核证据

对应 [R3 报告](../../REVIEW_STAGE1_V12_R3_20261113.md)。被测源码固定于 `4538fddf37bcc0e8ac608e5aa9b5f9093eca0b28`，对照 v1.1 `7b025ae5` 和 [R2 证据](../20261113_stage1_r2/README.md)。本目录4个便携脚本、4个结果及来源清单，无业务修改。

`source-manifest.json` 记录实际源码 SHA256/MD5、与 v1.1 是否逐字相同及提交前检查的 main 源码一致性；`evidence-manifest.json` 记录本目录其余文件身份。脚本中的基线版本固定用于重建旧格式/旧边界，不是部署指令。

## 运行方式

本轮 Python3.12.14、NumPy1.26.4、Torch2.5.1+cpu、SciPy1.17.0，解释器 `/workspace/carbon-review-venv/bin/python`。保留仓内原版游戏环境及其依赖、Git 中 `7b025ae5`/`4538fddf` 对象。无需正典模型或真实回放下载，不运行训练、云端任务、完整收益实验。

```bash
export CARBON_REPO=/path/to/checkout-at-4538fddf
export CARBON_REVIEW_OUT=/tmp/carbon-s1-r3
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S1R3_EVIDENCE=/path/to/reports/review_evidence/20261113_stage1_r3
python "$S1R3_EVIDENCE/terminal/teacher_terminal_probe.py"
python "$S1R3_EVIDENCE/method/probe_method_r3.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/method"
python "$S1R3_EVIDENCE/pilot/probe_pilot_r3.py"
python "$S1R3_EVIDENCE/acceptance/probe_acceptance_r3.py" --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT/acceptance"
```

被测源码 checkout 可以与新报告所在 checkout 不同；明确使用冻结源码，避免将未来修复结果归入本次证据。输出目录只用于临时工件。exit0 表示断言/观测按预期完成，部分断言确认残余缺陷存在，不能据此称全链验收通过。

## 结果与控制范围

| 结果 | 使用的真实路径与替代输入 | 结论范围 |
|---|---|---|
| terminal/results.json | 实际 teacher/guard/Feat/FastCarbon/原版规则；固定网络推理 | 5个299/300到时输入入口拒绝；16个合法终局边界、两臂排序及分支隔离保持 |
| method/results.json | AST原样执行 `_q_scene`、`_anchor_task`、main、list_replays/sampler/统计；明确的教师/终局控制表，回放只提供长度 | 原R21锚点例及R23、功效补正通过；不等每局根数下软场景收益仍root均值入裁决 |
| pilot/results.compact.json | 真实main/采样/工件资格与I/O；昂贵worker替换为确定性记录 | 28项观测，四个原缺口关闭、正路PASS、旧v1.1工件拒绝；du缺ds诊断异常P3 |
| acceptance/results.json | 原验收函数AST、真实drive_agent；合成fixtures/no-op policy | 15项输入路径，旧finite/重复/单条反例拒绝且正路能过；精确数量及teacher根唯一性仍未锁 |

method的混合长度语料不是实际完整比赛：25个300步长度fixture和75个90步长度fixture仅用于真实采样器产生1/3根的分布。终局表刻意保持锚点正收益，隔离主软场景评分聚合错误。不能把该反例写成线上教师真实收益为负。

pilot精简结果为derived，内嵌完整原始结果hash与精简说明；探针可复建完整与精简输出。运行路径、run_id、对象id、时间字段会变化，不要求跨机器整体JSON字节一致。归档结果只证明注明的执行，不替代作者Studio正典权重验收。

未变的 search/guided/np 等由源码身份核对保持先前验证状态，本轮未重复完整runtime探针。测试中临时复制的旧实现、权重、日志、pyc及完整冗余stdout不入仓。
