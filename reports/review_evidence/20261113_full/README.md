# 第五轮完整复核证据

主报告：[REVIEW_FULL_20261113.md](../../REVIEW_FULL_20261113.md)。记录时间：2026-11-13，北京时间。

本目录保存短定向探针、当次输出及分域说明，不保存生成的策略包、解量化权重、完整第三方页面或重复的生产回放。探针的成功退出可能表示“成功复现缺陷”，**不能把断言通过数当作全部系统健康**。

## 版本与范围

- 主基线：`98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c`。包含4b44362d修复，主域输出绑定该版本。
- 第一增量：`62e0dee16419725ece6f18a22f095c83bb6fcd0f`。新增网络头、独立smoke、自博弈producer；文件名含latest的第一批证据绑定此版本。
- 最终代码截止：`57557a83a73cd4d6b26b56ebebb4c0df8e13a155`。仅再增加完整head trainer与Studio probe；用575命名的增量说明/输出覆盖。
- [source-manifest.json](source-manifest.json) 固定审查文件身份与增量范围；[evidence-manifest.json](evidence-manifest.json) 固定本目录其他证据的SHA256。
- 各notes中的绝对scratch路径标识当次环境，不是交付文件下载地址；请使用下表及相邻相对链接。旧notes的“尚未新增/实现”等判断只适用于其顶部固定SHA，新增实现以主报告第5节与latest/575说明为准。

## 证据导航

| 域 | 说明 | 主要输出 |
|---|---|---|
| 模拟器/搜索 | [simulator/notes.md](simulator/notes.md) | [原版终局](simulator/results.json)、[搜索边界](simulator/search-boundaries.json)、[模拟器契约](simulator/simulator-contracts.json) |
| 数据 | [data/notes.md](data/notes.md)、[新增producer](data/latest-notes.md)、[Studio增量](data/latest575-notes.md) | [构建/发布](data/data-results.json)、[旧生产器标签](data/label-results.json)、[新增producer](data/latest-selfplay-results.json)、[575源码身份](data/latest575-sources.json) |
| 网络/PPO/AWR | [training/notes.md](training/notes.md)、[新头增量](training/latest-notes.md) | [网络接口](training/network-ppo-results.json)、[AWR](training/awr-results.json)、[新头兼容](training/latest-heads-results.json)、[完整训练出口](training/latest-full-heads-results.json)、[新模型接口重测](training/latest-parity/network-ppo-results.json) |
| 打包/认证 | [runtime/runtime-notes.md](runtime/runtime-notes.md)、[复跑说明](runtime/README.md) | [打包](runtime/pack-export-results.json)、[认证反例](runtime/certifier-probes.json)、[真实包短测](runtime/real-pack-certifier-results.json)、[末次推进计时](independent/certifier-completion.json) |
| 考试/自动化 | [automation/notes.md](automation/notes.md) | [helper/客户端反例](automation/results.json)、[shell语法](automation/syntax.json) |
| 训练方法 | [method/notes.md](method/notes.md)、[新loss](method/latest-notes.md)、[完整训练校准](method/latest575-notes.md) | [目标/判据](method/evidence.json)、[新头loss](method/latest-heads-evidence.json)、[校准误报/漏报](method/latest575-calibration-evidence.json) |
| 当日平台规则 | [来源记录](sources/verified-public-sources.json) | 北京时间11:23读回；只归档hash/时间/验证方式，原始页面未复制进仓库 |

## 安全、可迁移的复现方式

使用独立临时checkout固定到上述SHA，不在正在训练的目录切换版本。环境：Python3.12、NumPy1.26.4、Torch2.5.1+cpu；原版环境对照还需仓库原环境的Python依赖。本次使用CPU环境；没有GPU训练、远端命令、生产缓存重建、天梯提交或云资源操作。

将已入库的本证据目录放在固定checkout之外，避免切换旧SHA时文件消失。大多数probe使用以下变量；method probe使用显式参数：

```sh
export CARBON_REPO=/path/to/fixed-checkout
export CARBON_REVIEW_OUT=/tmp/carbon-review-replay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BC_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
python /path/to/evidence/data/recheck_data.py
python /path/to/evidence/data/recheck_labels.py
python /path/to/evidence/training/check_network_ppo.py
python /path/to/evidence/training/check_awr_pipeline.py
python /path/to/evidence/simulator/eval_terminal_probe.py
python /path/to/evidence/simulator/search_sync_probe.py
python /path/to/evidence/simulator/simulator_contract_probe.py
python /path/to/evidence/automation/probe.py
python /path/to/evidence/method/check_method.py --repo "$CARBON_REPO" --out "$CARBON_REVIEW_OUT"
```

runtime按其README顺序运行：pack_export会准备real_pack_certifier所需的解量化临时权重；不要单独调用第三步后把缺fixture当业务失败。`independent/check_certifier_completion.py`使用2秒测试阈值，真实短sleep验证整局计时控制流。它不是平台配额性能基准。

新增producer/new heads探针需要62e或明确兼容的最终575源码；`method/check_latest_heads.py --repo ... --out ... --ref ...`从指定Git版本读取。模拟器probe为新运行创建临时子目录；部分其他probe在指定输出目录写JSON。automation/probe.py默认写自身旁的results.json，故建议复制证据目录后复跑，不覆盖已归档观测。原JSON中包路径、临时目录及时间是当次事实，新的结果应保存为新证据。

所有策略异常、损坏文件、发布失败、NaN、重复seed、POST重放均在本地替身/临时目录中构造。明确的真实代码路径、替换范围和未验证条件见各notes。Torch小张量backward用于检验梯度/目标，不等于真实训练；没有执行生产optimizer更新。没有创建平台同等的1.5CPU/512MiB cgroup；真实n1r/Echo3仅作两步冒烟，不能作为强度或全局资源认证。
