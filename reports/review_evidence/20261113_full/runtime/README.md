# Runtime 第5轮证据复现

审查SHA：`98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c`。摘要见 `runtime-notes.md`。

依赖 Python3.12、NumPy1.26.4（已用 `/workspace/carbon-review-venv/bin/python`）；不需要GPU、torch或网络。脚本不修改仓库，仅在临时目录写桩、解量化fixture、包和JSON。设置 `CARBON_REPO` 为checkout；`CARBON_REVIEW_OUT` 默认为 `/tmp/carbon-review-round5/runtime`。未设repo时向cwd/脚本祖先找code/certify_pack.py。

运行顺序：

```sh
export CARBON_REPO=/path/to/carbon-league
export CARBON_REVIEW_OUT=/tmp/carbon-review-round5/runtime
python -B certifier_probe.py
python -B pack_export_probe.py
python -B real_pack_certifier_probe.py
```

最后一步依赖第二步生成的 `n1r_dequantized.npz`。certifier_probe按默认时间阈值真正sleep，最长约7秒，并行只为缩短小型故障测试耗时，不作为性能读数。两步真实包证书只是smoke；真实pack原文件只读。所有临时source漂移都发生在输出目录。

精选发布文件：四个Python（common.py及上述三脚本）、README.md/runtime-notes.md、三个结果JSON（certifier-probes.json、pack-export-results.json、real-pack-certifier-results.json）。原始每例certificate JSON可以保留，但已被聚合JSON完整包含。不要发布生成的npz、submission包、source-drift或standalone目录、pyc。

本轮运行命令输出保存到 `/workspace/carbon-review-round5/runtime`；JSON内绝对路径标识当次事实，复跑会使用新的环境变量路径。没有部署、天梯提交、长对局评测或cgroup资格认证。
