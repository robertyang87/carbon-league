# 截至 628e28d0 的前五新增六场

从 `56e1e613` 到 `628e28d0`，在截止点全仓确认 347 个来源文件。本次有变化的来源所含比赛引用从 5,312 条增至 5,347 条。其中 1 条补记已有局，24 条双方都未进入冻结前五，另有 10 条指向本批 6 局。6 局均为对外比赛，新增故障 0。累计 1,636 局有双方官方终分，其中互打 121 局、对外 1,515 局。另计 1 局历史故障。 [成绩逐条去重](score_delta.json) · [来源与回放身份](manifest.json)

| 前五策略 / 物理席 | 对手 | 官方终分（前五 / 对手） | 前五终场分差 | 最大十步窗口 | 窗口分差变化 | 窗口正常交仓（前五 / 对手） | 窗口树收入（前五 / 对手） |
|---|---|---:|---:|---|---:|---:|---:|
| [claude-AlphaCarbon-Master6 / 0](2bd31002ceddeec5c96081fe8a7fc981-1.md) | RyanX-AIPowercore-it920-evo31 | 5,041.68 / 7,002.92 | -1,961.24 | 190–200 | -687.50 | 137.40 / 722.21 | 89.13 / 90.60 |
| [claude-AlphaCarbon-Master5s / 0](2d5ac6e5ead10ca88f2816026fd80e8b-1.md) | RyanX-AIPowercore-it920-evo31 | 4,999.10 / 5,415.97 | -416.87 | 217–227 | -729.18 | 0.00 / 676.65 | 110.51 / 59.62 |
| [ClaudeFable5.2-v7 / 1](3f1215f6020138f028464efe9b9868c5-1.md) | RyanX-AIPowercore-it920-evo31 | 7,511.72 / 7,588.83 | -77.11 | 261–271 | +1,103.05 | 1,143.91 / 99.21 | 25.76 / 122.55 |
| [claude-AlphaCarbon-Master5s / 1](9b813c96fc9baa2f51989e9162e2b51f-1.md) | RyanX-AIPowercore-it920-evo31 | 7,728.39 / 7,973.64 | -245.25 | 190–200 | +793.56 | 905.25 / 0.00 | 57.96 / 170.91 |
| [claude-AlphaCarbon-Master5s / 0](c04108b99b53f3a941469f504431c301-1.md) | RyanX-AIPowercore-it920-evo31 | 6,729.81 / 6,716.77 | +13.04 | 281–291 | +769.80 | 1,027.26 / 257.67 | 23.25 / 77.41 |
| [claude-AlphaCarbon-Master6 / 0](d109b5fce5ea8da2d8e67fa1924d5dc0-1.md) | RyanX-AIPowercore-it920-evo31 | 7,630.28 / 7,550.02 | +80.26 | 214–224 | -898.22 | 0.00 / 650.84 | 71.77 / 204.46 |

## Master6 的胜、负和中游对手

- **胜局**　种子 609054917，Master6 席 0。最后 30 拍，Master6 和对方各交仓 2,132.15、1,750.60，现金分差变化 +321.22，最终分差 +80.26。[逐拍与收入](d109b5fce5ea8da2d8e67fa1924d5dc0-1.md)
- **负局**　种子 894846674，Master6 席 0。最后 30 拍，Master6 和对方各交仓 1,463.30、1,597.81，现金分差变化 -101.45，最终分差 -1,961.24。[逐拍与收入](2bd31002ceddeec5c96081fe8a7fc981-1.md)
- **对中游**　按 2026-11-19T02:53:08Z 冻结 heavy 榜的第 21–50 名计算，本批有 0 局。同一对手 it920-evo31 的 2 局在该快照均无可核验档位。[分档与原始名次表指纹](master6.json)

每个逐局页面列出现金差绝对变化最大的十拍窗口，并回到窗前 3、5、10 拍，查看双方的工人、树、基地、在途货、可见状态与实际指令。这些窗口按赛后现金变化选取，用于追溯收入构成。[双方状态](turning_points.json) · [逐拍八项现金](turn_cash.json.gz) · [逐拍指令](turn_actions.json.gz) · [进攻防守与树权](cases.json.gz) · [后续 10、30 拍收入抽核](future_economy_check.json)

官方 Board 与 FastCarbon 各复算 1,794 个联合回合，六场官方终分全部吻合。双方共 3,588 个行动前决策通过上下文核验。[双引擎收据](validation.json) · [双席上下文](context_receipt.json) · [完整官方回放](replays.tar)

6 个新增初盘按录像内 225 格碳量计算 SHA-256；历史全部 1,637 个初盘（含故障）各异。六局与冻结时已登记或排除的 4,542 个物理图及开发图 621758400、621758401 在种子和地形上均无碰撞。 [地形逐局指纹](development_maps.json) · [暴露与预留范围](exposure.json)

逐局读取状态、双方可见信息与结果可用 `context.Corpus(manifest=Path('reports/sapling_1200k/all_top5/appendix_628e28d0/manifest.json'))`。本批独立索引与收据放在同一附录内。

复现时在仓库根目录运行 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 2 nice -n 15 /workspace/carbon-review-venv/bin/python reports/sapling_1200k/all_top5/appendix_628e28d0/prepare.py`。程序可从本批归档恢复官方回放。
