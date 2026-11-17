# leg-1 学生预览 vs Master3s2 — 报告（2026-11-18）

- **性质**：预览分布读数，**不作正式校准**；不含通过/不通过判定，不作上榜依据。
- **学生包**：`gen1_leg1_student_int8.py`，md5 `e6f62c8b40325989f80941bf2c418aac`，4,440,296 B（L20 训练 5749/5749 步完训 03:44 后的 int8 打包件）
- **考官**：`opp_master3s2.py` 包装（md5 `b4cde262…`）＋ `master3s2.py` 本体（md5 `57f9a594…`）
- **口径**：bigeval.py（md5 `354bcd1b…`），8 worker，线程钉 1；种子块 **990200–990399**，200 局
- **时间**：04:04 开跑（烟雾 2 局先行通过，RC=0，err 0），首局 04:07:40，末局 05:21:28，**完赛 05:21**，进程已退出

## 结果（200 局，逐局重算与 bigeval 汇总行逐项对齐）

| 指标 | 值 |
|---|---|
| 胜 / 负 / 和 | 16 / 184 / 0 |
| 均分 | −885.85，SEM（样本）45.2，95% CI [−974.4, −797.3] |
| mean_u | 0.1934 |
| err | 0（FAIL=false） |
| 最差分 | −2819.64 |
| 最差步时 | 505.0 ms（wms 均值 430.9，p95 460.0） |
| dep 局均 | [8135, 8498] |
| tree 局均 | [2939, 3293] |
| deathcargo 局均 | [348, 177] |
| plants 局均 | [28, 29] |

## 工件

- 逐局线：`data/k0_calib/preview_leg1_vs_master3s2_live.jsonl`（200 行，md5 `b3019c259aa20001a89baaa44d6663f4`）
- 汇总：`data/k0_calib/preview_leg1_vs_master3s2_20261118.json`（md5 `1f8c631a7b73d3710e058d627a8ac1cb`）
- 本报告：`data/k0_calib/preview_leg1_vs_master3s2_report.md`
- 原始 live 文件与 run 日志留 DGX：`carbon/data/bigeval_live/preview_leg1_vs_master3s2.jsonl`、`k0_dist/preview_leg1/run200.log`

## 参考

快线学生预览（2026-11-17，种子 989000–989199）：27 胜 / 173 负，均分 −774.4 ± 51.3，mean_u 0.2340。两预览种子块不同、学生不同，仅并列参考，不作同口径比较。
