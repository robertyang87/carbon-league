# ppou2 主臂关账签收（策略组）

- 日期：2026-11-16
- 签收方：策略组
- 关账件：本目录 7 件（commit 5afb0f40），裁定：`reports/RULING_ADJ2_PRIORITY_RESOURCES_20261116.md`（716984fe）

## 验合结果：全合，关账销项

- 七件均经 API 读回，md5 与 L20 回执逐字合：league.json 5d59f0da605d4daa806445b02a50721c、config.json 05bec0c9、effective_params.json 7dae80b7、argv_environ.txt 20c10d65、train.jsonl f9128b4d、boundary_stop.log 5933176d、README 清单一致。
- 终局身份入档：state_last.pt md5 **fec8c39306c2c758b927c75c9df20a8e**（82,600,176 B，实体留 L20）；ckpt_last.npz md5 **4eb54584f557636e0e5fd4b266c23326**（留 L20）。
- train.jsonl 直读：58 行（it=1–58）；末轮 it=58 games 4426、kl 0.1324、winrate 0.653、roll_s 2467.4／upd_s 701.6，与回执合。ckpt_37 段内 kl 全程在 0.15 帽内（it=52 0.1071 → it=58 0.1324），无单轮跳升 >+0.03。
- **待核项关闭**：it=57 的 kl 0.1243 与 it=55 同数——train.jsonl 直读证实两轮均为 0.1243（其间 it=56 为 0.1291），系记录 kl 的批内估计噪声下的四位小数巧合，L20 回执无误，不立项。
- config／argv 与注册形态合：--ref＝ckpt_37.npz、--kl 0.05、λ0.95、--value-head tanh、--resume＝state_frozen_it37.pt、--init＝m3s2r1L.npz。
- 停机序列与 boundary_stop.log 合：15:20:57 布防 → it=58 轮行＋state 轮转 15:21:27 → TERM 进程组 368291 于 15:21:37 → 15:21:39 验不在册、GPU 0 MiB。轮次原子性保持（在飞轮次跑完才停）。
- 收尾联赛快照 league.json（5d59f0da）即 adj2 league 构建底本；其先验总和 7.45 与 adj1 口径一致，现压力成员（m6r1L＋caragor1_ep19）份额 0.725／7.45＝9.73%，adj2 份额线在此底本上重算。

## 关账结论

主臂（ppou2_m，it=1–58，锚 ckpt_37 段 it=38–58）**正式关闭**。轨迹关账读为 it=55 检查点考处置（`reports/PPOU2_IT55_EXAM_DISPOSITION_20261116.md`，5f473edbeb，U 0.5924）——L20 回执中“it=55 结果到后汇总并入关账”一项由该处置件直接承接，无需另行汇总。恢复只许新注册，不许直接 resume。本签收抄送作战室。
