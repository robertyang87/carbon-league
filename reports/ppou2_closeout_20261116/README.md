# ppou2 主臂关账（2026-11-16）

裁定：reports/RULING_ADJ2_PRIORITY_RESOURCES_20261116.md（commit 716984fe）——所有者 15:17/15:18 亲令 adj2 升最高优先级，主臂提前收尾、L20 让座 adj2。不执行 ckpt_45 锚切换、不跑盒末；it=60 不注册；盒末确认考（988000 块）作废；恢复只许新注册。it=55 检查点考在 C1 照旧跑完作轨迹关账。

## 停机实况（机上直读，boundary_stop.log）

- 在飞轮次 it=58 于 2026-11-16 15:21:27 CST 落地（轮行 t=1794813678，games 4426）。
- 轮行落盘＋state_last.pt 轮转双条件确认后，15:21:37 TERM 进程组 368291，15:21:39 验进程组不在册、GPU 显存 0 MiB。
- 终局轮 it=58：kl 0.1324（对 ckpt_37 锚，上限 0.15 内；it=57 为 0.1243，跳升 +0.0081）、winrate 0.653、roll_s 2467.4 / upd_s 701.6。末轮 pace（pace 口径 gpm 107.6＝games/(roll_s/60)；若含 upd_s 则 83.8）受并行 gate 包络腿与签名补测的 CPU 争用压低，非训练异常。
- train.jsonl 共 58 行（it=1–58），md5 f9128b4d2b45c7eaf9a6bc907f29b3a7。

## 关账三件套

| 件 | 内容 | md5 |
|---|---|---|
| 终局 state | L20 /root/carbon/data/rl3/ppou2_m/state_last.pt（82,600,176 B，实体留机、未入仓） | fec8c39306c2c758b927c75c9df20a8e |
| 终局 ckpt | L20 /root/carbon/data/rl3/ppou2_m/ckpt_last.npz（13,972,532 B，留机） | 4eb54584f557636e0e5fd4b266c23326 |
| league 快照 | league.json（本目录；adj2 league 构建底本） | 5d59f0da605d4daa806445b02a50721c |
| config | config.json（本目录） | 05bec0c9cd453975c1a8df446296c111 |
| config 附 | effective_params.json（本目录） | 7dae80b7b356c9072390b1622cdc86c3 |
| argv＋environ | argv_environ.txt（本目录；停机前 /proc/368291 直读） | 20c10d653d1359cc75968577c3ebb01d |

本目录文件：README.md、league.json、config.json、effective_params.json、argv_environ.txt、train.jsonl、boundary_stop.log。
