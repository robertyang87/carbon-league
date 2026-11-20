<!-- 落仓头（工程组，2026-11-20）：DGX 原件 GATE_COMBO_REPORT.md（md5 bcca5818387ce7ba0578e4f1d93c5c82，DGX /home/arm/junyang/lp2_weight/combo_gate/）逐字转录。主代理独立复核：compare_result.json 四腿 55/55、A 对 C 0/55；A/C0 epoch 统计与 ② 闸在册值逐字合；两件 md5 在 DGX 与本地双读合。判读：四腿全 PASS，组合件 train2_uspec_w.py（md5 79321c672650bba728563d1c6fee381c）可供 Gen-5 发车。 -->

# GATE REPORT — train2_uspec_w.py 组合等价闸（2026-11-20，DGX 执行）

组合件：`train2_uspec_w.py` md5 **79321c672650bba728563d1c6fee381c**（落盘读回对账合期望值）
基模块：`train2_uspec.py` md5 **aa1f8596c8101a146b6a230d9bfd3f54**（读回合期望值）
结论：**四腿全 PASS**。

## 环境与口径

- 座：DGX（spark-29db，100.84.148.73），venv `/home/arm/junyang/carbon/.venv/bin/python`；staging `/home/arm/junyang/lp2_weight/`，产物目录 `combo_gate/`（driver.log、logs/、outs/、compare.py、compare_result.json、ones_fixture_gen.json）。
- 与 ② 行权重闸完全同 fixture、同配方：每腿 `--epochs 1 --batch 128 --seed 0 --fast-policy-weight 1.0 --feed-workers 1 --device cuda`，init＝gen2 镜像 ckpt md5 `b944a2b9c96384187a22040a44dc8f98`，各腿均带 `--out`＋`--out-pt`，七腿顺序跑、全 rc=0（11:43:08–11:44:49）。
- 比对口径：导出的 state_dict 逐张量 `torch.equal` 计数（共 55 张）＋epoch 统计行逐字比。注：.pt 容器文件 md5 在跑次间不同（zip 序列化层元数据），张量层逐字同为准——与 ② 闸同一判读口径。
- 同座并行线（价值谱正式训 PID 1898309、vec2 消融）未受干扰，未 kill 任何进程。

## 文件 md5（DGX 读回）

| 文件 | md5 |
|---|---|
| train2.py | 06e3c87c932a762dba6a09ee1bd06117 |
| train2_w.py | cab1d22a20c34bfbe6d4e3b361a021b8 |
| train2_uspec.py | aa1f8596c8101a146b6a230d9bfd3f54 |
| train2_uspec_w.py | 79321c672650bba728563d1c6fee381c |
| layout.py | 6877bc73ca971e0110d19d54b04fcd7d |
| model.py | c102c29780b67035068e9a001005170e |
| init gen2 ckpt | b944a2b9c96384187a22040a44dc8f98 |
| combo_gate/run_gate.sh | f9d681a12297e2595cdf526e7b6885d8 |
| combo_gate/compare.py | c181c18d760e2d3b1b1ee72b25972774 |

## Fixture 出处与生成法

- `fixture/raw_flat/`（② 闸原物，未改）：4 局×299＝1,196 行，无 wpw/wvw 键。分片 md5：0e2cecb7…＝4845d519a2048b9ba375a527d62ceb2e、0f203344…＝17d76dd9e965ee1fc2cbb785d250e5f6、1c989743…＝85ff28c963b2f7b5e5095ffea2d18028、234319e1…＝9b37a468ebe1da37759514418b9d177e。
- `fixture/weighted_flat/`（② 闸原物，未改）：同上＋wpw/wvw 键（float32）；suppressed 两局（1c989743、234319e1）wpw 全 0.5（每片 sum 149.5），mistake 两局 wpw 全 1.0，wvw 全 1.0。分片 md5：72b713ed5ae119b8e0efb935f7cb72d0、081e0289e7ee517fd6fccc288192137b、d9f9f54f25ab4418b2d69f3b71dfb3cd、e2754884b84f8e77058253c4b36cf7e4（与本地 fixture 记录逐字合）。
- `fixture/ones_flat/`（本闸新生成）：由 raw_flat 逐片复制全部原键、加 wpw/wvw＝float32 全 1.0（dtype 对齐 weighted 版），逐片 299 行、全集 wpw 总和 1196.0；生成记录 `combo_gate/ones_fixture_gen.json`。分片 md5：0174d31e49543fb1dbde0b49a55fb643、4155b2c76e20168c69d2826ae9d5d4a9、53802d2f1cc75ebb603ea376bfcd299a、ce381b60399acb39e01c70100766c547。

## 逐腿结果

epoch 统计原文（各腿日志 `epoch 0:` 行）：

- A（train2 × raw）：`{'loss': 1.3058, 'pw': 0.4782, 'pb': 0.4926, 'v': 0.0826, 'delta': 0.147, 'fog': 0.524, 'win': 0.3106}`——与 ② 闸在册 A 跑统计逐字合（本闸为新跑，未复用旧产物）。
- B1（组合 × raw，--utility-scale 600）：同 A 逐字。
- B2（组合 × ones，--utility-scale 600）：同 A 逐字。
- C0（train2_w × weighted）：`{'loss': 1.2998, 'pw': 0.4786, 'pb': 0.4853, 'v': 0.0825, 'delta': 0.1452, 'fog': 0.5264, 'win': 0.3164}`
- C（组合 × weighted，--utility-scale 600）：同 C0 逐字。
- D0（train2_uspec × raw，--utility-scale 300）：`{'loss': 1.3222, 'pw': 0.4803, 'pb': 0.4963, 'v': 0.0841, 'delta': 0.1624, 'fog': 0.5545, 'win': 0.2994}`
- D（组合 × raw，--utility-scale 300）：同 D0 逐字。

| 腿 | 比对 | 张量 | 统计 | 判读 |
|---|---|---|---|---|
| B1 | vs A | 55/55 torch.equal | 逐字同 | **PASS** |
| B2 | vs A | 55/55 torch.equal | 逐字同 | **PASS** |
| C | vs C0 | 55/55 torch.equal | 逐字同 | **PASS**（×0.5 生效腿：组合件在带权重输入上与已过闸 train2_w 完全一致） |
| D | vs D0 | 55/55 torch.equal | 逐字同 | **PASS** |

旁证：
- C 相对 A 的位移：A vs C 张量 0/55 相同（全不同），统计 loss 1.3058→1.2998、pb 0.4926→0.4853，与 ② 闸 C 跑在册位移（loss 1.2998、pb 0.4853）逐字合——行权重在组合件里真实生效且幅度与 train2_w 一致。
- D 相对 A 的差异（刻度 300 生效旁证）：loss 1.3058→1.3222（Δ+0.0164）、pw 0.4782→0.4803、pb 0.4926→0.4963、v 0.0826→0.0841、delta 0.147→0.1624、fog 0.524→0.5545、win 0.3106→0.2994；D 与 A 张量层必然不同（统计已不同，未单独计数）。
- 各腿日志含 `utility scale: 600/300` 打印（B1/B2/C＝600，D0/D＝300），刻度确经基模块全局生效。

## 异常记录

- 首次驱动脚本经 ssh heredoc 写入时变量被引号层吃掉，七腿空转未真正跑批（无任何训练进程启动、无产物）；已改本地写文件＋scp 重跑，上表结果为第二次正式跑批。现行文件与在跑进程零改动。
