# it85 检查点定标 · Echo2 场（C2 / spot-31，2026-11-14）

§17 检查点 1，种子 999000–999199，同种子配对，新口径全量存档（bigeval dceb1ebe，逐局线带原始 margin 与 u）。

## 场一 · 候选 it85 vs Echo2

- 被测方：it85（snipe_n1 it85 检查点裸网包，code/exp/it85.py，md5 6bd5cc5a85ca90cb5426b545b7a9d84，4,598,446 B，main 737af89d）
- 对手：Echo2（我方已提交真身，exp/echo2.py，md5 15b6d754e6e21de33870074d3983a16f）
- 结果（被测方口径）：**118 胜、均 +232、se 50、lo2se 132**、worst −1525、worst_ms 118.0、err 0、FAIL false
- 独立重算（本地快照 200 行）：118 胜、mean 232.2、se 50.3、lo2se 131.6、worst −1525.34、U 均值 0.5719；种子 999000–999199 完整唯一、err 全 null、hh/ah/bh 全场一致
- 逐局线：bigeval_live/it85_echo2.jsonl（200 行，md5 004587a5d0f0a6bf99660fe67364f88d）

## 场二 · 基线 m3s2r1 vs Echo2（同种子配对）

- 被测方：m3s2r1（Master3s2 替身冻结裸网，exp/m3s2r1.py 包装 md5 73542cdd0ea50dce87e3638fd4c9bf5a，权重 cnet2_m3s2r1L.npz md5 b6db08c9b076257c54817377bb93130d，与其存档场同一包）
- 对手：Echo2（同上）
- 结果（被测方口径）：**145 胜、均 +466、se 61、lo2se 345**、worst −2290、worst_ms 156.0、err 0、FAIL false
- 独立重算：145 胜、mean 466.0、se 60.7、lo2se 344.5、worst −2289.74、U 均值 0.6632；种子完整唯一、err 全 null
- 逐局线：bigeval_live/m3s2r1_echo2_base.jsonl（200 行，md5 3952afb290b41afac6df259aff90af18）
- 参考：基线存档场（2026-11-12，旧版 harness）为 144 胜 / +472 / se 60，本次新口径重跑 145 / +466 / 61，基本一致。

## 配对差（原始值，判读口径归策略组）

- 同种子逐局配对：平均 ΔU（it85 − m3s2r1）＝ **−0.0913**（sd 0.4158，se 0.0294）；平均 Δmargin ＝ −233.8
- 即 Echo2 场上 it85 低于基线 m3s2r1；合并判读（含 C1 的 Master2 场）与裁决归作战室/策略组。

## 考前闸与过程记录

- 闸一（开考前直读，07:06）：fastsim code/ 与 packs/ 均 aefe6c83826efae4de83bef8c643c80b、bigeval dceb1ebeb2a804068c2263fc9206833a、机器空闲（load 0.00、无考试进程）；packs/ 保持原样（仅 np/search/fastsim）。
- 闸二（开考后）：两场 stderr 起始行 harness_md5 均 dceb1ebe；逐局线 u 字段为长小数（例：0.7706776532511666、0.1179626797219178），hh 全场一致。
- 偏差：07:08 上传链脚本时误以 stdin 执行（bash -s），场一在快照环启动前开考、无独立冒烟；闸二在实考上补验全过，快照环 07:11 拉起、07:11:29 首次落盘即场一全 200 行，逐局线无丢失。链壳于场一后随会话断开死掉，场二改 setsid 独立重拉（进程组独立、落盘可查），终局文件经双路核验为单套 200 局完整唯一。已同步作战室（07:12 回执）。
