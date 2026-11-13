# snap_70 fastsim A/B — Echo2 场（新 fastsim aefe6c83）

性质：模拟器效应测量，不是重考、不改任何 verdict。旧 fastsim（61859708）跑的原记录保持在账。

## run
- 被测方 snap_70（snipe_n1 it70 快照）vs 对手 Echo2（天梯考官）
- 种子 999000，200 局；2026-11-14 00:36:32–00:38:32 CST，spot-31
- pack exp/snap_70.py md5 ece3ff5294aaca0be0563be66571d9d7（4,598,882 B，链内预检通过）
- fastsim.py code/ 与 packs/ 两处皆 aefe6c83（链内预检通过），py_compile 通过
- 冒烟 n=2 先行：err 0，worst_ms 94.0
- 链 rc=0；2 分钟快照环全程在跑

## 结果（被测方口径）
- 新 fastsim：200 局、121 胜、均 +237.0、se 52（逐行重算 51.8）、lo2se 133.0（重算 133.3）、worst -2938、worst_ms 119.0、err 0
- 旧记录（fastsim 61859708，2026-11-13 19:32）：121 胜、均 +237.0、se 52.0、lo2se 133.0、worst -2938、worst_ms 118.0、err 0
- 逐行重算与 launcher JSON 一致；种子域 999000–999199 唯一齐全；err 全 null

## 配对 Δ（同种子逐局，n=200）
- 逐局 margin 不同的局数：0/200；最大 |Δmargin| = 0
- Δ胜局数 0；Δ均分 0.0
- ΔU（U = 0.5·W + 0.5·sigmoid(margin/1000)）：均值 0.0000，bootstrap 20,000 次 95% CI [0.0000, 0.0000]
- 结论：fastsim P3 修复（id 计数器＋set_state）在本场零可测效应；逐局 margin 200/200 全同

## 归档
- 逐局线 synced/h31/bigeval_live/snap_70_echo2_newsim.jsonl，839,661 B，md5 bac83d014c69f95f430e43f86931da12
- 旧记录对照：synced/h31/bigeval_live/snap_70_echo2.jsonl，md5 bbe8415e1ba5d4ba5a9f279f4d362915
- 说明：两文件字节不同（时间戳等字段），margin 逐局全同
