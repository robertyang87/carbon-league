# 确认：TOP4 v4 train 335 局回放集正式生效（2026-11-17）

署名：策略组。对象：DGX 组装交付 L20 的 train 335
回放集、reports/TRAIN335_INTEGRITY_CHECKLIST_
20261117.md（825d3d1f）与 data/train335_integrity_
checklist.json（fb6bcc15）。方法：仓内原件读回＋
策略组独立复算＋回放字节抽验（非采信自报）。

## 抽验结果

1. **集合全等**：清单 JSON 的 335 局 id 集与 v4
   train（696060d7）自原件复算**逐元素全等**；作
   废局不在集内。
2. **逐局全表**：335 行 classification 全 full、
   steps_played 全 299、entries 全 300、步连续／
   statuses／终局 rewards 三项全真，异常 0 行——
   与主清单正文（checked 335／full 335／short-
   explained 0／anomalies 0）逐数合。
3. **字节抽验（7 局分层）**：快线交集 2＋冻结增量
   2＋其余 2＋50d4b8cc（交付时唯一非仓内清单局）
   ——自仓内家族语料原件 fetch 解压直读：字节
   md5 与（更新后）家族清单逐局合、steps 清单
   300、statuses 双 DONE、终局 rewards 与清单表
   逐值全合，7/7。

## MANIFEST 本体替代口径（答作战室问）

交付 MANIFEST.md5 本体（7dc459b4）尚未入仓。本
确认的证据链为：本清单（上列独立抽验）＋交付三
跳 md5sum -c 全 335 OK 的在册回执（组装／DGX 中
转／L20 落盘，清单 md5 7dc459b4 已报）＋抽验把仓
内原件字节与清单表对死——已覆盖 MANIFEST 本体
的证据职能（交付集身份＋字节同一）。**MANIFEST
本体降为归档欠件**：DGX 仍须补入仓 data/ 存档，
但不再阻塞 train 集生效。

## 效力

- **train 335 局回放集自本件起正式生效**（与完整
  性总清单同批），L20 侧可按在册开工序凭本集推进
  （train 件直读验合的「清单 md5 对账」一项以本
  件＋fb6bcc15 为据）。
- 本确认只及回放集与完整性；leg-1 其余前置（标注
  合并代交付、join/build 于验通 CPU 座、L20 对表
  ＋冒烟）照在册口径各自成立后开训。

落字：策略组，2026-11-17。
