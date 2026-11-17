# TOP4 leg-1 train 集（335）完整性总清单 — 2026-11-17

执行：DGX 线。配套件：`data/train335_integrity_checklist.json`（逐局结果全表，md5 5254dba7dacbb5d4480674114a382712）。
本清单与 train 回放集同批生效（策略组分工、作战室转达）：train 集已交付 L20，本清单为其完整性总凭据，供策略组抽验出确认件。

## 身份

- train 清单：`data/top4_pool_v4_train.json`，md5 696060d7936ff73897de1fd89abc7b8f，335 局。
- 交付件：L20 `/root/carbon/katago/top4_leg1_replays/`，335 件回放＋MANIFEST.md5（md5 7dc459b4581f004130d8c4e4b4275bcf）；DGX 中转与 L20 落盘 md5sum -c 均 335/335 OK。
- 本清单的检查对象＝交付字节本身（组装集），非源件抽样。

## 集合全等核对（防组装错位）

组装集文件名集合（去 .json.gz 后缀）与 v4 train 清单逐元素比对：**全等**（335＝335，排序后逐项一致，无多、无缺、无错位）。

## 逐局完整性（所有者 SOP 2026-11-17 14:54 口径）

检查器：audit_integrity.py（gzip/JSON 完整性、步条目数、步连续 obs.step＝条目序、两侧状态终局前 ACTIVE／终局 DONE、终局 rewards 在位；短局按终局工人归零分类 ANNIHILATION 可解释／UNEXPLAINED 异常）。

对组装集全量直跑结果：**checked=335，full=335，short-explained=0，anomalies=0，exit 0**。

即 335 局全部为满 300 条目（299 步打满）的完整对局：步连续 335/335、状态形态 335/335、终局 rewards 335/335。无短局、无异常、无需处置项。逐局明细（条目数／连续性／状态／终局 rewards／分类）见配套 JSON 全表。

## 身份来源标注（策略组抽验要求，2026-11-17 补）

- 50d4b8cc634466307faae70afd999385-1：快线语料第 148 局（当时仅为快线在 DGX 留置，后裁定为 TOP4 池/train 合法成员）——其回放不在四家仓内清单中，身份以 DGX 在册记录为准：replay md5 e6381b5bc37972e4113f8643c770c6f6、252,513 B；交付 MANIFEST.md5 本体（已入仓 data/train335_delivery_MANIFEST.md5）含此局同值一行；本清单 JSON 全表该局条目带 identity_source 字段。审计结果：full PASS。

注：快线作废局 8759f8217d439cc5127f10aa2139935a-1 不在 train 内（冻结时已硬排除），本清单无短局处置事项；此前分项检查（快线 147 审计、冻结增量 50 审计）与本总清单口径一致，本清单以交付字节全量直跑为准、取代分项证据作为 train 集的完整性凭据。
