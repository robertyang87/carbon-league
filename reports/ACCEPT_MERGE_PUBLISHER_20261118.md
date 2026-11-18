# 验收：合并 publisher（merge_annot_generations）有条件 PASS（策略组，2026-11-18）

- 验收对象：工具 tools/merge_annot_generations.py（commit f2282aed，md5 f1f5ab9c，707 行）＋交付件 reports/BUILD_MERGE_ANNOT_GENERATIONS_20261118.md（3f87e729）。
- 验收依据三方一致：交付件自审 14/14 PASS（主代理独立复跑同数）；策略组亲读交付件全文；策略组独立抽查工具源码关键段（种子派生 sha256(注册种子‖源代 id) 前 8 字节、发布序 current.json 最后翻、teacher_config_hash＝canonical 来源清单 sha256、重叠根 die 点名「refusing to pick one」原文在实现内、已发布不覆盖多处在位）。

## §1 钉死项逐项核对

- 注册先行：发布序＝封印→预检→代落盘→REGISTRATION.json 落盘→current.json 最后翻；--verify 含注册件与代内清单 canonical 逐字一致＋派生种子复算。**合。**
- 合并身份：来源清单哈希三处同写、不借用源代哈希；解释口径实现（下游零改动、join_u 仍 9054aadc、钉版件零改动）。**合。**
- 采样器：逐源独立、稳定序无放回、floor(根数×率)、逐源模式声明（full／sample@0.2）、派生种子逐源入清单。**合。**
- 去重：逐字节同才去重、不同即 die 点名（T9 证据点名 (selfplay-900001, 25, 0)＋字段 Q、明示拒绝取一/平均/拼接）。**合。**
- 内容寻址、已发布不覆盖。**合。**

## §2 条件（段 1 首次正式合并前置补验）

- 唯一未闭环证据：真实代 meta 旁证未读（验收期间 L20 复读遇本机 known_hosts 轮换窗口；格式依据为一手源码＋早先读回的真实 manifest＋fixtures 经真 loader 全套验过——属强证据，但真实代最后一环不省）。
- **前置钉死**：环 3 段 1 首次正式合并前，工程组须在真实源代上跑一次合并冒烟＋--verify，真实 meta 逐字段对齐读回、回执报策略组备案后，方许正式合并。此补验不另开验收轮次；未补验前本工具只许合成/冒烟用途。
