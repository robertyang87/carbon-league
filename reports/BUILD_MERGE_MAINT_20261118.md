# 合并 publisher 维护批落地回执（工程组，2026-11-18）

销项件 CLEAR_MERGE_PUBLISHER_CONDITION（43907b6a）§2 裁定的维护批，已入仓。

- **件**：`tools/merge_annot_generations.py` md5 `63681663b1c5bf6395863e0d085b2ff6`（818 行；基底 f1f5ab9c，diff 131 行全在本批三项与配套 verify/注册逻辑内）。已发布代与身份哈希不动：规范化清单构成、派生规则本体未改义。
- **(a) games 逐局出处**：一致性校验后取合值——同局见于多源时 pass／selected_sha256／shard_sha256 须逐字段在所有记录该局的源中全等才逐字携带；值不同或一源有一源无，die 点名局名与字段；全源皆无不携带。
- **(c) seed 类型规范化**：存储形只在 meta /sampling/seed 与 REGISTRATION seed 两处——种子串为某 int 的规范十进制表示时存 int（"007"/"+3" 等非规范形保持 str）；REGISTRATION 另记 seed_canonical，派生永远只喂 canonical 串 utf8；legacy 代（无 seed_canonical）verify 走 str(存储值）回退。新旧两形复算同一派生值。
- **(b) 格式注记**：docstring 补一行明示合并抽样与标注器内 pi 两层之分（引 43907b6a §2.1 在册语义）。
- **自证**（主代理复跑同数）：既有 14/14 无回归；新增 M1–M8 全过——含单源/多源携带、两类 die 点名、新代 int 形 verify、**基底工具所产 legacy 代经新工具 --verify PASS**、同输入同种子下身份哈希与基底逐字同（71083f40…）。

段 2 首次正式合并的本工具前置已提前解除（维护批落地报备义务完成）。

—— 工程组 2026-11-18
