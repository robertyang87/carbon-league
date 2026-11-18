# 合并 publisher 维护批验收备案（策略组）

策略组，2026-11-18。依据：工程组落地回执 `reports/BUILD_MERGE_MAINT_20261118.md`（f6df43d7）；销项件 `reports/CLEAR_MERGE_PUBLISHER_CONDITION_20261118.md`（43907b6a）§2 维护批裁定。本件为裁定约定的常规验收抽查，不另开轮次。

## 抽查结论：PASS，维护批销项

- **件身份**：`tools/merge_annot_generations.py` 现行版 md5 `63681663b1c5bf6395863e0d085b2ff6`（策略组直读仓内文件复算合），自基底 f1f5ab9c 的 diff 只在本批三项与配套 verify/注册逻辑内。
- **(a) 逐局出处**：实现比「携带」更严——同局多源时 pass／selected_sha256／shard_sha256 须逐字段全源全等才逐字携带，值不同或有无不一即 die 点名局名与字段，全源皆无不携带。合裁定口径。
- **(c) seed 规范化**：规范十进制形存 int、非规范形保持 str，REGISTRATION 另记 seed_canonical、派生只喂 canonical 串；legacy 代 verify 走回退且复算同值——合「不改变已注册种子派生值、legacy 兼容」两条件。
- **(b) 注记**：docstring 已补两层抽样之分的明示行，引在册语义。
- **证据**：既有 14/14 无回归，新增 M1–M8 全过（主代理复跑同数），含基底所产 legacy 代经新工具 --verify PASS、同输入同种子下身份哈希与基底逐字同（71083f40…）——已发布代与身份哈希语义不动，已验。

**备案：维护批义务完成，段 2 首次正式合并的本工具前置提前解除；现行 publisher 版本以 63681663 为准。**

—— 策略组 2026-11-18
