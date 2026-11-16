# 附记：点 B 396 附卷结果件 identity_match.games_sha256＝false 的口径（DGX，2026-11-17）

应 RULING_POINTB_APPENDIX_RERULING_20261117（589a3ec7）附记补入卷备查，不作裁定条件。

compare_pointB396_result.json／compare_pointB396_deep_result.json 两局的 identity_match.games_sha256 均为 false，原因是构造性的：

- games_sha256 是对**整份 --games 列表**的哈希（标注器 LEGACY 身份键），不是逐局字段。
- 封口参照侧分片（eng_ref_20261116）出自全 200 局列表身份的运行：games_sha256＝7175b92149fd47db8ff16122c1f51cd79a9f1045d92726244c460a76849ef4a2（列表＝selfplay-994400–994599）。
- 候选侧（Studio 附卷运行）只跑被比两局：games_sha256＝c3649caaa222b0f4e128eb65fec1e6926d80c6eea3aca3ab3b7c7b88b5b87fdf（列表＝selfplay-994400、selfplay-994401）。selected_sha256 同理不同（d0b0b32c… vs 9117c25c…），同为整表量。
- 被比对象本身逐项全同：两局 game id 相同、逐局 replay_sha256 相同（结果件 replay_sha_match＝true）、根集全同（每局 198 根、步 200–298×双席、arms_set_diff_roots＝0）、config_sha256 相同（identity_match.config_sha256＝true）。

结论：该字段 false 只反映两边运行的列表范围不同，不反映被比两局有任何身份差；比对有效性由 replay_sha 合＋根集全同承载，与复裁口径一致。
