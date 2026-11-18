# 合并 publisher 真实代冒烟回执（工程组，2026-11-18，L20）

有条件 PASS（ACCEPT_MERGE_PUBLISHER fdf47383 §2）唯一条件的执行回执。产物隔离于 L20 `/root/smoke_merged/`，源数据全程只读。

## 选源（真实不同窗对）

- 源 A（full)：`loop_pilot/annot_a` = gen-c89f222be34e2458，config `3cc2e67e`，window [0,299]，29,900 根，权重 `06a03d9c`。
- 源 B（sample)：`katago/pilot200_annot` = gen-26d6fda809835820，config `9a2b2440`，window [200,299]，39,600 根，权重 `06a03d9c`。
- 两代 config 逐字段比对：H/K/M、temps、belief、guard_cfg、scene_contract、seed_teacher、teacher_style 全同，**仅 window 不同**——裁定语义的标准情形。

## 四步结果

- **合并 PASS**：工具自仓现行版取用、L20 落盘 md5 逐件验合（publisher f1f5ab9c／preflight f9582389／join_u 9054aadc／distill_u cfbf227e）。种子 20261118；产出 gen-1a99649e84ee2b69，teacher_config_hash `155e18f2d1a52136d6929b55b8816684f01e0b4cc521be9e58b269e85a8d0ef8`（来源清单哈希）；总根 37,820（29,900 full＋7,920＝floor(39,600×0.2))；250 局不交叠、dedup 0；REGISTRATION.json 随代落盘（文件 sha256 1081a1d5…）。
- **--verify PASS**（RC=0)：真 loader＋并集窗口记录闸＋注册件逐字一致＋派生种子复算全过。
- **consume_preflight PASS**（RC=0)：sidecar CONSUMPTION.json 封印（sidecar_sha256 dd73ce71…），绑定 target.manifest_sha256 a0ecff34…（与 current.json 一致），两源代逐个经真 loader 验讫，checks 六项全 true。
- **真实 meta 逐字段对齐**：身份槽与主体布局对齐（teacher 块、config 对象、config_sha256 槽位、games 表、字段文件布局同形；merged_from 与 merge 块为裁定设计的附加项）。三处差异如实记录：(a) 合并代 games 条目不带源代的 `pass`／`selected_sha256`／`shard_sha256` 三字段（逐局分片出处未携带）；(b) sampling 块：合并代无 pi0/pi1/sampling.selected_sha256、代之以 seed_derivation；(c) `/sampling/seed` 类型源代 int、合并代 str（CLI 种子字符串原样入册）。三处均不触 loader 契约（真 loader 与 distill 侧身份读法实测通过）；(a)(c) 是否在 publisher 后续维护中补齐，待策略组定夺。

## 通道异常注

本 VM→L20 代理管中途三度断裂（SIGPIPE／broken pipe／scp 超时），文件实际完整落盘且对端 md5 全合后才续跑；与既有「代理管慢而不稳」记录同型，非 L20 本机问题。

**结论：有条件 PASS 的唯一条件已满足，preflight 闭环四步在真实代上全通。**

—— 工程组 2026-11-18
