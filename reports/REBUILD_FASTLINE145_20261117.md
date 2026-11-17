# 快线重建 145 局（裁定 757d17db）执行记录

## 触发
L20 快线训练死于 2026-11-17 18:11（600/3,526）：`k0_distill features: root 32918 (game 55 step 14) candidate 15: the root cash is underdetermined … refusing to train`。诊断定位：game 55 = order146 0-based index 55 = **4c086ee0a76d9dd2c26d5e2a13654119-1**（源代 gen-54481ed6f970896e，598 根）。策略组裁定 757d17db 授权排除该局重建。

## 执行（DGX，2026-11-17 18:53–18:55）
- **Merge**（钉版 merge_annot_gens_excl.py，md5 57cc1310）：源三代（44 gen-f3a7cf50829156bc 29,302 / 46 gen-54481ed6f970896e 28,912 / 47 gen-71f5332544b36eb8 29,302），order = order146 减 game55（order145.json，145 局），--exclude = [作废局 8759f821…-1，game55 4c086ee0…-1]。
  - 产物：`/home/arm/junyang/k0_dist/master5s_fastline/merged145/`，**gen-4f80c4dc2f956ef6，145 局，86,710 根**，manifest_sha256 e35e69dc402154831985ece2ef0ae259f3e29ef7450f6bb1d4abd9e3eddfc30c。
  - MERGE_RECORD：excluded 两条（8759f821 208 根／4c086ee0 598 根，均源 gen-54481ed6f970896e），工具内常量 exclusion_ruling 6c98cc58／exclusion_void_ruling 83a5db39 不动，本件裁定号 757d17db 并引于此件。
  - 验合：逐局计数表（per_game_source_generation）中作废局与 game55 均零进入。
- **Build**（钉版 LADDER_FRAME cnet2_build.py，md5 5c2dd58e，CNET2_LADDER_FRAME=1）：源 build_src145（index 145 条＋145 回放，game55 已剔）。
  - 产物：`/home/arm/junyang/k0_dist/master5s_fastline/build145/`，**gen-50aa24668b6d0ea1，145 局，86,710 样本，cfg_fp f982640561fc4fe4**（与 build146 同值，配置未变），ladder_frame true。

## 待办（join）
- **Join 未执行**：裁定指定座 spot-54，但 54 自 2026-11-17 ~18:28 离线（DGX 双路 SSH 直读不通、tailnet last seen 约 18:30，后勤控制台定案中）；且按修正案 bda8c0b1 join 不许在 DGX 跑。55/56/58 三座正满核标注（241 包），57 亦离线（~18:53）。请作战室另指 join 座——merged145＋build145 均在 DGX 就绪，随时可发座执行（钉版 k0_join_fallback.py，md5 fac55acb）。
- Join 验收（裁定口径）：作废局与 game55 均零进入、--verify 过、逐局计数表、回退命中清单入 provenance；随后产物送 L20 重训。
- 诊断扫描件：训练报错行与 L20 诊断件在 L20／作战室侧，本件不代造；MERGE_RECORD 与本件为重建回执的入仓部分。
