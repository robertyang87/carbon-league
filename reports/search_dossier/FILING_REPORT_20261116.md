# 搜索线「最新数据与代码整理入仓」核对补齐报告（2026-11-16）

仓：robertyang87/carbon-league，main（sync/h29 只核对）。写入走 gh_api.put_file（branch='main' 关键字），每件提交后 get_file 读回比对 git blob sha＋md5，全验合。

## ① 新补交清单（24 件，全部 FILED-VERIFIED）

### reports/search_dossier/ 直下（4 件）

| 仓内路径 | 字节 | md5 | blob sha | commit |
|---|---|---|---|---|
| LOCALIZATION_REPORT.md | 9,311 | bb3e9e08df9faf61e2aa48729713a2c2 | b59ab6a4b42fa2331c886dfa00d1592ec6fb22dd | 0aaa528c4d3023c905d9290006dab6c7f2f1845d |
| numbers.json | 28,702 | 6fe4de860ea3619a763dc3e4fe9c9dc9 | 30748ccd978f533fadfee76f0e6619770c548969 | 4c07a3f0b81a25abc302733a1ebd3ef100d9ff4e |
| analyze.py | 8,996 | 69d621364d5dfb6b5b709470a9ab0411 | e1c8a9f3ecce1579b77871ca62aaef3da5fae05b | 0a9636d4227a21aedbac7ed973f248cd69204b6f |
| merge_verify.py | 6,048 | 5f7b7f7d24ca4c85c227b1a3fbc2e338 | 20ed7396607091d925792109ca37b58c4a9884a4 | 6a4cdd9f6bdc95bfa21af1a77c94468bd1306d24 |

（来源：winrate_screen/localization/ 三件＋s3_freeze/merge_verify.py。）

### reports/search_dossier/iter1/（20 件，微筛三变体已成形构建件）

| 仓内路径（iter1/ 下） | 字节 | md5 | blob sha | commit |
|---|---|---|---|---|
| echo4_search_int8_v1.py | 9,045,071 | 6f349672f264b50f2d4995fe12f729a4 | 93e2070c1fdca368ab98af107064c9270afc1cfa | be60639f1386f6d240724f2750e5232633c83521 |
| echo4_search_int8_v2.py | 9,041,707 | 99d4178870354f5c6c73b71f65ac3206 | 4ffed7794a612ac20706e472a8966de6b14591d0 | 56ec84e4369872859f609234d1c0f3985138a013 |
| echo4_search_int8_v3.py | 9,045,071 | fbad1e0e7f08d51ce9e3d1cf777453ac | 986fbd255667cdf22c1dcc708974439fa17eea64 | 6022addf76c799d56b40830e355af3fd9467b95e |
| echo4_search_int8_ctrl_tau0.py | 9,045,055 | 21197b51ee9d0e1111882ccb73ed2728 | 02ae258cf1421b2a5399d6c55a3725f3f227fd23 | 9b0ad459b98a75b63060e7509ee0753eaff902d2 |
| cnet2_comparator.py（变体，TAU 版） | 25,958 | 39000607b8df5ce85a56437dcc27398d | 88aae69b91aa4ddbfee8ec8c19fb8c761949e5fb | bbbaae6b1b410495c1301d9fd0679a1c18431c48 |
| cnet2_comparator.py.proof.json | 560 | 51851fcb619f2e2ec6fafe9cd5da6a0e | e9f14f18ab7e7812ec52682103b7a998c4364ea8 | 4498ac786c9b4ca359a3b5411943b3f37f6f4e85 |
| cnet2_search_cmp3.py | 132,494 | b8c77acacf936ee383fa5ab973e4ff73 | 5145a26558b78c315e429cac38229487ada2f16d | 01b2469e2f1c406d41766ecb70e5501abc69963b |
| make_comparator_tau.py | 14,588 | f9a82533f27377bc8870d12d06987278 | c9405eccd09ddfeab445be3b88bd3e36bbaa9473 | 8272d95417305c054ba65284ada4a67ce36d56e8 |
| make_search_cmp3.py | 25,692 | 588773e4de99f90b9a3437ca2cdd43af | e4b11c8a3372347e3c041960b5a2f6dbe4b47109 | b0e56b8cf5a4eea77e28818d274dfddf9982a085 |
| compare_equivalence.py | 3,237 | ccb138eaaea5a168fecee37fa52e4352 | da085301a023f1052cb5205d18e0b4489ca240f0 | 918f1936c40d111b309fb2307341cd804ef3eb84 |
| conf_v1.txt | 245 | e8c87ca0b8b15b09024332fe93084e7e | d7ed3a07736f5f94229807199cf174d494e93c1c | 669fd41219c7f4e65147dbfe193877e3a7cc71b6 |
| conf_v2.txt | 229 | d2433f8c7a3fb4f8d0d184a84643a092 | f3c15f970e97dc3f81cb25123d134bcd475b51d3 | af407cb3ff4183cd65e85ca63941c5d137a115ca |
| conf_v3.txt | 245 | f1d0b926c15f55740f75229cade2b359 | 38ff33cce1262888a3ec654319008248485cefdf | 0de0ad163153d7cfd265e0c7f6dd5622c345bdc3 |
| conf_resync.txt | 229 | b2c381f3a34ee7230114bde1dd947892 | 507ae6c3d1a4a463a08a041bcdec3c5fcbfdfe64 | 7f6d8ab95a7d52449ada385563ce073c3fa5f6e9 |
| conf_diffs.json | 665 | 8566504a2a6ef8ceac4e4c370a2e6dad | ba0fbb1231357d602be63fa5b81e0eb35b4d50e6 | ca745edc2487ea286c261678bd9ace195420e44f |
| equivalence_tau0.json | 995 | aa4d6607e1f8b4893e2f1e832ef60e5a | 2742f56ed3348a059629546609de3b5a0f8a83f3 | 269078e2818c36bb37255be5779a29a8e2becd83 |
| verify_v1.json | 970 | 5100aa6983c626f7b83bd22cd5b3f60c | c338dc42705800bfdf8966944d676b33833feac5 | 0f88724f29511fb49948c8e17ce435b52e1bf1f5 |
| verify_v2.json | 971 | 2ffc91a1c152c33ea604638395e395d4 | ebb9cc59ffcbb49c6731e8bc319d94aff132a487 | 5b7efcbee920cb1943c1d385638670a5969b5933 |
| verify_v3.json | 970 | cdb976f9d836bf25e686f2832ceebf46 | 5a7c163d3710021a0bd318c3f703317a514959a9 | 55e23f1cbced53f7c4f59e3f0963ad580508a783 |
| verify_ctrl.json | 979 | da14fab77519d4ad7e3d4b69b6c2f4b8 | d5651d2e05fb5a83a1463d8397bcd3d6f7f9b715 | ed583311cd2ea062480a04dadb0e54fd42011a8e |

iter1 补交口径：三包＋TAU=0 对照包均 verify-pack PASS（verify_*.json ok=true、mismatches []，包 md5 与 verify 记录逐字合）；TAU=0 等价性 equivalence_tau0.json 判 PASS（5/5 检查）；conf_diffs 与注册预期逐字合（V1 +CMP_TAU=150；V2 CMP_H 1→2；V3 两者）。文件经只读快照（2026-11-16 12:20 CST）提交，提交后复核原件 md5 未变。**注意：交接尚未收口——build_record_iter1.json 与交接清单尚不存在（构建代理仍在跑，V2/V3 诊断 probe 未出）；若构建代理重建任一包，仓内副本需随交接包刷新。**

## ② 已在仓验合清单

### 后继权威集＋构建器/验包器/量化器（逐件下载 blob 算 md5，全合）

| 仓内路径 | md5 | blob sha |
|---|---|---|
| code/cnet2_comparator.py | 385190bd2f59d90f4e68a0d2dc6091a1 | 94ddfbdc0fc9324afb939080556fa755f855ee17 |
| code/cnet2_search.py | c3222eea76ea9d42c160626c2e239ca0 | a4d9284cdc60997c0927e1cca61f9e3f5aa179e6 |
| code/cnet2_search_pack.py（builder） | a1382d0ede6e7fac7e605231f214f13b | 84ff0467fbf5bae3c97697ec5e85da7486b96622 |
| code/packs/cnet2_search_pack.py（verifier） | 453f543a609328bd387171800b4cac9c | e9f131f76a8b75b8227415f7b1ed046ef9f80ac4 |
| code/packs/cnet2_pack.py（quantizer） | 18bf47b76a642aea2fd565ed9ed971f7 | b827493197aa5e588d43ee24d204254f5648ba08 |
| code/search_proto/pack_bridge.py | ad522153d32b7cbc5008ef0b59638c0f（与本地 proto/pack_bridge.py 逐字同） | 6ff18d6865d32069933c024869c2f42b4613bb24 |

### search_dossier 本地↔仓内 blob sha 逐字合（32 件，节选关键件，余同法验合）

s0a/s2 证书在仓（s0a_echo4_cost.json blob 9b510c2a、s2_echo4_search_cert.json blob 0a73e344；本地工作区无副本可对）。本地有副本且全合的包括：PREREG_ECHO4_TACTICAL_SEARCH.md（19c6bc43）、REGISTRATION_SUCCESSOR_AUTHORITATIVE_20261116.md（005768fc）、S3_ROOT_LIST_FREEZE_PROTOCOL.md（4212f13a）＋DRAFT（8764eaa9）、S3_PROTOCOL_ADDENDUM_DRAFT.md（8167b5e5）、S3_ADDENDUM_REVISION_DRAFT.md（dc95e2c2）、S3_TRIGGER_LIST_EXEC.md（656e279f）、s3_enumerate_roots.py（1218d573）、REVIEW_RECORD_S3_ENUMERATION_20261116.md（5a8fa0b7）、RULING_S3_ENUMERATION_SCRIPT_20261116.md（e4a7d6b8）、试跑包五件（TRIAL_CHECK_TABLE 0f2040f7／register 58a39ad8／report ab3828ad／roots e6e2ba14／roots.jsonl 11f0baa9）、FEASIBILITY（仓名 FEASIBILITY_ECHO4_SEARCH_20261115.md，d110befc）、AMENDMENT_S1_POSITIVE_LEG（66fb90f9）、REVIEW_SEARCH_IMPLEMENTATION（463e1ed7）＋R2（09055653）、RULING_R2（e0ebb752）＋ADDENDUM（3e1797f7）、SEARCH_LINE_STATUS（89c7875f）、SWITCH_PLAN（06868a31）、LVNET33_ACC12_REPORT（38045307）、lvnet33_acc12_evidence.json（3cc6ff72）、RESYNC_BUILD_RECORD.json（3771d156）、RESYNC_R2_BATCH3.md（f0c61d40）、PACK_DEDUP_INVESTIGATION.md（8019f4c8）、DEDUP_ROUTEB_REPORT.md（3335898c）、S1_CLOSURE_ATTACHMENT_VERIFY3_VENUE_20261116.md（8cd7e7bb）、s1_verify3_results_20261116_venue.json（ae01fd10）、WIRING_MAP.md（7c17112b）。策略组 §10.4 确认件 RULING_S3_ENUMERATION_S104_ACCEPTANCE_20261116.md 在仓（blob 6f4aa5c8，4,699 B）。

### 快筛归档（sync/h29，只核对未搬运；本地副本 blob sha 与分支逐字合）

| 分支路径 | blob sha | md5 |
|---|---|---|
| synced/h29/bigeval_live/screen_search_a.jsonl | 201f226da2663d2643a565135817173194c50d8f | aee34f7f05ea554b698a831b40ce3820 |
| synced/h29/bigeval_live/screen_search_b.jsonl | 61664858b2272f8b7d6acfc193be621f61dbd7bc | e8f58b9b54097335e94a7d6a313fa406 |
| synced/h29/bigeval_live/screen_r2_a.jsonl | d5cfa6e645209e5c7c912f30e349e4f07cb61662 | 9cdec061a7f00671216796cc09a6d342 |
| synced/h29/bigeval_live/screen_r2_b.jsonl | 8585f5d9b8686e1e2ab1f32327fe78d0dac6bdd7 | 46ad6a0e8ea3ed96a2e6d52b55eea58e |
| synced/h29/summaries/winrate_screen.json | a7a1d71f56d5976d05860b0a4ca841e91b5c7731 | 56aa7ec7f2fe05d149c1e255b6cb70fa |
| synced/h29/summaries/winrate_screen_r2.json | 5f0808dbd5459f2f595d22bfb0e779161c2936d7 | 42e54fa505e64785f34f870364405f83 |

## ③ 跳过清单＋原因

- s3_freeze/trial_out/witness_trial.jsonl（19,881,352 B）：禁区明示，非冻结 sidecar，md5 已记于核对表。
- iter1/build_record_iter1.json、交接清单：**不存在**——构建代理尚未收口（out/ 最后写入 11:54 CST 的 equivalence_tau0.json；diag 仅 probe_ctrl30/probe_v1_30，V2/V3 诊断未出）。交接包成形后需再补交构建记录＋清单。
- iter1 其余：build_*/、tests_var/、tree_auth/、tree_var/、proto_src_var/、diag/ 为构建中间树/工作副本（含 ckpt npz 重复件，单树约 23 MB）；out/*.log、diag_driver.log 为日志（排除规则）；out/probe_ctrl30.json、probe_v1_30.json 为诊断中间产出且整套未齐（V2/V3 缺），未交。
- s2_run/RUN_BUNDLE.md、verify3_rerun/RUN_BUNDLE.md 及各 *.log：运行笔记/日志，不在本次枚举范围。
- s3_freeze/probe_settle_298.py（＋.log）、file_s3_evidence.py、merge_verify_trial.json：工作仪器，不在枚举范围。注：probe_settle_298.py 被审查记录 §3.4 与 §10.4 确认件引为非冻结验证仪器（md5 86b20725），如需证据链完整可另行裁定补交至 s3_enumeration_trial/。
- resync_recheck/REREG_WINRATE_SCREEN_RESYNC_DRAFT.md：已被正式 R2 注册取代的草稿存根，不交。
- proto/、battery_rerun/、resync/、switch/、bridge_fix/、landing/、lvnet_acc/ 等本地工作树：正本在仓 code/search_proto/ 等处（pack_bridge 已逐字验合），工作副本不重复入仓；*.log、__pycache__ 一律排除。

## ④ 内容不一致（只报不交，待裁定）

1. **LVNET33_ACCEPTANCE_PROTOCOL_DRAFT.md**：本地 3,849 B（md5 5c7bdc34e11d93730bd68968ac955932，blob a9cea0bd）≠ 仓内 5,223 B（blob 42c3d257b615d9e8488d7831452421eec34671c9）。本地为早期草稿，仓内为更长版本；未覆盖。
2. **switch/dedup/build_record_dedup.json**：本地工作版 4,702 B（md5 0e7ce87bf7320cc7b0ce31a38689b909，blob 65dd74f5）≠ 仓内 6,098 B（blob 747654d5）。但本地 build_record_dedup_FILED.json 与仓内逐字同（md5 bfa77e1a3d334bd1b8c539e8e8309db0）——入仓的是 _FILED 版，工作版仅本地留存。
3. **switch/dedup/regression_evidence.json**：本地工作版 14,405 B（md5 f7ddd555da3b28a36602d15018f86306，blob 8c66a343）≠ 仓内 16,183 B（blob 9148abe3）；_FILED 版与仓内逐字同（md5 378c7bc4f02953856b27d7708cdce888）。同上。

第 2、3 项实质无缺口（入仓版已在仓），仅本地工作版与入仓版并存；第 1 项为真分叉，待裁定。
