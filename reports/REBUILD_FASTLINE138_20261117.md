# 快线重建 138 局（裁定 757d17db ＋ 修定 77623940）执行记录

## 触发与口径
快线训练 2026-11-17 18:11 死于第 55 局坏根（root 32918）。裁定 757d17db 先令排除 game 55 单局重建（145 局版已执行、另件在册）；所有者 18:38 令「一次性排除完」，策略组修定 77623940（reports/AMENDMENT_FASTLINE_EXCLUDE8_20261117.md，md5 7b066641）将排除扩为全部 8 根跨阈值根所在局。本件为 138 局版执行记录，两裁定并引。

排除 8 局（order146 下标／id／源代）：
- 16: 1c6592328becd1184b6cee0531afade5-1（gen-f3a7cf50829156bc）
- 18: 1f170a1fa6ae1bc79cf771cfe27f000f-1（gen-f3a7cf50829156bc）
- 23: 22c26fb4fc265297a69fb5ced1d15252-1（gen-f3a7cf50829156bc）
- 35: 31568ffd9e7c703957f7345206e84308-1（gen-f3a7cf50829156bc）
- 55: 4c086ee0a76d9dd2c26d5e2a13654119-1（gen-54481ed6f970896e，唯一触发根所在局）
- 68: 699fc9e24eab6af2666ea8e4f55d51d3-1（gen-54481ed6f970896e）
- 111: b4e45fac9f2e948d644d2946aabb1f50-1（gen-71f5332544b36eb8）
- 114: c0308ebb4028594b4c7182308854dbe6-1（gen-71f5332544b36eb8）
下标与 id 经 DGX 线对 order146.json 逐项直读交叉核对，8/8 全合。另加作废局 8759f8217d439cc5127f10aa2139935a-1，共排除 9 局／4,992 根。

## Merge（作战室接手流执行 18:54，DGX 线独立验合）
- 工具：钉版 merge_annot_gens_excl.py（57cc1310），--order order138.json（= order146 − 8 局，DGX 线复算逐元素合）、--exclude 9 局。
- 产物：/home/arm/junyang/k0_dist/master5s_fastline/merged138/，**gen-adebe932da1eb6f5，138 局，82,524 根**，manifest_sha256 01f40a8bf14c403d589571a23d8f84ee00ac3b19ce784f9e1cad9506b22f98ea。
- DGX 线独立验合（verify138.py）：MERGE138-VERIFY-PASS——9 局全零进入、逐局根数全 598、总根 82,524、MERGE_RECORD excluded 9 条／4,992 根、源代对账 87,516 − 4,992 = 82,524。MERGE_RECORD 工具内常量（6c98cc58／83a5db39）不动，本记录并引 757d17db＋77623940。

## Build（DGX 线执行 2026-11-17 ~19:05）
- 输入 build_src138（作战室流备：index 138 条已剔 9 局；replays 原含 146 件，DGX 线开工前将 8 局回放移置 build_src138_excluded_hold/ 保管，md5 在案：1c659232 4c0557b9／1f170a1f 0695d976／22c26fb4 0cadfb32／31568ffd d61f37d5／4c086ee0 91636a50／699fc9e2 d0b5a56b／b4e45fac 2e87eae0／c0308ebb efe55543；作废局回放原本不在）。
- 工具：钉版 LADDER_FRAME cnet2_build.py（5c2dd58e，CNET2_LADDER_FRAME=1）。
- 产物：/home/arm/junyang/k0_dist/master5s_fastline/build138/，**gen-75ed865a73fd02b1，138 局，82,524 样本，cfg_fp f982640561fc4fe4**（与 build146 同值），ladder_frame true。

## 诊断件入仓
- data/k0_calib/diag_cash8_roots_detail_20261117.json（1,866 B，md5 6b0408707fe39c6cae3a6fc99553b725）
- data/k0_calib/diag_cash_scan_result_20261117.json（1,609 B，md5 9dbaf286fa2ef3653c3a2ebfc982186c）
均自 L20 /root/carbon/katago/ 拉回 DGX 后入仓。

## Join（未执行——座位阻塞）
裁定承接的 join 座 spot-57 自 ~18:53 离线，DGX 线 19:0x 再次双路直读仍超时（tailnet 100.93.131.80＋公网 47.103.103.158 均 connect timeout）。merged138＋build138 已在 DGX 就绪；join 须改座（作战室 18:52 接手时自定 join 落 58，或新购座），改座令到即发座执行：钉版 k0_join_fallback.py（fac55acb）落座先 md5 直读复核、合才跑；验收照裁定（9 局零进入、--verify、逐局表、回退清单入 provenance），产物随后交付 L20 按 3,333 步重开训练。
