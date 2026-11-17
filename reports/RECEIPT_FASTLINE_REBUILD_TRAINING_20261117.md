# 快线重建训练完训回执（L20 线，2026-11-17）

署名：L20 线。承接裁定 757d17db 与修定 77623940（AMENDMENT_FASTLINE_EXCLUDE8）。

## 训练

- 开训 2026-11-17 19:01:14 CST，完训 20:08 CST；3,333/3,333 updates，elapsed 4,012.3 s（1.204 s/步）。
- 命令：`bash /root/carbon/katago/code/train_pilot.sh /root/carbon/data/master5s_fastline_joined138 /root/carbon/katago/top1_fastline_student.npz 3333 128`
  内层 argv：k0_distill.py（2728fee6）--parent 06a03d9c --kl-beta 1.0 --lr 0.0005 --bs 128 --seed 7 --save-every 100 --device cuda。
- eff 行验合：arm S、teacher_config_hash 3cc2e67e66cff9a1d901b663a579abc2d62246d6bdf76017f00bcf998aea1ff3、data gen-56e91737a08c5fb6、roots 82,524、candidates 1,293,795。
- Final student：top1_fastline_student.npz，13,994,764 B，md5 **41bc69e6e3c8d75d0a541944a50a13c2**，sha256 8f9d2a685a799d50545dc2302d9763f31dbe0fedbd372fe65496287d831775a6。
- 末段 loss：update 3333 total 3.7277 / ce 3.2322 / anchor_kl 0.4955。训练日志中拒训（refusing to train）出现 0 次。
- 正式包：top1_fastline_int8.py，4,445,189 B，md5 c46f5ba34d5f330268229045d69b162a（cnet2_pack.py 18bf47b7；打包管线已于训前用 died-600 证据检查点预验：bigeval 2 局 FAIL=false、err=0）。
- 死亡 600 步检查点证据保全：top1_fastline_student_died600_evidence.npz（md5 9c062a8bd7e2fa686943dae7e214f98e，日志三件套同前缀保全），按裁定废弃、未续训。

## 语料代与重建链

- joined：gen-56e91737a08c5fb6，manifest_sha256 b91e6f2f04de34c978cb1968f32d7019b39cb7d7e163c5aaffd36726b1b81016，138 局 × 598 = 82,524 根，candidates 1,293,795，join --verify PASS（执行侧回执）。
- sources：annotation gen-adebe932da1eb6f5（merge 57cc1310，排除作废局 + 名单 8 局共 9 局）/ build gen-75ed865a73fd02b1（5c2dd58e，cfg_fp f982640561fc4fe4）；join 工具 fac55acb。重建由作战室在 spot-58 端到端执行，REBUILD_RECORD 在 DGX k0_dist/master5s_fastline/ 并引 757d17db + 77623940。
- L20 身份闸直读：138 局 id 唯一、逐局 598 根、作废局与名单 8 局零进入、teacher config 全窗 3cc2e67e…a1ff3、weights 06a03d9c。
- **三代 manifest 出处**：joined 代 manifest/meta/current 随本回执入仓（data/fastline_rebuild/）；merge/build 两代 manifest 原件在作战室 REBUILD_RECORD 链，由作战室补入（策略组 2026-11-17 19:05 确认此处置）；joined meta 内嵌两代 manifest_sha256：注解 01f40a8bf14c403d589571a23d8f84ee00ac3b19ce784f9e1cad9506b22f98ea、build 7f2e8d46ccb1b362f48e717c8eaaaecde5df3b904c9f61eca654b043d1e2eb81。

## 预扫（所有者 19:10 硬闸的本线落实）

- 对在训 joined 代 gen-56e91737 全集逐根预扫（k0_distill._root_cash_set 判据，82,524 根全扫、无抽样）：inconsistent 0 / straddle 0 / tripping 0。报告 data/fastline_rebuild/prescan_joined138_20261117.json（md5 7ec67ff284654da60bb813d8560027a6）。

## 随件入仓的诊断明细

- data/fastline_rebuild/diag_cash_scan_result_20261117.json：原 joined 代 gen-82a26d66 全扫原始结果（87,308 根；straddle 8、tripping 1）＋ pilot 对照代全扫（straddle 0）。
- data/fastline_rebuild/diag_cash8_roots_detail_20261117.json：8 根跨阈值根逐根复验明细（局 id / step / 现金集 / slot 10 招募候选数 / 是否触发）。
- data/fastline_rebuild/joined_gen56e91737_current.json / _manifest.json / _meta.json：joined 新代三件。

## 读数（进行中）

- 次读数（对父代 06a03d9c，σ̂ 块 991000–991049，k0_duel bead6367，同诊断口径）已由 L20 点火：并发实测 4×20 单局 3.62 分钟 vs 基线 2.67 分钟（劣化 +35.6% 超 20% 线），按作战室裁定退回 3×20 续跑，聚合结果另行入仓。
- 主读数（对 Echo4 819250bf，2,000 局，种子块 20350101–20352100，带注册 §0 上限口径）由作战室排三座分片执行，包件即上方正式包。

落字：L20 线，2026-11-17。
