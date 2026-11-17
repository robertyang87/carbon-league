# 快线主读数分片预注册（2026-11-17）

TOP1 快线（Master5s 全窗标注）验收主读数分片预注册。标注完成前落字，读数派单时按此执行，不再临时分段。

## 读数口径（注册绑定）

- 候选：快线学生（L20 训练产出后打包的 int8 包）；对手：冻结 Echo4（包 md5 819250bf，供件已预置三座 /root/carbon/readout_supply/）。
- 规模：2,000 局；种子块 20350101–20352100（与既往一切种子域零交集，派单前机械复验）。
- 判读：配对 ΔU + t CI；次读数对父权重 06a03d9c。
- 工具：bigeval.py 钉版 354bcd1b。

## 分片（667/667/666，连续无交叠无缺口）

| 座 | 种子段 | 局数 |
|---|---|---|
| spot-44 | 20350101–20350767 | 667 |
| spot-46 | 20350768–20351434 | 667 |
| spot-47 | 20351435–20352100 | 666 |
| 合计 | 20350101–20352100 | 2,000 |

- 三段首尾相接：20350767→20350768、20351434→20351435；并集恰为全块。
- 执行形态：学生包落盘后三座并行开跑，与次读数并行（作战室后链并行化令）；精确 argv 以读数派单为准，供件已预置、路径已定死。
- 任一座不可用时：其段整段改派另一座并在本文件追记修订，不许拆段混跑。

## join 产物路径（与 L20 对齐件）

- DGX 侧 join 产物目录：/home/arm/junyang/k0_dist/master5s_fastline/joined/
- 交付 L20 固定落点：/root/carbon/data/master5s_fastline_joined/（L20 预置开训命令 <JOINED_DIR> 按此定死，join 落字时零改动开训；L20 确认回执另记）

## merge/join 工具在位（2026-11-17 12:34 DGX 直读验合）

- merge_annot_gens.py：/home/arm/junyang/merge_annot_gens.py，md5 821fa78009e24986c15cabfb94cb01dd（钉版合）
- k0_join.py（钉版）：/home/arm/junyang/k0_join_patched_f9ee.py，md5 f9ee005af72855bf2d92f278e1042976（与仓内 main code/k0_join.py 逐字合，API 直读验）；快线 join 调用此路径。carbon/code 内的 27c11e16 为旧 r4 副本，非快线钉版，不用。
- cnet2_build.py（LADDER_FRAME 钉版）：/home/arm/junyang/ladder_frame_build/cnet2_build.py，md5 5c2dd58e734ccab8d71c5a6fce51dde9（与仓内 main 逐字合）
- DGX 产物目录已建：/home/arm/junyang/k0_dist/master5s_fastline/joined/
