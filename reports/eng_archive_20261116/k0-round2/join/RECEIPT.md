# K0 第二轮补件 4（join 打包步）回执 — 收尾线

前任代理实现并跑过大半验收后未落仓未写回执。本线盘点工件后**亲自复跑全链**（不采信前任日志），再落仓。

## 落仓

- 新文件：`code/k0_join.py`，md5 **78363299a7da04e09724c7327f0d1c86**（31,424 B），commit **bf2aac72a4139d3a55227eb9e90df9bc7920b97d**，API 读回逐字节一致。
- 仓内仅此一个新文件。`src/` 内其余副本经逐个 md5 对照 main：k0_annotate `cdee7c03`、k0_distill `d5fc464f`、cnet2_build `8598f26d`、exit_merge `49c32b25` 与 main 一致；`src/k0_train.py` 为旧的 W01 中间版副本（61ad3e2e，main 现行 4d5b3b67），只读且不在 join 链内（join 只 import k0_distill）。复跑环境用的是经 md5 验过的 main 现行字节（k0_distill/cnet2_train `a3ad86d4`/cnet2_model `3bb5857f`/cnet2_feat `2a45b08b`/cnet2_build）。

## 实现审计（对 src/k0_join.py 全文）

四项职责俱在：① 双输入只经 current 指针消费、直指封存代拒收，两侧各自验封（标注侧 manifest self-seal＋meta seal＋教师身份交叉核＋逐字段 sha/行数＋§13-C 记录门；build 侧 pointer/manifest 对应＋逐字段 sha256＋meta seal），特征按 (game 映射, seat, step) 1:1 连接，缺行/席位段歧义/无此席均 die 点名根；② policy/value 双 bundle 打包（game:=标注 gid），games 映射表入 meta 留痕（game/gid/build_gi/build_key/n_roots）；③ 哨兵 −2→−1 逐根断言（填充块须整齐 −2、真实臂内 −2 die、译后全域扫残留），翻译计数入 meta 与 producer；④ producer 双绑两源代（各 gen id＋manifest sha256，build 另带 cfg_fp）＋教师身份（config hash/weights md5/belief）＋本件 md5。发布前经真消费者 `k0_distill.load_annotation` 对暂存字节 preflight 全门，发布走内容寻址＋验后复用不覆盖＋指针最后翻转。
一处文档口径注记：文件头 docstring 称标注侧「k0_annotate.verify_generation 校准」，实现实为本地按同口径重写校验（未 import 标注器）；build 侧同理。功能等价，但标注器记录门口径将来若变，join 侧须同步镜像。

## 全链复跑（本线亲跑，标注器真实产物起）

两源代身份：标注 gen-**1dcc9611575bfbe3**（manifest sha256 6fca0cad…，教师 config hash 8b758f83…、weights md5 06a03d9c…）；build gen-**75de8ffbdd421e42**（manifest sha256 e5611e45…，cfg_fp fa04d3614c821115；由 gen1r_smoke 两局回放经 cnet2_build 小代，为 fixture 输入）。父代权重 ~/workspace/s12work/heads_v2.npz（md5 06a03d9c…，本线按 md5 自行定位，与产物记录的教师 weights md5 合）。

1. **join**：产出 gen-**f18f59064c91e5cd**（12 根、48 候选、翻译 entries 0）——与前任产物**同一 gen id**，且 18 个字段文件 sha、meta.json 字节、manifest 核心（除 created/self）、producer 逐项全等。
2. **消费者加载**：`k0_join --verify` 过（12 根/48 候选、ipw 未激活）；join 内置 preflight 同样通过。
3. **S 臂短训**（bs 4、2 updates、seed 7）：update 1 loss 2.6606968109857347、batch sha 99067fd4…与前任运行逐字相同；检查点 train_S.npz 落盘（saved sha256 35172bbe…）。
4. **C 臂短训**（--flatten-q，**前任未跑、本线补跑**）：2 updates 完成，train_C.npz＋complete.json 落盘。

## 负例复跑

- **seat 篡改**（annot_seat_flip，根 #0 seat 0→1）：拒，die 于标注侧重复根门——「duplicate root (gid, step, seat) = (0, 206, 1)」，零发布。
- **缺特征根**（前任无此 fixture，本线新造 build_missing_row：自 build 代删根 #6（gid 1、step 206、seat 0）对应的 build 行 804、重封字段表/total 1196→1195）：拒，die 点名——「root #6 … no feature row in the build generation … for this root」，零发布。
- **哨兵残留**（annot_real_neg2，真实臂内植 −2 一格）：拒，die——「sentinel -2 inside a real arm (candidate 0) … corrupt record」，零发布。
- **正例填充根**（annot_pad_root，根 #0 改真填充根 n_arms 4→3）：join 成功 gen-623404a90d056b58，翻译 entries=**12**、roots_with_padding=1、候选 47；产物 policy bundle 中该根填充槽整齐 −1、全 arms 残留 −2 为 0，消费者 --verify 通过。

## 真实产物的哨兵形态（负例盘点结论）

标注器真实 fixture（gen-1dcc9611575bfbe3）：12 根 **n_arms 全为 4=K，即零填充**；arms 内 −2 计数为 0（填充与真实臂皆无）；−1 出现 16 次，均为真实臂内的合法分量值（消费者「填充臂全 −1」指候选槽整臂，非分量）；Q 无 NaN。结论：**−2→−1 翻译路径未被真实产物触达**，其正确性目前仅由 doctored pad_root 正例承载（§13-C 一致重封、封印有效、内容缺陷隔离）；生产标注出现 n_arms<K 的根时该路径才首次吃真实数据，建议首个生产 join 对翻译计数做人工核对。

## r4 复审三项逐项核对（X02/X06，落仓前追加）

**(2) 连接键含 seat——现实现满足，实证。** 有效连接键＝（按名映射的局，seat，step）：join 先按局名配对，再在 build 局内按 side-major 行序切席位段，逐根以 (seat, step) 查行。真实根集 6 个 (gid, step) 组每组恰两根（双席各一）——裸 (gid, step) 处处二义。本线独立重导（不用 join 代码）核对：12 根的 joined 特征行与（名映射局，seat，step）对应的 build 行逐项相等；(gid 0, step 206) 双根分别落 build 行 206（seat 0）与 505（seat 1），互不串席。

**(3) 跨代 gid 不裸对应——现实现满足，实证。** 配对只走 games 表**局名**（build key 去后缀 ↔ 标注 game 名，标注表为权威 gid 空间）；build 行只用 build 代**自身** gi 在其代内选行（代内使用）；输出 gid 改写为标注 gid；映射全痕留 meta.games（game/gid/build_gi/build_key）与 producer.sources（双代 gen id＋manifest sha256）。本 fixture 两代 gid 恰同值（同 feed 序）正是 r4 警示的陷阱形态，故本线另做 **gi 互换实验**：build 代两局 gi 一致互换（gid.npy 值与 meta gi 同步换、重封）后重跑 join——18 个输出字段文件与正常跑**逐字节全等**，meta.games 如实记互换后的 build_gi（新 gen id 仅因源 manifest sha 与映射记录变化，属内容寻址应然）。证 join 不依赖跨代 gid 相等。

**(1) train_temp——join 侧记录如实，缺口归属 k0_distill.py（按令未改）。**
- 标注器在 manifest producer.config 与 meta.teacher.config **两处**记录 train_temp（config_sha256 绑定），本 fixture 值 **1.0**（behavior_temp 同为 1.0）。
- join 对温度无专门代码（全文无 temp 字样）：不读、不验、不改；标注代 meta.teacher 整块（含 config.train_temp）原样透传进 joined 产物 meta，并经 joined manifest 的 meta_sha256 封印——**温度记录在消费者-facing 产物中留存且可溯源**（producer.sources.annotation.manifest_sha256）。注：joined producer 块本身未复述 train_temp，记录只在 meta.teacher.config 一处。
- 消费侧：`code/k0_distill.py` 全文**无 train_temp**，目标一律以其自身 --q-temp（默认 0.05）由 Q＋priors 经 construct_target 重构、stored-q 交叉核亦用同一 q_temp。本线两次短训的 effective 记录 q_temp=0.05、与产物 train_temp=1.0 无冲突（1.0 即未 temper）。按治理文档（STAGE1_TEACHER_GATES §「按 q ** train_temp 归一化分布计算，结论只覆盖该分布；每个结果头记录 train_temp 与 certified_target」），若生产标注以 train_temp≠1（如 2）出代，蒸馏侧将按未 temper 口径消费——**实际训练分布≠认证分布的缺口在 code/k0_distill.py（消费侧），不在 join**；join 侧能做的记录义务已尽（值在产物内、封印在内）。本链 fixture 全程 train_temp=1.0，验收结论不受此项影响；该项为前瞻暴露，首个 train_temp≠1 的生产标注代出现前须由蒸馏侧或运行登记侧关闭。

## 未竟与如实注记

- fixture 规模仅 2 局 12 根，生产规模（全窗口根数、席位段切分在大局上、耗时）未验。
- 标注侧记录门为 join 内重写镜像（见审计注记），非 import 复用。
- W 轮拾取基线无变化（k0_distill d5fc464f、k0_annotate cdee7c03 未动）；join 为新文件，不与在飞修复线相撞。
- 复跑工件：~/workspace/k0-round2/join/rerun/（env＝main 字节、joined_out、joined_giswap、build_missing_row、build_gi_swap、train_S/train_C 检查点与日志、doctor 脚本两件）。
