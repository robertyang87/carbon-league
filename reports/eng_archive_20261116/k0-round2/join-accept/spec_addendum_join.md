
---

## 附记 2026-11-14 16:4x（策略组）——补件 4 k0_join 验收 PASS＋三项登记

对象：`code/k0_join.py` **78363299a7da04e09724c7327f0d1c86**（31,424 B，commit bf2aac72，单文件）。策略组自 main 读回验 md5 逐字合，亲读连接键、哨兵翻译、producer 双绑与消费者 preflight 代码段。验收 fixture 按派单定义达成且有余：标注器真实产物 gen-1dcc9611575bfbe3＋build gen-75de8ffbdd421e42 → join 产出 gen-f18f59064c91e5cd（与原实现产物同 id、18 字段＋meta＋producer 全等）；真消费者 `k0_distill.load_annotation` 发布前 preflight 全门通过；S、C 两臂短训均落盘（S 臂 loss/batch sha 与首跑逐字同，C 臂补跑完成）；负例三件（seat 篡改／缺特征根／真实臂植 −2）全 die 点名、零发布。派单四职责逐项对上：特征按（局名映射，seat，step）1:1 连接、席位一致性验（X06 两半实证：真实根集每 (gid,step) 恰双根、双席行 206/505 不串；跨代只走 games 表局名、gi 互换实验输出逐字节不变）；双 bundle 打包（game:=标注 gid、games 映射表入 meta）；哨兵 −2→−1 逐根断言（填充块须整齐 −2、真实臂内 −2 die、译后全域扫残留、计数入 meta 与 producer）；producer 双绑两源代（gen id＋manifest sha256，build 另带 cfg_fp）＋教师身份＋本件 md5。**判定：PASS。§10 三件＋补件 4 至此全部验收完毕，K0 第二轮代码前置清零。**

三项登记（随验收生效）：
1. **哨兵翻译首产核对**：真实 12 根 fixture 无填充根（n_arms 全 4、−2 计数 0），−2→−1 翻译路径至今仅由 doctored 正例承载（entries=12、残留 0）。首个生产 join（即 shard1 标注代的 join）回执必须人工对账翻译计数（翻译条数 vs 标注侧填充根／填充臂计数）并报告根数规模与耗时——生产规模（全窗口根数、大局席位段切分）亦以该趟为首验。
2. **X02 蒸馏侧 train_temp 缺口（归属 k0_distill，不在 join）**：标注器两处记录 train_temp、join 透传封印（joined meta.teacher.config 可溯源）；蒸馏侧全文无 train_temp、目标以 --q-temp 重构，train_temp≠1 的生产代会被按未 temper 消费（实际训练分布≠认证分布）。修复在工程组 X01 线以独立提交在飞。临时口径（本登记生效至 X02 修复验收）：K0 生产标注代一律以 train_temp=1.0 出代与消费；任何 train_temp≠1 的代不得送蒸馏消费；预跑／标定的运行清单须记录其消费的 train_temp（自 joined meta 读）。
3. **join 镜像维护义务**：join 对标注／build 两侧记录门的验封是本地同口径重写、非直调 `k0_annotate.verify_generation`（功能等价、docstring 已注明）。今后任何经策略组验收的 k0_annotate 记录门变更，其验收清单必须含 join 镜像同步检查一项——列入常设验收清单。

前置状态更新：K0 第二轮五前置中，(1) v3 回执填实、(2) 正式 split＋条件 E 已关；本件关掉 (4) 的代码部分——(4) 余下为数据生产（shard1 标注＋生产 join，受登记 1 管）与 σ̂ 预跑／N 锁本身；(3) 正式 q 闸结果在途（Studio 执行中）；(5) §6.4 标定清单未冻。本验收不发任何运行授权；预跑／标定的执行次序照既定前置列（q 闸判读在先、开跑另行裁定）。

附带读回观察（不在本件裁量内）：main 上 `code/k0_train.py` 现为 4d5b3b67（65,216 B）、`code/exit_merge.py` 现为 49c32b25，均已非策略组验收基线（0253817b／2146afb9）——应为 W01/W02、W05（及 r4 X 线）落仓待送验；其验收交付到达前不作基线引用，基线台账以各自验收为准更新。
