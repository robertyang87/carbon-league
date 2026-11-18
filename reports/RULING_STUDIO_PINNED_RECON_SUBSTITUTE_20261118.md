# 裁定：Studio 钉版全链件顶替正式环前置对账，销项（策略组，2026-11-18）

- 请示来源：作战室转发 Studio 冒烟件与工程组 §4 存疑 2 对账关闭报。前置出处：裁定 reports/RULING_KATAGO_CORE_U_HEAD_20261118.md §3（正式环开跑前以钉版六件套复跑一局对账）。

## §1 事实

- Studio 以钉版六件套（cnet2_search 94f7fb64，非 main 版）＋适配器 f22d9289，对试点局 300002（sha256 ed1c735a）走完四环：build RC=0（598 样本 exact 598／approx 0／unusable 0）；annotate RC=0（60 根、files_verified 14、config 79c44c1f 以 teacher_config 原函数独立复算逐字合、单价 5.87 秒/根）；join RC=0（60 根／960 候选、消费器预检 PASS、哨兵翻译 0）；预扫 RC=0（60 根 0 问题）。
- 工程组存疑 2 对账关闭：钉版 search 语义正确（q 为分布、priors 和 1、Q 为每臂值），与 main 版差异不在 search。

## §2 裁定：可顶替，前置销项；工程组两臂对账停恢复

- 前置的功能目的有二：① 确保正式环全程用钉版语义且全链无碍；② 接口冒烟（main 版 search）结论可迁移到钉版。Studio 件两项均达成——钉版全链四环 PASS、config 独立复算合、search 语义核验正确，且适配器同链在钉版上完整跑通即迁移验证。
- 工程组原协议的两臂逐字段对照（max 差／flips）回答的是「两版数值差多大」；正式环全程只用钉版、不混用 main 版，且差异已证不在 search，该数值差不构成正式环正确性风险。**两臂对照降为可选诊断**（后续若有人质疑冒烟结论迁移性再补），工程组可停恢复、检查点释放。
- 试点局不同（300002 vs 冒烟局 987001）不影响销项：对账验的是钉版链路与语义，非某一局数值。

## §3 平台缺陷口径（APFS 大小写不敏感撞名）升为全线口径

- Studio 实证：普通 APFS 大小写不敏感使 Q.npy／q.npy 撞名（manifest 两字段 sha256 逐字同、.npy 13/14），annotate 自验与 join 死在「q sums to 14.253, not 1」；与 2026-11-17 shard0 封印同款缺陷，test118 六批 RC=1 亦同因——**分片数据本身全对、Linux 侧验合闭合不受影响**，此归因收讫。
- 全线口径：凡在 macOS 上跑 annotate／join，out 目录必须落大小写敏感卷（Studio 已建卷重跑全通）；Linux 座不受影响。请作战室同步各线并入 SOP 教训。
