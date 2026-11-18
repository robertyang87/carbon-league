# Sapling 小预算闭环观测回执

这里的六份逐局回执来自仓内最终 `data/weights/sapling_bootstrap/{initial,full,natural,sqrt}.npz`，均经同一修正观察器实跑。`full` 是全强度角色/动作均衡；`natural` 是无均衡 BC；`sqrt` 是事后指定的一次开平方均衡消融，未做参数搜索或择优发布。训练来源和64次更新预算见上一级 `training.json`。

两块分开报告。原块 `620010000..620010003` 比较 initial/full/natural；新增块 `620020000..620020003` 比较 initial/full/sqrt。每块四张地图，各跑候选双席，共8局/臂；双席共享地图，不是8张独立地图。训练和离线验证地图仅 `620000000..620000005`，两块观测地图不参与训练；追加块是在原块结果已知后指定，不能把两块混起来伪装一次预注册显著性试验。

每局最多60次联合推进，t30/t60看行为；同一冻结 `policy_v0_5.make_agent()` 对手、标准 `cnet2_pack.py` int8、teacher=2、全部可选guard关闭、不搜索。对手与训练演示规则策略相同，因此是新地图上的行为检查，不是强手泛化测量。没有作完整终局强度或胜率认证。

## 原块：8局/臂

| 状态 | 臂 | 累计成功招募 | 累计新种树 | 当前正载采集工/当前采集工 | 正额交付次数 | 正额交付碳 | 首个采集→交付闭环局数 | 平均现金 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t30 | initial | 16 | 0 | 0/15 | 0 | 0.00000 | 0/8 | 20.0000 |
| t30 | full | 22 | 8 | 2/9 | 8 | 111.57660 | 2/8 | 13.2925 |
| t30 | natural | 0 | 0 | 0/0 | 0 | 0.00000 | 0/8 | 80.0000 |
| t60 | initial | 16 | 0 | 0/12 | 0 | 0.00000 | 0/8 | 20.0000 |
| t60 | full | 33 | 9 | 5/12 | 10 | 191.24574 | 4/8 | 14.0025 |
| t60 | natural | 0 | 0 | 0/0 | 0 | 0.00000 | 0/8 | 80.0000 |

## 新增块：8局/臂

| 状态 | 臂 | 累计成功招募 | 累计新种树 | 当前正载采集工/当前采集工 | 正额交付次数 | 正额交付碳 | 首个采集→交付闭环局数 | 平均现金 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t30 | initial | 16 | 0 | 0/13 | 0 | 0.00000 | 0/8 | 20.0000 |
| t30 | full | 16 | 4 | 0/8 | 3 | 57.40935 | 3/8 | 24.0925 |
| t30 | sqrt | 16 | 15 | 0/8 | 7 | 241.95375 | 7/8 | 20.4838 |
| t60 | initial | 16 | 0 | 0/10 | 0 | 0.00000 | 0/8 | 20.0000 |
| t60 | full | 32 | 8 | 6/8 | 16 | 300.87474 | 6/8 | 9.6325 |
| t60 | sqrt | 19 | 18 | 6/11 | 12 | 451.13625 | 7/8 | 28.4962 |

表中招募/树/交付为8局总量；现金为8局均值。新块 initial 的 `620020003/seat0` 在t32灭绝，因此该局t60列明确沿用终止时快照（`state_t=32`、`early_terminal_carried_forward=true`），没有终局后继续推进。其余47局到60步截断。自然BC的现金80来自从不招募、没有产出，不能按现金高判为更强。正载工是瞬时快照，已成功返航卸货也会使其归零，不能单独作为能力弱的判断。

原块 full 在2/4地图的双席共4/8局形成闭环；新块 full 为6/8，sqrt为7/8。这里只证明部分基础行为出现，sqrt仍有一局未闭环。这些小样本不足以认证稳定经济循环、收益提升或优胜模型；两个块和全部臂均保留，没有删掉自然BC的失败表现。

## 事件定义与复核

成功招募/新种树来自模拟器真实单调创建计数，包含后续死亡/被夺；夺树不计为新种树。正载工来自快照中采集工cargo>0。采集事件直接观察未修改 fastsim 实际增加cargo的执行行。正额交付要求实际执行了金额>0的本方工人→本方基地交付；敌方赠款和基地碰撞入账另列，友军合并不按“消失ID”误记损失。

闭环按当前携货来源记录：实际采集后，同工或仍持正货的友军接手者向本方基地实际正额交付。交空（含交至敌基地）、货物耗尽或工人死亡立即清来源标记；敌方货物转移不继承己方采集来源。此定义是行为事件历史，不是按每一单位碳做来源守恒核算。每场闭环指标只问是否出现过首个符合定义的闭环，原始事件次数单列。

自验包括有/无观察器逐步状态一致、友军合并30碳完整交付，以及“采完交空→只取得敌货→再次交付”必须是两次交付但仅一次采集闭环。此前观察器用终生标记的问题已修；修正后48局全部重跑，逐局最终状态哈希、双方t30/t60闭环布尔与原观察均完全一致（`observer_regression.json`），随后再用仓内最终权重全部重跑并核一致。两次重跑是对相同样本的工程核验，不能增加统计样本量。

`index.json`列六份回执SHA、权重SHA、实际包MD5/SHA、总调用/推进/错误及成本；各局含候选与对手座位、双边快照、双边调用与错误、截断/灭绝状态。最终这批为2852次联合推进、5704次agent调用，双方异常均0。早期灭绝在终局时停止；异常则保存诊断并停，不用空动作续局。

## 身份与成本

初始包MD5 `8a397c59022624d7cb4c5f06c923fb33` 与提交 `358063d1` 的 `reports/EXAM_SAPLING_VS_MASTER3S2_20261119.md` 考生完全一致：teacher=2、837152B。同机另有早期teacher=1文件，但它不是该考试包。

全均衡和自然BC曾先用早期导出的容器测量，最终源码导出修改了元数据，导致NPZ/包哈希不同。`weight_equivalence.json`独立读回确认各臂89个数值张量逐位相同、cfg相同、量化payload相同；`training.json`亦记录来源。为使外部重打包直接对上，当前公开回执已全部换成实际读取仓内最终NPZ重跑的结果，不仅靠等价性推断。

计量成本分开保留：首次诊断游戏CPU 23.921秒；修正观察器复跑 24.927秒；用最终归档权重复跑 23.823秒，游戏CPU合计 72.672秒。逐局wall加总分别 23.967/24.980/23.845秒（不是并发任务端到端历时），全部不含打包/自验/训练。每次都重复同48局，不增加独立样本数。

## 从仓库根目录重跑

需要NumPy。`--cache`只保存本次临时打包件；`--out`是新目录，不会覆盖既有回执。已有`--protocol`必须与源文件SHA、种子和配置一致，否则硬拒绝。两臂/三臂都通过重复`--weight 标签=权重路径`指定；`--seeds`最多四张地图，每张自动跑双席。

```sh
python reports/sapling_bootstrap/smoke_ab.py --self-test-only
python reports/sapling_bootstrap/smoke_ab.py   --weight initial=data/weights/sapling_bootstrap/initial.npz   --weight full=data/weights/sapling_bootstrap/full.npz   --weight natural=data/weights/sapling_bootstrap/natural.npz   --cache /tmp/sapling-smoke-cache-a --out /tmp/sapling-smoke-result-a   --protocol reports/sapling_bootstrap/receipts/original_protocol.json
python reports/sapling_bootstrap/smoke_ab.py   --weight initial=data/weights/sapling_bootstrap/initial.npz   --weight full=data/weights/sapling_bootstrap/full.npz   --weight sqrt=data/weights/sapling_bootstrap/sqrt.npz   --seeds 620020000 620020001 620020002 620020003   --cache /tmp/sapling-smoke-cache-b --out /tmp/sapling-smoke-result-b   --protocol reports/sapling_bootstrap/receipts/fresh_protocol.json
```

重复运行的时间字段会变化，故整个回执文件SHA不会固定；可逐局比较`final_state_sha256`及行为统计、权重/包哈希。研究观测器改变测量开销，不改变模拟转移；这里的时长不能当作1.5核/512MiB平台认证。
