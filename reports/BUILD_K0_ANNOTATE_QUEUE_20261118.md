# 建成回执与审查：标注优化第二批（k0_annotate_queue）

日期：2026-11-18。设计：`reports/DESIGN_ANNOTATE_BATCH2_20261118.md`（commit a35ec834）。状态：**建成闸全过、工程组审查通过；VALIDATION-ONLY，待验证座实测与策略组验收，生产链未动。**

## 交付件

| 件 | 仓内路径 | md5 | 规模 |
|---|---|---|---|
| 候选标注器 | `code/k0_annotate_queue.py` | d40e060c74fe9624ab80e2fe742cf3ca | 1443 行 |
| 逐根对账器 | `tools/compare_shards.py` | 9839fed9109191999a93fe4ca19d4e96 | 266 行 |
| RNG 流记录探针 | `tools/rng_stream_probe.py` | 034b988ae599a007ef59f349a09ee957 | 193 行 |

## 静态闸与身份

- 与 snapshot 底本（e9c7a0fb）diff 恰 5 个 hunk，与设计 §5.1 五处一一对应：①code_md5s 键名随自身文件名；②`_annotate_game` 头改 `_assemble_shard`（装配/写盘段逐字未动）；③根级 partial 账本＋队列机器整段插入；④publish producer 查找键改新文件名；⑤main 分发段改队列入口。五处外逐字（主代理抽验 hunk 位置与计数合）。
- code_md5s 自读回：annotator 键＝`k0_annotate_queue.py`＝d40e060c；六件共享逐件＝钉版值（teacher c6ade5ce、search 94f7fb64、np d8c15ad4、feat 2a45b08b、fastsim aefe6c83、gate e922d9d8）。配比：生产者 max(2, workers//8) 封顶、根队列界 4×评估者数、partial 账本每 16 根或 5 秒刷盘。config 函数未入任何调度字段。
- **O-1 身份核查结论**：直读 k0_join（fac55acb）/k0_join_u（9054aadc）全文——两者验标注代只查指针哈希、封印、{config_sha256, weights_md5} 交叉核对与逐字段记录闸，**不读 producer code_md5 的键名**；code_md5s 的真实消费者是标注器自身的 identity 整字典比对。故新键名只产生预期的新 generation 隔离，无固定键名冲突。
- 审查裁量（主代理裁定接受）：publish 的 `"k0_annotate_md5"` 字段标签与 docstring 用法行保留原文——前者是字段名非文件名、下游无人按名读，改之即越出五处范围；后者从逐字纪律。

## 建成自验读数

- 双跑 plan（300002+300003、窗 [200,299]）：config/selected/games/replay_sha256 与 n_roots **五项全 MATCH**（主代理抽验合）；候选实跑 config_sha256 与钉版同参逐字同。
- 小规模实跑对账（**缩样**：本 VM 单根成本实测 eval(步2)≈294 s、extract(步200)≈1453 s，深窗样本破 45 分钟闸，改浅窗 [2,4) 4 根）：priors/q/Q/Q_std max|Δ| **全恰 0.0**、flips 0、arms/belief/padding 差 0、逐数组 sha 全等。限定：浅窗走通同一 scene 快照＋队列＋装配路径，但深步前缀行为本机未证——其据为 Point A 对逐字继承的 `_game_root_scenes` 的既有恰零验证（cae8fcbb）＋待验证座框 A 全窗 [0,299] 补齐。
- 框 B RNG 流探针（同根集）：逐根 16 股流（belief 种子 [seed_base+m, 0xC0]、对手 seed_base+m、seed_base＝7000+step）钉版路 vs 快照路逐股全等（种子/次数/digest/sha256）；**B3 阴性对照 seed_base+1 后 16/16 股哈希全变**，探针确实接在流上。
- **杀 worker 冒烟抓真 bug 并闭环**：首跑 kill -9 评估者后主进程自杀（rc=1）——定位为死亡分支从不把死者移出 `evaluators` 字典、每 tick 重复计数同一尸体撞顶 6 次。修复一行 `del evaluators[wid]`（在五处范围内的队列机器段，diff 仍 5 hunk）。重验：只重生一名、done_roots 逐根补齐、rc=0、发布 generation id 与未受干扰跑逐字同（内容哈希身份＝恢复字节级一致）、分片对账仍全 0.0、直调 `_verify_shard` 通过。
- 事故如实记：本机 VM 重启曾吞首趟探针约 25 分钟计算（/tmp 证据），后改分块落盘重跑通过。

## 验证协议计时修订案（设计 §6 数字作废）

- annot_a 实测全波 **4h38m55s**（钉版、c8a、50 局单波），墙时≈最长局约 4.65 h。设计「全波约 3–3.4 h」低估约 40%；「钉版 12 局约 0.8 h」是量级错误——按局 map 下墙时由最长局决定、与局数无关。
- 修订口径：框 C 钉版基线不重跑，改用 annot_a 已录逐局时长（分片 mtime − 起跑时刻；口径：同档异座，跨座硅差未单独标定）；框 C 只跑候选形态。
- 修订总账（估计）：框 A 候选 12 局全窗约 0.8 h；候选 50 局整波约 3.2–3.8 h；框 B 约 0.4 h；对账成文约 0.5 h；**①单独形态（snapshot 按局 map）12 局约 3.9 h**——此档为总账主成本：含它总计约 8.8–9.4 h（估计），去它约 4.9–5.6 h（估计）。原约 3.5 h 验证窗作废。
- **O-4 复核请求（策略组裁）**：①单独档的成本在形态不在局数（3 局也要约 3.9 h），其价值仅在三路归因。工程组建议**裁减**：候选对钉版基线的总效应已由框 A＋50 局整波给出；若候选增益不及预期，归因可由逐根计时后补。保留与否请策略组按新数字裁定，验证 manifest 冻结时附 annot_a 逐局时长表重算钉死。

## 后续

数据组按建成回执购 c8a 验证专机；验证执行按设计 §6（经本修订案）＋冻结 manifest；结果回工程组审后转策略组验收。
