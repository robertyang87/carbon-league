# Round 4 runtime 复核

基线：`5463f504048bcca28868ebce0a063ac8d6c51b8e`；参照 `reports/REVIEW_FIXES_20261113.md`。业务源码只读；未部署、上榜、训练、运行远端任务或大规模性能测试。Python 3.12 / NumPy 1.26.4，BLAS单线程；所有结果在本目录。

## 结论

**C03、bigeval 小样本健康记录及特殊 seed 数量问题均通过当前真实包/短局探针。** 旧报告的这些缺陷不能继续标为未修。新整合有两项需要区分的风险：唯一生产打包路径带入旧 search 副本（限定配置下的潜在回归）；运行手册承诺的全套资源认证尚不能由其指定工具完成。

### P2 / 配置条件风险：新唯一生产源包含未合并已审查修复的 search 副本

位置：`reports/live/pack_submit_runbook.md:28`；`code/packs/cnet2_search_pack.py:135–138`；`code/packs/cnet2_search.py:49–55,935,1100,1176,1365,1447,1488`。

手册规定未来打包只用 packs 下的 packer。该 packer 从自身 HERE 读取模块，实际嵌入 `code/packs/cnet2_search.py`，不是权威 `code/cnet2_search.py`。六个副本中五个完全相同，只有 search 不同：旧版没有 R12 enc_for/完整缓存签名、终局判断/冻结与 prev_cash reward，并保留两处 left+1；C01 的全终局深度修复也不在旧副本。

动态证据（`pack-source-results.json`）：分别从两个入口以同一历史 b128 权重重新打包，生成包中的 search SHA 均准确等于各自来源。真实 TTA 推理后，按各自实际 consumer 取编码：权威 `enc_for` 与原坐标编码误差 **0**，旧 `last_enc` 路径误差 **0.5873794555664062**。因此将来的 TTA+guard/search 配置会再次配错坐标编码，搜索启用时还会绕开已修终局逻辑。simulator 子审查也复现：根步全终局主文件停101并选NONE；旧副本预算断时 shallow1 返UP，预算足够仍推进到105；step298边界主停299而旧副本走300。对应独立证据 `../simulator/search-boundaries.json`。

**不能扩大结论：** n1r reference 的四份内嵌源码与 packs 副本逐字相同，保留 reference 的历史字节是合理的；当前手册固定13键为 SEARCH=False，未开启TTA。两种新打包器在这一标准13键配置下的同一b128两步动作完全一致、errors=0；没有证明当前 n1r/Echo3 配置已发生上述运行错误。问题是把冻结历史实现同时指定为唯一未来生产源、却没有修复状态边界，不是要求改写历史参考包后沿用旧hash。

建议：保持 n1r reference 不变；将历史复现工具明确标为冻结版本，新的通用发布入口依赖权威源码，或同步修复后重新验收。构建记录嵌入模块 hashes，加入源一致性/有意差异 allowlist，避免更新 code 而发行仍使用旧副本。

### P2：runbook 的资源认证承诺与指定工具不闭合

位置：`reports/live/pack_submit_runbook.md:35`；`code/cnet2_search_timing.py:11–17,37–39,60–63`。

手册要求以这个工具验收 init20秒、单步硬3秒、慢步次数和额外池、整局1500秒、内存峰值，并输出p95。但当前工具只量 `agent()` 内部时长，未记录顶层 exec/init、整局或内存；只有p50/p90/p99/max；失败门仅检查累计错误，没实现资源超限退出。

动态证据：真实 `sleep(3.05)` 的无模型合法动作桩，工具的 range(299) 仅在内存AST中缩为1；测到 max=3050ms、a_errors=0，仍 **exit0**。见 `timing_resource_gate_probe.py/.json`。这不是模型性能测试，也不是否定已修 C03。exit0 当前只表示内部错误计数为零；手册要求人工读实测值，人工可以按 max 拦下这一例。**该证据不证明某个已上榜包超时，更不证明执行者已把它误判通过**。独立缺口是手册要求的 init/p95/slow_count/overage/game_wall/RSS 等没有完整输出或判定。

建议给完整外层认证 runner/明确补充命令，统一初始化、IPC、逐步/慢步池、整局与cgroup峰值判定；保留当前工具为局部健康/函数耗时子测试。不能把它单独的 exit0 当全平台资源通过。

## 已通过的旧问题复测

| 检查 | 当前结果 | 证据 |
|---|---|---|
| C03 健康包 | 两步 errors=0/root_errors=0/a_errors=0，exit0，纯网 SIMS 空时照常打印 wall | `timing_short.jsonl` |
| C03 根错误 | 错误teacher999，root_errors=2/a_errors=2，FAIL=true，exit1 | 同上 |
| C03 分类规则错误 | TG_MAX='invalid'，tg_errors=2/a_errors=2，FAIL=true，exit1 | 同上 |
| bigeval常规 | 健康2局exit0；内部异常2局exit3；3局里1个外层异常exit2 | `gates.jsonl` |
| 0个有效局 | exit2、FAIL=true，完整计数与有内容的first_error | `bigeval-boundaries.json` |
| 1个有效局＋外层失败 | exit2、FAIL=true，完整计数与有内容的first_error | 同上 |
| 1个内部异常局 | exit2、FAIL=true、a_errors_total=2、root_errors_total=2、first_search_error_seed=970500 | 同上 |
| EVO_SEEDS长度冲突 | N=3但2seed，明确stderr，exit2，不写新summary；不会再存FAIL=false | 同上 |
| 外层first_error稳定 | 完成顺序970502,970503,970501,970500，仍报告最低seed970500首因 | `first_error_order.json` |

源码对应：C03 `cnet2_search_timing.py:28–32,45–63`；seed参数预拒绝 `bigeval.py:22–24`；健康先汇总再判小样本 `bigeval.py:48–85`；外层首因排序 `bigeval.py:64–65`。`first_error` 仍专指外层异常；内部失败单独给计数和 first_search_error_seed，未新增内部 traceback。后者按完成顺序而非排序取值，这里不冒称所有字段都确定性排序。

## 当前真实参考包与较小入口边界

- 新入仓真实 `code/packs/n1r_int8_reference.py` 已可直接验证，不能沿用旧轮“n1r不在main”的限制。MD5 **ba7f84c3652ea2cd14a2243f2b1eace2**、**4,579,266**字节、13键CONF均与ledger/template一致；exec无`__file__`两步smoke合法己方动作且errors=0。证据 `n1r-reference-smoke.json`、`pack-doc-identities.json`、`pack-embedded-source-identities.json`。不是完整平台资格或强度结论；Echo3仍未用真实产物复测。
- P3局部CLI：`code/packs/cnet2_search_pack.py:106–108` 沿用 HERE/../data/weights，移动后默认查 `code/data/weights`。显式传权重可用；建议强制显式权重或改按repo root定位。
- P3来源链：ledger:7的n1r源权重没有文件名/MD5；最终包身份已核并不补齐原始权重来源。可明确标未核并保持reference冻结，不能据此认定包损坏。
- `PACK_CONF_TEMPLATE.txt` 是赋值档案，不是dict literal；手册没有完整CLI。未将未承诺的 `$(cat template)` 用法报告为确定故障。
- `build_v17.sh` 明确是历史重建入口，不把其调用根packer当成现行Echo3已走错链的证据。`.txt/.sh`此前只是sparse未检出，Git中存在，不列缺文件问题。
- 资源与台账的主要结论已包含在本节。

## 复现与限制

当前包错误探针复用了仓内 `reports/review_evidence/20261113_fixes/runtime/{prepare,timing_short,gates,first_error_order}.py`，设置 `CARBON_REPO=/workspace/carbon-league`、`CARBON_REVIEW_OUT=/workspace/carbon-review-round4/runtime`。本轮新增 `pack_source_probe.py`、`bigeval_boundary_probe.py`、`n1r_reference_smoke.py` 和 `timing_resource_gate_probe.py`；脚本/产物均未改业务源。

错误探针和源码比较包使用历史b128：仓内 `code/exp/cnet2_search_final_packed.py` 提取权重，出处见 `fixture_provenance.json`。这不是当前Echo3权重，也不替代新n1r直接smoke。所有计时仅用于短错误门复现，未做1.5CPU/512MiB限制下性能认证、全局胜率回归或外部历史结果核验。

本目录生成的 `.npz` 与 `*_standard.py/valid.py/invalid.py/rule_invalid.py` 等包是临时探针制品，发布报告时应排除；保存复现脚本与小型JSON/JSONL即可。
