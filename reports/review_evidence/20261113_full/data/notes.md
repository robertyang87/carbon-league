# Round 5 数据全链复核

基线固定为 `98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c`，对照 R4 `5463f504048bcca28868ebce0a063ac8d6c51b8e`、报告提交 `641f944`。本轮 builder 修复提交为 `4b44362`。所有代码行号指向该固定基线。业务源码和历史证据均未改动，没有训练、策略提交、远端资源操作或生产 corpus 重建。**代码验收与机器部署状态分开：本地原复现通过，不表示已在训练机部署。**

## 结论与已修项

**N03 原缺失 replay 例已经修好；P3 原“缓存缺字段导致合并过程覆盖旧数组”的例子也已修好。** 新临时目录只覆盖合并阶段的故障隔离，最终逐字段发布不是整个数据集的原子切换。以下新发现区分新改动边界和已有未覆盖造数入口，不能把旧问题的通过结果抹掉。

`recheck_data.py` 执行真实 builder/merger、从原模块 AST 原样抽取的 `cnet2_train.load/fetch`，不运行训练；12 个预期观测断言均通过。`recheck_labels.py` 运行真实造数函数和 FastCarbon，只有网络推理替为确定动作 stub；4 个定向观测断言通过。断言中含“证实问题存在”，因此不能把总通过数等同于系统全部无缺陷。

| 复核项 | 实测 / 判定 | 证据 JSON 键 |
|---|---|---|
| N03 eligible 缺回放默认关闭 | 两局少一文件时 exit 1，旧 meta/基础数组撤下 | `data-results.json:missing_default` |
| 显式 partial | exit 0，partial=true，valid_games=1，failed_games 保存原 ID 与 reason=missing_replay | `missing_partial` |
| partial 后恢复 | 同一输入补回文件，exit 0，两局恢复、失败声明为空，gid/metadata 重新一致 | `missing_restored` |
| 合法过滤区别于输入缺失 | short 与非 teacher 的缺文件分别记 min_steps/no_teacher_side，各 1；不产生伪失败 | `eligible_filter_accounting` |
| R4 P3 mid-merge | 缓存 shard 删除 maps，y 行数仍有效：exit 1；旧 meta 和所有旧数组逐字节保持；临时目录清除 | `mid_merge_failure` |
| index JSON 解析失败 | 非零退出，旧发布版本完整保留；这是失败时保留完整旧版本，不再描述为合并覆盖错误 | `scan_parse_failure` |
| 原局 split 继承 | 带 orig_id 的验证局衍生样本仍 VAL；已知 lb_cp 无 orig_id 非零拒绝且不创建输出 | `split_with_orig`、`split_missing_orig` |
| 实际新增 corpus schema | 从 Git blob 读取一局 M9XJ-it460 真回放，按 ladder_corpus 的 a/b schema 建 index；order=[b,a]，正确选 side=1，299 行、value=0.396240234375 | `actual_corpus_smoke` |

真实 corpus 源：`data/m9xj460_corpus/0bebaa4d73009b498af099841c4163d2-1.json.gz`，SHA256 `bcd4509622ba1c4ac1e856ee59f5fe96e10545b4480ecea9b599d47977cdb898`；配套 `data/m9xj460_matches.json`。该局回放 300 帧，最后两方 DONE；这是一局真实接口兼容性验证，不代表全量新增 corpus 已验收。其余小 fixture 使用三步真实观察，index.steps=300 仅用于通过既有 min_steps 过滤，不能冒充完整实战数据。

## D5-01：造数入口仍越过终局并用现金冒充结果（P2，已有入口漏修）

位置：`code/dagger_gen.py:51–81`；`code/lb_relabel.py:123–152`。对照当前正确的 `code/eval_v53.py:474–495`。

触发条件是自博/DAgger 轨迹提前灭绝，或固定 300 次推进包含平台实际不存在的第 300 次动作。两个生成器完全不判断终局，最后直接取 FastCarbon cash 差。`lb_relabel` corpus 模式从已录制 index 取 margin 不属于此复现；本问题针对 `play_selfplay`。

可执行 fixture 从合法初始状态开始：我方在 step 0/1 都在第一个基地招 collector；对方在 step 0/1 分别在两个基地招 collector。原版环境 `zerosum_env.make(...).run` 在 **step 2** 终止，rewards `[-299,50]`，分差 `-349`，旧 tanh 标签应为约 `-0.11578`。

- 真实 `dagger_gen.play` 保存 **300 行**（step 0–299），其中 **298 行在真实终局之后**；全部 value=0，结果 margin=0。
- 真实 `lb_relabel.play_selfplay` 默认 LB_FROM=270 保存 **30 行**（step 270–299），**全部在真实终局之后**；value=0、margin=0。
- 本例 DAgger 没有 planter，因此 policy uw 为零；但 `cnet2_train.py:156` 的 value MSE 不按 uw 屏蔽，错误价值标签仍会参与训练。LB 的基地样本还有正 uw。

这是确定动作 stub 隔离的真实控制流/特征/标签反例，没有执行真实网络推理，不测自然发生频率，也不推断已发布模型受影响程度。原版 authority 的输出和 stub 范围均在 `label-results.json:reference/lb_selfplay/dagger`。

建议让所有生产器共用 GameResult/终局判定，在最后合法动作停止，记录 raw rewards、termination 和真实 final margin；禁止生成终局后样本。根任务的 AWR reviewer 正在独立验证同类入口，可合并为“数据生产未共用终局契约”，不要重复列成多个总问题。

## D5-02：DAgger 特征前缀与训练动作前缀不一致（P2，已有逻辑问题）

位置：`code/dagger_gen.py:56–76`，尤其 `:72`；消费者 `code/cnet2_model.py:273`、`:279–291`（decode/forward_seq 也用 b[y]）。

当较早 planter 的 labeler 动作不同于实际动作时，生成器的 labeler 调用与 `Prefix.advance` 都基于实际 `y`，却把重标后的 `ym` 存为样本 y。训练 GRU 以存下的 ym 作为前一单位动作，于是同一输入中的 GRU 前缀与 mf/uf 所描述的前缀不同。

两 planter 受控观察：实际动作为 `[NONE,NONE]`，重标为 `[UP,LEFT]`。原样执行 `dagger_gen.py:64–72` 的 AST，第二个单位 `mf[LEFT,0]` 与按存储 ym 重算的正确前缀特征相差 **0.5**（预定占位不同）；见 `label-results.json:dagger_prefix`。不需要网络训练才能构成这项输入契约矛盾。

建议明确训练目标：若学的是实际前缀下每一单位的独立纠错，分别保存 prefix_actions 和 target_actions，让 GRU 与特征都读取实际 prefix；若学整套改进联合动作，则重标应顺序推进改进前缀，并按最终训练前缀重算特征。不能只替换标签数组而让两个前缀来源混用。

## D5-03：逐文件 replace 不能实现整版原子发布（P2，新修复的发布边界）

位置：`code/cnet2_build.py:571–575`；现有消费者 `code/cnet2_train.py:32–45`。R4 原 mid-merge 例已通过，不应说原例仍失败。

真实 builder 先发布六行数据，再以 STEP_MIN=2 重建两行。在第一份 maps.npy 已被替换、第二份 scal.npy 尚未替换时，通过原样抽取的当前 loader 读取，loader **正常返回**：`maps` 两行，其余字段（含 y/gid）六行，旧 meta.total=6。在第二次 os.replace 注入 OSError 后，builder exit 1，finally 删除临时目录，但这个混合版本仍留在正式目录。证据 `data-results.json:publication_replace_failure`。

这包含两个明确范围：

1. 并发消费者无需 builder 失败，只要读取落在两次 replace 之间，就可读到混合版本。
2. 发布阶段失败/进程中断会令该混合版本持久存在；本次用确定性 OSError 注入验证错误路径，不声称生产磁盘已经发生故障。

当前 replica_chain 在构建前删除 DS、按非零 rc 停止，因此本例**没有证明该串行链越过门启动训练**；受影响的是可复用输出目录的独立 builder / 并发读者。现有 loader 没有读 generation manifest、字段长度检查或一次性 pin 版本。

建议每次发布完整不可变 generation 目录，原子更新一个指针；消费者在打开任何数组前解析/固定该 generation 的真实路径，最后读取相同版本 metadata。只把 out 换成 symlink 仍不够：现有 loader 对每个文件重新拼接路径，链接若在两次 open 间切换，仍可能跨版本。也可使用读写共享锁或带校验并重试的显式版本协议。

## D5-04：关闭 weighted 后，旧 w 仍被自动读入（P2，已有派生产物失效问题）

位置：`code/cnet2_build.py:396–398`、`:571–573`；消费者 `code/cnet2_train.py:35–38`、`:150–155`。

同一输出目录先 `CNET2_WEIGHTED=1` 构建双 teacher 样本，再取消该开关重建。第二次构建 cfg 指纹已变化、基础数组重建且 exit 0；但只覆盖本轮 fields，没有清除旧 `w.npy`。当前 loader 默认 `BC_USE_W=1` 且按文件存在判断，于是“不加权”数据仍加载旧 `[2,2,2,1,1,1,...]`，赢家样本继续有两倍 policy loss 权重。所有数组仍同为 12 行，所以行数检查也拦不住这个语义问题。

证据 `data-results.json:weighted_seed/unweighted_rebuild`，包括原样 loader 的实际加载结果。类似风险还适用于取消 UW 或改变行布局后残留的 set_uw/aux 派生数组；本轮只对 w 开关做确定性复现，不扩大声称其他情形均已实测。

建议 manifest 显式声明本版本包含哪些权重/辅助字段，消费者只按声明加载；新 generation 不携带旧目录未重算的派生产物。当前 replica_chain 的 rm -rf + set_uw 完整重建避免这条复用路径，但不能替代通用 builder 的缓存/输出契约。

## D5-05：先遇到缺失副本会遮蔽后续完好副本（P3，N03 新回归，失败关闭）

位置：`code/cnet2_build.py:277–280`、`:303–316`。

支持多个 REPLAY_DIR 的 scanner 在文件存在检查前先 `seen.add(key)`。同一合法原局在输入目录 A 的 index 中存在、文件缺失，在目录 B 有完整文件；按 A,B 顺序构建得到 missing_replay / exit 1；仅交换为 B,A 即成功。见 `duplicate_missing_first` / `duplicate_good_first`。上一版先检查文件再 seen.add，不存在这一错误拒绝。

这是保守失败，不会偷偷输出错误完整数据。建议按原局归并全部来源，优先选择可用且一致的副本；只有所有可用来源都缺失才报告 missing_replay，并另记重复来源。若多个副本内容/索引不一致，应显式冲突处理，不能仅按输入顺序任意覆盖。

## 其他审查范围与限制

- cache 仍只验证 y 第一维行数（`cnet2_build.py:455`），并不验证完整字段 schema。缺字段会被临时合并阶段拒绝且保护旧版本，本轮没有把这个失败关闭路径升级成新数据污染结论。缓存证书在合并前写入，坏缓存不会自动失效重建，是可改进的恢复性边界。
- 普通 BC builder 的 teacher forcing 按修正后的同一 y 推进（`:192–202`），与上述 DAgger 的分离问题不同。
- 检查了训练几何增强 `cnet2_train.py:75–92`、对称表 `cnet2_feat.py:517–532` 与动作合法掩码来源：worker 五向动作均合法、base 招募动作不做方向旋转，不能仅因 legal 未旋转就报错。未运行大规模增强分布测试或网络质量实验。
- 原局派生路径的已修 split 用例再次通过；未把设计文档中尚未实现的统一 lineage schema 当作现有代码已具备。原局 hash 规则仍是 basename CRC，所有新衍生入口应明确继续继承。
- 新增 Git corpus 是原始 gzip 与 matches 列表；测试只在临时目录按现有 ladder_corpus 约定构造 index，未写回、修补或部署原始 corpus。源码新增导致 cfg_fp 改变、旧 cache 将重建，是显式代码行为；没有在任一生产机器触发重建。

## 当日公开规则来源

本轮确实在 **2026-11-13 UTC** 匿名 GET 当前用户游戏地址，未使用登录态：`public-sources.json` 记录 URL、抓取时间、字节数、SHA256，原文分别为 `replay-api.txt`、`agent-guide.md`、`public-home.html`。不是沿用昨天文件冒充今天读回。

`agent-guide.md:38`：1.5 核 CPU / 512 MiB；基准每步 1 秒、单步最多 3 秒、3 次慢步、总额外 5 秒；初始化上限 20 秒包含源码编译/模块导入/顶层加载，**写在 agent() 内的延迟加载仍计入该步 1 秒**。抓取时间 03:18:30 UTC，SHA256 `f589ab9bb13c368ced64aafcf590a16eaad46cd97cde408fcb7e548b17226965`。可供运行时 reviewer 直接引用。

## 复跑

```bash
CARBON_REPO=/path/to/carbon-league CARBON_REVIEW_OUT=/path/to/results \
  /path/to/python recheck_data.py
CARBON_REPO=/path/to/carbon-league CARBON_REVIEW_OUT=/path/to/results \
  /path/to/python recheck_labels.py
```

依赖与项目一致；本次解释器 `/workspace/carbon-review-venv/bin/python`。脚本默认只 stdout，可由 CARBON_REVIEW_OUT 外置证据。全部合成输入/构建输出在临时目录并在结束清理；真实回放只用 git show 取一个 blob。JSON 中源码 SHA 固定，policy stubs 和故障注入范围均明确。没有训练或远端修改。
