# 575 Studio 脚本静态补查（本轮冻结截止）

截止 SHA：`57557a83a73cd4d6b26b56ebebb4c0df8e13a155`。新增脚本本身提交 `249e037c276c339c6ceb87d76a347f6678a9ef8a`。只用 git show 静态读取，未执行 Studio/S29 脚本、SSH、rsync、scp 或任何训练。文本快照与 hash 在 `latest575-sources.json` / `latest575-*.txt`。

**与 62e 的结论一致：生产器没有再次改变。** `selfplay_gen.py` MD5 仍为 `b275e952da14d644e6fab41e4ce941c8`，两个 shell 的固定 producer hash 均匹配；两者也都对权重执行同一个 MD5 门 `b6db08c9b076257c54817377bb93130d`。62e 已报告的四个 P2 和预算/统计 P3 仍适用，未在本次重复运行或扩展历史审查。

| 核对点 | 静态结果与范围 |
|---|---|
| producer 路径 | Studio `sp_probe_studio.sh:55–64` 复制到 `$HOME/junyang/carbon/code/selfplay_gen.py`，与 S29 复制到 code 根目录相同；能避开仓内 `code/producer/` 直接执行导入失败。已有旧 producer 不会自动替换，而是 hash 不符就停止；这是失败关闭，不是使用旧脚本继续跑。 |
| 记录 seed 范围 | Studio `:68` 是 `[991000,991200)`，S29 `:57` 是 `[991200,991400)`；范围不交叉，符合按文件名去重的约束。warmup 不写回放。实际两机器吞吐并非同一 seed 负载的配对比较。 |
| 权重来源 | Studio `:45–52` 缺权重时从 DGX 的 `cnet2_m3s2r1_last.npz` 拉到本地 `...r1L.npz`，随后校验精确 MD5；文件名不同不等于模型不同，门已覆盖字节一致性。本轮未访问两台机器核实文件。 |
| 代码源一致性仍未锁定 | Studio `:37–44` 只在列举文件有缺失时同步，其余情况保留现有代码；五个文件仅输出 MD5，没有和固定源码 manifest 比较。同步来源是 DGX 工作目录，不是 SHA 575。S29 也只打印这些源码 hash。因此 **producer+weights 相同并不能证明两机的 search/feat/simulator/env 版本一致**；跨机产物/吞吐比较前须核对源代码及依赖 manifest。 |
| 同步范围不只是 data | Studio `:40` 是整个 DGX `code/` 的 rsync，会写/覆盖本地 code 中同名文件；`:49` 写 weights，`:59` 写 producer，`:68` 写 `/tmp` 日志。文件头 `:25` 的“Writes only to .../data/selfplay_probe/”不准确。脚本没有保证只改变探针数据目录，也没有证明复制前后业务代码完全一致。需修正文档并把探针依赖同步到独立固定版本目录。 |
| 预算/超时 | Studio `:66–68` workers=ncpu−2，未按剩余 games 取 min；仍触发 62e 已复现的批次超 max-games。没有脚本内总 timeout，`:26` 把限时交给外部 SSH 调用方，本文件不足以验证那个外部限时。`--max-seconds=180` 仍只在批次外检查，不能当作180秒硬上限；S29 另有420秒进程 timeout，二者 wall 口径不能混称相同。 |
| 退出状态 | Studio `:69–74` 保存 producer 返回码，并在缺少 SPGEN-RESULT 时失败；本轮未见新增“producer 非零却 shell 报成功”的路径。不能用存在 marker 替代 errors=0 / 数据索引完整性校验，后两者问题已在62e报告。 |

建议本轮报告把“代码源未绑定”和“同步范围声明不符”作为 **P3 复现/执行范围边界** 加在两机 probe 的说明里，不重复计为 producer 四个 P2 的修复失败。实际远端代码状态和性能均未验证；本轮代码审查截止固定在 575，不追后续 main。
