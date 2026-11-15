# Stage 0 缺口清单（2026-11-13 本地复验）

环境：本机 Python 3.12.3、numpy 1.26.4、无 torch。六件全部
`python3 -m py_compile` 通过（stage0 五件＋selfplay_gen.py）。
「可直接上机」＝脚本本体无需改、按其 docstring 的环境纪律跑即可；
所有上机动作仍依赖云助手通道（故障中）。

## 阻塞项（1 个）

- **sp-probe 两脚本与产局器定版脱节**：selfplay_gen.py 定版后 md5
  由 0767d73e9ff1c979a97e97d032e7a138 变为
  b275e952da14d644e6fab41e4ce941c8。sp-probe-s29.sh 第 28 行、
  sp-probe-studio.sh 第 53 行硬校验旧 md5，且两脚本内嵌的 b64
  （s29 第 24 行起、studio 第 49 行起）是旧版全文——探针趟照现状
  派发必 FAIL（script-md5-mismatch）。修法：本地以定版重生成内嵌
  b64＋同步改三处 md5 期望（注释行、断言行、SCRIPT-OK 行），
  bash -n 复验（两脚本现语法均 OK，sp-probe-driver.py py_compile OK）。
  这一步不需要通道，是通道恢复前应先做完的本地准备。

## 逐件结论

### stage0_measure_throughput.py — 可直接上机
- 测的是 PPO rollout 路径（rl2_ppo_rollout.play_batch、temp 1.0
  采样自对弈），与产局器路径（selfplay_gen、greedy pack CONF）
  不是同一个吞吐——两个数字都要、不可混用，汇报时注明口径。
- 环境纪律只写在 docstring（第 5 行：env -u CNET2_TEMP_KIND
  CNET2_TEACHER=1），脚本内无 env 断言；裸跑会让 R04 的
  TEMP_KIND 全局量污染测量。上机按 docstring 口径跑即可；
  若要加固，加一行 env 断言是可选小改（非阻塞）。

### stage0_teacher.py — 可直接上机（测成本）；语义简化留 Stage-1
- 第 108 行 `fz[:s] = greedy[:s]` 是无操作死代码（下一行才真正改
  fz[s]），无害，可随手删。
- H 口径：evaluate_root 先走根步、再走 H 步续局、然后取叶值
  （总深度 1＋H）。与设计文档对 H 的定义是否一致，Stage-1 接
  教师进产局环前须对齐一次——成本与 H 成正比，口径差一层，
  每局教师成本差约一倍量级，不可含糊。
- 终局叶用现金差近似、未加 F01 灭绝罚分（docstring 已自述）；
  _extinct_state（约第 41 行）用硬编码 cash＜30，与
  cnet2_search._extinct 的 min(收藏者/种植者成本) 口径在配置
  成本非 30 时不一致。两者都只影响 Q 的尾部精度，不影响成本
  测量结论。
- 分支 Feat 历史清零、belief 未接线（docstring 已自述），
  Stage-1 事项。

### stage0_measure_teacher.py — 可直接上机（一个汇总小缺口）
- 第 102 行 summary 的 "act_calls_per_step" 恒为 None、从未回填
  （每位置的打印行里有 n_act_calls，原始数据不丢）。成本结论
  靠 ms 已足，此项非阻塞；补填＝对 r["n_act_calls"] 求均值一行。
- 每局折算需要外部传入 steps_per_game（argv[2]），来源是吞吐
  趟的 steps_per_game——两脚本靠人工接力，任务书环 4 已写明。

### stage0_heads_patch.py — 可直接上机（锚点先 grep）
- 相对路径操作 cnet2_model.py / cnet2_train.py，只能在 code/
  目录下跑（docstring 已述）。
- 三处锚点（MODEL_ANCHOR / CTX_ANCHOR / TRAINER_ANCHOR）与
  main tip 现行文件是否逐字一致，本地无从验证（本地没有 main
  tip 的 cnet2_model.py / cnet2_train.py 副本）。补丁自带保护：
  锚点计数≠1 非零退出、标记幂等、改后 import＋权重形状自验，
  失败形态是「停住」不是「改坏」。上机顺序：先 grep 锚点计数，
  再跑补丁。

### stage0_smoke_heads.py — 可直接上机（有顺序与设备条件）
- 必须在 heads_patch 之后跑：它调 net.forward_heads，未补丁的
  模型没有此方法，直接 AttributeError——这是预期内的顺序依赖，
  不是 bug，但派发清单必须把两件排成同一趟的先后步。
- 第 29 行 BC_DEVICE 默认 cuda；Studio 上跑须显式
  BC_DEVICE=mps（DGX 用默认）。
- 标签走 atanh(b["value"]) 反推（head_losses），是存量 ds 的
  口径；待环 2 的 builder 定点改动落地后改为直读 raw margin。
  |M|≳11400 饱和的尾部失真在 smoke 规模下可接受。

### selfplay_gen.py（定版 b275e952…）— 本地 stub 已验、真网待探针
- 本地验证（foundation_stub/）：SPGEN_STUB=1 跑 6 局
  （991000–991005），index/replay 的 margin/u 记录一致性全过；
  build_top 等价口径建出 3,588 样本（598/局）、
  teacher_sides {m3s2r1: 12}、零报错。stub 局空动作全平局
  （margin 0、U 0.5），非零 margin 的端到端读数要等真网探针。
- 真实路径（cnet2_search 真实现＋真权重）本地从未跑过——这正是
  探针趟的用途，不算脚本缺口，但探针前先解上面的阻塞项。
- 续跑语义：已存在 replay 的种子跳过、旧 index 条目原样保留；
  若对定版前产出的 corpus 续跑，新旧条目会混杂（旧条目无
  margin/u）——docstring 已注明，须整corpus 重产或回填。

## dataset 级 raw margin（环 2 边界，非脚本缺口）
cnet2_build 现行口径下 dataset 分片只有 value＝tanh(M/3000)。
定点改动三处已定位（FIELDS 加字段、build_game 加 append、
消费端改读），按本轮口径**未动** build 主逻辑，详见
foundation_taskbook.md 环 2。corpus 层（index/replay）即日起
已有 raw margin＋U 可读回，不依赖该改动。
