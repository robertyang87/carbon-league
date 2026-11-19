# REVIEW_TRAIN2 — train2.py / gate_train2.py 独立逐行深审

- 审查人：独立审查（非作者），只审不修；审查期间未修改任何被审文件，全部实测在 /tmp 下进行。
- 日期：2026-11-20。
- 对象（md5 开审前实测，与派单值逐字合）：
  - `train2.py` `d4eaac1234bbaf4ac5dbcc91a1db526c`（492 行）
  - `gate_train2.py` `876ae8df5cb2e1bec3d3dc80bf4512f6`（304 行）
  - 对照原件 `src_read/train.py` `34898e7c0132b1834404bff01f0cde21`（341 行）。另经 `gh_api.get_file` 读仓内 `tools/katago_m3s2/train.py` 比对：md5 同为 `34898e7c…`，存档与仓内一致，对照基准成立。
- 行号约定：`T2:` = train2.py，`G:` = gate_train2.py，`T1:` = src_read/train.py，`M:` = src_read/model.py。
- 本机无 torch/CUDA：凡只能代码级判断的项均标注「代码级」，并汇总进文末未覆盖清单，不默示通过。

---

## 重点 1 · RNG / 索引等价

### 1.1 permutation 的时点与次数 — 合
- T1 L276–277：每 epoch 开头恰一次 `order = rng.permutation(len(rows))`，全程无其他 permutation。
- T2：`build_epoch_plan` L221 `order = rng.permutation(n_rows)`；main L440 每 epoch 调用一次。时点（epoch 头、先于该 epoch 全部 randint）与次数（每 epoch 一次）逐项同。

### 1.2 randint(4) 消费顺序与交错关系 — 合
- T1 L282：`[augment(rows[i], int(rng.randint(4))) for i in idx]`——按 batch 切片升序、切片内按 idx 顺序、每行恰一次，与训练步交错执行。
- T2 L225：`gs = np.array([int(rng.randint(4)) for _ in idx], …)`，在 `build_epoch_plan` 内按同一切片升序、同行序整 epoch 预消费。
- 推理链：numpy `RandomState` 是单一顺序流，流状态只由调用序列决定。两边每 epoch 的调用序列同为 `[permutation, randint×n]`，跨 epoch 顺序拼接亦同（见 1.4）。T1 中与 randint 交错的只有训练步（torch 运算＋纯 numpy 的 stack/astype，见 1.3 的全文件扫描：训练路径上再无任何 numpy RNG 消费），交错的非 RNG 工作不改变流状态。故「提前整段消费」与「交错消费」产出的 (idx, g) 序列逐字同——预生成不改变序列，只改变消费发生的墙钟时刻。
- 实测（我方独立参考实现，按 T1 L276–282 语义另行转写，并在 batch 间插入异流 PRNG 消费以模拟 torch 流交错）：8 组参数 (n, batch, epochs, seed) ∈ {(5003,512,3,0), (5003,512,3,7), (100,512,2,1), (512,512,3,42), (37,8,3,3), (1024,100,2,5), (1,512,3,0), (225,225,2,9)} 与 `generate_index_sequence` 全部逐字 MATCH；闸自带 `check_sequence` 另 9 组（含 max_steps 中段截断 21/25/29/30/31/47）全 PASS。

### 1.3 训练路径上无其他 numpy RNG 消费 — 合
- 全文件扫描：T1 中 `np.random` 仅 L260 构造 `RandomState` 一处；M:（model.py）无 `np.random.*`、无 torch 随机算子调用（初始化走 nn 默认 init，属 torch 流）；`load_dataset`（T1 L197–239 / T2 L178–210，代码逐字同、仅 docstring 差）只有 `np.load/np.empty`。
- 附带排查：仓内 `layout.py`（md5 `6877bc73…`，本审自仓内取至 /tmp 核读）L145–164 `new_game_state` 确有全局流消费（`np.random.seed/uniform/choice` L158/163/164）。但 ① 两训练器对 layout 的使用仅 `layout.SID_NAME` 与 `layout.to_f16`（T1 L262/332，T2 L416/483），`new_game_state` 不在训练路径上、无调用；② 即便被调用，它驱动的是全局单例流，与独立构造的 `RandomState(seed)` 实例互不影响。此路已排除，不构成反例。

### 1.4 跨 epoch 流连续性 — 合
- 两边 rng 均只在 main 开头构造一次（T1 L260 / T2 L413），epoch 间不重播种。
- 实测：seed 0、n=50 时第 2 epoch 的 permutation 与「同流续读」值逐字合、与 fresh-seed 首 epoch 值不同；`generate_index_sequence(50,512,2,0)` 第 2 步 idx 与续读 permutation 逐字合。

### 1.5 torch.manual_seed 与 numpy 流的相互独立性 — 合
- 两边顺序同为先 `torch.manual_seed(args.seed)`（T1 L259 / T2 L412）后 `rng = np.random.RandomState(args.seed)`（T1 L260 / T2 L413）。torch 与 numpy 是两套独立 PRNG，互不消费对方流——此结构未被 T2 改动。
- 唯一消费 torch 流的是 `load_main_npz` 的网络构造（aux 头随机 init，M: L140 注释自述）；两边调用点、种子、顺序全同，且发生在首个 numpy 消费（permutation）之前。T2 新增的 Pool 创建（fork，L319–321）在 net 加载之后、epoch 循环之前，fork 不改父进程任一流的状态。此项为代码级判断＋调用顺序比对（torch 本体执行列未覆盖 U4）。

### 1.6 worker 侧「零 RNG」— 合（逐个列明）
T2 全文件随机源扫描结果：
| 位置 | 内容 | worker 侧消费 |
|---|---|---|
| T2 L236 | `generate_index_sequence` 内 `RandomState(seed)` | 闸用纯函数，主进程；worker 不调用 |
| T2 L413 | main 的 `rng` | 仅主进程；仅经 `build_epoch_plan`（L213–227）消费 |
| T2 L221/L225 | permutation / randint | 只在 `build_epoch_plan`，主进程执行 |
| `materialize_numpy` L251–285 | stack/astype/`score_utility`（tanh/clip） | 纯确定性运算，无随机源 |
| `augment` L103–118 | V4 表查表置换 | 确定性，无随机源 |
| `_worker_init` L288–296 | 仅写 BLAS 线程环境变量 | 无 |
| `_worker_task` L298–300 | 仅调 `materialize_numpy` | 无 |
| torch 随机算子 | worker 全程不 import torch（torch 仅在 main L408 起与 `_to_device` 形参中出现） | 无 |
| `random` 模块 | T2 全文无 `import random` | 无 |
| 多进程种子播撒 | Pool 用 fork（L319），子进程继承父进程 RNG 状态副本，但上列 worker 代码路径无任何抽取点 | 继承≠消费 |
结论：worker 侧零 RNG 成立——不是「播了种但没用」，而是根本无抽取代码路径。

### 1.7 计划数值的传递完整性 — 合
- plan 中 idx 为 permutation 的切片视图、gs 为 int64 数组（L224–226），经 pickle 送 worker 后逐元素 `int(g)`（L262）使用，值不变。
- 附带查明：`generate_index_sequence`（L230–245）在 max_steps 落在某 epoch 中段时，会先整 epoch 生成 plan（多消费该 epoch 余下 randint）再截断返回，与闸参考实现的「中段即返回」消费量不同；但该函数返回即弃 rng，多消费无任何可观察后果，截断点前序列逐字同（1.2 实测含 6 个中段截断点全 PASS）。不构成不合，记录备查。

**重点 1 小计：7 项全合。**

---

## 重点 2 · 预取保序（OrderedFeeder，T2 L303–358）

### 2.1 乱序完成下严格按步号交付 — 合（实测）
- 代码：`run_epoch` L333–351 只 yield 当前 `want`：或从重排缓冲 `results` 取（L340–342），或阻塞 `pending.pop(want).get()`（L343）并 `assert step == want`（L344）。已早到结果只进 `results` 缓存（L347–351 的 drain），永不越序 yield。
- 实测（桩物化注入步号相关延迟，3 workers / prefetch 4，n=37、batch=8、5 步计划）：worker 实际完成序 **[2,1,0,3,4]**，交付序严格为计划序，且每步 11 个数组与内联物化逐数组 `array_equal` 全合。

### 2.2 预取窗口语义 — 合
- 提交条件 L335 `submitted < want + self.prefetch`：任一时刻未消费的在途步数 ≤ prefetch；窗口随 want 前移滑动。drain（L347–351）只把已完成结果移入缓冲，不扩大提交窗口。缓冲规模受在途数上界约束，无无界积压。

### 2.3 首步 / 末步 / epoch 边界 / 尾批 — 合（实测）
- 尾批：n=37、batch=8 时 plan 步长 [8,8,8,8,5]，末步交付 x 形状 (5,54,15,15) 正确，无丢步重步。
- epoch 边界：同一 feeder 连续跑两遍同一 plan（模拟跨 epoch 复用），两遍结果均与内联参考逐数组合。
- 空 plan：yield 零次、立即返回、不挂。
- 首步：want=0 时先提交满窗口再阻塞取步 0，无特殊路径缺陷（2.1 实测含首步）。

### 2.4 prefetch 深度大于总步数 — 合（实测）
- prefetch=100 对 5 步 plan：全部提交后按序交付，结果逐数组合，无死等。prefetch=1（串行窗口）同样全合。`prefetch` 下限钳制 `max(1, …)` 在 L315。

### 2.5 workers=1 / 0 退化路径 — 合（实测）
- `__init__` L311–322：仅 `workers > 1` 建 Pool，否则 `_pool is None`，`run_epoch` L326–329 走内联物化。实测 workers=1 与 workers=0 均 `_pool is None` 且结果与参考逐数组合。

### 2.6 worker 普通异常的传播 — 合（实测，附注）
- 注入桩物化在步 2 抛 `ValueError`：异常 0.1 秒内传播到消费端（经 L343 或 drain 的 L350 `.get()` 重抛），无挂死、无静默跳步；T2 main 的 try/finally（L438/L479）保证 feeder 被 close。
- 附注：drain 循环（L347–351）会在 yield 后、下一 want 取数前顺手 `.get()` 已 ready 的后续步——后续步的异常可能因此「提前」在前一步刚交付后冒出，而非等到该步轮次。仍属传播而非吞没，训练会以异常中止，不产生错误结果；仅中止时点与步号归因可能错位一步，记录备查。

### 2.7 worker 硬死亡 / 结果丢失 — **不合（实测坐实）**
- L343 `pending.pop(want).get()` **无 timeout**；Pool 对执行中被硬杀的 worker 不重投该任务。
- 实测：注入桩物化在步 2 执行 `os._exit(42)`（模拟 OOM killer / 段错误）：主进程在 `.get()` 上无限阻塞，30 秒后被外部 timeout 杀掉（exit 124），无报错、无任何输出落盘。
- 影响定性：物化 worker 持有全量数组视图、单步峰值内存可观，OOM 硬杀是现实失效模式；一旦发生，训练表现为静默假死（进程在、日志停），与本线「验活之外须验效」的既往教训同型。此为本审在 train2 本体上发现的唯一实质缺陷。修法不在本审范围（只审不修），仅指出：`.get(timeout=…)`＋超时 die、或结果心跳，二者取一即可把假死变成显性失败。

### 2.8 消费端中途弃用生成器 — 合（代码级）
- `run_epoch` 自身无 try/finally，清理责任在调用方：T2 main 以 try/finally 包裹并 `feeder.close()`（L479，`close` L353–358 做 terminate+join）；闸 `_run_steps` 亦在 finally 中 close（G: L170–172）。两处已知调用点均已兜底。本项未单独实测弃用路径，列代码级。

**重点 2 小计：8 项，7 合 1 不合（2.7）。**

---

## 重点 3 · 异步拷贝（_to_device，T2 L360–373）

### 3.1 pin 的启用条件 — 合（桩实测）
- L367–373：`pin = (device != 'cpu' and str(device).startswith('cuda'))`；仅 pin 时 `.pin_memory()` 且 `.to(device, non_blocking=True)`，否则普通 `.to(device)`。
- 桩 torch 实测（记录 from_numpy/pin_memory/to 调用）：device='cpu' 时 pin 调用 0 次、`.to` 不带 non_blocking 参数；device='cuda' 与 'cuda:0' 时每张量均 pin＋`non_blocking=True`；device 以对象形式传入时 `str(device)` 路径同样成立。

### 3.2 同流顺序语义 — 合（代码级）
- T2 全文无任何 `torch.cuda.Stream` / stream 上下文（通读确认）：全部拷贝与计算都发在进程默认流上。CUDA 同流内操作按签发序执行，故每步 forward 读到的必是该步 DMA 完成后的数据，不存在跨流竞态的可能性。异步拷贝在此结构下是安全的，无需额外事件/同步点。CUDA 实际时序本机不可测，列未覆盖 U2。

### 3.3 loss 读数 / 日志打印 — 合（代码级）
- 每步 L471 `float(val)` 对 device 上的 loss 分量逐个取宿主值——`float()` 对 CUDA 张量隐含同步，读到的必为完成值，无「未同步就读」。epoch 末打印（L477）用的是已累加的宿主 float。代价是每步一次同步（吞吐问题、非正确性问题），与 T1 L322 的原有行为一致（T1 本就每步 float 取数），不构成 T2 引入的回归。

### 3.4 检查点保存路径 — 合（代码级，附注）
- 导出在训练循环结束、`feeder.close()` 之后（L481–486）：`net.export_source_dict(snames)`＋`np.savez_compressed(…, **layout.to_f16(out))`，`--out-pt` 时 `torch.save(net.state_dict(), …)`，与 T1 L330–336 同式。参数的最后更新（opt.step）与导出时的宿主侧转换同在默认流序上，转换调用自身隐含同步；文件中无显式 `torch.cuda.synchronize()`——依赖的是单流顺序＋转换隐含同步这一标准单流模式。代码级判断为合；CUDA 实机行为列未覆盖 U2。

### 3.5 异步 DMA 的源内存生命周期 — 合（代码级）
- pin 路径先 `pin_memory()` 生成钉住副本（L372）再异步拷贝，DMA 源不是 feeder 的 numpy 数组本体：numpy 批次在 yield 后被释放/复用也不会踩到在途 DMA。钉住副本的释放由 PyTorch 宿主分配器的流记录语义兜底（DataLoader 同款用法）。代码级判断为合。

### 3.6 CPU 目标误用 pin — 合（桩实测）
- 见 3.1：CPU 路径 pin 零调用、non_blocking 零传参，不存在把 pin 用到 CPU 目标上的误用。

**重点 3 小计：6 项全合，其中 3.2/3.3/3.4/3.5 为代码级判断（本机无 CUDA，时序与数值不可实测）。**

---

## 重点 4 · 闸自身（gate_train2.py，照 gate_vec2 翻车病类逐项排查）

### 4.1 病类① 零维/标量参与 concatenate/stack — 合
- 全文件 grep：`np.concatenate`、`np.stack` **零出现**。序列检查比的是元组（G: L71–84）；loss 记录在 `_run_steps` L162–165 以 `float(...)` 逐分量建 dict，标量从不进任何拼接/堆叠。无此病。

### 4.2 病类② 布尔混入数值 max/min/mean — 合
- 全部 max/min 站点逐个核：L76 `min(len, len)` 整数；L116 `min(1.0, …)` 浮点；L259 `max(ABS_TOL, REL_TOL*abs(o[k]))` 浮点；L261 `worst = max(worst, d)`，worst 初值 0.0、d 为 `abs(n[k]-o[k])`，n/o 均来自 `float()` 构造的 dict（L162–165），无布尔可混入；L282–286 步速 max 均为浮点。与 gate_vec2 第 223–224 行 `max(d.values())` 含布尔的形态逐项对照：gate_train2 无「对混合类型 dict 取 max」的写法。此病不成立。

### 4.3 病类③ 异常穿透（一）· 导入期 — **不合（实测）**
- G: L45 `import train2 as t2` → T2 L44 `import layout` 在模块顶层。而交付目录 `~/workspace/trainopt/` 内**没有 layout.py**（目录清单实测：仅 train2.py / gate_train2.py / src_read/ / BOTTLENECK_NOTE.md）。
- 实测：在该目录裸跑 `python3 gate_train2.py --mode seq` → `ModuleNotFoundError: No module named 'layout'`，exit 1，**无任何结论行**（输出原文见附录 A1）。
- 连带：闸 docstring（G: L10–14）称 seq 模式 "Runs on any machine"——实际前提是 layout.py 与之同目录（在仓内 `tools/katago_m3s2/` 成立、在交付审查目录不成立）。本审将仓内 layout.py（md5 `6877bc73…`）取至 /tmp 并以 PYTHONPATH 注入后闸才可跑。定性：可移植性声明过宽＋导入期异常无结论行。

### 4.4 病类③ 异常穿透（二）· main 体内 — **不合（实测＋代码）**
- main 内唯一的 try 只包 `import torch`（L241–246）。以下调用全部裸奔：`check_sequence`（L226–227）、`t2.load_dataset`（L248）、`_run_steps` 两处（L250–253、L276–280）。任一抛错（数据目录不存在时 load_dataset 的 SystemExit、_run_steps 内 worker 异常、OOM 等）都直接穿透 main → traceback，**无 FAILURE 记录、无 VERDICT 行**，与 docstring L23–26「any failed check prints a final line」的自述不符——该自述只覆盖「检查返回失败」的路径，不覆盖「检查本身抛异常」的路径。
- 本地实测坐实：`--mode seq --batch 0` → L224 `ZeroDivisionError` traceback，exit 1，无结论行（附录 A4）。另注：L247 `'torch' in dir()` 的写法只兜 ImportError；torch 导入若抛其他类型异常（如 CUDA 库损坏的 OSError）同样穿透。
- 对照 4.3：两处穿透都是「进程死了但没留结论」，正是病类③要防的形态。

### 4.5 病类④ 结论行可产出 — 合（实测）
- seq 模式实跑（layout 注入后）：`[a] sequence check: n_rows=5003 batch=512 steps=21 -> PASS`＋`VERDICT: PASS (within the checks actually run; skipped parts are printed above and do not count as passed)`，exit 0（附录 A2）。
- equiv 无 data 路径与 equiv 给假 data 但本机无 torch 路径：均打印 `[b] SKIPPED: …` 明示跳过原因与 L20 命令形状，随后给出带限定语的 PASS 结论行，exit 0（附录 A3）——跳过不冒充通过，措辞合格。
- 失败路径的结论行（VERDICT: FAIL -> ABANDON，L296）本机未能实跑触发（seq 检查全合、无 torch 跑不了 loss 检查）；其代码路径为 failures 非空即打印并 return 1，结构成立。此半项列未覆盖 U1 的一部分。

### 4.6 报告文件落盘 — **不合**
- 全文件 grep：gate_train2.py 内 `open(`/`write(` **零出现**——闸不写任何报告文件，全部输出只在 stdout。本审附录中的「运行记录文件」是我用 tee 捕获的，不是闸的产物。按本线既定纪律（回归/验合输出必须当场落 durable 路径、不许只留易失处），闸的结论持久化完全依赖调用方自觉重定向，闸自身不满足落盘要求。

### 4.7 (b) 的「old 臂」不是 train.py — **不合（覆盖缺口＋措辞）**
- `_run_steps` L131 对**两个臂**都用 `t2.build_epoch_plan` 生成计划；old 臂（L134–136）是 `t2.materialize_numpy` 的内联生成器，new 臂是同一物化函数的池化版。即 (b) 比的是 train2 自身物化函数的「内联 vs 池化」，**不是** docstring L11–13 所称的 "the OLD inline feeding path"（train.py 的内联喂数）。
- 后果：`build_epoch_plan` 或 `materialize_numpy` 若与 T1 不符，(b) 两臂会一起错、互比仍 PASS；能兜住 RNG 的只有 (a) 的转写参考（其本身也是转写、非执行 T1）；而 T1 L283–296 的内联组装与 `materialize_numpy` 的逐式对照，闸内**没有任何**检查执行——本审在重点 5.8 以人工逐式比对补上了这一格，但闸自身缺位属实。

### 4.8 [b] 行 PASS/FAIL 标签混淆 — **不合（轻微）**
- L270–272：[b] 行尾的 PASS/FAIL 取自全局 `failures`（含先前 (a) 的失败）。若 (a) 已失败而 (b) 损失差在容差内，[b] 行会打印 FAIL，与该行自报的 max|diff| 数据矛盾。最终 VERDICT 因此仍为 FAIL、总判不错，但分段标签失真，读报告者会被误导。属标签准确性问题，轻微。

### 4.9 bench 模式无判读线 — 合（设计如此，记录备查）
- bench 段（L274–290）只打印新旧步速、比值与基线锚，从不向 failures 追加任何项：新路径再慢，总 VERDICT 也只取决于 (a)。docstring 未声称 bench 有通过线，故按「测量件」定性为合；但使用者须知：**bench 结果不进闸门判定**，采纳与否靠人读比值。

### 4.10 (b) 步数不足不设防 — **不合（轻微）**
- L257 `zip(old, new)` 按短者截断；若数据量使总步数 < `--loss-steps`（默认 100），闸只比实际跑到的步数并以 `len(old)` 打印（L269–272），不判失败。打印值使细心读者可见，但 VERDICT 可在仅比了个位数步数的情况下给出 PASS。属证据强度不设防，轻微。

### 4.11 容差 / 判读线来源标注 — 合
- ABS_TOL=2e-2、REL_TOL=0.02：在常量处（L34–35）、docstring（L28–38）、[b] 输出行（L270）三处均标注「待策略组钉 / PLACEHOLDERS pending the pin」，无冒充已钉值。
- BASELINE_STEPS_S=11.78 标注来源「L20 Gen-2 measured, war-room supplied」（L36 及 bench 输出 L286–287）。
- (a) 序列检查为零容差逐字比，docstring L38 明示。标注齐全。

### 4.12 GPU 利用率采样段的异常处理 — 合
- `_gpu_util_sample` L176–190：nvidia-smi 缺失返回 None；subprocess 异常与输出为空导致的 IndexError 均被 `except Exception` 兜为 None，bench 打印 "not-measured" 语义明确。此段是全闸少有的异常兜底完整处。

**重点 4 小计：12 项，6 合 6 不合（4.3、4.4、4.6、4.7、4.8、4.10；其中 4.8/4.10 为轻微）。**

---

## 重点 5 · 配方对照（独立重核）

### 5.1 CLI 参数个数 — 合
- AST 枚举实测：T1 共 **13** 个 `add_argument`——作者所报「13 个」属实。T2 共 16 个 = 同 13 个＋新增 3 个（`--feed-workers` 默认 8、`--prefetch` 默认 4、`--compile` store_true），新增项均为喂数/可选编译，不动配方。

### 5.2 13 个共有参数默认值逐字对照 — 合
AST 字面量逐项比对（type/default/required/action/help），13/13 全 MATCH：
| 参数 | T1 | T2 | 默认 |
|---|---|---|---|
| --data | L238 | L383 | required |
| --init | L240 | L385 | required |
| --out | L241 | L386 | required |
| --out-pt | L242 | L387 | None |
| --epochs | L243 | L388 | 3 |
| --batch | L244 | L389 | 512 |
| --lr | L245 | L390 | 1.5e-4 |
| --lr-final | L246 | L391 | 3e-5 |
| --weight-decay | L247 | L392 | 0.01 |
| --fast-policy-weight | L248 | L393 | 0.0 |
| --n-extra | L249 | L394 | 0 |
| --device | L251 | L396 | 'cpu' |
| --seed | L252 | L397 | 0 |
（行号为 grep 实测的 add_argument 所在行；T2 新增三项在 L399/L401/L403。）

### 5.3 loss 权重与组成 — 合
- 总 loss 行逐字同：T1 L314 ＝ T2 L463 `loss = lp_w + lp_b + 1.5 * l_v + 0.3 * l_d + 0.2 * l_f + 0.2 * l_win`。
- 训练步代码块（自 `for gp in opt.param_groups` 至 `opt.step()`，两边各 18 行）逐行 diff：**唯一差异**是 forward 句柄 `net` → `train_net`（T2 L452 区），其余 lp_w/lp_b 的掩码归一、l_v/l_d/l_f/l_win 的定义行全同。`train_net` 在 `--compile` 关闭（默认）时就是 `net` 本人（L430）。

### 5.4 AdamW 参数 — 合
- T1 L265–266 ＝ T2 L418–419：`torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)`，betas/eps 均用库默认、两边同。注意 T2 的 optimizer 仍建在未编译的 `net.parameters()` 上，`--compile` 不改变优化器对象。

### 5.5 lr 调度 — 合
- `lr_at` 函数体逐字同（T1 L270–272 / T2 L423–425）：余弦自 lr 衰至 lr-final；`steps_per_epoch = math.ceil(len(rows)/args.batch)`、`total_steps` 公式同（T1 L267–268 / T2 L420–421）。

### 5.6 BF16 autocast — 合
- 逐字同：T1 L302 ＝ T2 L451 `with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):`。

### 5.7 导出格式 — 合
- T1 L330–332 ＝ T2 L481–483：`out = net.export_source_dict(snames)` 后 `np.savez_compressed(f, **layout.to_f16(out))`；`--out-pt` 的 `torch.save(net.state_dict(), …)` 亦同（T1 L334–336 / T2 L485–487）。T2 导出走未编译的 `net`（非 train_net），`--compile` 开启时导出路径仍为原模块——处置正确。

### 5.8 组装公式（materialize vs T1 内联）— 合
- T1 L283–296 内联组装与 T2 `materialize_numpy` L251–285 逐式对应：x/100、sc 原样、tw/tb/wm/bm 原样、vt=`score_utility(margin)`、aux_delta/1000、carbon/100、win 原样、pw=`searched + (1-searched)*fast_policy_weight`，dtype 均为 float32。
- 唯一写法差：T2 对 pw 多一层 `np.ascontiguousarray(…, dtype=np.float32)`（L285）——T1 侧 `searched` 本为 float32 数组、表达式结果同为 float32，连续化不改值。合。

### 5.9 共享函数逐字性 — 合
- AST 源码段比对：`_v4_coords`、`build_v4_tables`、`augment`、`score_utility`、`_rss_gb`、`STORE_DTYPES` 六件**逐字相同**；`ReplayData`、`load_dataset` 代码相同、仅 docstring 缩写不同（diff 已核，无代码行差异）。

### 5.10 其余有意差异盘点 — 合
- ArgumentParser 的 description：T1 取 `__doc__` 首行、T2 硬编码同串，结果文本相同。
- T2 训练主循环外加 try/finally 关 feeder、步内张量改经 `_to_device`（重点 3 已审）、epoch 末打印与 200 步进度打印格式与 T1 逐字同（正则比对 SAME）。以上差异均不触配方。

**重点 5 小计：10 项全合。作者「13 个全同」的自报经独立重数重核成立。**

---

## 总表

| 重点 | 项数 | 合 | 不合 |
|---|---|---|---|
| 1 RNG/索引等价 | 7 | 7 | 0 |
| 2 预取保序 | 8 | 7 | 1（2.7 worker 硬死亡永久阻塞） |
| 3 异步拷贝 | 6 | 6（其中 4 项代码级） | 0 |
| 4 闸自身 | 12 | 6 | 6（4.3 导入穿透、4.4 体内穿透、4.6 无报告落盘、4.7 old 臂非 T1、4.8 标签混淆*、4.10 步数不设防*） |
| 5 配方对照 | 10 | 10 | 0 |
| **合计** | **43** | **36** | **7** |

*为轻微项。

一句话结论：train2 本体的等价性构造（重点 1/3/5）经逐行比对与实测全部成立，唯一实质缺陷是 OrderedFeeder 对 worker 硬死亡无超时、会静默假死（2.7）；gate_train2 的两类 gate_vec2 旧病（标量拼接、布尔混 max）确实没有，但病类③④以新形态复发——异常穿透无结论行（4.3/4.4）、不写报告文件（4.6）、且 (b) 的 old 臂并非 train.py（4.7），使闸的实际覆盖小于其 docstring 的宣称。

## 未覆盖项（需 torch / L20 才能验，逐项列明，不默示通过）

- U1：闸 equiv 模式 (b) 损失轨迹比对的实跑——需 torch＋真实 --data/--init；本机仅跑通 seq 与两条 SKIPPED 路径，FAIL 结论行（G: L296）亦未实跑触发。
- U2：CUDA 上 pinned＋non_blocking 的实际时序行为、每步 float() 同步的真实代价、检查点导出的实机正确性（重点 3 的 3.2/3.3/3.4/3.5 为代码级判断）。
- U3：`--compile` 开启路径的数值等价——闸按设计在 OFF 下运行，ON 路径两文件内均无任何验证。
- U4：`load_main_npz` 构造期 torch 流消费（aux 头随机 init）在两实现间的实机一致性——以调用顺序与种子相同推理，未执行。
- U5：bench 模式在 L20 上的真实步速与 GPU 利用率（对 11.78 steps/s 锚）——本机无 torch/GPU，未跑。
- U6：OrderedFeeder 在生产规模（512 行/步、全量数组）下的内存与行为，特别是 2.7 的硬死亡场景在 L20 上的实际触发概率与表现——本审桩数据为 37 行小数组。
- U7：闸 `_run_steps` 中途弃用 feeder 生成器时的 Pool 清理实效（2.8 为代码级判断，未在 torch 路径实测）。

## 附录 · 实跑记录（命令与输出原文）

### A1 闸裸跑（交付目录、无 layout.py）— 对应 4.3
命令：`cd ~/workspace/trainopt && python3 gate_train2.py --mode seq`
```
Traceback (most recent call last):
  File "/home/hatch/workspace/trainopt/gate_train2.py", line 45, in <module>
    import train2 as t2
  File "/home/hatch/workspace/trainopt/train2.py", line 44, in <module>
    import layout
ModuleNotFoundError: No module named 'layout'
EXIT=1
```

### A2 闸 seq 模式实跑（layout.py 自仓内取至 /tmp 注入 PYTHONPATH；layout.py md5 6877bc73ca971e0110d19d54b04fcd7d）
命令：`PYTHONPATH=/tmp/gate_run python3 gate_train2.py --mode seq`
```
[a] sequence check: n_rows=5003 batch=512 steps=21 -> PASS
VERDICT: PASS (within the checks actually run; skipped parts are printed above and do not count as passed)
EXIT=0
```

### A3 闸 equiv 的两条跳过路径 — 对应 4.5
命令：`PYTHONPATH=/tmp/gate_run python3 gate_train2.py --mode equiv`（无 --data/--init）
```
[a] sequence check: n_rows=5003 batch=512 steps=21 -> PASS
[b] SKIPPED: --data/--init not given (or torch unavailable here). Run on L20, next to model.py/layout.py:
    python gate_train2.py --mode equiv --data <dirs> --init <ckpt> --device cuda --n-rows <dataset rows>
VERDICT: PASS (within the checks actually run; skipped parts are printed above and do not count as passed)
EXIT=0
```
命令：`PYTHONPATH=/tmp/gate_run python3 gate_train2.py --mode equiv --data /nonexistent --init /nonexistent`
```
[a] sequence check: n_rows=5003 batch=512 steps=21 -> PASS
[b] SKIPPED: torch not installed on this machine. Run on L20 (command shape above).
VERDICT: PASS (within the checks actually run; skipped parts are printed above and do not count as passed)
EXIT=0
```

### A4 异常穿透演示 — 对应 4.4
命令：`PYTHONPATH=/tmp/gate_run python3 gate_train2.py --mode seq --batch 0`
```
Traceback (most recent call last):
  File "/home/hatch/workspace/trainopt/gate_train2.py", line 304, in <module>
    sys.exit(main())
  File "/home/hatch/workspace/trainopt/gate_train2.py", line 224, in main
    steps_per_epoch = -(-args.n_rows // args.batch)
ZeroDivisionError: integer division or modulo by zero
EXIT=1
```
（无 VERDICT 行。）

### A5 OrderedFeeder 实测汇总（桩物化，/tmp/feeder_test）
- 乱序完成：完成序 [2,1,0,3,4]，交付序＝计划序，11 数组/步逐数组相等，末步形状 (5,54,15,15)。
- 边界：prefetch=100（>5 步）、prefetch=1、workers=1、workers=0、空 plan、同 feeder 连跑两 epoch——全部与内联参考逐数组合、无挂死。
- 异常注入（步 2）：`raise` → 「exception propagated after 0.1s: ValueError('injected worker failure at step 2')」，feeder 正常关闭；`die`（os._exit(42)）→ 主进程阻塞，30s timeout 杀掉，exit 124，无输出。

### A6 _to_device 桩 torch 实测汇总（/tmp/feeder_test/test_todevice.py）
- 'cpu'：pin_memory 调用 0 次，`.to` 无 non_blocking 参数。
- 'cuda' / 'cuda:0'：每张量 pin_memory 1 次＋`.to(device, non_blocking=True)`。
