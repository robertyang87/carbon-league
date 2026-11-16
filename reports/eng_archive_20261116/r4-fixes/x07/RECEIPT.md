# r4 · X07 趟回执（k0_duel：X07＋X04 双半边＋X08）

**落仓**：仅改 `code/k0_duel.py`，commit `cbbe002c`，md5 `215b18e06d7cf2ae0883407df53966ee`（16,785 B）→ **`bead6367e7650894afdda7717ff649a5`**（26,855 B），sha 钉基线 blob 提交、API 读回逐字节一致。闸门 `stage1_teacher_gate.py` 前后均 `e922d9d8`，未动一字。证据在 `~/workspace/r4-fixes/x07/{probes,fixtures,evidence}/`，最终全链日志 `evidence/final_chain_stage2.log`（f0–f4）＋`evidence/final_tail_stage2.log`（尾段）。

## X07 · 加载门＋推理门（原条目＋细化）

**复现（基线 215b18e0，真实 CLI，根 8573…@285）**：正典 heads v2 五张量（stem/blocks.3.c1/tattn.1.qkv/value/whead）整张置 NaN——(A) 双臂同坏：exit 0、ΔU=0.0、U=0.7928345470416605（有限且貌似正常）；(B) 学生坏/父代健康：exit 0、ΔU=−0.1031018943400631（凭空造出一个「测量值」；父代 U 与健康真值逐位同，证明坏的只是学生侧消费）；(C) 五张量置 Inf：exit 0、ΔU=0.0。Inf 变体仅 stderr 一行 RuntimeWarning。
**细化复现（bhead.bias 单张量 NaN，报告原案）**：`pol.act` 输出 logits 第 10/11 行（bhead 槽位面）为 NaN、value 有限；以该策略直调续弈（绕过任何加载，全程无门）返回 U=0.16894850197364647——**与健康真值逐位相同**：输出侧任何检查（终局 U、ΔU、labels）都看不见这类损坏，只有推理表面能。
**修法**：① 加载门 `_policy_from_bytes`：npz 全部浮点张量逐一全有限校验，任一非有限 die 点名文件＋张量名＋坏元计数；文件缺失/不可解析（截断）同样 clean die 点名（基线是 FileNotFoundError/BadZipFile 裸 traceback）。② 推理门 `_greedy_checked`：把闸门 `_guarded_greedy` 在 2pol 改编内**逐调用展开**（act→检查→reset_step→apply，调用与次序不变），每步 logits 与 value 非有限即 die 点名根/臂/席/步。③ 逐根 U 门＋落盘前整行复检（u_student/u_parent/du)。④ 场景特征门：根 x_me/x_opp 浮点字段非有限 die 点名根与字段（基线实测：毒化后返回有限但错误的 U=0.1740081776655953、无报错）。
**验收**：坏权重 A/B/C 三案 CLI 全 exit 1、无产物落盘；bhead 直调续弈在 step 0 即 die（logits 非有限元 2 个）。正常路径：f1 同权 ΔU 逐根恰 0、重跑逐位同，U=[0.1555330393366056, 0.16894850197364647] 与原验收逐位一致；CLI 扰动学生 ΔU=0.054987669606519574 与原验收逐位一致。

## X04 · 单读绑定（权重半边＋replay 半边，独立成段）

**权重半边复现**：run_duel 按路径算 md5（:332 口径）、worker 再按路径加载——探针先记 md5=06a03d9c（正典），换盘为扰动权重（cee6b4ea）后 `_load_policy` 消费的 whead.weight 是替换件的：记 A 测 B 坐实。
**replay 半边复现**（报告 :273/:300/:307/:327）：load_roots 校验后把回放文件原地换成另一局字节（md5 bb57b1cb→ffb23969），run_duel 仍带旧 roots_sha256=57f2edc7… 出数，u_parent=0.7583044435450692（真值 0.16894850197364647）——校验后换 replay、结果仍带旧 SHA，与报告「现金已变」同型。
**修法**：权重——文件只读一次，run_duel 对捕获字节算 md5，worker 经 BytesIO 解析同一批字节（AST 断言 np.load 全模块唯一调用点且喂 BytesIO）；replay——load_roots 单次读字节（md5/ft/长度同源），返回 replay_blobs，worker 在进程内 scratch 目录按原 basename 物化已验字节，extract_scene 只消费 scratch 路径；run_duel 无 blobs 或有根未被 blobs 覆盖即 die。探针：捕获后换盘（含运行中途换盘）全惰性，记录 md5 恒为实际消费字节的身份；replay 换盘后 u_parent 与真值逐位相同。

## X08 · 同局一内容（P3）

**复现**：同名不同内容两文件（bb57b1cb / ffb23969）同挂 game 8573…，基线 load_roots 静默全收（2 根、同一 game 绑 2 份内容，聚合即成 1 局）。
**修法**：load_roots 建 game→replay_md5 登记，同 game 异 md5 即 die 点名两路径与两 md5；同内容异路径放行（实测 2 根接受、md5 唯一）。

## 回归（最终字节 bead6367 全链复跑）

f0 五种坏清单全 die 点名、roots_sha256=c5d0530e… 与基线同；**f1 同权 ΔU=0 恰、逐位重跑同**；**f2 父代臂 U 与闸门 continue_game 直调逐值相等（两根）**——内联展开保真由此坐实；f3 无教师（静态＋import 阻断动态）过；f4 聚类统计与 scipy/独立 bootstrap 逐位同；workers=2 池路径 ΔU=0 exit 0（新 initializer 签名实测）；坏权重/缺失/截断/特征 NaN/X08 冲突全 die 点名且零产物。

## 如实注记

- **内部 API 形态变更**：load_roots 返回三元组（增 replay_blobs）、run_duel 增必填口径的 replay_blobs 参数（缺省 None 即 die）；证据区 fixture 副本（f0/f1/f2/f3 及其 runner）已同步更新并全链复跑，原验收证据树 `~/workspace/k0-round2/duel/` 未动。仓内无其他调用方。
- 有限权重造非有限 logits 的尝试（3 张量 ×1e20，加载全有限）未造出非有限 logits（归一化层吸收）——推理门的实证承载是「加载下游注入」（直构 CachePolicy 的 bhead 案）；其定位是测量边界不变量，不依赖权重来源。
- scratch 成本：每 worker 进程每唯一 replay 一份副本（临时目录），超大根集请把 TMPDIR 指到磁盘——已写入代码 docstring。
- 中间版（仅 X07＋X04 权重半边）曾整链跑过一轮（f0–f4＋CLI＋workers2＋坏权重全过，日志 `evidence/final_chain.log`；其尾段特征探针因探针先行升级三元组签名报 ValueError，是探针/签名错位、非驱动问题，该探针在最终字节上已复跑过）。
- 本趟未跑任何真实测量、未碰机器进程；X04 的标注器半边与 X05/X09 归标注器趟，不在本回执范围。
