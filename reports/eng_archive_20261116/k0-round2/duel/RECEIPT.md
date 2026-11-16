# K0 第二轮件 3 · 根续弈测量驱动（k0_duel）回执

**交付**：`code/k0_duel.py`，commit `3631e12139b9`，md5 **215b18e06d7cf2ae0883407df53966ee**（16,785 B）。落仓后 API 读回与本地受测字节逐字节一致。只新建此一文件，未改任何既有文件。

**护栏（闸门钉住）**：`stage1_teacher_gate.py` 动手前与交付时两次 API 读回均为 **e922d9d890c3e1eff6643e947289acb0**，未动；`stage0_teacher.py` 同为 c6ade5ced42e2c54a77c1499163aa151。其余 staged 参照副本（feat/np/search/eval/fastsim/env carbon.json）均按 main 字节取得、md5 与在案锚点逐一对上。

## 实现要点

- **语义**（草稿 §4.2）：学生臂＝根席学生网＋对手席父代网；父代臂＝双席父代网；同根同种子（seed＝seed_cont＋step，默认 995400＝闸门 du 口径；裸网路径上数值惰性、仅作口径与落账，与闸门 greedy 臂一致）；生产 guard 双臂齐备；du 按根配对、按局聚类（t 区间为主＋聚类 bootstrap 并报，均为闸门自身实现）。
- **为何不直调 continue_game**：它把单个 pol 绑双席，无法表达「对手＝父代」。`continue_game_2pol` 是其裸网路径（guided_KMH=None、first_labels=None 的 greedy 臂调用形）的逐行映射改编，仅改每席策略一处；保真由验收 2 逐值证明。
- **无教师隔离**：驱动从不 import 闸门/教师模块；所需纯原语按 AST 白名单从源码提取、受控命名空间编译——闸门 15 个纯函数＋教师仅 `_extinct_state`/`_terminal_u`/字面量 `TEACHER_GUARD_CFG`，以 shim `T` 绑定（无 evaluate_root，guided 调用结构性不可能）；导入时 AST 审计断言被提取的闸门函数对 `T` 的引用仅限 TEACHER_GUARD_CFG。产物头记录两源文件 md5 钉每次运行的提取基。
- **根集硬校验**：显式清单输入；窗口默认 step 200–299（可配）、step 必须 < 该局首终局帧（padding 根直接 die 点名）、game id 与回放文件名一致、(game, step, me) 不许重复；根集身份 roots_sha256（{game, step, me, replay_md5}）入头且逐行落盘。行含 game/step/me/seat、u_student、u_parent、du、seed、双权重 md5、roots_sha256。

## 验收逐项（证据在 evidence/，脚本在 fixtures/）

1. **同权重 ΔU 严格为 0 — PASS**（f1）：双臂同为正典 heads v2（06a03d9c…），2 根（step 260/285）每根 du 恰为 0.0、ΔU == 0.0；整跑重跑逐行逐位相同（f1_run1/f1_run2.json）。
2. **终局计分逐值一致 — PASS**（f2）：同 2 根，驱动 u_parent 与 fixture 直调闸门 `continue_game`（闸门自身 extract_scene＋greedy 臂调用形）逐位相等：0.1555330393366056 / 0.16894850197364647；后者与独立计时探针三度吻合。CLI 学生跑的 u_parent 与此再同（四度）。
3. **无教师证明 — PASS**（f3）：(a) 静态 AST 扫描：k0_duel 的 import 集合无任何 teacher/stage 模块；(b) 动态：子进程以 import 钩子硬阻断 stage0_teacher 与 stage1_teacher_gate，驱动照常完成一局根 duel，且两模块始终不在 sys.modules。
4. **聚类统计自检 — PASS**（f4）：构造 4 局已知差值，t 区间与独立 scipy 参考、聚类 bootstrap 与按规格独立实现，均逐位一致（口径：按局均值、df=G−1、default_rng(995400) 游戏级重采样百分位）。
- **附带**（f0）：五种坏清单（越窗/step=FT/重复/局名不符/席位非法）逐一按名 die；CLI 全路径（扰动 student vs 父代，2 根）ΔU=+0.0550、产物字段齐全（cli_student.json）——此数为 fixture 扰动权重对父代的示意值，非任何真实测量。

## 如实注记

- 本 VM（1.5 核共享、同期多线争用）单条续弈墙时约 50–190 s（step 285 约 14 步续弈 CPU 约 15 s），fixture 因此只取晚窗 2 根；生产规模（约 2,000 局）须在 DGX/Studio 或以 --workers 并行跑，本机不作数。
- 首轮 fixture 链的 CLI 段曾被环境静默杀死一次（无 traceback、进程树消失），改由跟踪后台单独重跑完成，exit=0；驱动代码与此无关。
- G=2 时聚类 bootstrap 区间天然过窄（只能重采样 2 个局均值）——统计实现即闸门原实现、按注册口径双报，口径本身不在本件改动范围。
- 本件只实现＋fixture 自验，未跑任何真实测量、未碰机器进程。
