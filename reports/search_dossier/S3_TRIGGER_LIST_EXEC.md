# S3 执行版触发清单（搜索组落字 · 2026-11-16）

依据：预注册 PREREG_ECHO4_TACTICAL_SEARCH（现行版 md5 19c6bc43）覆盖表＋ S3 协议附则（8167b5e5）§1／§7＋策略组 §7 裁定（不开变体 CONF 补测；生产 CONF 下不成立的类如实记「不覆盖（触发不成立）」；S3 裁定级范围收窄为实际发生的触发类）。本清单把三类注册战术的触发条件逐类写死为可执行判据，并逐类判定其在生产 CONF 下是否实际发生。判定全部以代码证据为准（文件＋行号），不凭印象。

## 0. 口径基座

**生产 cfg 的组装**：`make_search_agent(conf)` 内 `cfg = dict(DEFAULTS); cfg.update(conf or {})`（cnet2_search.py:621–623，源码 md5 94f7fb64，下同）。即生产 cfg ＝ DEFAULTS ⊕ Echo4 包 CONF，后者覆盖前者。

**Echo4 包 CONF 逐字**（code/packs/echo4_int8.py:28，包 md5 819250bfc82632786063dabd46a616bd；包内 :76 原样传给 make_search_agent）：

```python
CONF = {"SEARCH": False, "ROLL": "self", "GUARD": True, "GUARD_NARROW": True, "MODE": "sweep", "MIX": False, "TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240, "BUDGET": 0.7, "HARD": 0.85, "ABORT_DL": 0.95, "START_DL": None}
```

**DEFAULTS 相关项**（cnet2_search.py:38–44）：`KEY_R=2, MAX_KEY=3, MINC=10.0, HUNT=True`。DEFAULTS 与 CONF **均不含**下列键：`KEY_LATE_HOME`、`KEY_LATE_CARGO`、`KEY_LATE_STEP`、`MAX_KEY_LATE`、`THREAT_FIRST`、`SPLIT`、`PLANTER_KEY`、`BASE_KEY`、`EH_STEP`、`LEAF_THREAT`、`WT_LEFT`。这些键在代码中一律以 `cfg.get(...)` 读取，缺省即 None／假——对应变体路径不执行。

**触发的唯一入口**：根枚举的触发面就是 `key_units(obs, cfg)`（cnet2_search.py:386–478）；生产调用点在 `_search` 保留段 :1506（以生产 cfg 原样调用）。S3 根枚举（基线包 SEARCH=False 自博弈、种子块 986200–986499、每步筛根，附则 §1）即以本清单判据对每步 obs 复算 key_units 的分类结果。

**可观测量**（全部取自 obs 与 cnet2_feat 的静态表，无隐藏量）：

| 量 | 来源 | 说明 |
|---|---|---|
| `step` | `obs["step"]` | 当前步；终局步 299（权威口径，最后可行动根步为 298）；`left = 299 - step`（key_units:439 同式） |
| `me` | `obs["player"]` | 本席 |
| 工人表 | `obs["players"][p][2]` ＝ `{uid: (pos, cargo, kind)}` | pos＝格号 0–224、cargo＝载货量、kind ∈ {COLLECTOR, PLANTER} |
| 基地表 | `obs["players"][p][1]` ＝ `{槽: 格号}` | 本席基地格集合 |
| 距离 | `F.DIST[a, b]`（cnet2_feat.py:54） | 15×15 环面曼哈顿距离矩阵 |
| 现金 | `obs["players"][p][0]` | 仅 PLANTER_KEY／BASE_KEY 变体用，本清单三类不用 |

## 1. 类一 · 即时载货碰撞／逃逸 —— **实际发生（裁定级照注册成立）**

**触发判据**（key_units:389–405 基础段，无步数门，全步带 0–298 均可触发）：对本席每个工人 `w=(pos, cargo, kind)`，当且仅当 `kind == "COLLECTOR"` 且存在敌方 COLLECTOR `e`（`obs["players"][1-me][2]` 中 kind 为 COLLECTOR 者）满足下列任一方向时，该工人成为关键根（stake 取两方向最大值，>0 才入选）：

- **1a 受威胁／逃逸向**（:399–400）：`DIST[pos, e.pos] <= KEY_R(=2)` 且 `e.cargo <= cargo` 且 `cargo >= MINC(=10.0)` → stake ＝ `cargo × PRIO_THREAT`（`cfg.get("PRIO_THREAT", 1.0)`，CONF／DEFAULTS 均未设 → 1.0）。语义：我方载货车在敌车 2 格内、敌货不重于我货——碰撞时我方货物处于可被夺状态。
- **1b 猎杀向**（:401–402）：`DIST[pos, e.pos] <= KEY_R(=2)` 且 `e.cargo > cargo` 且 `e.cargo >= MINC(=10.0)` → stake ＝ `e.cargo`。此向多一道门 `cfg["HUNT"]`：DEFAULTS 为 True、CONF 未覆盖 → **门开**。

入选后按 stake 降序（:404），最终截断 `keys[:MAX_KEY]`，MAX_KEY＝3（:452、:463）。根的分类即按产出它的工人登记；同一步可产出多根（≤3）。

**变体子情形（均为类一内部的排序／选取变体，不另立类）**：

- **1c THREAT_FIRST**（:407–419）：门 `cfg.get("THREAT_FIRST")` ——键缺省 → **关**。若开，会以「受威胁且 cargo ≥ THREAT_MIN（默认 100）」的字典序优先整体替换排序。生产口径下不生效，类一按基础段排序。
- **1d SPLIT**（:420–435）：门 `cfg.get("SPLIT")` ——键缺省 → **关**。若开，选取改为至多 1 名逃逸者＋1 名猎杀者。生产口径下不生效。

**判定**：类一是 key_units 的默认触发面，不依赖任何缺省关闭的变体键；1a／1b 两向在生产 cfg 下均生效。S3 裁定级范围含类一，照注册执行。

## 2. 类二 · 返航交付 —— **不覆盖（触发不成立）**

**唯一触发路径**：KEY_LATE_HOME 末盘返航段（key_units:437–451）。其判据（假设门开时）为：`step >= KLH_STEP`（`cfg.get("KLH_STEP", 270)`）且本席存在 COLLECTOR 满足 `cargo >= KLH_MIN`（默认 100.0）、基地表非空、设 `d = min DIST[pos, b]`（b 遍历本席基地格）、`left = 299 - step`，且 `0 < d` 且 `0 <= left - d <= KLH_SLACK`（默认 4）——即「载重车离家步数恰与剩余步数紧相差 0–4 步」。命中集合非空时该段**整体替换**当步关键集并提前返回（:449–451；`KLH_SHARE` 缺省为假，不做奇偶步分摊）。

**子情形拆分**：

- **2a 紧迫返航（上列 KLH 段）**：门 `cfg.get("KEY_LATE_HOME")`（:437）——该键不在 CONF（包 :28 逐字如上）、不在 DEFAULTS（:38–44）→ cfg 中为缺省 None → **整段不执行**。
- **2b 非紧迫返航**（载货车一般性回家交付、不带「紧相差」条件）：key_units 内**不存在**此触发路径——基础段只认对敌 COLLECTOR 的捕获几何（§1），不认与自家基地的距离。除 KLH 段外没有任何以「距自家基地」为条件的关键根触发。故此类无第二子情形可活。

**同名机制备忘（非触发）**：规则层 `endgame_home`（cnet2_search.py:1807 定义、:2044 调用）是搜索后动作覆写，不是根枚举触发；且其门 `cfg.get("EH_STEP") is not None` 中 EH_STEP 同样不在 CONF／DEFAULTS → 生产下亦关。此条仅防混淆，不改变判定。

**判定**：类二在生产 CONF 下触发不成立。本趟 S3 记「不覆盖（触发不成立）」，理由＝唯一触发路径被缺省关闭的 KEY_LATE_HOME 门挡住；不以变体 CONF 的根充数（策略组 §7 裁定）。

## 3. 类三 · 近步限现金清算 —— **不覆盖（触发不成立）**

**子情形拆分**（此类的「晚期」机制在 key_units 内有两处，须分开判）：

- **3a KEY_LATE_CARGO 主段**（:455–462）：判据（假设门开时）为 `step >= KEY_LATE_STEP` 且本席每个 `cargo >= KLC_MIN`（默认 10.0）的 COLLECTOR（未被类一选入者）按 cargo 降序追加进关键集（与类一根同受 `keys[:mk]` 截断）。**双重门均关**：① 门 `cfg.get("KEY_LATE_CARGO")`（:455）——键不在 CONF／DEFAULTS → 关；② 步门 `cfg.get("KEY_LATE_STEP", 999)`（:455）——该键同样不在 CONF／DEFAULTS → 取默认 999，而步数域为 0–299，`step >= 999` 永不成立。任一门单独即足以使本段永不执行。
- **3b MAX_KEY_LATE 扩容**（:453–454）：`cfg.get("MAX_KEY_LATE") is not None` 且 `step >= KEY_LATE_STEP` 时把截断上限 mk 由 3 换成 MAX_KEY_LATE。它**不是独立触发**（不新增任何根，只放宽类一根的保留数），且 MAX_KEY_LATE 键缺省为 None、步门同为默认 999 → 生产下关，mk 恒为 3。
- **3c 与类二的分界**：近步限且「回家路紧」的根属类二 KLH 段（§2），不计入类三，避免双计；类三的语义是「无威胁条件下把全部载货车纳入搜索做末盘清算」，与类二的紧迫返航判据不同。

**易混键澄清**：CONF 中的 `"LATE_STEP": 240` 不是 KEY_LATE_STEP——它在 :843–857 被用于对手教师相位切换（TEACHER_LATE 自第 240 步起），与 key_units 的任何触发门无关，不得据此认为晚期触发已开。

**判定**：类三在生产 CONF 下触发不成立（主段双门皆关、扩容项同关）。本趟 S3 记「不覆盖（触发不成立）」，理由如上；不以变体 CONF 的根充数。

## 4. 三类之外的 key_units 段（备忘，不在注册覆盖内）

- **PLANTER_KEY**（:464–473）：门 `cfg.get("PLANTER_KEY")` 缺省 → 关；其对象是种树位选取，属注册覆盖表「不覆盖」类（长期种树等）。
- **BASE_KEY**（:474–477）：门 `cfg.get("BASE_KEY")` 缺省 → 关；其对象是晚期招募槽搜索，不属注册三类。
- **叶内晚期计价变体**（LEAF_THREAT :681、WT_LEFT :684）：是启发式叶的计价开关，不是根枚举触发；两键亦不在 CONF／DEFAULTS。仅备忘，防与触发混读。

## 5. 结论汇总（S3 执行口径）

| 类 | 注册等级 | 生产 CONF 下判定 | 决定性证据（一行） | S3 处置 |
|---|---|---|---|---|
| 一 · 即时载货碰撞／逃逸 | 裁定级 | **实际发生**（1a／1b 两向均活） | key_units:389–405 基础段无变体门；KEY_R／MINC／HUNT 由 DEFAULTS:38–44 给定（2／10.0／True），CONF 未覆盖 | 裁定级照注册执行；根按类一登记 |
| 二 · 返航交付 | 裁定级 | **不覆盖（触发不成立）** | 唯一路径 key_units:437 门 `cfg.get("KEY_LATE_HOME")`，该键不在包 CONF（echo4_int8.py:28）亦不在 DEFAULTS:38–44 | 记不覆盖＋本理由；裁定级范围收窄，不充数 |
| 三 · 近步限现金清算 | 裁定级 | **不覆盖（触发不成立）** | 主段 key_units:455 双门：KEY_LATE_CARGO 键缺省、KEY_LATE_STEP 缺省默认 999 超出步数域 0–299 | 记不覆盖＋本理由；裁定级范围收窄，不充数 |

据此，本趟 S3 的裁定级触发范围先行声明收窄为**类一（即时载货碰撞／逃逸）**一类；类二、类三以「不覆盖（触发不成立）」入册，与附则 §7 及策略组裁定一致。根冻结名单（附则 §1）按本清单判据在种子块 986200–986499 上枚举：预期入册根全部属类一，若枚举中出现任何自称类二／类三的根，即判据实现与本清单不符、按附则 §3「触发判据误报」记无效并停报核查。
