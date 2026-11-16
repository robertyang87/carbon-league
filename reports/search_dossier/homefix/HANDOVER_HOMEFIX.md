# HANDOVER — HOMEFIX composite package (D1 search + endgame return-home rule layer)

依据：`reports/REGISTRATION_COMPOSITE_RULE_LAYER_20261116.md`（commit f0a512764c，md5 ea5a2d793381b5b0d18cc575fcd46329）§1 钉版规则＋§2 复合形态。**按注册文件建，不按 3s2 原 go_home 语义建**（无 min_cargo、无 spare、无死等检查、无策略概率扫描）。

## 1. 成品身份

| 项 | 值 |
|---|---|
| 文件 | `echo4_search_int8_d1_homefix.py` |
| blob-ready 路径 | `/home/hatch/workspace/search-group/homefix/build_comp/echo4_search_int8_d1_homefix.py` |
| md5 | **25f144f3d9ecc87be9073d135d3ab8c8** |
| 大小 | **9,051,753 B**（≤ 10,000,000 ✓） |
| sha256 | c3085cd14a17c6c8f8389e48e84ea29a2a21840dc555fed3995597cfc6199798 |
| git blob sha1 | fa67f6c72b5a39cc09dfb6b9b84bae7051ca819a |

## 2. 内嵌六源 md5（5+1）

| 源 | md5 | 与 D1 关系 |
|---|---|---|
| cnet2_feat | 2a45b08b6f97508541e62867e408748e | 逐字同 |
| cnet2_np（actor） | d8c15ad4fdd8ee079363cf215d52daf3 | 逐字同（见 §5） |
| fastsim | aefe6c83826efae4de83bef8c643c80b | 逐字同 |
| cnet2_comparator | 39000607b8df5ce85a56437dcc27398d | 逐字同 |
| **cnet2_homefix（新）** | **9bb454703c830feb061821ded3c4e289** | 规则层模块 |
| cnet2_search（派生） | a9ab60a88cb13335721d2e0dee8f2e98 | cmp3 b8c77aca 的最小 diff 派生 |

## 3. 钉版规则实现口径（注册 §1）

- **触发**（每步、每名自方工人重估）：`cargo > 0` 且 `moves_available ≤ d + HOME_BUFFER`，`moves_available = 300 − obs["step"]`，`HOME_BUFFER = 2`。无其他参数。
- **动作**：覆盖为 RETURN STEP——目的格距离恰为 d−1 的移动；多解时按**钉死方向序**取首个；无此移动则原地等待（NONE）。
- **方向序（本构建钉死）**：游戏动作索引序（`cnet2_feat.MOVES` 下标）k=1,2,3,4 → **UP, RIGHT, DOWN, LEFT**。
- **最短路/静态障碍判定（查游戏代码得出，非假设）**：`fastsim.py` 移动为四向＋逐坐标环绕（`_MOVE_OFFSETS`/`_move_nbr`），step 阶段 1 无条件施加；树/基地/单位从不阻挡移动本身（碰撞是动态的、阶段 2 事后结算）。**棋盘无静态障碍**，故移动规则下最短路＝环面曼哈顿距离＝`cnet2_feat.DIST`，规则模块直接读该表（不自造度量），目的格读 `cnet2_feat.DEST`。
- **d=0 不触发的验证**：fastsim 阶段 3——站在自家基地的收集者即存仓（cash 增加、cargo 清零），故真实 obs 里基地上工人 cargo 必为 0；规则内另显式跳过 d==0。
- **形态**：出口 overlay——只作用于比较器最终联合动作；候选生成与比较不带规则。确定性纯函数（模块级 `LAST` 仅诊断侧信道、每调用覆写、规则从不读它）。
- **参数形态选择**：`HOME_BUFFER=2` **硬钉在模块内，不加 CONF 键、无关闭开关**。理由：注册 §1 只钉这一个参数；且 CONF 与 D1 逐字同使 F3 同块对照干净。**CONF diff vs D1 包 = 空**（解包后字面比对确认：BUDGET 0.7、CMP_H 1、CMP_TAU 30 等全同，无 HOME_* 键）。

## 4. 挂钩点（最小 diff）

- 生成器：`homefix/make_search_homefix.py`（md5 b4c7374b），锚点变换自 pristine cmp3（断言 md5 b8c77aca），可复现、同步门通过。Diff vs cmp3：**+25/−3 行**＝头部说明块＋`import cnet2_homefix as HF`＋三处返回挂钩，无 DEFAULTS 改动、无其他行。
- 挂钩位置：`cnet2_search.make_search_agent` 内层 `agent()` 的**三条最终动作出口**——无搜索早退（底本 1109 行）、搜索路径返回（1145 行）、异常兜底返回（1149 行）——各自在 return 前过 `HF.apply_homefix(obs, ·)`。agent.stats/cmp_stats 接线未动。
- 挂钩处可用数据：仅 obs（player、step、自方基地位、自方工人 [pos, cargo, kind]）＋最终联合动作 dict；几何表 DIST/DEST/MOVES 由规则模块读 feat。

## 5. Actor 字节比对（注册 §2 要求，显式结论）

解包 D1 包与复合包逐项比对（记录 `out/actor_compare.json`）：**cnet2_np 源码字节相同；权重组件 _B1/_B2/_B3、量化元数据 _Q1/_Q2/_Q3、_M1/_M2/_M3、_ENS 全部相同**；权重 blob（_B1）sha256 a4e70bafd68aa192dac804d05b7aac5cb530854c50a4a60c46ab24877c536d62。**actor 字节未变，结论 PASS。**

## 6. 自检结果（全数）

- **verify-pack PASS**：校验器 453f543a 的 MODS 扩展版（bf15b3a4，扩展方式与构建器同——第六个模块须进校验集）对 `tree_comp/code` 权威树：exit 0、ok=true、mismatches []、missing_embedded []、六源内嵌 md5 全合。记录 `out/verify_homefix.json`。
- **构建链可信**：复制树＋D1 CONF 先原样复现了 D1 包逐字（md5 e8ea1e0e 对上）才动工；`iter2/build_d1` 全程未改（副本在 `build_comp/` 作业）。
- **诊断等价（注册口径）**：probe_runner 形态、种子 7000、包坐 seat 0、诊断预算 BUDGET=30（仅诊断副本，交付包 0.7 未动），与 D1 诊断流（`iter2/out/probe_d1_30.json`）逐歩比对——**步 0–295（296 步）双座动作流逐字相同**；首个流差异在步 296，恰为触发工人 player-0-worker-46：D1 DOWN → 复合 RIGHT（钉版回家步，d=2），该步对手座动作相同；比较器遥测总量两边相同（calls 263、changed 1）。首触发（步 295）选步恰与策略自选相同（changed=False），流未变——恒等主张以更强形式成立（覆盖全为 no-op 的步也相同）。记录 `out/probe_homefix_30.json`、`out/equivalence_homefix.json`。
- **触发证据（自然局）**：同局 checked=1,341（带货工人评估数）、triggered=9、跨 5 步（295–299）、实际改值覆盖 3 次。**首触发元组 (step, cargo, d, moves_available) = (295, 24.2622, 3, 5)**（恰在 5 ≤ 3+2 边界）。worker-46 在规则下步 295–297 连续回家（d 3→2→1）。
- **合成状态测试 15/15 PASS**（`test_homefix_synth.py`，对解出的内嵌模块跑；日志 `out/test_homefix_synth.log`）：触发＋唯一缩距方向 LEFT；精确边界（moves=d+3 不触发）；cargo=0 不触发；d=0 不动；中局不触发；方向序平局 UP 先于 LEFT；环面绕行 RIGHT；无缩距移动→NONE 等待（防御分支——真实度量下 d>0 必有缩距邻格、该分支不可达，用临时替换的不一致度量强制行使后已恢复）；seat 1 对称；对手工人不动；确定性＋输入不被改写。
- **内嵌四套件**（成品抽源入 `tests_comp/`、iter1 同法）：tests_s1 **7/7**、tests_s1_fix **15/15**、tests_s1_fix_r2 **10/10**、tests_r2_fix **11/11**，均 exit 0（日志 `out/tests_*.log`）。
- **交付包冒烟**：成品包独立加载、与基线包对打 12 步无异常。

## 7. 口径备忘（非阻塞，供裁决/后续注册注意）

1. 触发式**无可达性条件**（注册公式原文）：d > moves_available 时也触发、只朝家走（如本局 worker-83 步 297：d=5、moves=3 照样触发）。按注册字面实现，实测在案。
2. 等待分支在真实度量下不可达（图距离 d>0 必有缩距邻格），仅作防御保留。
3. 方向序按派单授权由构建方选定并在此钉死（UP, RIGHT, DOWN, LEFT）；F3/F4 若显示方向序敏感，属新注册事项，本包不自改。

构建记录：`build_record_homefix.json`（同目录）。无 git 提交（归档由上层办理）。
