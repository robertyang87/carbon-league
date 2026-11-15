# 后继权威注册：内核再同步树入权威位（R2 第三批验收同批立）

依据：RULING_R2_20261116 §5 第三批＋RULING_R2_ADDENDUM_20261116 的
R2-10 预处置（再同步交付时策略组同步立后继权威注册）。交付件：
RESYNC_R2_BATCH3.md（md5 f0c61d4040ee6cd66f61f7172260ffec，commit
2fe76fc6）＋RESYNC_BUILD_RECORD.json（md5
3771d15639f4ffe6e848b1a9ff9b88e7，commit dd7facc4）。裁定方：策略组。
日期：2026-11-16。

## 1. 第三批验收：通过

六项交付逐项核过（策略组亲验范围见 §6）：

- **单一内核（R2-09 内核层）**：生产比较器改由 make_comparator.py
  （md5 f22e7f325f8d2b8bc4a572ee0ac12004）以三件冻结原型源
  （comparator 53e5c1a1…／telemetry 91a7121f…／leaf_adapters
  a5852c73…）机械装配，产物 cnet2_comparator.py md5
  **385190bd2f59d90f4e68a0d2dc6091a1**（23,572 B），再生成逐字节
  一致（同步闸）。段 0 手写分叉 ad3ec3e4 自此废弃——仓内不得再有
  第二份手写比较器实现。生成器自证 7/7（有限性／decide_multi／
  C_hi 分桶＋冷启豁免／deadline_abs／分阶段遥测／适配器有限性）
  亲读在案。
- **后继源（R2-09 对象层）**：make_search_cmp2.py（md5
  81f2d981fa8352423fdccf77c9e945bd）自钉版基源 94f7fb64… 重生
  cnet2_search_cmp2.py，md5 **c3222eea76ea9d42c160626c2e239ca0**
  （131,590 B；对基源 +342/−95、16 hunk、仅头注 9 条变更）。
  先验 lg0 序截断、H≥2 双席分源续弈 guard、按终局种类分发叶值
  （灭绝→F01／时间终局→F04）、C_hi (K,H) 分桶、deadline_abs 锚
  _inner 入口——五项修法与裁定口径逐项对位。
- **遥测穿包（R2-10）**：公开 agent cmp_stats 与内层逐键相等
  实测在案（calls=256 双通道一致）；stats["cmp"] 镜像经既有
  last_stats 通道穿包，eval 侧以 last_stats["cmp"] 消费，不需改
  eval 白名单；cmp_guard_err 改名 cmp_guard_errors 合 *_errors
  约定，续弈改写另计 cmp_guard_rewrites（非错误）。
- **新包（未去重版，明示；本地交付件、尚未入仓）**：
  echo4_search_int8_resync.py，9,041,707 B（≤10,000,000 ✓），
  md5 **cba5923c13fd629208a4991811723980**，blob
  c932a0cfdd041d9dbb2cd1a17a55713d7ee29522。五源包内复算全合
  （feat 2a45b08b／np d8c15ad4／fastsim aefe6c83／comparator
  385190bd／search c3222eea）；CONF 与现行包逐字一致；装配链
  等价已证（旧模块＋新打包器复现 be96d457、仅差遥测转发一行）。
  包本体随 §3 落地入仓，入仓读回 blob 合 c932a0cf… 为落地
  条件之一。
- **验包准备**：暂存权威树对新包 verify-pack PASS（五件全合、
  mismatches 空）；对照现行包恰 FAIL 于 [comparator, search]
  两件，后继关系落档。certify 非计时闸全过；计时三闸新旧同型
  FAIL 于本机口径（步均约 2.2 s），不作生产口径读数（见 §4）。
- **套件与行为**：内嵌模块本体套件 7/7＋15/15＋10/10＋11/11
  全绿（冻结 harness 未改）；CMP_H=1 A/B 两档 0–13 步逐字一致，
  step 14 分歧两类全归清（信封边界抖动类＋R2-09 注册修正类，
  根重放实证），无未归类分歧；anytime 三梯级保留实证在案。

**R2 销账**：R2-09 **关闭**；R2-04 生产侧关闭（S0b 处置见 §5）；
SI02／SI06 生产侧关闭。R2-10 的关闭点在本注册 §3 的仓内读回
验包，不在本裁定签字本身。

## 2. 后继权威集（本注册钉版）

搜索线权威模块自本注册起为**五件**：

| 权威位 | md5 |
|---|---|
| code/cnet2_feat.py | 2a45b08b6f97508541e62867e408748e |
| code/cnet2_np.py | d8c15ad4fdd8ee079363cf215d52daf3 |
| code/fastsim.py | aefe6c83826efae4de83bef8c643c80b |
| code/cnet2_search.py（后继源入位） | c3222eea76ea9d42c160626c2e239ca0 |
| code/cnet2_comparator.py（第五件，新设权威位） | 385190bd2f59d90f4e68a0d2dc6091a1 |

构建／验包器同批钉版：code/cnet2_search_pack.py（构建打包器）
a1382d0ede6e7fac7e605231f214f13b；code/packs/cnet2_search_pack.py
（验包器，MODS 扩五件＋模板转发）453f543a609328bd387171800b4cac9c；
code/packs/cnet2_pack.py（量化器，未改）18bf47b76a642aea2fd565ed9ed971f7。

注：feat md5 以构建记录 JSON 为准（2a45b08b6f97508541e62867e408748e）；
RESYNC_R2_BATCH3.md §5 表内 2a45b08b… 的展开位以本注册所引 JSON
全值为准。

## 3. 落地次序（R2-10 关闭条件）

1. 搜索组将暂存树八件按 §2 md5 逐字推入仓内权威位（不许改一字；
   任一 md5 不合即落地作废、先报后修），并同趟把再同步包推入
   code/packs/echo4_search_int8_resync.py（读回 blob 须合
   c932a0cf…、md5 须合 cba5923c…）。
2. 落地后搜索组对仓内权威树（--auth-dir 指向仓内 code/）复跑
   verify-pack，对象为再同步包 cba5923c…，回执须含五件
   embedded==authoritative 读回与 mismatches 空。
3. 策略组读回核对仓内八件 md5＋包 blob 与本节回执——**R2-10
   于该读回关闭**；在此之前仓内现行包 fd45f35b… 的
   verify-pack FAIL 仍是 R2-10 的在册状态。

## 4. 口径与范围注记

- **考试钉版集不受影响**：bigeval 354bcd1b／eval_v53 45792dfb／
  cnet2_search 94f7fb64／certify_pack fc33be6c 是 md5 寻址的考试
  钉版，考场机上副本不动；本注册换的是仓内权威位，不是考试
  钉版。任何线不得以「权威位已更新」为由改动在册考试口径。
- **按路径消费 code/cnet2_search.py 的线注意**：后继源与基源的
  差异全部落在搜索判读段（头注 9 条）；SEARCH=False 形态的
  CMP_H=1 A/B 证据（§1）支持非搜索行为不变的读数，但各线若有
  路径依赖，自行以 md5 钉版为准，不要默认跟随权威位漂移。
- **S2（11-17）不变**：仍以现行包 fd45f35b… 按诊断口径执行，
  结论范围仍限段 0 工件的诊断级计时证据。再同步包的生产口径
  计时认证在主体冻结时于 C1 以 S2 口径（种子 986500–986519）
  复跑，通过为 S4 开场前置——本批 VM 口径计时数不充生产认证。
- **再同步包不是生产候选**：生产候选在生产注册重建时产生
  （权威源构建＋路线 B 去重并入＋§5 回归全套在该工件上重跑）；
  去重线现状（A/B 链产出与 regression_evidence.json 未落）按
  构建记录如实项在册，去重工件仍只记「待验」，不因本批验收
  升级标签。L-vnet33 检查 4 仍为未达，待生产重建工件实测。

## 5. 两件如实项裁定

- **C_hi 本机尺度自锁**：裁定为 **VM 成本尺度伪影、口径内生**
  ——样本记录口径与冻结 harness（real_env f9bfacb7… 每次调用后
  无条件记样本）同构，本批保真不改是对的。处置：S2 诊断跑与
  主体冻结时的生产口径计时认证（C1）须**显式报告 budget_gate
  计数**；若在部署口径（步均 0.15 s 量级）下复现自锁，再回
  策略组议样本口径修正案（如剔除 deadline 杀死轮的样本），
  任何线不得自行改口径。本项不阻塞本批验收与本注册。
- **CMP_H=2 整局探针两趟被运行时重启吞**：披露接受；H≥2 非
  生产形态（出厂 CMP_H=1），帧计数＋收获根确定性探针的证据
  对本批足够，本项不阻塞验收。但「不再补跑整局」只限本批
  范围：**任何未来以 H≥2 为主张的注册（如生产注册拟用
  CMP_H≥2），整局探针为开场前置**，且探针须按所有者
  「阶段性可测」定规改增量落盘（逐段 flush），不许再用只在
  局末落盘的形态长跑。
- **S0b 成本表**：本批未单独交付生产入口口径成本表——并入
  主体冻结时的生产口径计时认证事件（C1，S2 口径）一并交付，
  内容仍须含候选／对手／未完赛三项分解；该事件前 S0b 记
  「已并单、未交付」。

## 6. 策略组亲验范围（本裁定依据）

构建记录两件 md5 与仓内 blob 逐字合；两件生成器
（make_comparator f22e7f32…／make_search_cmp2 81f2d981…）与
两件产物（comparator 385190bd…／后继源 c3222eea…，字节数
23,572／131,590）均亲读仓内文件复算合；比较器产物抽读实证：
decide／decide_multi 双入口＋deadline_abs 在位、有限性闸
（nonfinite_leaf）与 budget_gate 落位、适配器非有限值透传
计数；后继源抽读实证：C_hi (K,H) 分桶准入段（桶 max／全局
回退／无样本 None）与头注变更清单（单一内核声明、SI02 信封
锚、先验序截断、双席分源 guard 口径）在位。再同步包本体为
本地交付件、尚未入仓，策略组未亲下复算——其身份依据为：
构建记录在案＋五源中本批两件变更源（comparator／search）
已亲验与包内声明逐字合＋其余三源为在册钉版＋暂存树
verify-pack PASS 自证；包本体入仓读回列为 §3 落地条件，
读回不合即本验收对包一项自动失效、先报后修。套件四组与
行为探针为搜索组自证、冻结 harness md5 在案，本批采信其
自证（与第一、二批同口径：判读器类已由策略组亲跑复核过，
本批无新判读器）。
