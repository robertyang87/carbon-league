# 合并 publisher 建成交付（工程组，2026-11-18）

- 工具：`tools/merge_annot_generations.py`，md5 `f1f5ab9c79ffd6fb099866686b18861e`，707 行。只 import 复用 `k0_join_u`（9054aadc，未动）；钉版/已验收文件零改动。
- 规格链：可行性件 bd36c4fe ＋ 裁定件 7283a3f3（合并身份/采样器）＋ 策略组三点回裁（注册先行/一种子派生/内容寻址）＋ 去重口径背书，逐项实现、自审 14/14（主代理独立复跑同数）。

## 实现要点

- 前置：各源经真 loader 全套自验；跨源 die 点名——format、code_md5s 集、weights_md5、config 逐字段仅 window 可不同、字段 dtype/尾形。
- 采样：逐源独立、(game, step, seat) 稳定序无放回均匀抽 floor(根数×率)；full 源全取。
- 种子：全代一种子必填；逐源派生种子＝sha256(注册种子‖源代 id) 前 8 字节，规则入注册件 `seed_derivation`、派生值逐源入清单。
- 身份：teacher_config_hash＝规范化来源清单（canonical JSON）的 sha256，三处同写（producer.config_sha256／producer.teacher_config_hash／meta.teacher.config_sha256），不借用任一源代哈希；来源清单同入 producer 与 meta 的 merged_from。
- 注册先行：发布序＝封印→真 loader＋并集窗口记录闸预检→代落盘（同代复用不覆盖）→REGISTRATION.json 落盘→current.json 最后翻；--verify 含注册件与代内清单 canonical 逐字一致＋派生种子复算两查。
- **去重口径确认（应策略组背书单列）**：重叠根**逐字节同才可去重、不同即 die 点名**已实现——自审证据 T9：构造同根不同字节样本（(selfplay-900001, 25, 0) 的 Q 改 +0.5、源自身记录闸仍过），die 原文点名根与字段并明示「refusing to pick one, average, or splice」，原文存工程组本地 `annotate-batch2/mixbuild/tests/evidence_overlap_die.txt`。

## 自审（合成小样本，未碰生产数据）

14/14 PASS：合并 full+sample 根数与实抽（40＋8=48)、--verify PASS、真 loader 直读合并代 PASS、双跑确定性（同 gen id＋逐字段 sha 全等）、同 out 复用不覆盖；负例：config 非 window 字段 die 点名 train_temp、weights die 点名两值、重叠同字节正例去重 4 根、重叠不同字节 die（上行）、删注册件 verify die、篡改注册清单 verify die。

## 一点说明

格式依据为一手源码（k0_annotate_queue 的 publish 原文＋join_u loader 契约）与本线早先读回的真实合并代 manifest（字段逐项吻合）；自审 fixtures 能被真 loader 全套验过。验收期间 L20 真实代 meta.json 复读遇本机 known_hosts 轮换窗口未成行，如需该旁证可于窗口过后补读。

—— 工程组 2026-11-18
