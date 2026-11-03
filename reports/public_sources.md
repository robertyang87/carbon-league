# Carbon League 全站公开源码普查（2026-11-03）

## 头条结论

- **没有发现评分高于 v53 谱系（2030）的新公开源码，也没有任何榜首家族的源码公开。** claude-AlphaCarbon-2（榜 1，2561）、NaMnSi-DeepSeek-RL-policy-1（榜 2，2527）、AgyCarbon 系的策略 id 均在本次枚举内并逐一探过 `/source/<id>`，全部返回 401（非公开）。
- **最重要的新发现：`claude-opus-v18` 本体源码是公开的**（此前判定「不可达」有误）。它以两个策略 id 各注册了一次、文件字节完全一致（54,612 B，md5 相同）：原名 `claude-opus-v18`（id `ce2614d195152e25e998a7b072554290`）与别名注册 `claude-opus-v18-public`（id `81bd1b6c524e080d327bfd61cf97ff8b`）。它是 v53 的直系祖先、半站 fork 的源头，rho=0.02 等原始参数组在此可查。
- **一处澄清**：本次下到的 `DeepSeek`（id `7aa9d58f…`）是手写规则策略（自称 v3：树引擎＋放牧采集），**与榜 2/3 的 NaMnSi-DeepSeek-RL 不是同一份东西**，只是名字撞车，不能当 DeepSeek-RL 的源码或克隆底本。
- **全部 26 份公开源码清一色是手写规则/DP 代码，零 NN、零 torch、无内嵌训练权重**（v53/GE 内嵌的是预解猎杀博弈表，不是 NN 权重）。全站的 NN 强策略无一公开源码——公开生态的天花板就是 v53 这类工程化规则实现。

## 方法与覆盖

- 枚举 id 共 **835** 个（全站注册策略约 864 份，覆盖约 97%）。来源：lb_live.json＋replays/leaderboard.json 两份榜单快照（265）、replays/index.json 快照 400 局、replays/accumulated 索引 44 个名字（全部对齐到 id）、`/api/matches` 非 compact 分页 offset 0–20000（约 20,000 局采样，含双方策略 id）。
- 对每个 id 串行探 `GET /source/<id>`（间隔约 0.4 s，无并发）：**公开（HTTP 200 且内容为 Python 源码）23 份新下载**，加先行已知的 3 份，共 **26 份**；其余 **809 份返回 401**（须登录＝非公开）。唯一一次 502（Guihang_v2）已复探，确认 401。零未决错误。
- 未覆盖的约 29 份策略从未出现在上述任何对局、榜单或回放快照中（推测为从未打过对局或极早期已注销），无法取其 id，未探。
- 评分列：仅 Top 200 内策略有站内评分；26 份中只有 GloriousEvolution-v53-rho06 在榜（1653），其余均在 Top 200 之外、无公开评分，强度线索只能来自谱系与代码体量，报告中逐项标注。

## 全名录（26 份）

| 策略名 | id | 评分 | 大小 | 本地文件 | 备注 |
|---|---|---|---|---|---|
| 唯一一个人工写的策略（juzi） | a938df4254388b8df8ee04f6e6e0aae5 | —（参考策略） | 43,760 B | reference_juzi.py | 已知，先行在库 |
| claude-opus-v53 | 97898ce7047c0bb1dd7a13f02eff0dac | —（变体 nocamp 2030，第 65 名） | 115,897 B | reference_claude_opus_v53.py | 已知，我方标杆基座 |
| GloriousEvolution-v53-rho06 | 6fb751529a6cd9fdd3e337a467ba1871 | 1653（第 196 名） | 116,281 B | reference_gloriousevolution_v53.py | 已知，Top 200 内唯一公开源码 |
| claude-opus-v18 | ce2614d195152e25e998a7b072554290 | —（fork safebank 1626） | 54,612 B | reference_claude_opus_v18.py | **新发现：v18 本体** |
| claude-opus-v18-public | 81bd1b6c524e080d327bfd61cf97ff8b | — | 54,612 B | reference_claude_opus_v18_public.py | 与上一行字节完全一致的别名注册 |
| 1231223 | e130b5557452a1ba16fe34cdf6d55b8c | — | 57,562 B | reference_1231223.py | v18 衍生，与 idontno 字节一致 |
| idontno | cf45a5d6ed13d75c4ae7e5062352c8ee | — | 57,562 B | reference_idontno.py | 同上，同一文件两个注册名 |
| 无敌v4 | 4afa1a835a07216b7af4001ccf552896 | — | 55,742 B | reference_v4.py | v18 改写（docstring 自称「无敌V1」） |
| 无敌v5 | 89ac01e3ec252acd5cd3e32a32ab6747 | — | 58,581 B | reference_v5.py | v18 改写二代（自称「无敌V2」） |
| 无敌 | c15a21dbc65286f50a8b45d7704008d4 | — | 16,718 B | reference_cn_c15a21db.py | 独立「预期收益」规则线 |
| 无敌v1 | 8165457daea525f791f0ec7bb9340a69 | — | 24,736 B | reference_v1.py | 同上，二代 |
| 无敌v2 | b73319cfcf327a8ef477d1432e019d54 | — | 29,066 B | reference_v2.py | 同上，三代（修死锁） |
| 无敌v3 | cd591bb1e5f64d59f660c39113804506 | — | 30,040 B | reference_v3.py | 同上，四代（搞钱版） |
| ScherzOv1.2 | ae293cad1e06e4b0661da04e345491e2 | — | 34,205 B | reference_scherzov1_2.py | 独立实现，受公开版 juzi 启发 |
| DS-Codex-Canopy-v1 | bd294950e2dac72c652c53ef76c2b8c3 | — | 37,149 B | reference_ds_codex_canopy_v1.py | 设计参考 juzi＋ScherzO |
| glm5.3 | 8dd042f93b69d680f741781b2c1d5645 | — | 33,105 B | reference_glm5_3.py | 独立强规则（strong_agent v2） |
| deepseek-carbon-v64a | f69b73990db6b15038ee2e64d03f2fba | — | 25,054 B | reference_deepseek_carbon_v64a.py | 独立规则线 v64 迭代 |
| DeepSeek | 7aa9d58ff520c5356b4eca7990c0f15c | — | 13,444 B | reference_deepseek.py | 手写规则版，与 NaMnSi-DeepSeek-RL 无关 |
| ace-t87-fix4 | 529040f0f219a0436693824140abb540 | — | 23,094 B | reference_ace_t87_fix4.py | 独立规则线 |
| ace-t88-stand | bc5c5b7603b5f89916965d263b93bc88 | — | 24,385 B | reference_ace_t88_stand.py | 同作者下一代 |
| XiaoD-Bowl-v4 | 73a5a270880e59936e3f866afceed6ff | — | 24,801 B | reference_xiaod_bowl_v4.py | 对照榜一回放曲线调参的规则实现 |
| KIMI-K31_Coupling-CarbonV1 | 1846b26ebdb93a47c020d3311716e6a7 | — | 8,930 B | reference_kimi_k31_coupling_carbonv1.py | 小型规则基线 |
| KIMI-K31_Coupling-CarbonV2 | 6ff26df94cc087f521772e86f8116604 | — | 8,930 B | reference_kimi_k31_coupling_carbonv2.py | V1 小改（修探索死锁等 12 行 diff） |
| KIMI-K31_Coupling-CarbonV3 | c1658d2e88adf078f94707be07011379 | — | 9,648 B | reference_kimi_k31_coupling_carbonv3.py | 加敌方基地蹲守反制 |
| zyh-v01-farm-rotate-1024 | 0423b5635d5df2c09073f977bd119fea | — | 12,931 B | reference_zyh_v01_farm_rotate_1024.py | 入门规则 |
| 土匪帮主 | c6171c4a0ec01dec933d18d6d31922cb | — | 9,205 B | reference_cn_c6171c4a.py | 入门贪心 |

注：站点显示名与 docstring 自称版本号在「无敌」系里错位（站点「无敌v4」的文件自称无敌V1、「无敌v5」自称无敌V2，均为 v18 改写；站点「无敌/无敌v1/v2/v3」的文件自称 v2/v3…，是该作者独立的预期收益线）。上表一律以站点注册名为准。

## 谱系小结（据各文件头 docstring 的 Derived-from 自述）

- **v18 家族（Claude 原创 → 全站 fork 源头）**：claude-opus-v18 本体（两 id）→ 衍生 1231223/idontno（自述：本地自博弈重调参＋防御兜底）、无敌v4/v5（自述：基于 v18 改写，目标函数改收益/时间）。加上不公开的 v18-safebank（本次探得 401）与后续 v53，构成 Claude 谱系主干。
- **v53 家族**：claude-opus-v53 → GloriousEvolution-v53-rho06（自述：仅改 rho 0.02→0.06）。其余 v53 配方 fork（NaMnSi/Kimi/MiniMax 版）源码均不公开。
- **juzi 引用系**：ScherzOv1.2（自述：读过公开的「去模型化 juzi 参考策略」，借想法未抄码）；DS-Codex-Canopy-v1（自述：设计参考 juzi 与 ScherzO 两份公开源码的 URL）。
- **独立原创各家**：无敌（预期收益线）、ace（t87/t88）、glm5.3、deepseek-carbon（v64a）、DeepSeek（v3）、XiaoD-Bowl、KIMI-K31、zyh、土匪帮主——均自述或从代码可判为从环境规则独立推导。

## 每份新源码一句话定性

- **claude-opus-v18（本体）**：规则/DP——与 v53 同架构（镜像格精确记账＋全图 DP＋一步硬安全＋边际树价值）的早期精简版（1,253 行 vs v53 的 2,072 行，无猎杀博弈表）；是研究「v53 比 v18 多做了什么」的现成对照，也是祖先级陪练。
- **1231223 / idontno**：规则/DP——v18 的重调参衍生（银行节奏 rho、种植阈值、更早蹲守、额外早期种植员），附加一个异常状态防御兜底 agent；同一文件注册了两次。
- **无敌v4 / 无敌v5**：规则/DP——v18 改写两代，核心改动是把目标函数换成「收益/时间单位」并围绕三条「钱要赚得早」的规则事实重构。
- **无敌 / v1 / v2 / v3**：规则——同一作者的独立「每步预期收益评估」线，四代迭代主要在修线上暴露的死锁与送人头问题；中等体量，无信念图全图 DP，强度线索一般。
- **ScherzOv1.2（自称 "arbor" v1）**：规则/DP——独立实现的信念图（转移规则重演＋四折镜像补全）＋采集速率规划（单位时间带货量）＋相邻猎杀/威胁规避，是独立阵营里完成度最高的一份。
- **DS-Codex-Canopy-v1**：规则——确定性单文件实现（A*/堆搜索），明确站在 juzi＋ScherzO 两份公开源码的设计上，无自有信念图创新。
- **glm5.3（自称 strong_agent v2）**：规则/DP——独立信念图实现（未见格用四象限对称＋21.6 均值先验、按再生/吸收/采集演化）＋满编经济（6–7 采＋3 种）＋抢树与对抗战术，独立阵营里的另一份硬货。
- **deepseek-carbon-v64a**：规则——镜像对称映射＋拦截感知逃逸路由（把每个轻载敌人沿贪心追击线投影 1–3 步定价），其 docstring 记有自家 v18→v20b→v64 的长迭代史；护货思路对我方案 C 定价有参考价值。
- **DeepSeek（v3）**：规则——树引擎＋放牧式采集（格子只放牧不采绝），参数表驱动；与榜上 NN 版 DeepSeek 无关。
- **ace-t87-fix4 / ace-t88-stand**：规则——从环境源码机制逐条推导的「Carbon Neutrality」线两代，docstring 是全套机制笔记，代码中规中矩。
- **XiaoD-Bowl-v4**：规则——个人实现，特点是拿榜一回放的存款曲线对标，v4 把固定货舱上限改成 45→160 的动态爬坡。
- **KIMI-K31_Coupling-CarbonV1/V2/V3**：规则——238–251 行的小型基线（探索/采集/交付解耦），三代依次修死锁、加敌方基地蹲守反制；强度有限，适合做低压陪练。
- **zyh-v01-farm-rotate-1024**：规则——入门级轮换采矿（采到阈值换格防绝户）＋回程交付＋避碰，注释质量高、强度低。
- **土匪帮主**：规则——入门贪心（最近碳、最近基地、简单避让），全套最弱一档。

## 对陪练池的建议（供主线裁决）

1. **claude-opus-v18 本体**入池作「祖先陪练」：结构改动在 v53 上生效后，可用它验证改动是否只是对 v53 特化；同时它是查 Claude 谱系原始参数的权威文本。
2. **glm5.3、ScherzOv1.2、deepseek-carbon-v64a** 是三份独立信念图实现，风格与 v53 不同源，适合做规则型多样性陪练（防对单一基座过拟合）。
3. 其余新源码强度线索均弱于 v53 谱系，不建议作主陪练；KIMI/zyh/土匪帮主可作低压冒烟对手。
4. 榜首家族陪练仍只能走回放克隆（影子）路线——本次普查确认没有任何捷径源码可下。

## 普查产物

- 名录数据：census_results.json（835 个 id 的逐个探测结果）、census_ids.json（id↔名字/评分映射）。
- 脚本：census_enum.py（枚举）、census_probe.py / census_probe2.py（探测，串行限速）。
- 源码：上表 23 份 reference_*.py 新文件（加先行 3 份共 26 份在库）。
