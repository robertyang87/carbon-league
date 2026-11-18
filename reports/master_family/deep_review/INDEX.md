# Top 2：Master5s 与 Master6 逐局深读索引

代码与仓内回放冻结提交：`37965ea578851b5f3ff88ffc42d7119ff716b5cc`。范围：Master5s、Master6；如有官网补充记录，以实际来源栏区分。

元数据并集 **352 局**；有逐局分析 **352 局**；缺回放或分析失败 **0 局**。每局唯一键为完整 `matchid-game`，按每页约 12 局均衡分页。

[每局一行 CSV](all_games.csv) · [模型成绩 CSV](model_results.csv) · [状态截面 CSV](stage_summary.csv) · [完整汇总 JSON](summary.json)

## 读法与限制

- 逐局卡的 tN、交付、碰撞、窗口和状态表均指动作执行后可见的状态步；只有明确标为“决策步”的字段使用推进前源步（决策 t 产生 state t+1）。位置为 15×15 observation 行优先索引。
- 卡片关注方在前；CSV 与卡片采用同一视角。家族内对战只占一行，但在模型成绩中分别记入双方，因此模型出场次数之和可大于独立局数。
- 官方成绩来自元数据；阶段现金、空间与事件只来自有回放的局。缺回放不补零，提前终局不延长。
- 对手与席位不平衡，版本/时间不同；下表是语料描述，不是固定考官的配对实力实验。窗口、事件及胜负是事后观察，不能单凭它们声称某动作的反事实收益。
- 树 ID 前缀不表示当前所有权；击杀、捕获载货、银行交付是不同事件。完整隐藏图不构成策略可见输入。

## 按模型成绩（官方元数据）

| 模型 | 出场 | 有回放 | 胜 / 负 / 平 / 未知 | 平均分差 | 中位分差 |
|---|---:|---:|---|---:|---:|
| Master5s | 191 | 191 | 180 / 11 / 0 / 0 | +2,080.52 | +1,818.64 |
| Master6 | 163 | 163 | 144 / 18 / 1 / 0 | +4,892.77 | +2,050.07 |

## 状态截面（有回放的实际观察）

每一行各自报告有效局数。final 为每局真实最后状态，具体步数可能不同；均值不能与固定 t299 无条件互换。

| 模型 | 状态步 | n | 平均现金差 | 工人 / 对手 | 树 / 对手 | 载货 / 对手 |
|---|---:|---:|---:|---:|---:|---:|
| Master5s | 25 | 191 | -1.90 | 6.32 / 5.95 | 1.54 / 1.72 | 61.95 / 59.43 |
| Master5s | 50 | 191 | +5.63 | 9.47 / 8.62 | 6.22 / 3.95 | 232.20 / 185.73 |
| Master5s | 100 | 191 | +583.07 | 9.69 / 9.21 | 4.93 / 4.91 | 577.62 / 374.30 |
| Master5s | 150 | 190 | +997.01 | 9.63 / 9.30 | 5.50 / 4.68 | 530.69 / 401.58 |
| Master5s | 200 | 190 | +1,376.67 | 9.62 / 9.38 | 5.63 / 4.88 | 526.30 / 433.79 |
| Master5s | 250 | 190 | +1,694.32 | 9.68 / 9.59 | 5.89 / 4.53 | 502.96 / 459.07 |
| Master5s | 299 | 190 | +2,078.98 | 6.49 / 6.08 | 3.13 / 3.21 | 89.09 / 33.22 |
| Master5s | final | 191 | +2,079.63 | 6.50 / 6.05 | 3.17 / 3.19 | 91.04 / 33.05 |
| Master6 | 25 | 162 | -0.54 | 5.67 / 5.28 | 1.39 / 1.75 | 62.78 / 47.56 |
| Master6 | 50 | 162 | +26.93 | 9.21 / 7.31 | 4.87 / 4.83 | 143.69 / 112.10 |
| Master6 | 100 | 160 | +487.32 | 9.72 / 6.49 | 6.92 / 3.78 | 413.73 / 271.80 |
| Master6 | 150 | 160 | +1,478.57 | 9.82 / 6.54 | 7.09 / 3.44 | 502.44 / 274.15 |
| Master6 | 200 | 160 | +2,633.33 | 9.86 / 6.60 | 7.32 / 3.40 | 550.64 / 297.76 |
| Master6 | 250 | 160 | +3,801.97 | 9.82 / 7.21 | 7.26 / 3.21 | 571.64 / 341.95 |
| Master6 | 299 | 159 | +4,980.77 | 6.41 / 5.31 | 4.23 / 3.09 | 57.86 / 37.78 |
| Master6 | final | 163 | +4,889.57 | 6.44 / 5.18 | 4.26 / 3.02 | 69.16 / 36.85 |

## 实际回放来源

| 来源 | 局数 | 已分析 | 缺失 | 汇总 |
|---|---:|---:|---:|---|
| `data/krypton-earlyhome_corpus` | 2 | 2 | 0 | [JSON](by_source/data__krypton-earlyhome_corpus.json) |
| `data/master5s_corpus` | 154 | 154 | 0 | [JSON](by_source/data__master5s_corpus.json) |
| `data/master6_corpus` | 102 | 102 | 0 | [JSON](by_source/data__master6_corpus.json) |
| `data/master6s_corpus` | 1 | 1 | 0 | [JSON](by_source/data__master6s_corpus.json) |
| `data/xiaofeiyu5-d5_corpus` | 1 | 1 | 0 | [JSON](by_source/data__xiaofeiyu5-d5_corpus.json) |
| `https://117.72.210.134/replay` | 92 | 92 | 0 | [JSON](by_source/https_____117.72.210.134__replay.json) |

## 全部对局

| # | 完整身份 | 关注模型 / 对手 | 席位 | 结果 | 官方分差 | 回放 |
|---:|---|---|---:|---|---:|---|
| 1 | [`7edaeef85acdd6175b5151acfa9e4e16-1`](pages/games-001.md#g-7edaeef85acdd6175b5151acfa9e4e16-1) | Master5s / 哈基米 v3 | 0 | 胜 | +1,602.03 | 已分析 |
| 2 | [`e46fc0a3b033ea2662683665fc37c92d-1`](pages/games-001.md#g-e46fc0a3b033ea2662683665fc37c92d-1) | Master5s / 小肥鱼v5 | 0 | 胜 | +775.17 | 已分析 |
| 3 | [`ecdd3ad9e7a16a53aba5a4ed51e10b6e-1`](pages/games-001.md#g-ecdd3ad9e7a16a53aba5a4ed51e10b6e-1) | Master5s / stephen-K4TacSearch-v6 | 1 | 胜 | +2,310.13 | 已分析 |
| 4 | [`2ebe44bee72d74b66c5b69f09ce99575-1`](pages/games-001.md#g-2ebe44bee72d74b66c5b69f09ce99575-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 0 | 胜 | +4,062.51 | 已分析 |
| 5 | [`34822028068c905ca1bf90f45a005aef-1`](pages/games-001.md#g-34822028068c905ca1bf90f45a005aef-1) | Master5s / 小肥鱼v5 | 0 | 胜 | +1,698.44 | 已分析 |
| 6 | [`30330fcb8982d57b36f4c168fb663ced-1`](pages/games-001.md#g-30330fcb8982d57b36f4c168fb663ced-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +544.07 | 已分析 |
| 7 | [`a1de950526b7bfa32c15301e55ae8420-1`](pages/games-001.md#g-a1de950526b7bfa32c15301e55ae8420-1) | Master5s / ClaudeFable5.2-BC-v3 | 1 | 胜 | +1,189.95 | 已分析 |
| 8 | [`56eba1d3291f5e825fbbace1ab8bff3e-1`](pages/games-001.md#g-56eba1d3291f5e825fbbace1ab8bff3e-1) | Master5s / ClaudeFable5.2-BC-v2 | 0 | 胜 | +5,414.58 | 已分析 |
| 9 | [`b0de87e741fd0279b0430033a1e7abaf-1`](pages/games-001.md#g-b0de87e741fd0279b0430033a1e7abaf-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 1 | 胜 | +1,361.97 | 已分析 |
| 10 | [`85aa69e9678f9bd97020e27a8ef943e2-1`](pages/games-001.md#g-85aa69e9678f9bd97020e27a8ef943e2-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +1,660.48 | 已分析 |
| 11 | [`c2fd0dcd97929f12fe6bb3be60a89415-1`](pages/games-001.md#g-c2fd0dcd97929f12fe6bb3be60a89415-1) | Master5s / ClaudeFable5.2-BC-v3 | 0 | 胜 | +8,112.08 | 已分析 |
| 12 | [`2da59e846e326c701cabcd15fa6439bd-1`](pages/games-001.md#g-2da59e846e326c701cabcd15fa6439bd-1) | Master5s / ClaudeFable5.2-BC-v2 | 1 | 胜 | +3,763.54 | 已分析 |
| 13 | [`1717d1fced6509c6c4d224172b8c5beb-1`](pages/games-001.md#g-1717d1fced6509c6c4d224172b8c5beb-1) | Master5s / 哈基米基准 | 0 | 胜 | +1,729.87 | 已分析 |
| 14 | [`4524e00b340d74ca29a1195d16e2a5a3-1`](pages/games-002.md#g-4524e00b340d74ca29a1195d16e2a5a3-1) | Master5s / stephen-K4TacSearch-v6 | 0 | 胜 | +2,689.74 | 已分析 |
| 15 | [`8fe94754c17c07ab3a959ef7e3314d70-1`](pages/games-002.md#g-8fe94754c17c07ab3a959ef7e3314d70-1) | Master5s / 哈基米基准 | 1 | 胜 | +493.69 | 已分析 |
| 16 | [`db0193406ee795dba5018c02e6ebf506-1`](pages/games-002.md#g-db0193406ee795dba5018c02e6ebf506-1) | Master5s / 哈基米 v3pro | 1 | 胜 | +2,623.64 | 已分析 |
| 17 | [`36643173a8d08f3b96cd2818569c23a3-1`](pages/games-002.md#g-36643173a8d08f3b96cd2818569c23a3-1) | Master5s / stephen-K4TacSearch-v3 | 0 | 胜 | +3,209.58 | 已分析 |
| 18 | [`9bb4ea68a0933e630057edc70f8f5c29-1`](pages/games-002.md#g-9bb4ea68a0933e630057edc70f8f5c29-1) | Master5s / Fable5.2-Max-V7 | 1 | 胜 | +2,658.47 | 已分析 |
| 19 | [`1fd5e494424c26bd81a9e5f7ba2e6613-1`](pages/games-002.md#g-1fd5e494424c26bd81a9e5f7ba2e6613-1) | Master5s / 哈基米 v3pro | 1 | 胜 | +1,760.53 | 已分析 |
| 20 | [`3ed956099b0566c29e29e4847397fa4a-1`](pages/games-002.md#g-3ed956099b0566c29e29e4847397fa4a-1) | Master5s / claude-AlphaCarbon-Master4s | 1 | 胜 | +2,215.13 | 已分析 |
| 21 | [`3b5442e3b0936940f343013d4cb2179b-1`](pages/games-002.md#g-3b5442e3b0936940f343013d4cb2179b-1) | Master5s / Fable5.2-Max-V9 | 0 | 胜 | +14,528.74 | 已分析 |
| 22 | [`c0308ebb4028594b4c7182308854dbe6-1`](pages/games-002.md#g-c0308ebb4028594b4c7182308854dbe6-1) | Master5s / stephen-K4TacSearch-v4 | 0 | 胜 | +1,409.08 | 已分析 |
| 23 | [`7410f3684f58a457810e9b24835adea2-1`](pages/games-002.md#g-7410f3684f58a457810e9b24835adea2-1) | Master5s / Fable5.2-Max-V6 | 1 | 胜 | +2,619.95 | 已分析 |
| 24 | [`dbfe0909e055a4becb14f29d06d818ef-1`](pages/games-002.md#g-dbfe0909e055a4becb14f29d06d818ef-1) | Master5s / AgyCarbon-P2C-Godslayer-CVI | 0 | 胜 | +2,061.89 | 已分析 |
| 25 | [`00f7e221872bf7bc75a3847676e17b56-1`](pages/games-002.md#g-00f7e221872bf7bc75a3847676e17b56-1) | Master5s / 哈基米 v3 | 1 | 胜 | +2,658.63 | 已分析 |
| 26 | [`502cbf1c92561b42c2fa5b7135aa3daf-1`](pages/games-002.md#g-502cbf1c92561b42c2fa5b7135aa3daf-1) | Master5s / stephen-K4TacSearch-v2 | 1 | 胜 | +2,151.99 | 已分析 |
| 27 | [`8f9d8139832d7879ef40268826673ddd-1`](pages/games-003.md#g-8f9d8139832d7879ef40268826673ddd-1) | Master5s / Echo4 | 1 | 胜 | +3,683.61 | 已分析 |
| 28 | [`22c26fb4fc265297a69fb5ced1d15252-1`](pages/games-003.md#g-22c26fb4fc265297a69fb5ced1d15252-1) | Master5s / Echo5 | 0 | 胜 | +3,106.63 | 已分析 |
| 29 | [`48eacdd8b11157127897278abf814953-1`](pages/games-003.md#g-48eacdd8b11157127897278abf814953-1) | Master5s / Fable5.2-Max-V8 | 1 | 胜 | +7,338.32 | 已分析 |
| 30 | [`f39aed0b73cb0af85ac40496bc999a11-1`](pages/games-003.md#g-f39aed0b73cb0af85ac40496bc999a11-1) | Master5s / AgyCarbon-P2C-Godslayer-CVI | 1 | 胜 | +3,431.94 | 已分析 |
| 31 | [`dc3b757bc9fe90a93e6f377d255a70fc-1`](pages/games-003.md#g-dc3b757bc9fe90a93e6f377d255a70fc-1) | Master5s / Krypton-M128Search-v1 | 1 | 胜 | +3,594.38 | 已分析 |
| 32 | [`70cd568b536cf94ab703c8830cbd293e-1`](pages/games-003.md#g-70cd568b536cf94ab703c8830cbd293e-1) | Master5s / Echo3 | 0 | 胜 | +2,596.48 | 已分析 |
| 33 | [`b219068ea91e04b1052c627147944446-1`](pages/games-003.md#g-b219068ea91e04b1052c627147944446-1) | Master5s / 哈基米 v2 pro | 1 | 胜 | +3,837.39 | 已分析 |
| 34 | [`04c4595fe42c3df466d898cf91ef89fc-1`](pages/games-003.md#g-04c4595fe42c3df466d898cf91ef89fc-1) | Master5s / stephen-K4TacSearch-v1 | 1 | 胜 | +1,192.57 | 已分析 |
| 35 | [`5a29ce1cf123531152fa0b074aa1af1b-1`](pages/games-003.md#g-5a29ce1cf123531152fa0b074aa1af1b-1) | Master5s / stephen-K4TacSearch-v4 | 0 | 胜 | +1,515.96 | 已分析 |
| 36 | [`2f90dcc6a2eb0ee40cd49bf10b800d6c-1`](pages/games-003.md#g-2f90dcc6a2eb0ee40cd49bf10b800d6c-1) | Master5s / stephen-K4TacSearch-v3 | 1 | 胜 | +2,540.91 | 已分析 |
| 37 | [`b4c800a8147e3affd47317dd124f401c-1`](pages/games-003.md#g-b4c800a8147e3affd47317dd124f401c-1) | Master5s / Krypton-M144Search-v1 | 1 | 胜 | +5,024.79 | 已分析 |
| 38 | [`e253a5af21dc27609f2972d4b62a29fd-1`](pages/games-003.md#g-e253a5af21dc27609f2972d4b62a29fd-1) | Master5s / Fable5.2-Max-V8 | 1 | 胜 | +4,008.49 | 已分析 |
| 39 | [`d1d31ee963f6d9d35490c2a10bf9e023-1`](pages/games-003.md#g-d1d31ee963f6d9d35490c2a10bf9e023-1) | Master5s / Fable5.2-Max-V9 | 0 | 胜 | +5,795.50 | 已分析 |
| 40 | [`4d414c14de657213fb672a3022d2ac34-1`](pages/games-004.md#g-4d414c14de657213fb672a3022d2ac34-1) | Master5s / Carago-V2微调 | 0 | 胜 | +955.44 | 已分析 |
| 41 | [`dfade934a41b69385a7143b586c75759-1`](pages/games-004.md#g-dfade934a41b69385a7143b586c75759-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +925.41 | 已分析 |
| 42 | [`b01e870a1892cd87e9bd4bbcb45b9b90-1`](pages/games-004.md#g-b01e870a1892cd87e9bd4bbcb45b9b90-1) | Master5s / 小肥鱼v5 | 0 | 胜 | +1,448.03 | 已分析 |
| 43 | [`eef68438eba8b4db68f1bb4b8fb918a3-1`](pages/games-004.md#g-eef68438eba8b4db68f1bb4b8fb918a3-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +1,205.43 | 已分析 |
| 44 | [`8759f8217d439cc5127f10aa2139935a-1`](pages/games-004.md#g-8759f8217d439cc5127f10aa2139935a-1) | Master5s / ClaudeFable5.2-BC-v2 | 0 | 胜 | +2,372.55 | 已分析 |
| 45 | [`dd6aff420b28edf6960d7b467e529df3-1`](pages/games-004.md#g-dd6aff420b28edf6960d7b467e529df3-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +332.04 | 已分析 |
| 46 | [`737be0fd4cb3b9698190d1631af2d042-1`](pages/games-004.md#g-737be0fd4cb3b9698190d1631af2d042-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 1 | 胜 | +2,006.37 | 已分析 |
| 47 | [`caa5901be2e08abd0d666697dc542a84-1`](pages/games-004.md#g-caa5901be2e08abd0d666697dc542a84-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 1 | 胜 | +2,437.71 | 已分析 |
| 48 | [`d48fd1719afcfee8b77a6b8cb7ae2997-1`](pages/games-004.md#g-d48fd1719afcfee8b77a6b8cb7ae2997-1) | Master5s / ClaudeFable5.2-BC-v3 | 1 | 胜 | +2,459.74 | 已分析 |
| 49 | [`ff09b13031cfcd11d59631affed007d6-1`](pages/games-004.md#g-ff09b13031cfcd11d59631affed007d6-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +1,087.81 | 已分析 |
| 50 | [`37768a0cb18b1a14d5b8103e0041de6b-1`](pages/games-004.md#g-37768a0cb18b1a14d5b8103e0041de6b-1) | Master5s / 小肥鱼v5 | 0 | 胜 | +2,164.71 | 已分析 |
| 51 | [`6ce0e3516c513493910bd7d4cdc86361-1`](pages/games-004.md#g-6ce0e3516c513493910bd7d4cdc86361-1) | Master5s / ClaudeFable5.2-BC-v3 | 1 | 胜 | +1,882.45 | 已分析 |
| 52 | [`23e6019f321ed539386263c3e85b8e42-1`](pages/games-004.md#g-23e6019f321ed539386263c3e85b8e42-1) | Master5s / 小肥鱼v5 | 1 | 胜 | +1,501.36 | 已分析 |
| 53 | [`ca8a8a9cb60115f0514d249d92d048c9-1`](pages/games-005.md#g-ca8a8a9cb60115f0514d249d92d048c9-1) | Master5s / 小肥鱼v5 | 1 | 胜 | +587.93 | 已分析 |
| 54 | [`c4d761ec5af1ee73ab7cce2b9c6a2b08-1`](pages/games-005.md#g-c4d761ec5af1ee73ab7cce2b9c6a2b08-1) | Master5s / ClaudeFable5.2-BC-v3 | 1 | 胜 | +2,332.63 | 已分析 |
| 55 | [`1fedcf34baee77b3e866de8065948aca-1`](pages/games-005.md#g-1fedcf34baee77b3e866de8065948aca-1) | Master5s / 哈基米基准 | 1 | 胜 | +1,528.13 | 已分析 |
| 56 | [`a0cf8dd343aba2e42e6970c919e20da5-1`](pages/games-005.md#g-a0cf8dd343aba2e42e6970c919e20da5-1) | Master5s / ClaudeFable5.2-BC-v2 | 0 | 胜 | +2,038.02 | 已分析 |
| 57 | [`7981780bf487dcc1c83d7c94d5faea25-1`](pages/games-005.md#g-7981780bf487dcc1c83d7c94d5faea25-1) | Master5s / ClaudeFable5.2-BC-v3 | 0 | 胜 | +1,237.46 | 已分析 |
| 58 | [`a4cbf1fa9a498523f02de6c84dbdf14e-1`](pages/games-005.md#g-a4cbf1fa9a498523f02de6c84dbdf14e-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +1,691.63 | 已分析 |
| 59 | [`104c209f6b2076777f6f33df15edb997-1`](pages/games-005.md#g-104c209f6b2076777f6f33df15edb997-1) | Master5s / ClaudeFable5.2-BC-v2 | 0 | 胜 | +2,648.04 | 已分析 |
| 60 | [`3a3d08498acbb03c5d8ea98479e9bb3b-1`](pages/games-005.md#g-3a3d08498acbb03c5d8ea98479e9bb3b-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +540.87 | 已分析 |
| 61 | [`3263a59e9c56a0eee1fd769ff742e58b-1`](pages/games-005.md#g-3263a59e9c56a0eee1fd769ff742e58b-1) | Master5s / Grok4.7-Search-v5 | 0 | 胜 | +1,289.28 | 已分析 |
| 62 | [`f0795ca7e2d82fe3174ee167ea8585b4-1`](pages/games-005.md#g-f0795ca7e2d82fe3174ee167ea8585b4-1) | Master5s / Grok4.7-Search-v5 | 1 | 胜 | +5,774.61 | 已分析 |
| 63 | [`b4e45fac9f2e948d644d2946aabb1f50-1`](pages/games-005.md#g-b4e45fac9f2e948d644d2946aabb1f50-1) | Master5s / Grok4.7-Search-v5 | 0 | 胜 | +2,453.25 | 已分析 |
| 64 | [`699fc9e24eab6af2666ea8e4f55d51d3-1`](pages/games-005.md#g-699fc9e24eab6af2666ea8e4f55d51d3-1) | Master5s / Grok4.7-Search-v5 | 0 | 胜 | +5,260.67 | 已分析 |
| 65 | [`63cf3ff281804e868d6f838873a23ed4-1`](pages/games-006.md#g-63cf3ff281804e868d6f838873a23ed4-1) | Master5s / Grok4.7-Search-v5 | 0 | 胜 | +2,787.95 | 已分析 |
| 66 | [`17cf6cd3cc8ea8a27269047e2b56be8c-1`](pages/games-006.md#g-17cf6cd3cc8ea8a27269047e2b56be8c-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 1 | 胜 | +1,546.36 | 已分析 |
| 67 | [`4d36ff29c656ab7063f10e83734a2dc5-1`](pages/games-006.md#g-4d36ff29c656ab7063f10e83734a2dc5-1) | Master5s / 哈基米基准 | 1 | 胜 | +1,137.18 | 已分析 |
| 68 | [`1c6592328becd1184b6cee0531afade5-1`](pages/games-006.md#g-1c6592328becd1184b6cee0531afade5-1) | Master5s / ClaudeFable5.2-BC-v2 | 0 | 胜 | +3,938.70 | 已分析 |
| 69 | [`13811486e41e9a9b363469dba2bb378e-1`](pages/games-006.md#g-13811486e41e9a9b363469dba2bb378e-1) | Master5s / stephen-K4TacSearch-v6 | 0 | 胜 | +250.24 | 已分析 |
| 70 | [`5f2083216180fc1cf815f8d90abfc62c-1`](pages/games-006.md#g-5f2083216180fc1cf815f8d90abfc62c-1) | Master5s / stephen-K4TacSearch-v6 | 0 | 胜 | +2,160.26 | 已分析 |
| 71 | [`392116cf7bf821f974683908f2924353-1`](pages/games-006.md#g-392116cf7bf821f974683908f2924353-1) | Master5s / 哈基米基准 | 1 | 胜 | +882.40 | 已分析 |
| 72 | [`ad630693c26ae17b724092518d883ae1-1`](pages/games-006.md#g-ad630693c26ae17b724092518d883ae1-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 1 | 负 | -610.36 | 已分析 |
| 73 | [`3e5b1e2e17d8b956b21c6e378d1cf861-1`](pages/games-006.md#g-3e5b1e2e17d8b956b21c6e378d1cf861-1) | Master5s / 小肥鱼v5 | 0 | 胜 | +1,085.25 | 已分析 |
| 74 | [`20ed2afd8245781230e23f73b9bddbfa-1`](pages/games-006.md#g-20ed2afd8245781230e23f73b9bddbfa-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 0 | 胜 | +6,772.70 | 已分析 |
| 75 | [`13015a90cc59fc48d8b26337df6b8161-1`](pages/games-006.md#g-13015a90cc59fc48d8b26337df6b8161-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +54.73 | 已分析 |
| 76 | [`2ca1a6e639af71704b87d036fefb18f4-1`](pages/games-006.md#g-2ca1a6e639af71704b87d036fefb18f4-1) | Master5s / stephen-K4TacSearch-v6 | 1 | 胜 | +2,270.45 | 已分析 |
| 77 | [`2d712db8c7e48ee1c2ef8876c3866195-1`](pages/games-007.md#g-2d712db8c7e48ee1c2ef8876c3866195-1) | Master5s / 小肥鱼v5 | 1 | 胜 | +1,258.37 | 已分析 |
| 78 | [`81602de8cc2ab08aebf643830c60b33e-1`](pages/games-007.md#g-81602de8cc2ab08aebf643830c60b33e-1) | Master5s / 哈基米基准 | 1 | 胜 | +491.94 | 已分析 |
| 79 | [`3fa7ad1b32db4e8f57bb74b00ddb8fe4-1`](pages/games-007.md#g-3fa7ad1b32db4e8f57bb74b00ddb8fe4-1) | Master5s / 小肥鱼v5 | 1 | 胜 | +1,234.23 | 已分析 |
| 80 | [`f7831333de4a7380c0527cf2af42701f-1`](pages/games-007.md#g-f7831333de4a7380c0527cf2af42701f-1) | Master5s / ClaudeFable5.2-BC-v3 | 1 | 胜 | +1,817.18 | 已分析 |
| 81 | [`902c19f0d9c27f6f2df9b80a09a2ec57-1`](pages/games-007.md#g-902c19f0d9c27f6f2df9b80a09a2ec57-1) | Master5s / ClaudeFable5.2-BC-v2 | 0 | 胜 | +1,576.78 | 已分析 |
| 82 | [`f405b1a0f7d7bb1c5b4e7a3e011e9b95-1`](pages/games-007.md#g-f405b1a0f7d7bb1c5b4e7a3e011e9b95-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 0 | 胜 | +4,140.76 | 已分析 |
| 83 | [`eca9e0ff749ffd031fbc451255d158f1-1`](pages/games-007.md#g-eca9e0ff749ffd031fbc451255d158f1-1) | Master5s / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +807.64 | 已分析 |
| 84 | [`98b57c58a16d11b4fb0286774e18ba68-1`](pages/games-007.md#g-98b57c58a16d11b4fb0286774e18ba68-1) | Master5s / Grok4.7-Search-v5 | 1 | 胜 | +1,173.38 | 已分析 |
| 85 | [`bdc8435261e23a96b1931a53c4c325f8-1`](pages/games-007.md#g-bdc8435261e23a96b1931a53c4c325f8-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 1 | 胜 | +3,103.64 | 已分析 |
| 86 | [`8ff6e97e347abe2310e999bf7558acd5-1`](pages/games-007.md#g-8ff6e97e347abe2310e999bf7558acd5-1) | Master5s / ClaudeFable5.2-BC-v3 | 0 | 胜 | +3,346.94 | 已分析 |
| 87 | [`86c41fd46ad2b05fd5a732b06c587fb9-1`](pages/games-007.md#g-86c41fd46ad2b05fd5a732b06c587fb9-1) | Master5s / Grok4.7-Search-v5 | 1 | 胜 | +2,177.98 | 已分析 |
| 88 | [`9e94753554981699b6547e96ec02898b-1`](pages/games-007.md#g-9e94753554981699b6547e96ec02898b-1) | Master5s / 小肥鱼v5 | 1 | 胜 | +66.43 | 已分析 |
| 89 | [`a65c0a4a0c38eac49927b1fbf875dc6a-1`](pages/games-008.md#g-a65c0a4a0c38eac49927b1fbf875dc6a-1) | Master5s / stephen-K4TacSearch-v6 | 1 | 胜 | +1,819.10 | 已分析 |
| 90 | [`9581b889b5ffc67ed420ccc0b96d00b0-1`](pages/games-008.md#g-9581b889b5ffc67ed420ccc0b96d00b0-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 1 | 胜 | +1,818.64 | 已分析 |
| 91 | [`f84911d34421003cb704d4c8b82314d5-1`](pages/games-008.md#g-f84911d34421003cb704d4c8b82314d5-1) | Master5s / Grok4.7-Search-v5 | 0 | 胜 | +1,870.04 | 已分析 |
| 92 | [`2b83af8293847872ec97be3d98d0f9b0-1`](pages/games-008.md#g-2b83af8293847872ec97be3d98d0f9b0-1) | Master5s / ClaudeFable5.2-BC-v2 | 0 | 胜 | +2,455.50 | 已分析 |
| 93 | [`ab65ffdc07226a0ab6d7bee5997c719a-1`](pages/games-008.md#g-ab65ffdc07226a0ab6d7bee5997c719a-1) | Master5s / ClaudeFable5.2-BC-v3 | 1 | 胜 | +1,929.20 | 已分析 |
| 94 | [`0ef7bb0a863a98c15a802fb6e08f5731-1`](pages/games-008.md#g-0ef7bb0a863a98c15a802fb6e08f5731-1) | Master5s / Grok4.7-Search-v5 | 1 | 胜 | +1,938.84 | 已分析 |
| 95 | [`afb1c0a51bddc547bbd00df00564aed1-1`](pages/games-008.md#g-afb1c0a51bddc547bbd00df00564aed1-1) | Master5s / 哈基米基准 | 1 | 胜 | +1,480.04 | 已分析 |
| 96 | [`39637cfc931f6c15cf156f2f4c26ac51-1`](pages/games-008.md#g-39637cfc931f6c15cf156f2f4c26ac51-1) | Master5s / ClaudeFable5.2-Sym8-v4 | 0 | 胜 | +1,896.22 | 已分析 |
| 97 | [`f3f127ffa006568bbfe06d2d84e51da8-1`](pages/games-008.md#g-f3f127ffa006568bbfe06d2d84e51da8-1) | Master5s / master3 | 1 | 胜 | +3,309.64 | 已分析 |
| 98 | [`7f008c50c2922c2e67e1a24f93fb8c31-1`](pages/games-008.md#g-7f008c50c2922c2e67e1a24f93fb8c31-1) | Master5s / ClaudeFable5.2-BC-v2 | 1 | 胜 | +3,060.32 | 已分析 |
| 99 | [`085fb2c7fa4673cb62a88582dd232fb1-1`](pages/games-008.md#g-085fb2c7fa4673cb62a88582dd232fb1-1) | Master5s / Grok4.7-Search-v5 | 0 | 胜 | +2,096.07 | 已分析 |
| 100 | [`a229117417d0cefbcc646b589530e8b3-1`](pages/games-008.md#g-a229117417d0cefbcc646b589530e8b3-1) | Master5s / 小肥鱼v5 | 1 | 胜 | +1,209.53 | 已分析 |
| 101 | [`139ce9f90d1cdbfbdaebe112cd95c8b8-1`](pages/games-009.md#g-139ce9f90d1cdbfbdaebe112cd95c8b8-1) | Master5s / Echo5 | 1 | 胜 | +3,544.12 | 已分析 |
| 102 | [`2bb0546c0fa8b5de7b4576b31a5b6383-1`](pages/games-009.md#g-2bb0546c0fa8b5de7b4576b31a5b6383-1) | Master5s / ClaudeFable5.2-v7 | 1 | 负 | -318.70 | 已分析 |
| 103 | [`3509e5b66ffc3e67f03739023d38a03b-1`](pages/games-009.md#g-3509e5b66ffc3e67f03739023d38a03b-1) | Master5s / ClaudeFable5.2-v7 | 1 | 胜 | +288.37 | 已分析 |
| 104 | [`203462ddecf2ff126c4322c8250e77aa-1`](pages/games-009.md#g-203462ddecf2ff126c4322c8250e77aa-1) | Master5s / ClaudeFable5.2-v7 | 0 | 胜 | +1,109.19 | 已分析 |
| 105 | [`0d0b6f119652e111da30364ca877a5d0-1`](pages/games-009.md#g-0d0b6f119652e111da30364ca877a5d0-1) | Master5s / Fable5.2-Max-V9 | 0 | 胜 | +3,152.11 | 已分析 |
| 106 | [`ef8c2fbd6a5f7c9b58bb2b091a96d0c1-1`](pages/games-009.md#g-ef8c2fbd6a5f7c9b58bb2b091a96d0c1-1) | Master5s / ClaudeFable5.2-v7 | 1 | 胜 | +1,227.52 | 已分析 |
| 107 | [`dc91a09f4e84928870bf76e09af19bc5-1`](pages/games-009.md#g-dc91a09f4e84928870bf76e09af19bc5-1) | Master5s / claude-opus55-cloud-Echo6 | 1 | 胜 | +2,952.69 | 已分析 |
| 108 | [`f306679c3694d94f4bdb3e868fb15955-1`](pages/games-009.md#g-f306679c3694d94f4bdb3e868fb15955-1) | Master5s / ClaudeFable5.2-v7 | 1 | 胜 | +1,829.31 | 已分析 |
| 109 | [`ac364a420640522907349a4f31ea09ef-1`](pages/games-009.md#g-ac364a420640522907349a4f31ea09ef-1) | Master5s / ClaudeFable5.2-Search-v7 | 0 | 胜 | +1,248.01 | 已分析 |
| 110 | [`6208b896bf5cfb902605cf049560d0e4-1`](pages/games-009.md#g-6208b896bf5cfb902605cf049560d0e4-1) | Master5s / ClaudeFable5.2-Search-v7 | 0 | 胜 | +1,272.47 | 已分析 |
| 111 | [`01b56d406361dbb513d6d7417dc0184f-1`](pages/games-009.md#g-01b56d406361dbb513d6d7417dc0184f-1) | Master5s / ClaudeFable5.2-v7 | 0 | 胜 | +2,998.34 | 已分析 |
| 112 | [`5a186082f07f6f0eca9094cfe678508c-1`](pages/games-009.md#g-5a186082f07f6f0eca9094cfe678508c-1) | Master5s / ClaudeFable5.2-v8 | 0 | 胜 | +1,837.75 | 已分析 |
| 113 | [`f27bf25a57669091127e4eeaabea7b1b-1`](pages/games-010.md#g-f27bf25a57669091127e4eeaabea7b1b-1) | Master5s / ClaudeFable5.2-v8 | 0 | 胜 | +1,816.29 | 已分析 |
| 114 | [`cc55d20939336bf84566156754665648-1`](pages/games-010.md#g-cc55d20939336bf84566156754665648-1) | Master5s / 小肥鱼v5-late1 | 1 | 胜 | +756.66 | 已分析 |
| 115 | [`f5a8339ea7846cd64b1a27a526341ba0-1`](pages/games-010.md#g-f5a8339ea7846cd64b1a27a526341ba0-1) | Master5s / 小肥鱼v5-late1 | 1 | 胜 | +1,207.08 | 已分析 |
| 116 | [`8f459a86f11b9b539c9b192448fe6bda-1`](pages/games-010.md#g-8f459a86f11b9b539c9b192448fe6bda-1) | Master5s / ClaudeFable5.2-v8 | 1 | 胜 | +7,502.29 | 已分析 |
| 117 | [`23353c8f6d169b431d678b6653804992-1`](pages/games-010.md#g-23353c8f6d169b431d678b6653804992-1) | Master5s / ClaudeFable5.2-Search-v7 | 1 | 胜 | +934.10 | 已分析 |
| 118 | [`9123289768bfb40633b0c00668825ab2-1`](pages/games-010.md#g-9123289768bfb40633b0c00668825ab2-1) | Master5s / 小肥鱼v5-late1 | 0 | 负 | -129.35 | 已分析 |
| 119 | [`16246b994fa62016c8f673db3f47426a-1`](pages/games-010.md#g-16246b994fa62016c8f673db3f47426a-1) | Master5s / ClaudeFable5.2-v8 | 0 | 胜 | +1,559.21 | 已分析 |
| 120 | [`1b3177fc72dcd87cf955642782c41a80-1`](pages/games-010.md#g-1b3177fc72dcd87cf955642782c41a80-1) | Master5s / 小肥鱼v5-late1 | 0 | 胜 | +576.74 | 已分析 |
| 121 | [`a203c2ece52f5d44e229e6f6e3a4a93a-1`](pages/games-010.md#g-a203c2ece52f5d44e229e6f6e3a4a93a-1) | Master5s / 小肥鱼v5-late1 | 0 | 胜 | +1,292.43 | 已分析 |
| 122 | [`e0c126311823190a38dc2bb5fabff800-1`](pages/games-010.md#g-e0c126311823190a38dc2bb5fabff800-1) | Master5s / 小肥鱼v5-d5 | 1 | 胜 | +1,580.44 | 已分析 |
| 123 | [`3505ed9b6d524d71d58de58f2eabc315-1`](pages/games-010.md#g-3505ed9b6d524d71d58de58f2eabc315-1) | Master5s / 小肥鱼v5-d5 | 0 | 负 | -441.93 | 已分析 |
| 124 | [`9d2ed003f12fa024fd5e05c8be1137d7-1`](pages/games-010.md#g-9d2ed003f12fa024fd5e05c8be1137d7-1) | Master5s / 小肥鱼v5-d5 | 1 | 胜 | +733.15 | 已分析 |
| 125 | [`0a815a4c86941b6d187c57727e3e6090-1`](pages/games-011.md#g-0a815a4c86941b6d187c57727e3e6090-1) | Master5s / ClaudeFable5.2-v7 | 1 | 胜 | +931.56 | 已分析 |
| 126 | [`94c0826751e649dcc2cfd54a8553cdd0-1`](pages/games-011.md#g-94c0826751e649dcc2cfd54a8553cdd0-1) | Master5s / ClaudeFable5.2-Search-v7 | 1 | 胜 | +3,155.02 | 已分析 |
| 127 | [`a763c2da406430867718486972898996-1`](pages/games-011.md#g-a763c2da406430867718486972898996-1) | Master5s / 小肥鱼v5-late1 | 0 | 胜 | +1,922.42 | 已分析 |
| 128 | [`33ac4fb13a0bd6b4c118cdbeb5d43e5e-1`](pages/games-011.md#g-33ac4fb13a0bd6b4c118cdbeb5d43e5e-1) | Master5s / ClaudeFable5.2-v8 | 1 | 胜 | +2,793.36 | 已分析 |
| 129 | [`1f170a1fa6ae1bc79cf771cfe27f000f-1`](pages/games-011.md#g-1f170a1fa6ae1bc79cf771cfe27f000f-1) | Master5s / ClaudeFable5.2-v7 | 1 | 胜 | +1,976.00 | 已分析 |
| 130 | [`8297406e1276d00ed0357d1b978b1e1b-1`](pages/games-011.md#g-8297406e1276d00ed0357d1b978b1e1b-1) | Master5s / 小肥鱼v5-late1 | 1 | 胜 | +649.82 | 已分析 |
| 131 | [`bbe51a3424ac2226ed8d2252fe2201b9-1`](pages/games-011.md#g-bbe51a3424ac2226ed8d2252fe2201b9-1) | Master5s / ClaudeFable5.2-v8 | 1 | 胜 | +1,004.64 | 已分析 |
| 132 | [`fe666a3489fc8b53b72bb64e5af9a18c-1`](pages/games-011.md#g-fe666a3489fc8b53b72bb64e5af9a18c-1) | Master5s / ClaudeFable5.2-Search-v7 | 0 | 胜 | +11,496.14 | 已分析 |
| 133 | [`1d8a5e60edfdb28ee1801eccbf220f6b-1`](pages/games-011.md#g-1d8a5e60edfdb28ee1801eccbf220f6b-1) | Master5s / ClaudeFable5.2-Search-v7 | 0 | 胜 | +3,504.05 | 已分析 |
| 134 | [`4c086ee0a76d9dd2c26d5e2a13654119-1`](pages/games-011.md#g-4c086ee0a76d9dd2c26d5e2a13654119-1) | Master5s / 小肥鱼v5-d5 | 1 | 胜 | +1,392.25 | 已分析 |
| 135 | [`8caa0fdab6a29ebb41ddbe377dfe28ea-1`](pages/games-011.md#g-8caa0fdab6a29ebb41ddbe377dfe28ea-1) | Master5s / ClaudeFable5.2-v8 | 0 | 胜 | +2,829.07 | 已分析 |
| 136 | [`54ae0807115990474ee47ad8c8641b54-1`](pages/games-011.md#g-54ae0807115990474ee47ad8c8641b54-1) | Master5s / 小肥鱼v5-late1 | 0 | 胜 | +961.97 | 已分析 |
| 137 | [`4f522ee8ab66b4ffc68bcf3822389e89-1`](pages/games-012.md#g-4f522ee8ab66b4ffc68bcf3822389e89-1) | Master5s / ClaudeFable5.2-v8 | 0 | 胜 | +4,525.70 | 已分析 |
| 138 | [`31568ffd9e7c703957f7345206e84308-1`](pages/games-012.md#g-31568ffd9e7c703957f7345206e84308-1) | Master5s / ClaudeFable5.2-Search-v7 | 1 | 胜 | +2,117.60 | 已分析 |
| 139 | [`12e2ff04a56cd951318d912e819b91ab-1`](pages/games-012.md#g-12e2ff04a56cd951318d912e819b91ab-1) | Master5s / 小肥鱼v5-d5 | 1 | 胜 | +477.42 | 已分析 |
| 140 | [`841d11240fa7482c6d21a4185099f049-1`](pages/games-012.md#g-841d11240fa7482c6d21a4185099f049-1) | Master5s / ClaudeFable5.2-Search-v7 | 0 | 胜 | +2,219.39 | 已分析 |
| 141 | [`3b40cb24f17cc3852f7b5f234ffef623-1`](pages/games-012.md#g-3b40cb24f17cc3852f7b5f234ffef623-1) | Master5s / 小肥鱼v5-d5 | 0 | 胜 | +1,005.69 | 已分析 |
| 142 | [`3025ca573d299f0d387c22fddc1fa2fd-1`](pages/games-012.md#g-3025ca573d299f0d387c22fddc1fa2fd-1) | Master5s / 小肥鱼v5-d5 | 0 | 胜 | +2,440.72 | 已分析 |
| 143 | [`333fd46d4296ba6efbaf68d77450790d-1`](pages/games-012.md#g-333fd46d4296ba6efbaf68d77450790d-1) | Master5s / 小肥鱼v5-late1 | 0 | 负 | -116.81 | 已分析 |
| 144 | [`6c3380709cf63756a0b8b2f8cc58601d-1`](pages/games-012.md#g-6c3380709cf63756a0b8b2f8cc58601d-1) | Master5s / ClaudeFable5.2-v8 | 1 | 胜 | +952.59 | 已分析 |
| 145 | [`4cab29b766a5a8f60b9a4347adb05863-1`](pages/games-012.md#g-4cab29b766a5a8f60b9a4347adb05863-1) | Master5s / ClaudeFable5.2-Search-v7 | 0 | 胜 | +1,160.32 | 已分析 |
| 146 | [`ad6b0225a805a79d58a8c3cb47d20ea0-1`](pages/games-012.md#g-ad6b0225a805a79d58a8c3cb47d20ea0-1) | Master5s / 小肥鱼v5-late1 | 1 | 胜 | +982.88 | 已分析 |
| 147 | [`3199ea6fa6700a1cd596c1ffd96b34ec-1`](pages/games-012.md#g-3199ea6fa6700a1cd596c1ffd96b34ec-1) | Master5s / ClaudeFable5.2-v8 | 1 | 胜 | +2,120.93 | 已分析 |
| 148 | [`50d4b8cc634466307faae70afd999385-1`](pages/games-012.md#g-50d4b8cc634466307faae70afd999385-1) | Master5s / 小肥鱼v5-d5 | 1 | 胜 | +2,166.17 | 已分析 |
| 149 | [`abb23c7796d827b211c26e46b99785b3-1`](pages/games-013.md#g-abb23c7796d827b211c26e46b99785b3-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 胜 | +128.32 | 已分析 |
| 150 | [`0a4b378d35c63912fd4bebe264ffdb43-1`](pages/games-013.md#g-0a4b378d35c63912fd4bebe264ffdb43-1) | Master5s / RyanX-AIPowercore-it720-preview | 1 | 胜 | +1,686.49 | 已分析 |
| 151 | [`edb585cf77ed804bcf68276515b1f78e-1`](pages/games-013.md#g-edb585cf77ed804bcf68276515b1f78e-1) | Master5s / RyanX-AIPowercore-it720-preview | 1 | 胜 | +373.93 | 已分析 |
| 152 | [`6b77b77327fb186f176c76b876c1aea9-1`](pages/games-013.md#g-6b77b77327fb186f176c76b876c1aea9-1) | Master5s / RyanX-AIPowercore-it720-preview | 1 | 负 | -280.92 | 已分析 |
| 153 | [`8028b5874ca6fdbf50c83b7f3eecd461-1`](pages/games-013.md#g-8028b5874ca6fdbf50c83b7f3eecd461-1) | Master5s / ClaudeFable5.2-Search-v7 | 0 | 胜 | +1,249.13 | 已分析 |
| 154 | [`a9b1fcf0a020a42efcf51b3d7c359b47-1`](pages/games-013.md#g-a9b1fcf0a020a42efcf51b3d7c359b47-1) | Master5s / ClaudeFable5.2-v7 | 0 | 胜 | +2,019.30 | 已分析 |
| 155 | [`22df9f53842ce68080a6aeb3be2361ea-1`](pages/games-013.md#g-22df9f53842ce68080a6aeb3be2361ea-1) | Master5s / Krypton-M3s2-EarlyHome | 0 | 胜 | +2,800.68 | 已分析 |
| 156 | [`c83854a8bb70a683b57de37af0f0b66e-1`](pages/games-013.md#g-c83854a8bb70a683b57de37af0f0b66e-1) | Master5s / Krypton-M3s2-EarlyHome | 1 | 胜 | +3,889.73 | 已分析 |
| 157 | [`04e2bd65b4b6e81980135e06cfadd7b0-1`](pages/games-013.md#g-04e2bd65b4b6e81980135e06cfadd7b0-1) | Master5s / ClaudeFable5.2-v7 | 0 | 胜 | +2,464.64 | 已分析 |
| 158 | [`a5d9c8dee0d99122bc8b62d10fbf75c1-1`](pages/games-013.md#g-a5d9c8dee0d99122bc8b62d10fbf75c1-1) | Master5s / 小肥鱼v5-d5 | 0 | 胜 | +2,678.44 | 已分析 |
| 159 | [`0f74bbe00b507097182c2cb9655006b9-1`](pages/games-013.md#g-0f74bbe00b507097182c2cb9655006b9-1) | Master5s / Krypton-M3s2-EarlyHome | 0 | 胜 | +912.74 | 已分析 |
| 160 | [`cf409979a5c2cef1f2d0894fbfb1512f-1`](pages/games-013.md#g-cf409979a5c2cef1f2d0894fbfb1512f-1) | Master5s / 小肥鱼v5-d5 | 1 | 胜 | +1,068.98 | 已分析 |
| 161 | [`1e193dab0083d1623dd8facfcf34d1ca-1`](pages/games-014.md#g-1e193dab0083d1623dd8facfcf34d1ca-1) | Master5s / ClaudeFable5.2-v8 | 1 | 胜 | +7,669.60 | 已分析 |
| 162 | [`73a884b4a7adc8d2feb37d845e50648a-1`](pages/games-014.md#g-73a884b4a7adc8d2feb37d845e50648a-1) | Master5s / ClaudeFable5.2-v7 | 1 | 胜 | +1,239.90 | 已分析 |
| 163 | [`7adfb38706a6855c057758e37a1914c8-1`](pages/games-014.md#g-7adfb38706a6855c057758e37a1914c8-1) | Master5s / Krypton-M3s2-EarlyHome | 0 | 胜 | +1,434.11 | 已分析 |
| 164 | [`b2fe58e3f2001c06ae53a9e33315a01e-1`](pages/games-014.md#g-b2fe58e3f2001c06ae53a9e33315a01e-1) | Master5s / Krypton-M3s2-EarlyHome | 1 | 胜 | +3,432.74 | 已分析 |
| 165 | [`4994add4c0523c907c03810fc8eb405c-1`](pages/games-014.md#g-4994add4c0523c907c03810fc8eb405c-1) | Master5s / 小肥鱼v5-late1 | 0 | 胜 | +792.69 | 已分析 |
| 166 | [`de5aeb591c283a149796cb0a200ab943-1`](pages/games-014.md#g-de5aeb591c283a149796cb0a200ab943-1) | Master5s / Krypton-M3s2-EarlyHome | 1 | 胜 | +7,193.53 | 已分析 |
| 167 | [`009446f3e4b5231b5bcc5650aeba7214-1`](pages/games-014.md#g-009446f3e4b5231b5bcc5650aeba7214-1) | Master5s / 安生N29 | 1 | 胜 | +2,136.17 | 已分析 |
| 168 | [`45f77f0d7876a94ddb910c4a04982b0c-1`](pages/games-014.md#g-45f77f0d7876a94ddb910c4a04982b0c-1) | Master5s / Krypton-M3s2-EarlyHome | 0 | 胜 | +2,615.10 | 已分析 |
| 169 | [`00b853aabfec65bd9ecee24c0b8bb7dc-1`](pages/games-014.md#g-00b853aabfec65bd9ecee24c0b8bb7dc-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +693.51 | 已分析 |
| 170 | [`f75e3739888938ca1df2d6a29228c046-1`](pages/games-014.md#g-f75e3739888938ca1df2d6a29228c046-1) | Master6 / 哈基米 v3 | 1 | 胜 | +10,637.22 | 已分析 |
| 171 | [`02a73d9effe7d04131e46c61a036b0cb-1`](pages/games-014.md#g-02a73d9effe7d04131e46c61a036b0cb-1) | Master6 / ClaudeFable5.2-v8 | 0 | 胜 | +10,597.45 | 已分析 |
| 172 | [`538d11b6048b88feda3310b3e73f8f30-1`](pages/games-014.md#g-538d11b6048b88feda3310b3e73f8f30-1) | Master6 / 小肥鱼v5 | 1 | 负 | -474.83 | 已分析 |
| 173 | [`f64fa44f1c884f04b4b3941c040c3b39-1`](pages/games-015.md#g-f64fa44f1c884f04b4b3941c040c3b39-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 0 | 负 | -1,194.86 | 已分析 |
| 174 | [`d2c7d061739bef723dc4abfffb22de2a-1`](pages/games-015.md#g-d2c7d061739bef723dc4abfffb22de2a-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 负 | -38.37 | 已分析 |
| 175 | [`2c62cc5c7d86a9bd7554001a972f15b2-1`](pages/games-015.md#g-2c62cc5c7d86a9bd7554001a972f15b2-1) | Master6 / 小肥鱼v5-d5 | 0 | 胜 | +1,967.25 | 已分析 |
| 176 | [`0241b43180b3840ce9e07ccdeddda374-1`](pages/games-015.md#g-0241b43180b3840ce9e07ccdeddda374-1) | Master6 / Krypton-M3s2-EarlyHome | 1 | 胜 | +8,096.52 | 已分析 |
| 177 | [`5b4f5ceaa627db4ff6228b2b18e3aecb-1`](pages/games-015.md#g-5b4f5ceaa627db4ff6228b2b18e3aecb-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 胜 | +8,578.74 | 已分析 |
| 178 | [`dfade732264f7716034817877dfaa308-1`](pages/games-015.md#g-dfade732264f7716034817877dfaa308-1) | Master6 / 小肥鱼v5-d5 | 0 | 胜 | +196.12 | 已分析 |
| 179 | [`8e983cb27ec89013fc28873674712c21-1`](pages/games-015.md#g-8e983cb27ec89013fc28873674712c21-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 0 | 负 | -135.38 | 已分析 |
| 180 | [`472971224e2271b4da33a1def70e7d0e-1`](pages/games-015.md#g-472971224e2271b4da33a1def70e7d0e-1) | Master6 / 小肥鱼v5-late1 | 0 | 胜 | +999.87 | 已分析 |
| 181 | [`a547b189361afe25e5f0682c73e572e8-1`](pages/games-015.md#g-a547b189361afe25e5f0682c73e572e8-1) | Master6 / ClaudeFable5.2-v7 | 0 | 胜 | +10,181.93 | 已分析 |
| 182 | [`5f5daa03787e1af574075785c5c3c276-1`](pages/games-015.md#g-5f5daa03787e1af574075785c5c3c276-1) | Master6 / stephen-K4TacSearch-v6 | 1 | 胜 | +10,436.40 | 已分析 |
| 183 | [`29744abbc65fc6cb45cb1c020feca97e-1`](pages/games-015.md#g-29744abbc65fc6cb45cb1c020feca97e-1) | Master6 / 小肥鱼v5-late1 | 0 | 胜 | +2,196.15 | 已分析 |
| 184 | [`f3d51b78d834bc998582f6657012bd20-1`](pages/games-015.md#g-f3d51b78d834bc998582f6657012bd20-1) | Master6 / ClaudeFable5.2-v7 | 1 | 胜 | +6,670.45 | 已分析 |
| 185 | [`e39fcc4b89b2994853bb8e8b897487eb-1`](pages/games-016.md#g-e39fcc4b89b2994853bb8e8b897487eb-1) | Master6 / 哈基米基准 | 0 | 胜 | +892.17 | 已分析 |
| 186 | [`e80e7d3886d13fea2bb412fa5122831a-1`](pages/games-016.md#g-e80e7d3886d13fea2bb412fa5122831a-1) | Master6 / ClaudeFable5.2-v8 | 1 | 负 | -84.06 | 已分析 |
| 187 | [`80edafd08cd808c222ba57fe83d15048-1`](pages/games-016.md#g-80edafd08cd808c222ba57fe83d15048-1) | Master6 / Krypton-M3s2-EarlyHome | 1 | 胜 | +1,849.03 | 已分析 |
| 188 | [`35906b33fd22a900c46b1bcf28af3f8f-1`](pages/games-016.md#g-35906b33fd22a900c46b1bcf28af3f8f-1) | Master6 / 哈基米基准 | 0 | 胜 | +1,484.30 | 已分析 |
| 189 | [`62393017e805fa7147e10fc3670da0b0-1`](pages/games-016.md#g-62393017e805fa7147e10fc3670da0b0-1) | Master6 / 哈基米 v3pro | 1 | 胜 | +10,504.99 | 已分析 |
| 190 | [`77d8c3ae9b4c52234ace5d61f137c5d7-1`](pages/games-016.md#g-77d8c3ae9b4c52234ace5d61f137c5d7-1) | Master6 / stephen-K4TacSearch-v6 | 0 | 胜 | +1,127.03 | 已分析 |
| 191 | [`a8e0e210ed73a3b52d49b7503f37d015-1`](pages/games-016.md#g-a8e0e210ed73a3b52d49b7503f37d015-1) | Master6 / 安生N29 | 1 | 胜 | +8,964.09 | 已分析 |
| 192 | [`7e4939bdbff00f87c862ba6ceab0d3a8-1`](pages/games-016.md#g-7e4939bdbff00f87c862ba6ceab0d3a8-1) | Master6 / 哈基米 v3pro | 1 | 胜 | +293.15 | 已分析 |
| 193 | [`20f31dbccbfe279e2893fcfda0ae8298-1`](pages/games-016.md#g-20f31dbccbfe279e2893fcfda0ae8298-1) | Master6 / 哈基米 v3 | 1 | 胜 | +5,348.32 | 已分析 |
| 194 | [`1841c9fb5bf77087b4129693495faac3-1`](pages/games-016.md#g-1841c9fb5bf77087b4129693495faac3-1) | Master6 / First111 | 1 | 胜 | +14,596.67 | 已分析 |
| 195 | [`4da2642d8d22b1fa336bcff2174d2e35-1`](pages/games-016.md#g-4da2642d8d22b1fa336bcff2174d2e35-1) | Master6 / Echo5 | 0 | 胜 | +11,540.21 | 已分析 |
| 196 | [`dc0190bece37880c391663a537397013-1`](pages/games-016.md#g-dc0190bece37880c391663a537397013-1) | Master6 / Fable5.2-Max-V9 | 0 | 胜 | +9,389.00 | 已分析 |
| 197 | [`a68916e8094ea33450423fbff12cf52f-1`](pages/games-017.md#g-a68916e8094ea33450423fbff12cf52f-1) | Master6 / 安生N27 | 0 | 胜 | +10,851.69 | 已分析 |
| 198 | [`826b5156312c8daa3e5137854816b997-1`](pages/games-017.md#g-826b5156312c8daa3e5137854816b997-1) | Master6 / 安生N27 | 1 | 胜 | +9,530.78 | 已分析 |
| 199 | [`9f27c951e34659a048dc3f0afb864fa2-1`](pages/games-017.md#g-9f27c951e34659a048dc3f0afb864fa2-1) | Master6 / Fable5.2-Max-V7 | 0 | 胜 | +13,676.16 | 已分析 |
| 200 | [`5b5e53f50e478dc437fd64aff92fe1d6-1`](pages/games-017.md#g-5b5e53f50e478dc437fd64aff92fe1d6-1) | Master6 / Echo4 | 0 | 胜 | +12,382.80 | 已分析 |
| 201 | [`5444e7e5540b75cddfd73ce02c479a30-1`](pages/games-017.md#g-5444e7e5540b75cddfd73ce02c479a30-1) | Master6 / stephen-K4TacSearch-v3 | 1 | 胜 | +12,120.26 | 已分析 |
| 202 | [`5f69ba737b6b4dfcfffcd08c6c6fbf5d-1`](pages/games-017.md#g-5f69ba737b6b4dfcfffcd08c6c6fbf5d-1) | Master6 / Fable5.2-Max-V8 | 1 | 胜 | +10,941.53 | 已分析 |
| 203 | [`a76284dd75a05d851d42e1d97582fb17-1`](pages/games-017.md#g-a76284dd75a05d851d42e1d97582fb17-1) | Master6 / 安生N30 | 1 | 胜 | +7,507.16 | 已分析 |
| 204 | [`f54daea3e868c4f8aa7e4ca241e51f78-1`](pages/games-017.md#g-f54daea3e868c4f8aa7e4ca241e51f78-1) | Master6 / AgyCarbon-P2C-Godslayer-CVI | 0 | 胜 | +9,376.69 | 已分析 |
| 205 | [`f59c4a6ea2d96ae70a90440c893c949b-1`](pages/games-017.md#g-f59c4a6ea2d96ae70a90440c893c949b-1) | Master6 / stephen-K4TacSearch-v2 | 0 | 胜 | +1,462.01 | 已分析 |
| 206 | [`498610b364b806b298782c4d2e3614d9-1`](pages/games-017.md#g-498610b364b806b298782c4d2e3614d9-1) | Master6 / claude-opus55-cloud-Echo6 | 1 | 胜 | +9,896.99 | 已分析 |
| 207 | [`e90b9e02d0187e39388af883e6ba37d6-1`](pages/games-017.md#g-e90b9e02d0187e39388af883e6ba37d6-1) | Master6 / stephen-K4TacSearch-v4 | 0 | 胜 | +1,262.41 | 已分析 |
| 208 | [`8215da6e9d67928dc6c918742566c953-1`](pages/games-017.md#g-8215da6e9d67928dc6c918742566c953-1) | Master6 / 安生N31 | 0 | 胜 | +526.43 | 已分析 |
| 209 | [`b30ab36f856969cac5d2dae5565d9197-1`](pages/games-018.md#g-b30ab36f856969cac5d2dae5565d9197-1) | Master6 / 小肥鱼v5-d5 | 0 | 胜 | +455.16 | 已分析 |
| 210 | [`a15fcfb9c152def4684cbb236d8823d6-1`](pages/games-018.md#g-a15fcfb9c152def4684cbb236d8823d6-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 胜 | +437.17 | 已分析 |
| 211 | [`d0f5e89c975909cd491a987567d751bb-1`](pages/games-018.md#g-d0f5e89c975909cd491a987567d751bb-1) | Master6 / ClaudeFable5.2-v7 | 0 | 胜 | +7,975.18 | 已分析 |
| 212 | [`30cdbe96f8d6f2c5677483f5c87f4a0a-1`](pages/games-018.md#g-30cdbe96f8d6f2c5677483f5c87f4a0a-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 胜 | +177.50 | 已分析 |
| 213 | [`16fe44ae57a6707c88cb25fac96cb7a9-1`](pages/games-018.md#g-16fe44ae57a6707c88cb25fac96cb7a9-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +506.00 | 已分析 |
| 214 | [`1fce7056bf01172a1bdf026799141f24-1`](pages/games-018.md#g-1fce7056bf01172a1bdf026799141f24-1) | Master6 / 小肥鱼v5-late1 | 1 | 胜 | +630.03 | 已分析 |
| 215 | [`20ab1325ba01b9b75373635ee3d7b1ef-1`](pages/games-018.md#g-20ab1325ba01b9b75373635ee3d7b1ef-1) | Master5s / RyanX-AIPowercore-it720-preview | 1 | 胜 | +1,132.78 | 已分析 |
| 216 | [`3686922ec2b2128fbd1ff6bfd7d04d81-1`](pages/games-018.md#g-3686922ec2b2128fbd1ff6bfd7d04d81-1) | Master6 / ClaudeFable5.2-v7 | 0 | 胜 | +10,772.46 | 已分析 |
| 217 | [`1bc491f7ef4ac8f24ac17a7a70717c02-1`](pages/games-018.md#g-1bc491f7ef4ac8f24ac17a7a70717c02-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 胜 | +394.30 | 已分析 |
| 218 | [`d77adda4082645819a97e5f71e2c541e-1`](pages/games-018.md#g-d77adda4082645819a97e5f71e2c541e-1) | Master6 / ClaudeFable5.2-v8 | 0 | 胜 | +1,301.41 | 已分析 |
| 219 | [`a349f113d8cd4665b6da581cd7833500-1`](pages/games-018.md#g-a349f113d8cd4665b6da581cd7833500-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +829.86 | 已分析 |
| 220 | [`d1982baa7f56397eca02d7fc882e9a33-1`](pages/games-018.md#g-d1982baa7f56397eca02d7fc882e9a33-1) | Master6 / 小肥鱼v5-late1 | 1 | 胜 | +522.31 | 已分析 |
| 221 | [`54b1d3e11b3a9a83b488a738de241517-1`](pages/games-019.md#g-54b1d3e11b3a9a83b488a738de241517-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 胜 | +4.44 | 已分析 |
| 222 | [`1359247345c9f23c19af46240c47ffa5-1`](pages/games-019.md#g-1359247345c9f23c19af46240c47ffa5-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +632.93 | 已分析 |
| 223 | [`590146b91a6be88c74092d74b54d1e6c-1`](pages/games-019.md#g-590146b91a6be88c74092d74b54d1e6c-1) | Master6 / Krypton-M3s2-EarlyHome | 0 | 胜 | +1,150.96 | 已分析 |
| 224 | [`df1002840c44b1b3055ceaad5a5c061e-1`](pages/games-019.md#g-df1002840c44b1b3055ceaad5a5c061e-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +376.08 | 已分析 |
| 225 | [`c4fa4b449c4d3c5f62a29e5a4475cf62-1`](pages/games-019.md#g-c4fa4b449c4d3c5f62a29e5a4475cf62-1) | Master6 / 小肥鱼v5-d5 | 1 | 胜 | +904.15 | 已分析 |
| 226 | [`af7c078e9a0001364d3ce4025b23e1ed-1`](pages/games-019.md#g-af7c078e9a0001364d3ce4025b23e1ed-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 胜 | +4,876.82 | 已分析 |
| 227 | [`011008e4dee32c2fdbfa37044c3c2608-1`](pages/games-019.md#g-011008e4dee32c2fdbfa37044c3c2608-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 负 | -127.08 | 已分析 |
| 228 | [`b58c6f50466bff30fc183256041de011-1`](pages/games-019.md#g-b58c6f50466bff30fc183256041de011-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 胜 | +337.31 | 已分析 |
| 229 | [`d1ecca5e86110600b6dd0add400fe723-1`](pages/games-019.md#g-d1ecca5e86110600b6dd0add400fe723-1) | Master6 / ClaudeFable5.2-v7 | 1 | 负 | -145.16 | 已分析 |
| 230 | [`5a5f72bc66bdafbc4d8d90edc9870f8e-1`](pages/games-019.md#g-5a5f72bc66bdafbc4d8d90edc9870f8e-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +8,661.76 | 已分析 |
| 231 | [`14919682e511adc8098c338d7a01be90-1`](pages/games-019.md#g-14919682e511adc8098c338d7a01be90-1) | Master5s / RyanX-AIPowercore-it720-preview | 1 | 胜 | +60.70 | 已分析 |
| 232 | [`2b36d9c3899abac5f4601a09508fe2da-1`](pages/games-019.md#g-2b36d9c3899abac5f4601a09508fe2da-1) | Master6 / ClaudeFable5.2-v8 | 0 | 胜 | +11,430.61 | 已分析 |
| 233 | [`b25e54576d8e4ca28ccd266a3e531a21-1`](pages/games-020.md#g-b25e54576d8e4ca28ccd266a3e531a21-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 1 | 胜 | +4,558.68 | 已分析 |
| 234 | [`176aab183f1b1f586ed4a0a6f6fece30-1`](pages/games-020.md#g-176aab183f1b1f586ed4a0a6f6fece30-1) | Master6 / 哈基米基准 | 1 | 胜 | +134.71 | 已分析 |
| 235 | [`8c7bd6ea5904d6984735a528833d5ed6-1`](pages/games-020.md#g-8c7bd6ea5904d6984735a528833d5ed6-1) | Master6 / Krypton-M3s2-EarlyHome | 1 | 胜 | +10,007.84 | 已分析 |
| 236 | [`79960a8111a1698d069e3c58ff986487-1`](pages/games-020.md#g-79960a8111a1698d069e3c58ff986487-1) | Master6 / 小肥鱼v5-d5 | 0 | 胜 | +6,948.73 | 已分析 |
| 237 | [`dcc847457524cef15b6cc068f29ea616-1`](pages/games-020.md#g-dcc847457524cef15b6cc068f29ea616-1) | Master6 / ClaudeFable5.2-v7 | 0 | 负 | -94.65 | 已分析 |
| 238 | [`b807deb80d8cc55df578a02236428db1-1`](pages/games-020.md#g-b807deb80d8cc55df578a02236428db1-1) | Master6 / 小肥鱼v5-late1 | 0 | 胜 | +885.87 | 已分析 |
| 239 | [`ee48fd77f50b11aace52810f6eb32540-1`](pages/games-020.md#g-ee48fd77f50b11aace52810f6eb32540-1) | Master6 / ClaudeFable5.2-v8 | 0 | 胜 | +9,788.29 | 已分析 |
| 240 | [`20cf86d80e4621391c705158ad187584-1`](pages/games-020.md#g-20cf86d80e4621391c705158ad187584-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +1,162.05 | 已分析 |
| 241 | [`b4b03f5ecc6781a9c45a8d55ae96c764-1`](pages/games-020.md#g-b4b03f5ecc6781a9c45a8d55ae96c764-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 0 | 负 | -696.81 | 已分析 |
| 242 | [`6a978d3f2ded0608d1b52c9b17ab0986-1`](pages/games-020.md#g-6a978d3f2ded0608d1b52c9b17ab0986-1) | Master6 / Krypton-M3s2-EarlyHome | 1 | 胜 | +148.15 | 已分析 |
| 243 | [`9723c173fb2c3d7d8fee64dcd64b0ce6-1`](pages/games-020.md#g-9723c173fb2c3d7d8fee64dcd64b0ce6-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 胜 | +1,009.63 | 已分析 |
| 244 | [`989823698f16fb6dbeb8673cc3cc47e6-1`](pages/games-020.md#g-989823698f16fb6dbeb8673cc3cc47e6-1) | Master6 / stephen-K4TacSearch-v6 | 0 | 胜 | +12,037.93 | 已分析 |
| 245 | [`669523683958722b320356f498e02709-1`](pages/games-021.md#g-669523683958722b320356f498e02709-1) | Master6 / ClaudeFable5.2-v8 | 1 | 胜 | +9,548.46 | 已分析 |
| 246 | [`bf1723a977f4ae066fed92292ec7e6a7-1`](pages/games-021.md#g-bf1723a977f4ae066fed92292ec7e6a7-1) | Master6 / 小肥鱼v5-d5 | 1 | 胜 | +548.12 | 已分析 |
| 247 | [`31747f3c43c927b887e6368d2ac623fe-1`](pages/games-021.md#g-31747f3c43c927b887e6368d2ac623fe-1) | Master6 / 小肥鱼v5 | 1 | 负 | -788.02 | 已分析 |
| 248 | [`b0596cbf9a1da6d8ee0db064fccf850c-1`](pages/games-021.md#g-b0596cbf9a1da6d8ee0db064fccf850c-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 负 | -699.34 | 已分析 |
| 249 | [`c7a43acf0b27bec94831acf5f63e2c1d-1`](pages/games-021.md#g-c7a43acf0b27bec94831acf5f63e2c1d-1) | Master6 / ClaudeFable5.2-v8 | 1 | 胜 | +894.53 | 已分析 |
| 250 | [`6c7d3c49cad8818146c636dbab05b9e5-1`](pages/games-021.md#g-6c7d3c49cad8818146c636dbab05b9e5-1) | Master6 / Krypton-M3s2-EarlyHome | 0 | 胜 | +8,071.36 | 已分析 |
| 251 | [`bef6d5664071eef80955b1a5855f05f7-1`](pages/games-021.md#g-bef6d5664071eef80955b1a5855f05f7-1) | Master6 / 哈基米 v3pro | 1 | 胜 | +12,329.89 | 已分析 |
| 252 | [`9015648c70f1d7628a1d43c2fb1b4260-1`](pages/games-021.md#g-9015648c70f1d7628a1d43c2fb1b4260-1) | Master5s / 安生N30 | 1 | 胜 | +2,134.74 | 已分析 |
| 253 | [`bd11332a58f618d3f3750b1193eace81-1`](pages/games-021.md#g-bd11332a58f618d3f3750b1193eace81-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +10,236.79 | 已分析 |
| 254 | [`09a12e7f6169d13d2b6a924468e0f205-1`](pages/games-021.md#g-09a12e7f6169d13d2b6a924468e0f205-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +9,104.63 | 已分析 |
| 255 | [`a0b06c2df17f21dc9821fc6175c7d7a8-1`](pages/games-021.md#g-a0b06c2df17f21dc9821fc6175c7d7a8-1) | Master6 / 小肥鱼v5-late1 | 1 | 胜 | +8,679.16 | 已分析 |
| 256 | [`e7e42ca5896934a3a0115f3e1baa136c-1`](pages/games-021.md#g-e7e42ca5896934a3a0115f3e1baa136c-1) | Master6 / stephen-K4TacSearch-v6 | 1 | 胜 | +918.83 | 已分析 |
| 257 | [`a3cd09d94c8d5719ab564f53c68db493-1`](pages/games-022.md#g-a3cd09d94c8d5719ab564f53c68db493-1) | Master6 / 哈基米基准 | 0 | 胜 | +12,240.44 | 已分析 |
| 258 | [`997232d2523741adbd0403e3750efd03-1`](pages/games-022.md#g-997232d2523741adbd0403e3750efd03-1) | Master6 / ClaudeFable5.2-v8 | 1 | 胜 | +1,528.90 | 已分析 |
| 259 | [`0e4aac28e06ddc63f8b082966d71f12f-1`](pages/games-022.md#g-0e4aac28e06ddc63f8b082966d71f12f-1) | Master6 / ClaudeFable5.2-v7 | 0 | 胜 | +245.13 | 已分析 |
| 260 | [`e71d5897d9c7908e11c4fb3a870c994f-1`](pages/games-022.md#g-e71d5897d9c7908e11c4fb3a870c994f-1) | Master5s / claude-AlphaCarbon-Master6 | 0 | 负 | -7,078.55 | 已分析 |
| 261 | [`9e0f21cfd1b7cb04e90174cd255b7d86-1`](pages/games-022.md#g-9e0f21cfd1b7cb04e90174cd255b7d86-1) | Master6 / ClaudeFable5.2-v8 | 0 | 胜 | +9,322.44 | 已分析 |
| 262 | [`02ab03aeb1de04e89d030d0e71b280ec-1`](pages/games-022.md#g-02ab03aeb1de04e89d030d0e71b280ec-1) | Master5s / claude-AlphaCarbon-Master6 | 0 | 负 | -671.93 | 已分析 |
| 263 | [`b455a0c842ab04a2f37917d0f155b1a5-1`](pages/games-022.md#g-b455a0c842ab04a2f37917d0f155b1a5-1) | Master6 / ClaudeFable5.2-v7 | 0 | 胜 | +7,421.89 | 已分析 |
| 264 | [`f6bfa602792e97292aa7d84788606363-1`](pages/games-022.md#g-f6bfa602792e97292aa7d84788606363-1) | Master6 / 小肥鱼v5-late1 | 1 | 胜 | +2,319.37 | 已分析 |
| 265 | [`b7bf53df0b46e2aaab9ae34f244b49a0-1`](pages/games-022.md#g-b7bf53df0b46e2aaab9ae34f244b49a0-1) | Master6 / 哈基米基准 | 1 | 胜 | +545.62 | 已分析 |
| 266 | [`4ed9d90532588246c93c2705e92ca86f-1`](pages/games-022.md#g-4ed9d90532588246c93c2705e92ca86f-1) | Master6 / 小肥鱼v5-d5 | 0 | 胜 | +636.63 | 已分析 |
| 267 | [`3d375fad2dd0c7d6e168c5316563e621-1`](pages/games-022.md#g-3d375fad2dd0c7d6e168c5316563e621-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 1 | 负 | -68.63 | 已分析 |
| 268 | [`774d051ac0beeca433b63362f249ad0f-1`](pages/games-022.md#g-774d051ac0beeca433b63362f249ad0f-1) | Master5s / 哈基米 v3pro | 0 | 胜 | +1,824.89 | 已分析 |
| 269 | [`be73806ef6f30deb3e08f926c7ff4c30-1`](pages/games-023.md#g-be73806ef6f30deb3e08f926c7ff4c30-1) | Master6 / Krypton-M3s2-EarlyHome | 1 | 胜 | +7,847.63 | 已分析 |
| 270 | [`d3c73bba6b5492b947256c7d33f0605b-1`](pages/games-023.md#g-d3c73bba6b5492b947256c7d33f0605b-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +7,945.30 | 已分析 |
| 271 | [`809bac27ad68862dada5a1dbbcdb7809-1`](pages/games-023.md#g-809bac27ad68862dada5a1dbbcdb7809-1) | Master5s / 安生N29 | 1 | 胜 | +2,569.99 | 已分析 |
| 272 | [`d2a3b420acce68d396b69e350814139e-1`](pages/games-023.md#g-d2a3b420acce68d396b69e350814139e-1) | Master6 / ClaudeFable5.2-v8 | 0 | 胜 | +9,625.40 | 已分析 |
| 273 | [`2d7af73e124320d425f26eac812e0ad0-1`](pages/games-023.md#g-2d7af73e124320d425f26eac812e0ad0-1) | Master6 / 哈基米 v3pro | 0 | 胜 | +11,561.54 | 已分析 |
| 274 | [`97c4147f9e22966495d499ae97443dd2-1`](pages/games-023.md#g-97c4147f9e22966495d499ae97443dd2-1) | Master5s / 安生N27 | 0 | 胜 | +1,558.13 | 已分析 |
| 275 | [`c78a918ebd92a08149b394ef8959d161-1`](pages/games-023.md#g-c78a918ebd92a08149b394ef8959d161-1) | Master6 / ClaudeFable5.2-v7 | 1 | 胜 | +3,866.92 | 已分析 |
| 276 | [`cc5df60d50f8dd9f4e527acb7373a187-1`](pages/games-023.md#g-cc5df60d50f8dd9f4e527acb7373a187-1) | Master6 / 小肥鱼v5-d5 | 0 | 胜 | +9,790.76 | 已分析 |
| 277 | [`5fb0f13dacd5a31934a64dcb7917720b-1`](pages/games-023.md#g-5fb0f13dacd5a31934a64dcb7917720b-1) | Master6 / RyanX-AIPowercore-it720-preview | 0 | 负 | -1,250.40 | 已分析 |
| 278 | [`d840e4de591fb29628091673c049b6a0-1`](pages/games-023.md#g-d840e4de591fb29628091673c049b6a0-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 负 | -563.80 | 已分析 |
| 279 | [`feb6c4c2240544609a3926903f7681d0-1`](pages/games-023.md#g-feb6c4c2240544609a3926903f7681d0-1) | Master5s / First111 | 0 | 胜 | +4,745.11 | 已分析 |
| 280 | [`ddd8bf4ef175473cec7ff9d7480be558-1`](pages/games-023.md#g-ddd8bf4ef175473cec7ff9d7480be558-1) | Master6 / RyanX-AIPowercore-it720-preview | 0 | 胜 | +202.20 | 已分析 |
| 281 | [`dfb6af5c020694f6e682187e48303ae7-1`](pages/games-024.md#g-dfb6af5c020694f6e682187e48303ae7-1) | Master6 / RyanX-AIPowercore-it720-preview | 0 | 负 | -1,525.52 | 已分析 |
| 282 | [`88b749bbc6652916deca09c1a44cfad9-1`](pages/games-024.md#g-88b749bbc6652916deca09c1a44cfad9-1) | Master6 / 安生N30 | 1 | 胜 | +894.23 | 已分析 |
| 283 | [`4d6f8e4e72a28532c2b4e9b82e317390-1`](pages/games-024.md#g-4d6f8e4e72a28532c2b4e9b82e317390-1) | Master6 / 安生N29 | 0 | 胜 | +7,041.67 | 已分析 |
| 284 | [`95d18ab1be530b142cc29fe159ad1823-1`](pages/games-024.md#g-95d18ab1be530b142cc29fe159ad1823-1) | Master6 / stephen-K4TacSearch-v6 | 0 | 胜 | +10,083.77 | 已分析 |
| 285 | [`c96ce8e7c46f59aeabe0e490884fb228-1`](pages/games-024.md#g-c96ce8e7c46f59aeabe0e490884fb228-1) | Master6 / 哈基米基准 | 1 | 胜 | +176.57 | 已分析 |
| 286 | [`6bfc439e37be544e61bfbe5cf8f70efa-1`](pages/games-024.md#g-6bfc439e37be544e61bfbe5cf8f70efa-1) | Master6 / 哈基米 v3 | 1 | 胜 | +11,392.62 | 已分析 |
| 287 | [`fe6fff0b1df597dcbf8bafa23c8c8572-1`](pages/games-024.md#g-fe6fff0b1df597dcbf8bafa23c8c8572-1) | Master6 / 哈基米 v3pro | 1 | 胜 | +1,237.64 | 已分析 |
| 288 | [`6c2d7cfcfe3ca9d55ab128e166369794-1`](pages/games-024.md#g-6c2d7cfcfe3ca9d55ab128e166369794-1) | Master5s / 安生N30 | 0 | 胜 | +4,076.82 | 已分析 |
| 289 | [`c682745d307bf8bec8fbbd2cdf419d18-1`](pages/games-024.md#g-c682745d307bf8bec8fbbd2cdf419d18-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 负 | -52.07 | 已分析 |
| 290 | [`8dd781b85a33e6a28d8476c0dcac9e11-1`](pages/games-024.md#g-8dd781b85a33e6a28d8476c0dcac9e11-1) | Master5s / 哈基米 v3 | 0 | 胜 | +2,329.71 | 已分析 |
| 291 | [`3c26541db076cd630c203e0a41e21cdd-1`](pages/games-024.md#g-3c26541db076cd630c203e0a41e21cdd-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +804.51 | 已分析 |
| 292 | [`51f34379b8018af261ce057ca75692a0-1`](pages/games-024.md#g-51f34379b8018af261ce057ca75692a0-1) | Master6 / RyanX-AIPowercore-it720-preview | 1 | 胜 | +974.25 | 已分析 |
| 293 | [`e68c53ccdd731443fdbc0493a46ce999-1`](pages/games-025.md#g-e68c53ccdd731443fdbc0493a46ce999-1) | Master6 / AgyCarbon-P2C-Godslayer-CVI | 0 | 胜 | +6,930.14 | 已分析 |
| 294 | [`b67eedf5ab9ec4ffbd941e3cbab635a5-1`](pages/games-025.md#g-b67eedf5ab9ec4ffbd941e3cbab635a5-1) | Master5s / 哈基米 v3pro | 1 | 胜 | +1,315.53 | 已分析 |
| 295 | [`79a6f726539d4747b696666900578ef6-1`](pages/games-025.md#g-79a6f726539d4747b696666900578ef6-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +367.12 | 已分析 |
| 296 | [`791ae149851d64f1b7252c0e44fa5b99-1`](pages/games-025.md#g-791ae149851d64f1b7252c0e44fa5b99-1) | Master6 / 小肥鱼v5-late1 | 0 | 胜 | +7,328.25 | 已分析 |
| 297 | [`1105b8f37c1ab04df11ba512dfce35f5-1`](pages/games-025.md#g-1105b8f37c1ab04df11ba512dfce35f5-1) | Master6 / 安生N29 | 0 | 胜 | +8,301.21 | 已分析 |
| 298 | [`785db4b4353aea1f2e61c8493ba425b3-1`](pages/games-025.md#g-785db4b4353aea1f2e61c8493ba425b3-1) | Master6 / RyanX-AIPowercore-M9XJ-630 | 0 | 胜 | +512.67 | 已分析 |
| 299 | [`a5339b36de9977c9a825bf6de5691cac-1`](pages/games-025.md#g-a5339b36de9977c9a825bf6de5691cac-1) | Master6 / 安生N30 | 0 | 胜 | +1,688.06 | 已分析 |
| 300 | [`7d2f125ddec4e3fb476907dadec6883e-1`](pages/games-025.md#g-7d2f125ddec4e3fb476907dadec6883e-1) | Master6 / ClaudeFable5.2-v7 | 0 | 胜 | +8,250.17 | 已分析 |
| 301 | [`4d590a1b51696b0d9983585ac8efbb51-1`](pages/games-025.md#g-4d590a1b51696b0d9983585ac8efbb51-1) | Master5s / 安生N27 | 1 | 胜 | +2,336.94 | 已分析 |
| 302 | [`481a728f3c0bfdca4497ed20cf869254-1`](pages/games-025.md#g-481a728f3c0bfdca4497ed20cf869254-1) | Master6 / Fable5.2-Max-V8 | 1 | 胜 | +15,206.00 | 已分析 |
| 303 | [`0892c1671ab180361a184b6c0ddb0abe-1`](pages/games-025.md#g-0892c1671ab180361a184b6c0ddb0abe-1) | Master6 / Fable5.2-Max-V6 | 0 | 胜 | +11,129.69 | 已分析 |
| 304 | [`b9f76da9aaa1349b95eb8d767a059f53-1`](pages/games-025.md#g-b9f76da9aaa1349b95eb8d767a059f53-1) | Master6 / stephen-K4TacSearch-v6 | 0 | 胜 | +1,528.11 | 已分析 |
| 305 | [`b4b95a9c8bf827a2684ef801ebd8b920-1`](pages/games-026.md#g-b4b95a9c8bf827a2684ef801ebd8b920-1) | Master6 / Krypton-M3s2-EarlyHome | 0 | 胜 | +958.08 | 已分析 |
| 306 | [`16c18b65adabdee40d6a79fc9def85b1-1`](pages/games-026.md#g-16c18b65adabdee40d6a79fc9def85b1-1) | Master6 / ClaudeFable5.2-v7 | 1 | 胜 | +9,807.43 | 已分析 |
| 307 | [`679bb68235c516308da2acf36b130c00-1`](pages/games-026.md#g-679bb68235c516308da2acf36b130c00-1) | Master6 / 安生N27 | 0 | 胜 | +8,698.04 | 已分析 |
| 308 | [`c42f8d6140839e3974cfbdf2b9a51922-1`](pages/games-026.md#g-c42f8d6140839e3974cfbdf2b9a51922-1) | Master6 / Echo5 | 0 | 胜 | +1,801.21 | 已分析 |
| 309 | [`0f976fbab719fdc1377137109920f960-1`](pages/games-026.md#g-0f976fbab719fdc1377137109920f960-1) | Master6 / 小肥鱼v5-d5 | 0 | 胜 | +789.30 | 已分析 |
| 310 | [`1efab038c5dd46eb1451a08658a30fbc-1`](pages/games-026.md#g-1efab038c5dd46eb1451a08658a30fbc-1) | Master6 / Echo4 | 0 | 胜 | +11,749.32 | 已分析 |
| 311 | [`8ace8ed61af9859060c6b29f66931b92-1`](pages/games-026.md#g-8ace8ed61af9859060c6b29f66931b92-1) | Master6 / stephen-K4TacSearch-v3 | 0 | 胜 | +11,369.88 | 已分析 |
| 312 | [`f8a6bc149397325799ff80e2de38804d-1`](pages/games-026.md#g-f8a6bc149397325799ff80e2de38804d-1) | Master6 / Fable5.2-Max-V7 | 0 | 胜 | +12,925.20 | 已分析 |
| 313 | [`7d651b81bb589a01eca3ff86a2b5f3c7-1`](pages/games-026.md#g-7d651b81bb589a01eca3ff86a2b5f3c7-1) | Master6 / 小肥鱼v5-late1 | 1 | 胜 | +652.47 | 已分析 |
| 314 | [`9df8e96aa0d450a42960627f0c273eb7-1`](pages/games-026.md#g-9df8e96aa0d450a42960627f0c273eb7-1) | Master6 / Krypton-M3s2-EarlyHome | 0 | 胜 | +6,410.22 | 已分析 |
| 315 | [`337d835612fef5b624dcf4d232334a48-1`](pages/games-026.md#g-337d835612fef5b624dcf4d232334a48-1) | Master6 / Fable5.2-Max-V9 | 0 | 胜 | +10,562.90 | 已分析 |
| 316 | [`8fefb27600ec429ba04a3850e38cb98e-1`](pages/games-026.md#g-8fefb27600ec429ba04a3850e38cb98e-1) | Master6 / Carago-V2微调 | 1 | 胜 | +1,030.44 | 已分析 |
| 317 | [`640cf4d5ac8e464b66bf93455c6deba2-1`](pages/games-027.md#g-640cf4d5ac8e464b66bf93455c6deba2-1) | Master6 / Fable5.2-Max-V7 | 0 | 胜 | +1,348.50 | 已分析 |
| 318 | [`3cefb5267a6300ebe841eed28fbd9766-1`](pages/games-027.md#g-3cefb5267a6300ebe841eed28fbd9766-1) | Master6 / Fable5.2-Max-V9 | 1 | 胜 | +1,418.42 | 已分析 |
| 319 | [`2f09025d3cfbd4028af4efaff1813dbe-1`](pages/games-027.md#g-2f09025d3cfbd4028af4efaff1813dbe-1) | Master6 / Fable5.2-Max-V8 | 0 | 胜 | +10,757.38 | 已分析 |
| 320 | [`6e0935557f8406a96b29dce58659e670-1`](pages/games-027.md#g-6e0935557f8406a96b29dce58659e670-1) | Master6 / ClaudeFable5.2-v8 | 0 | 胜 | +13,770.40 | 已分析 |
| 321 | [`5256711b9f8cbe46189d1f683cc928b3-1`](pages/games-027.md#g-5256711b9f8cbe46189d1f683cc928b3-1) | Master5s / 安生N31 | 0 | 胜 | +8,948.32 | 已分析 |
| 322 | [`72754f2abee913f2c18d7f67a02026b4-1`](pages/games-027.md#g-72754f2abee913f2c18d7f67a02026b4-1) | Master6 / 哈基米 v3 | 0 | 胜 | +1,609.44 | 已分析 |
| 323 | [`ea0da721a26d892c456fe7b0054bb018-1`](pages/games-027.md#g-ea0da721a26d892c456fe7b0054bb018-1) | Master6 / RyanX-AIPowercore-it720-preview | 0 | 负 | -911.17 | 已分析 |
| 324 | [`21548060ac200dbec3df57d35ee513c3-1`](pages/games-027.md#g-21548060ac200dbec3df57d35ee513c3-1) | Master6 / Krypton-M128Search-v1 | 1 | 胜 | +1,752.62 | 已分析 |
| 325 | [`5ddcefaf07f1af139387a8ec0a3eb96f-1`](pages/games-027.md#g-5ddcefaf07f1af139387a8ec0a3eb96f-1) | Master6 / claude-opus55-cloud-Echo6 | 1 | 胜 | +784.23 | 已分析 |
| 326 | [`50cac5af3793a0b9fb69fa44f19de0b1-1`](pages/games-027.md#g-50cac5af3793a0b9fb69fa44f19de0b1-1) | Master6 / stephen-K4TacSearch-v2 | 1 | 胜 | +1,580.41 | 已分析 |
| 327 | [`00b490c13c7f3afd6419725df29a09a4-1`](pages/games-027.md#g-00b490c13c7f3afd6419725df29a09a4-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +9,562.39 | 已分析 |
| 328 | [`f1e4f4bd9c59bc73c14dc61aef686eb5-1`](pages/games-027.md#g-f1e4f4bd9c59bc73c14dc61aef686eb5-1) | Master6 / stephen-K4TacSearch-v4 | 0 | 胜 | +7,474.07 | 已分析 |
| 329 | [`1fa76261ead2623276bcc7363e802630-1`](pages/games-028.md#g-1fa76261ead2623276bcc7363e802630-1) | Master5s / 安生N30 | 0 | 胜 | +2,063.44 | 已分析 |
| 330 | [`0a93f327e9ef8200ba227adfe97bd8a8-1`](pages/games-028.md#g-0a93f327e9ef8200ba227adfe97bd8a8-1) | Master6 / 安生N31 | 0 | 胜 | +9,054.41 | 已分析 |
| 331 | [`b89da539fda8c75bad5c0c54d28827b0-1`](pages/games-028.md#g-b89da539fda8c75bad5c0c54d28827b0-1) | Master6 / 安生N31 | 0 | 胜 | +1,511.71 | 已分析 |
| 332 | [`036cb350850848e47b66b5032fab60e9-1`](pages/games-028.md#g-036cb350850848e47b66b5032fab60e9-1) | Master6 / Echo3 | 0 | 胜 | +2,407.63 | 已分析 |
| 333 | [`cbdb85bb656b406dd338629b90b6f816-1`](pages/games-028.md#g-cbdb85bb656b406dd338629b90b6f816-1) | Master6 / 小肥鱼v5 | 0 | 胜 | +22.59 | 已分析 |
| 334 | [`704a532aab1f77dd3ba0210f9cf8b55b-1`](pages/games-028.md#g-704a532aab1f77dd3ba0210f9cf8b55b-1) | Master6 / ClaudeFable5.2-v7 | 0 | 胜 | +10,588.85 | 已分析 |
| 335 | [`06591c01cd12b7f46c7215d5b464fb9b-1`](pages/games-028.md#g-06591c01cd12b7f46c7215d5b464fb9b-1) | Master5s / RyanX-AIPowercore-it720-preview | 0 | 负 | -23.24 | 已分析 |
| 336 | [`afac7d66b6ca8a4d13967df93d77cad9-1`](pages/games-028.md#g-afac7d66b6ca8a4d13967df93d77cad9-1) | Master6 / First111 | 0 | 胜 | +1,172.90 | 已分析 |
| 337 | [`8a563a5e59432d54074bbef70c9d8a54-1`](pages/games-028.md#g-8a563a5e59432d54074bbef70c9d8a54-1) | Master6 / 哈基米基准 | 0 | 平 | +0.00 | 已分析 |
| 338 | [`e96d4dcf4e69f5e9d9af66a9000256e6-1`](pages/games-028.md#g-e96d4dcf4e69f5e9d9af66a9000256e6-1) | Master6 / claude-AlphaCarbon-Master6s | 0 | 负 | -810.01 | 已分析 |
| 339 | [`c6e8900903560d742b48733a54a20ef5-1`](pages/games-028.md#g-c6e8900903560d742b48733a54a20ef5-1) | Master6 / 哈基米 v3 | 0 | 胜 | +10,538.91 | 已分析 |
| 340 | [`217b7ffa09cd5838cedbe2cd7b6d45dd-1`](pages/games-028.md#g-217b7ffa09cd5838cedbe2cd7b6d45dd-1) | Master6 / stephen-K4TacSearch-v6 | 0 | 胜 | +160.52 | 已分析 |
| 341 | [`ba8f2a3377e8f67ec08cecd603790e43-1`](pages/games-029.md#g-ba8f2a3377e8f67ec08cecd603790e43-1) | Master6 / stephen-K4TacSearch-v4 | 0 | 胜 | +11,371.37 | 已分析 |
| 342 | [`7951977b9832d6c694e59caf0df53f83-1`](pages/games-029.md#g-7951977b9832d6c694e59caf0df53f83-1) | Master6 / Fable5.2-Max-V6 | 1 | 胜 | +1,072.05 | 已分析 |
| 343 | [`043181cc87f4e8d569a46681c3c617a7-1`](pages/games-029.md#g-043181cc87f4e8d569a46681c3c617a7-1) | Master6 / master3 | 1 | 胜 | +2,050.07 | 已分析 |
| 344 | [`2ba36ce809c070807f1df8ce9d9896a5-1`](pages/games-029.md#g-2ba36ce809c070807f1df8ce9d9896a5-1) | Master6 / 哈基米 v2 pro | 0 | 胜 | +8,220.76 | 已分析 |
| 345 | [`1c4129dd97facd73433a6f1127f07b66-1`](pages/games-029.md#g-1c4129dd97facd73433a6f1127f07b66-1) | Master6 / First111-1 | 0 | 胜 | +2,979.65 | 已分析 |
| 346 | [`ebaaf39c01dadae4059fb32b8508fc85-1`](pages/games-029.md#g-ebaaf39c01dadae4059fb32b8508fc85-1) | Master6 / 安生N27 | 0 | 胜 | +7,192.14 | 已分析 |
| 347 | [`3136eab484b5d37bf5699ab34fc9e3a7-1`](pages/games-029.md#g-3136eab484b5d37bf5699ab34fc9e3a7-1) | Master5s / claude-AlphaCarbon-Master6s | 0 | 负 | -7,459.05 | 已分析 |
| 348 | [`27b612db5474263ecb2909a57df24505-1`](pages/games-029.md#g-27b612db5474263ecb2909a57df24505-1) | Master6 / claude-AlphaCarbon-Master6s | 1 | 负 | -1,606.19 | 已分析 |
| 349 | [`c7b6871ddaa4ae5462f8a96e7f7743a1-1`](pages/games-029.md#g-c7b6871ddaa4ae5462f8a96e7f7743a1-1) | Master6 / Echo5 | 1 | 胜 | +7,548.12 | 已分析 |
| 350 | [`29a5d11dd8709bee1e35d534a72c8eca-1`](pages/games-029.md#g-29a5d11dd8709bee1e35d534a72c8eca-1) | Master6 / 安生N29 | 0 | 胜 | +11,784.38 | 已分析 |
| 351 | [`8576857b881006a65658502aa0fce156-1`](pages/games-029.md#g-8576857b881006a65658502aa0fce156-1) | Master6 / ClaudeFable5.2-v8 | 1 | 胜 | +11,065.73 | 已分析 |
| 352 | [`081e11502d87f9fe69ced1e56620e891-1`](pages/games-029.md#g-081e11502d87f9fe69ced1e56620e891-1) | Master6 / stephen-K4TacSearch-v1 | 0 | 胜 | +1,694.07 | 已分析 |
