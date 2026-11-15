# Echo4 战术搜索比较器 · 原型实现（搜索组）

状态与全部数据见 `reports/search_dossier/SEARCH_LINE_STATUS_20261116.md`；注册文件 `reports/search_dossier/PREREG_ECHO4_TACTICAL_SEARCH.md`（判据以注册为准）。

## 文件

- `comparator.py` — 比较器核心（纯标准库）：K≤4 候选、整轮完成才提交、中断回上一完整轮（首轮未完回 A0）、守卫否决过滤、入口起算的调用内时限。
- `telemetry.py` — 独立遥测 StatsCollector（显式发布，针对旧实现 stats 字典未发布到 agent 的死链教训）。
- `leaf_adapters.py` — 叶适配：叶①启发式叶 affine 直通（原始分差尺度）；vnet 适配骨架（身份＝（检查点, state md5），U=(tanh(z)+1)/2，未过四项验收禁用）。
- `pack_bridge.py` — 只读桥接（SI08）：钉版四源（search/feat/np/fastsim）全量拉取、逐件按钉版 md5 校验后落 `vendor/`；加载前逐件复核在盘缓存（不合即停）；另提供离线 bootstrap（从本地冻结 checkout 复制校验，不依赖 gh_api 在场）。
- `real_env.py` — 真实环境接线：把 Comparator 装进钉版 cnet2_search 的 `_search` 形态（保留候选生成段、替换推进＋判读段；嵌套 leaf/guard 以闭包按名直引，漂移风险见文件头注）。
- `tests_s1.py` — 桩级机制用例（7 例）。`smoke_real_s1.py` — 真实环境冒烟。`s1_battery.py` — S1 电池 v1（自然根＋精确终局正反例；判读闸过松，见 SI03，只作历史工件）。`s1_battery2.py` — S1 电池 v2：正式/诊断双模式、注册断言绑退出码、自描述证据包，含 `--replay` 对旧证据回放判读。

## 运行

```
python3 tests_s1.py          # 桩级 7/7
python3 tests_s1_fix.py      # 修复回归（SI01/02/04/05/06＋anytime）15/15
python3 tests_s1_fix_r2.py   # R2 修复回归（正式闸/桥/计时锚点）
python3 smoke_real_s1.py [--pack PATH]  # 真实环境冒烟（需先经 pack_bridge 备齐 vendor/；包取 --pack→缓存→checkout→在线，整包 hash 恒验）
python3 s1_battery2.py       # 正式 S1 电池 v2（正式闸：FAIL 退出 2、未验退出 3），证据包落 s1_battery2_evidence.json
python3 s1_battery2.py --mode diagnostic   # 诊断模式：同采集同断言、只记账不裁决
python3 s1_battery2.py --replay <旧证据 JSON>  # 对既有证据回放正式判读，不跑实局
```

`s1_battery.py` 为 v1 历史工件（判读闸过松，见 SI03），不再作正式运行口径。

钉版：基体包 `code/packs/echo4_int8.py`（md5 819250bf…）；四源 feat 2a45b08b／np d8c15ad4／fastsim aefe6c83／search 94f7fb64。vendor/ 为本地拉取的校验副本，不入仓。
