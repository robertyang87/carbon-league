# S3 枚举试跑核对表（2026-11-16）

脚本：`s3_enumerate_roots.py`，md5 `1218d573e81f6f9b614a61597968ea45`。
试跑：诊断块 986200–986202，3 局，基线包 Echo4（md5 819250bf），
本 VM 普通进程（不设 0.7 s 预算、不设 systemd scope），总墙钟 2241 s，退出 0。

## 证据件 md5

| 件 | 字节 | md5 |
|---|---|---|
| roots_trial.jsonl（3920 行） | 3,693,068 | 11f0baa9cc2cc172fc706413d302a9d5 |
| roots_trial.json（同根 JSON 数组） | 4,513,552 | e6e2ba1453c526f63868da66f83a9236 |
| register_trial.json | 1,777 | 58a39ad8bd036789f97fc75423bd72cd |
| report_trial.json | 1,166 | ab3828ad9c227e17cd36a45d2fb565b4 |
| witness_trial.jsonl（sidecar，非冻结件，未入仓） | 19,881,352 | 4609087301209eefa902067758d509eb |

## 独立重算（搜索组，试跑 JSONL 直读）

- 根数 3920；root_id 唯一 3920/3920。
- trigger_class 全 class1（3920/3920）。
- 结算可辨根 10 根，全在 step 298，settlement_argmax 全 0。
- cand_count ∈ {5, 9, 13}（4k+1 型，守卫零否决）。
- 步域 15–298；子方向 1a 2602 / 1b 1318；双席 1960/1960。
- 局：986200 根 1450（725/725，764.0 s）、986201 根 1030（515/515，690.6 s）、
  986202 根 1440（720/720，785.1 s）；产根局 N=3，零根局 M=0。
- 18 根人工抽查全合（审查记录 §3.2）；结算独立验证 8/8（探针
  probe_settle_298.py，md5 86b20725e2b67d10541a4c9be521fd00）。
- §1.4 触发事件「无」；无 HALT。

## 门 2 未覆盖说明

试跑 3 局全以时限终局，无灭绝局；灭绝分支叶 F01 交叉核对未实际触发。
该分支代码与 s1_verify3 的 make_terminal_score_fn 同源（battery rerun
A6 B9 例已验 −102.0），全量遇灭绝局时自动执行、不等即停。
记为「试跑未覆盖、代码同源已验」，不构成开跑阻拦。
