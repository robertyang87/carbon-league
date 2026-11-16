# S3 枚举 Studio 分片 §10.5 回执（局 270–299）

- 开跑令：作战室 2026-11-16 12:09；开跑 2026-11-16 12:17:18 CST
- 完赛：最末 report mtime 2026-11-16 12:27:50 CST（lane3 / lane6）
- **墙时：632 秒（10 分 32 秒），12:17:18 → 12:27:50**
- 局数：30 局（270–299），register 并集恰为 270–299，无缺无重
- **总根数：36,128**（roots_lane<N>.jsonl 行数与各 report total_roots 逐路一致，witness 行数亦逐路一致）
- §1.4 事件：无（8 路 report 均为「无」，halts 均为空）
- HALT：无（Studio 目录无 HALT_*.json，watch.json halted=false，8 路 done=true）
- 零根局 M：0；产根局 N：30；N+M=30 成立
- 全部 30 局 reached_step 299，stop_reason 均为 time_limit_299
- 日志报错：8 路日志 grep error/traceback/exception/HALT 均为 0 行；各路末行均为「分片完成」
- 结算可决根：68（lane0 12、lane1 6、lane2 10、lane3 8、lane4 4、lane5 20、lane6 8、lane7 0）；settlement_argmax_nonzero 合计 6（仅 lane5）
- 内存峰值：最高 lane0 212,368 KB（≈207.4 MiB），8 路峰值 187,248–212,368 KB，均 < 512 MiB 上限；无 SIGKILL
- agent_call_s：lane0–6 n=2392，mean 0.010–0.011 s，max 0.012–0.015 s；lane7 n=1196，mean 0.011 s，max 0.014 s

## 分路明细

| lane | 局 | 种子 | 根数 | 墙时（日志） | 峰值 RSS KB | 可决根 |
|---|---|---|---|---|---|---|
| 0 | 270–273 | 986470–986473 | 5,146 | 71 s | 212,368 | 12 |
| 1 | 274–277 | 986474–986477 | 4,406 | 67 s | 206,640 | 6 |
| 2 | 278–281 | 986478–986481 | 4,634 | 80 s | 203,600 | 10 |
| 3 | 282–285 | 986482–986485 | 5,176 | 85 s | 208,032 | 8 |
| 4 | 286–289 | 986486–986489 | 4,310 | 77 s | 199,456 | 4 |
| 5 | 290–293 | 986490–986493 | 4,942 | 83 s | 203,440 | 20 |
| 6 | 294–297 | 986494–986497 | 5,110 | 85 s | 202,288 | 8 |
| 7 | 298–299 | 986498–986499 | 2,404 | 41 s | 187,248 | 0 |
| 合计 | 270–299（30 局） | 986470–986499 | 36,128 | — | 最高 212,368 | 68 |

## md5（Studio 直读）

- script_md5（8 路一致）：1218d573e81f6f9b614a61597968ea45
- package_md5（8 路一致）：819250bfc82632786063dabd46a616bd
- protocol_md5：4212f13afe71b4008579df5f0c605bf0（S3_ROOT_LIST_FREEZE_PROTOCOL.md）
- key_units code_md5（cnet2_search.py 386–483）：94f7fb6438bb7577b141e388f068e122

### roots_lane<N>.jsonl

- lane0 be76e8a5fbd2a6f04552803a8acca2bf
- lane1 98dce433590870a4a1de54a64bbf8a1c
- lane2 1ccd501874b891d9677aaed6d7cf3ea5
- lane3 8eb61afdf6a5fe36a319eb7261261acc
- lane4 01cfb5238d8e53d0424fd53d2d036d84
- lane5 484773377b154556016c00d8d1c32427
- lane6 cc40dbaf48443f2529eccd896dd3aad5
- lane7 eb00bff86cf381ea9a8d025bf6731f9d

### register_lane<N>.json

- lane0 846dc799d072a1bd42e10a3b099c3624
- lane1 664be3d7e5bcad8134ad2d833050820d
- lane2 3aa5d0e384edb68faafe0e276adf0ab7
- lane3 229195d72bf9cae5bfd6fcfe66e88f17
- lane4 658511b6131d4af889ac44bcd65c5a2e
- lane5 33d15f95f60a752cf0b12b9993f9c867
- lane6 20c0e07558ac9b197e3d1e9b47a40300
- lane7 d3f49c314b9d96876e8fcfb35c75aa86

### report_lane<N>.json

- lane0 d80abac5e12aca302c1a41fe31ee92cc
- lane1 e8360bb193fcb812f937989f2e2a8c43
- lane2 cbbee840cf61798af5c7e3d2ea80a72b
- lane3 e0b034d378f80abb5c4bfec8578ad7ca
- lane4 e955b4872d5e442aeb33497844bd047a
- lane5 b0f07043593cc367ae5b86e652b1148d
- lane6 88771bce455ed6672036563222cd392a
- lane7 3bee270f34d3a77c77f948243d78e6be

### witness_lane<N>.jsonl

- lane0 288de310d7e189b5421fcf7a7c80760f
- lane1 17daeebf21469d7d6faf8dee86741a14
- lane2 47630d26e6a3e41a29d0367a9f591fd8
- lane3 2986671f5b073df179360cd59b08cb1a
- lane4 c7b031f47848c88d0aa9d6f650103828
- lane5 9041da4095d2ffdcc4f6a24b63f0f7a2
- lane6 df9c92962697db4b537e295b2d26e08c
- lane7 8d7e9f89d0c8403729bc68596d4d4100

### roots_lane<N>.json

- lane0 3bd7c8753a0cc399189f060f80997bcf
- lane1 9ef3c0ebf59c916652cabd5f58f79bbb
- lane2 4363cf704a5de5de01ebf5c1c0190ddc
- lane3 5e4e917102136734f390831755a6b862
- lane4 539edeb65e65231b20daf24a5ada20e6
- lane5 e78ced87af0af9b138fd380b049f6d08
- lane6 43315e7cd4a07667224fc2c2f8041676
- lane7 75d6f0949f2cfbfa5d8b4db138230b3b

## 落盘

- Studio：/Users/arm/junyang/carbon/data/s3_enum_studio/out/lane<N>/（每路 lane<N>.log / register / report / roots.json / roots.jsonl / witness.jsonl）
- 本地副本：~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/s3-enum-studio-shard/out/（全份下载，md5 与上列 Studio 直读值逐一核对一致）
- 合并：未跑 --merge，归搜索组
