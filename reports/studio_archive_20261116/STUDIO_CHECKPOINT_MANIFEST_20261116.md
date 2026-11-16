# Studio 检查点件清单（2026-11-16）

范围：ppou2 训练探针（Mac Studio，2026-11-16 09:15:49–10:12:27 CST）产生的检查点件。
目录：Studio `/Users/arm/junyang/carbon/data/ppou2_probe_studio/`。二进制件留 Studio 本机、不入仓，本清单记其身份（字节数＋SHA-256＋MD5）。

口径注：`rl2_checkpoint.py` 的封印哈希为 SHA-256（checkpoint_complete.json 与 train 行 checkpoint.files 均记 SHA-256）；MD5 仅作辅核。

| 件 | 角色 | 字节 | SHA-256 | MD5 |
|---|---|---|---|---|
| state_frozen_it40.pt | 输入：主臂 it=40 冻结态（L20 拉取） | 82,600,176 | 6c0c4910b9d562b302a8041042817fdc5d848e3f7283c8d5c39f9a4d512ef588 | 73f649aa98993f783396fa91e88554f6 |
| out/state_last.pt | 产出：it=41 训练态（net/opt/vnet/vopt） | 82,585,520 | 06218eb21db17f3c793fece36a3d0c9c091fede775dd4d4299d5218338e2c1cc | — |
| out/ckpt_41.npz | 产出：it=41 策略检查点 | 13,972,532 | d401b064550a6613e4defd681948167ed2b55a0fa1d6af45151e36c93adcc976 | b2e456a232701251a1b2a78b55fa5cf8 |
| out/ckpt_last.npz | 同 ckpt_41.npz（字节同、哈希同） | 13,972,532 | d401b064550a6613e4defd681948167ed2b55a0fa1d6af45151e36c93adcc976 | b2e456a232701251a1b2a78b55fa5cf8 |
| out/checkpoint_complete.json | 封印件（it=41，两件 SHA-256 如上） | 209 | —（全文随本批入仓：ppou2_probe_checkpoint_complete.json） | — |

验合：加载测试通过（ckpt meta it=41；state keys 含 net/opt/vnet/vopt，it=41）；封印件所记两件 SHA-256 与落盘文件直读逐字合。

处置（策略组＋作战室 2026-11-16 裁定，PPOU2_STUDIO_PROBE_RULING_20261116）：探针检查点为探针工件，**不入主臂链**；慢速第二席不成立，ppou2 正训轮不再派 Studio；单轮 KL 0.1913 待查、MPS 训练挂起（L20 读口径后另裁）。

附：S3 枚举 Studio 分片（局 270–299）无检查点件，其产出为根单数据件，已随本批入仓 data/s3_enum_studio/。
