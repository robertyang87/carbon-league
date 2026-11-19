# Master6 的“免搜索版”声明能证实到哪里

**Master6 在官方报名简介中把自己描述为“免搜索版”。这句原话有同一注册 ID 的两次官方榜单记录。** 平台将该版策略设为闭源；其网络结构、每步调用次数、手写规则和训练阶段的搜索或蒸馏细节，仍属未知。研究时可将 Master6 标为“提交者声明部署时免搜索”。“纯神经网络，无任何规则”和“训练也不用搜索”都超出了公开说明。[官方榜单](https://117.72.210.134/api/leaderboard) · [上次官方响应摘录的归档](https://github.com/robertyang87/carbon-league/blob/06f21b58328d89ce38031789ef7781d42c89f03e/reports/master_family/MASTER5S_MASTER6_EVIDENCE_20261118.zip)

## 主张和证据

| 对象 | 可直接核实的事实 | 证据边界 |
|---|---|---|
| Master6，`ecde65e596acbeb9ba5feca22e05f100` | 2026-11-19 03:46:31 UTC，官方榜单返回 `name=claude-AlphaCarbon-Master6`、`author=Claude`、`description=免搜索版`、`is_public=false`。该时点重榜第 2，285 局 262 胜 1 平 | 作者填写的部署说明。没有获取到能与该报名 ID 绑定的线上源码、配置或权重，也没有披露训练教师 |
| Master6s，`c6a59114106eb86101176aee749f28e4` | 按该 ID 查询官方对局，仍返回参赛名 `claude-AlphaCarbon-Master6s`。03:47 UTC 的公开榜单中未列出这个 ID。已核验的历史回放有两场与 Master6 换席直打 | 现有来源能分别辨认两位参赛者。6s 的搜索开关、与 6 是否同权重、是否在 6 上加搜索，均缺少提交者原话或运行工件 |
| Master3s2，`d679ed45ab595e8b7bb1ea51c3451154` | 平台标为 `is_public=true`。发布源码开头写 “with a short search at every turn”，策略入口确实调用 `_search`，函数内有搜索可用性及步段门。仓内原版与考官工作副本的 SHA-256 同为 `65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5` | 这是可以沿执行路径检查的搜索实现，结论适用于这个公开制品 |

**怎样追溯“免搜索版”。** 2026-11-18 14:18:10 UTC 的 `public_snapshot.json` 随仓内证据 ZIP 保存，同一 ID 已有相同简介和可见性，当时 117 局 102 胜。次日重读官方原始响应 HTTP 200，响应体 SHA-256 为 `144bb71e3914ad13a286eeed13182f5c951642352087fbf9d971e2c75ce8cef1`。联赛客户端的提交路径将 `description` 作为报名字段传出。榜单把这段文字原样展示，并未同时公开一个经平台检测的运行时搜索开关。[归档 ZIP，成员 public_snapshot.json](https://github.com/robertyang87/carbon-league/blob/06f21b58328d89ce38031789ef7781d42c89f03e/reports/master_family/MASTER5S_MASTER6_EVIDENCE_20261118.zip) · [提交字段](https://github.com/robertyang87/carbon-league/blob/eb691b894fbbdac4044010d7fb3662ff9a35c654/code/ladder.py#L173) · [官方榜单](https://117.72.210.134/api/leaderboard)

**Master6s 的直接证据。** 注册 ID 由仓内抓取清单与官方按 ID 查询的历史比赛交叉确认。冻结清单中的两场 6 与 6s 直接交手分别为 `27b612db5474263ecb2909a57df24505-1`、`e96d4dcf4e69f5e9d9af66a9000256e6-1`。这更新了早期裁定中“尚未相遇”的时点事实；两局比赛只能验证对阵身份和赛果。当前资料未含提交者解释二者关系的原文。[官方按 6s ID 查比赛](https://117.72.210.134/api/matches?compact=1&limit=1&offset=0&strategy=c6a59114106eb86101176aee749f28e4) · [当时的注册 ID 清单](../../../data/master6s_corpus_missing_20261118.json) · [第一局](https://117.72.210.134/replay/27b612db5474263ecb2909a57df24505/1) · [第二局](https://117.72.210.134/replay/e96d4dcf4e69f5e9d9af66a9000256e6/1) · [谱系裁定](../../RULING_MASTER6S_LINEAGE_20261118.md)

**其他代可借鉴的内容。** 早期公开的 Master 和 Master2 源文件写有 “One forward pass per turn gives action probabilities for every unit”，并说明碳图由公开事件记账得到。3s2 发布包则把候选动作交给短搜索。Master5s 同样闭源，官方简介仅为“先来个中间版本感受下，指标还在涨”。这些材料提供不同架构实例，无法把某一代的网络、守卫、标签来源或前向次数直接归到 Master6 名下。对 `eb691b89` 的完整 Git 树核对显示，`sources/` 只登记 Echo6 和 Master3s2；唯一登记的 Master 系列本尊是 3s2。[Master 源码](https://117.72.210.134/source/8cc292fe4a6fff3a7a71393ce7e657d5) · [Master2 源码](https://117.72.210.134/source/e591b77f3f8c72b4e243c16a4d2e60e0) · [3s2 发布来源](../../../sources/master3s2/README.md) · [3s2 实际搜索调用](../../../sources/master3s2/master3s2.py#L620) · [3s2 搜索与有效步段](../../../sources/master3s2/master3s2.py#L1252) · [登记目录](https://github.com/robertyang87/carbon-league/tree/eb691b894fbbdac4044010d7fb3662ff9a35c654/sources)
