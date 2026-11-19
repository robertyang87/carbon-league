# 前五追加两场逐局复核

截至 `87688aac`，两场均为前五对外赛。累计有 1,638 场获双方官方终分，其中互打 121 场、对外 1,517 场。另有 1 场历史执行故障，共 1,639 个实测初盘。[逐索引差分](score_delta.json) · [地图指纹](exposure.json)

| 前五选手 / 席位 | 对手 | 官方分（前五 / 对手） | 最大十步现金摆动 | 十步分差变化 | 正常交仓（前五 / 对手） | 树收入（前五 / 对手） |
|---|---|---:|---:|---:|---:|---:|
| [claude-AlphaCarbon-Master6 / 1](4a8912fffc316669d63682a50de1f7b1-1.md) | RyanX-AIPowercore-it920-evo31 | 5,176.69 / 6,401.40 | 95–105 | -590.38 | 120.54 / 591.81 | 53.49 / 146.24 |
| [RyanX-AIPowercore-it720-preview / 0](c94ce6f5882f7141736fc65769d45fdf-1.md) | LJM-TidalSearch-v1 | 8,983.44 / 7,686.59 | 225–235 | +809.65 | 818.09 / 0.00 | 80.62 / 210.04 |

两局均保留现金转折前 3、5、10 步双方具体工人、基地、树、货量及实际命令。逐拍现金包含正常交仓、树吸收、碰撞入账、双方支出。双方同时选择动作，后续结果仅作标签。 [回溯快照](turning_points.json) · [逐步动作](turn_actions.json.gz) · [收入账](turn_cash.json.gz) · [攻防与树权](cases.json.gz)

claude-AlphaCarbon-Master6 在物理席 1 对 RyanX-AIPowercore-it920-evo31，本场终局分差 -1,224.71。此前同一对阵、双边物理席位也相同的比赛共有 3 场，胜／平／负为 0／0／3。冻结榜未查到对手档位。 [同对手逐场证据](same_opponent.json)

RyanX-AIPowercore-it720-preview 在物理席 0 对 LJM-TidalSearch-v1，本场终局分差 +1,296.85。此前同一对阵、双边物理席位也相同的比赛共有 0 场，胜／平／负为 0／0／0。冻结榜未查到对手档位。 [同对手逐场证据](same_opponent.json)

双引擎分别复算 598 个联合回合，两席共 1,196 个行动前状态可从公开历史重建。另独立抽核 64 个席位前瞻窗口，区分完整 10／30 步和被终局截断的未来收入。 [规则复演](validation.json) · [无泄漏索引](context_receipt.json) · [前瞻收入](future_economy_check.json) · [原始回放](replays.tar)
