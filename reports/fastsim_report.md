# FastCarbon 对拍验证报告

生成时间：2026-11-02 16:19:45（本地）  
总检查项：36，通过：36，失败：0  
对拍总耗时：25.1s

## 结果明细


### map

- [PASS] generation#2 == env.reset() full_carbon — equal=True, sums 1956.0 vs 1956
- [PASS] reset(seed) deterministic & 4-fold symmetric — gen1 sum=2422.0 nonzero=99

### game

- [PASS] seed=7 obs() visibility parity (every 25 steps) — carbon visibility, players, trees, step all match interpreter rule
- [PASS] seed=7 300 steps full parity — final cash=[20, 9.92], plants>=1, max_workers=4, max_trees=1
- [PASS] seed=123 obs() visibility parity (every 25 steps) — carbon visibility, players, trees, step all match interpreter rule
- [PASS] seed=123 300 steps full parity — final cash=[20, 104.63], plants>=15, max_workers=10, max_trees=7

### scenario

- [PASS] S1a collision capture parity — 1 step(s)
- [PASS] S1a collision capture: winner is P1 collector with 60 cargo — {'player-1-worker-0': (1, 112, 60, 'COLLECTOR')}
- [PASS] S1b collision tie parity — 1 step(s)
- [PASS] S1b collision tie: both dead, cell carbon 40 — ({}, 40.0)
- [PASS] S1c collision vs planter parity — 1 step(s)
- [PASS] S1c collision vs planter: planter dead, collector cargo 35 — {'player-1-worker-0': (1, 112, 35, 'COLLECTOR')}
- [PASS] S1d planter pileup on tree parity — 1 step(s)
- [PASS] S1d planter pileup on tree: both planters dead, P0 cash 110 — ({}, [110, 100])
- [PASS] S2a own deposit parity — 1 step(s)
- [PASS] S2a own deposit: cash 180, cargo 0, worker alive on base — ([180, 100], {'player-0-worker-0': (0, 64, 0, 'COLLECTOR')})
- [PASS] S2b enemy deposit parity — 1 step(s)
- [PASS] S2b enemy deposit: P0 cash 150, P1 collector alive cargo 0 — ([150, 100], {'player-1-worker-0': (1, 64, 0, 'COLLECTOR')})
- [PASS] S2c enemy empty on base dies parity — 1 step(s)
- [PASS] S2c enemy empty on base dies: worker deleted, cash unchanged — ({}, [100, 100])
- [PASS] S3 plant + absorb timing parity — 2 step(s)
- [PASS] S3 plant + absorb timing: cash 995 after absorb step, neighbours 96.25 — ([995.0, 100], 96.25)
- [PASS] S3b plant cost inflation parity — 1 step(s)
- [PASS] S3b plant cost inflation: cash == round(1000-26.175,2), 2 trees — ([973.83, 100], 2)
- [PASS] S4 seize parity — 1 step(s)
- [PASS] S4 seize: tree flipped to P0 age 21, cash 100-20+15=95 — ({'player-1-tree-0': (0, 112, 21)}, [95.0, 100])
- [PASS] S5 wither parity — 1 step(s)
- [PASS] S5 wither: tree gone, collector cargo 9, cell carbon 21 — ({}, {'player-0-worker-0': (0, 112, 9.0, 'COLLECTOR')}, 21.0)
- [PASS] S6 absorption & siphon parity — 1 step(s)
- [PASS] S6 absorption & siphon: P0 cash 535.5 (23+5 siphon, 3.75+3.75 cells) — [535.5, 500]
- [PASS] S6 absorption & siphon: siphoned collector 230 collected -> 207 after 10% — {'player-0-worker-0': (0, 97, 70.0, 'COLLECTOR'), 'player-1-worker-0': (1, 127, 207.0, 'COLLECTOR'), 'player-1-worker-1': (1, 98, 45.0, 'COLLECTOR')}
- [PASS] S6 absorption & siphon: blocked cell carbon 70 (collected, no tree absorption) — 70.0
- [PASS] S7 worker limit parity — 1 step(s)
- [PASS] S7 worker limit: exactly 10 workers after double recruit attempt — 10

### agents

- [PASS] example.py runs 30 steps on FastCarbon — cash=[80, 80]
- [PASS] reference_juzi.py runs 10 steps on FastCarbon — cash=[8.79, 20], 7.4s for 10 steps (agent-bound)

## 测速（300 步一局，随机招募/移动 agent 自对弈）

- 原版 zerosum_env `env.run`（随机 agent）：1.90s，最终 reward=[-211, 20]。注意：该局因一方被淘汰（compute_next_board 的 DONE 判定）提前结束，实际步数 < 300，此数字仅作量级参考。
- FastCarbon（同 agent 逻辑，obs+step，满 300 步）：中位数 0.185s（5 次：0.099, 0.101, 0.185, 0.195, 0.189），最终 cash=[20, 20]
- **同策略满 300 步对比（chaos 策略，含招募/种树/吸收的真实负载）：原版 Board 链 4.38s vs FastCarbon 0.605s → 加速比 7×**
- FastCarbon 纯转移（空动作 300 步）：0.033s（0.11 ms/步）
- 注：env.run 与 FastCarbon 的 agent 随机数流不完全一致，故最终分数不要求相同；语义一致性由上面的逐步对拍保证。

## 语义对齐范围与已知边界

- 已逐条对齐 Board.next()：招募成本（按全局工人数计价）与 worker_limit=10、环绕移动、碰撞消解（最少携带碳采集员胜出/并列全灭/胜者缴获/败者碳落树·基地·格子）、基地存碳（含敌方基地规则）、停留采集 30%、种植成本（首棵 20，之后 20+5·1.235^存活树总数）、抢树 20、树龄 50 枯萎留 30 碳、树吸收（4 邻 3.75% 带 quad 条件、8 邻+树格敌方采集员 10%）、未受影响格再生 3%、cash 2 位/格碳 3 位四舍五入、新生树当步不吸收。
- 复刻的原版怪癖：抢树时 _delete_tree 会清掉抢树者所在格的 worker 标记（影响当步吸收/再生判定）；工人死亡时其所种树的 trees_dict creator 字段置 None；quad flag 字典的键（任意树的 4 邻格）当步一律不再生，即使 flag 集合为空。
- 地图生成：populate_board 使用全局 np.random 流（原版从不自行播种），FastCarbon.reset(seed) 等价于 np.random.seed(seed) 后第一次 populate；env 的 make()+reset() 会 populate 两次，本报告已按此校验（第二次生成与 env full_carbon 完全一致）。
- 未覆盖（属 env core 而非 Board.next 的职责，未做也不影响对拍）： agent 超时与 remainingOverageTime 扣减（本实现原样透传）、compute_next_board 的淘汰/DONE 判定、无效动作的 INVALID 状态处理（本实现按 Board 行为将未知动作字符串视为无动作）。
- obs() 的 trees 字典条目顺序与 Board.observation 在个别步骤可能不同（本实现每步按玩家顺序重建，与 Board 构造顺序一致；吸收阶段的重插入顺序也已复刻），字典内容逐项一致——对拍按字典内容比较。
