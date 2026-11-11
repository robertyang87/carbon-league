# Echo2 候选：配置和考试（2026-11-11 晚）

背景（天梯，Echo1 有回放的 42 局）：输赢几乎完全由"对手种了多少树"决定。对手种 34 棵以上时 6 战 1 胜、平均 −701；
少于 20 棵时 15 战全胜、平均 +990。入库在赢局和输局里几乎一样。M6 赢 Master3s 的那一局也是靠种树 33 对 13。
所以候选都围绕"抢种树"这一点。主线仍是 L20 上的 `snipe_a`，这里是今晚能出结果的小修。

公共约定：
- 基准 E0 = Echo1 纯网络（权重 `cnet2_m3w3L`，全程 2 号风格行，防撞保护开）。
- 每个候选 200 对，同种子减 E0；对手：`opp_master2r`、`opp_m3echo`、M 风格开局网络。
- 每组报：分差、我方和对方的种树数、树收入（全局和前 100 步）、入库、种植员数、损失的车数。
- 某候选对 Master2 真身和 `opp_m3echo` 都不为负、且至少一个显著为正（超过 2 倍标准误）：补到 600 对，过了打限时包，由用户决定上榜。

## 1. E2-W：加权重训（DGX）

数据集：DGX 上已建好的 100 局 Master3s 数据集（下面写作 `$DS`）。训两份，区别只在第 2 步做不做。

```bash
cd ~/carbon/code
export BC_DEVICE=cuda BC_RAM=1 BC_THREADS=8 BC_INIT=<Grove1 策略网浮点权重> \
       BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
unset BC_TRAIN_PREFIX BC_CACHE_TRUNK BC_SELF_KL BC_EXIT_W BC_AUX_W BC_COMPILE
CFG='{"C":128,"NB":8,"D":128}'

# W5：100 局，不加权（对照：看多出来的 11 局值多少）
python3 set_uw.py $DS 1 1 1
python3 cnet2_train.py $DS ../data/weights/cnet2_m3w5.npz 20 "$CFG"

# W4：100 局，Master3s 对"种树多 / 会打架"的对手的那些局，权重乘 3
python3 set_uw.py $DS 1 1 1
python3 set_gw.py $DS 3 RyanX-AIPowercore Master2 Carago Fable5.2-Max-V3 Echo1 大肥鱼 哈基米
python3 cnet2_train.py $DS ../data/weights/cnet2_m3w4.npz 20 "$CFG"
```

- 和 W3（Echo1）完全相同的训练设置：从 Grove1 出发、整网、学习率 3e-4、20 轮、批量 256。
- `set_uw.py`、`set_gw.py` 都会改写数据集里的 `w.npy`，所以两份要先后跑，或者各用一份数据集拷贝。
- `set_gw.py` 会打印一行（加权了多少局、分别是哪些对手、平均权重），请原样回报。
- 两份各取最后一轮（`_ep19.npz` 或 `_last.npz`）送考。
- 预期不高：这些回放里 Master3s 始终是抢赢的一方，教不会"被抢先时怎么办"。它便宜，所以做。

## 2. P2 / P3 / P4 / PR：多招种植员的开关（代码 `16b0b03` 或更新）

在 E0 的包配置上只加一项 `MACRO`，其余不变：

| 代号 | MACRO |
|---|---|
| P2 | `{"PL_MIN_E":2,"PL_MIN_M":2}` |
| P3 | `{"PL_MIN_E":2,"PL_MIN_M":3}` |
| P4 | `{"PL_MIN_E":3,"PL_MIN_M":4}` |
| PR | `{"PL_MIN_E":2,"PL_MIN_M":2,"PL_MIN_REACT":0.5}` |

- E 段是前 60 步，M 段是第 60–240 步。场上种植员少于设定数时，网络想招车的那一次改成招种植员；不额外多招人。
- PR：对手的树比我们多时，下限再加 0.5 ×（对方树数 − 我方树数）。
- 考试记录里的 `mac_push` 是每局触发次数，一并报；为 0 说明开关没接上。
- 预期不高：以前所有规则开关都是零或负。

## 3. E2-S / E2-Q：Echo1 加搜索

- E2-S：单车搜索，`fx_echoS20.py`。
- E2-Q：多车搜索 `SEQ_KEYS=3`，`fx_echoQ3.py`。
- 只对 Master2 真身 200 对（搜索在走子固定的对手身上分数虚高，另两个对手不考）。
- 各打 10 局量每步耗时：中位数、99% 分位、最大值。
- 已知的折扣：推演用的小网络是按老打法训的，和 Echo1 不完全匹配。

## 4. 已关闭

- E2-L（第 240 步后换 7 号风格行）：−138 ± 34。Echo 系列全程用 2 号行。
- 给收尾加权（W2）：比不加权低约 100 分。

## 5. 可以叠加

单项有效的先上榜。两项都有效时（例如 W4 的权重加 P3 的开关），叠加后再考 200 对，作为下一个版本。
