# 正式考卷排除的本地用图

仓库快照 `4be3bc19b` 可从本地实验成绩和登记记录恢复 **1,053 个不同种子和 1,053 张不同初盘**。其中 36 张还留有比赛记录器写入的初盘哈希，逐件与本地官方生成方式一致。其余初盘按原地图种子重建。考卷按种子和 225 格行优先碳量同时排除。同期天梯清单覆盖 1,549 张图，两类合计 2,602 张，互不重合。[全部来源、逐种子棋盘与哈希](exposure.json)

本次枚举 65 份 `reports/sapling_1200k/**/results.json`，取带逐局种子的数组或分臂数组，核对 20 份带来源动作哈希与实测地图哈希的逐局凭据并读早期本地地图登记与三份考试原始压缩包。三份压缩包各含 1,000、3,000、200 条逐席成绩，同图跨变体和跨机器复算只记一张地图。清单逐件记录来源文件摘要与种子，缺逐局数组的文件也留零行计数。其他自定义格式的内部实验和仓外训练仍须追加对应的曝光地图。另外两张只为开发场预检过的地图（717043911、730178127）在正式抽取入口按已登记的计划单独排除。即使策略筛选在双席整场前就停止，这两张图仍计入曝光。[已冻结的开发图](../master6_actor_dev/stratified_action_value_dev/source/pilot_plan.json)

正式考卷未抽取。

复算使用以下命令。生成器只读取给定提交中的历史记录与已出现的游戏种子。[复算程序](build.py)

```sh
python reports/sapling_1200k/known_local_maps/build.py 4be3bc19b --out /tmp/sapling-local-map-check.json
cmp /tmp/sapling-local-map-check.json reports/sapling_1200k/known_local_maps/exposure.json
```
