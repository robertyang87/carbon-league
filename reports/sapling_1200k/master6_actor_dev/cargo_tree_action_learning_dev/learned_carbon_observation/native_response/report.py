"""Render only the observed roots, coverage, public search clocks and signed official settlements."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
q=json.loads((P/'result.json').read_text());n=len(q['maps']); roots=[v for v in q['maps'] if v['changed']]
notes=[]
for data in q['maps']:
    if not data['changed']:
        notes.append(f"地图 {data['seed']}、席 {data['seat']} 从 0 到 {data['last_t']} 共 {data['n_prefix']} 次真实对手调用，在预登记区间未出现任何己方工人指令差异。按协议记录零干预，未分叉，没有该图的分支终局。")
        continue
    a=data['arms']['student'];b=data['arms']['parent'];s=a['seat'];o=1-s
    d=data['component_net_own_minus_rival_student_minus_parent'];units=[]
    for uid,change in b['root_changed'].items():
        x=b['root_workers'][uid];y=a['root_workers'][uid]
        units.append(f"{uid}，{x['role']}，物理格 {x['cell']}。父包下令 {x['action']}，当步存活 {int(x['survived_first_joint'])}；学生下令 {y['action']}，当步存活 {int(y['survived_first_joint'])}。两席行动前现金 {x['wallets_pre']}。该工人当步现金事件，父臂 {x['root_cash_events']}，学生臂 {y['root_cash_events']}。")
    details=[]
    for title,arm in [('父代',b),('学生单步',a)]:
        score=arm['score'];stats=arm['suffix_clock'];l=arm['ledger']['root_to_terminal']
        timing=arm['whole_arm_elapsed_s'];hist=stats['max_completed_depth'];branches=stats['return_types']
        details.append(f"| {title} | {score[s]:,.2f} | {score[o]:,.2f} | {arm['official_margin']:+,.2f} | {l[s].get('own_delivery',0):,.2f} / {l[o].get('own_delivery',0):,.2f} | {l[s].get('ground_rent',0):,.2f} / {l[o].get('ground_rent',0):,.2f} | {-l[s].get('plant',0)-l[s].get('seize',0):,.2f} / {-l[o].get('plant',0)-l[o].get('seize',0):,.2f} | {arm['remaining_original_calls']} | {timing:.1f} |")
    notes.append(f"""地图 {data['seed']}、席 {s}，**第 {data['root_t']} 步第一次出现工人改令**。原版在看到任何分支指令前由正常入口独立返回一次，根命令为 `{b['root_orders'][o]}`。后续第一个不同的对手联合命令位于第 {data['first_later_enemy_command_difference']} 步。两臂各自走到官方终局。

{''.join(units)}

| 各臂 | 己方终分 | 原版终分 | 官方分差 | 自家回库 己 / 敌 | 地租 己 / 敌 | 树购置支出 己 / 敌 | 根后原版调用数 | 分支墙钟 s |
|---|---:|---:|---:|---|---|---|---:|---:|
"""+'\n'.join(details)+f"""

学生单步相对父臂的官方终局分差变化为 **{data['student_minus_parent_margin']:+,.2f}**。从根到终局，逐项计算「己方减对手」的两臂变化，正常回库 {d.get('own_delivery',0):+,.2f}、纯地租 {d.get('ground_rent',0):+,.2f}、种树与夺树的签名现金流合计 {d.get('plant',0)+d.get('seize',0):+,.2f}，其余项目见 result.json。实测搜索后缀深度直方图（完成深度→调用次数）父包 `{b['suffix_clock']['max_completed_depth']}`，学生臂 `{a['suffix_clock']['max_completed_depth']}`；两臂原版原生计时截止次数分别 {b['suffix_clock']['native_aborted']}、{a['suffix_clock']['native_aborted']}。""")
msg="""# 学习型观测对原版 Master3s2 的单步响应实验

冻结候选 `bc69b55e`，父包 `8da4b465`，原版考官 `65f268b2`。本实验只在两个预登记的已曝光开发图寻找第 24–128 步的首次己方工人改令。初始地形来自已签名地图，双方从零步真实执行；影子学生只接收与父代相同的本席公开历史。通过入口证据及绑定的 SHA 见 qualified.json。

"""+'\n\n'.join(notes)+f"""

{n} 张登记图中 {len(roots)} 张出现干预。原版温度 1，未设置随机种子，保留 0.58 秒自适应搜索。每张图各绑一枚物理 CPU，保留自然冷启动，在同一进程顺序完成该图分支。除根步外两臂己方使用完全相同的父代包。每个分支从同一个根后原版状态还原，包括追踪器、前次观察、搜索成本统计、策略 RNG 和进程内随机状态，重新核对语义摘要；其后双方收到各自变化的观测，搜索深度仍可随实际墙钟抖动。这里的终局差仅属于该干预和这次配对执行，不能推断总体胜率、上榜资格或新地图首胜。

run.py 产生按实际地图与席位的公共前缀、根状态及所有物理指令、对手时钟与每步观测哈希。settle.py 另起官方解释器逐个动作复演，并用 FastCarbon 独立分解每步双方现金，核对每步官方现金与完整终局。result.json 保存复核后的逐臂得分与资金流，原版每拍的候选、深度、墙钟及 CPU 见各局压缩文件。
"""
(P/'README.md').write_text(msg)
