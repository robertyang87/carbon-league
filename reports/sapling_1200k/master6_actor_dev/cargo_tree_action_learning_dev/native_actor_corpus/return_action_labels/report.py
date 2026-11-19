"""Write a Chinese report from the signed root list, verified settlements and frozen leave-map-out fit."""
import json
import gzip
from pathlib import Path

HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_bytes())

def report():
    registry,labels,fit=map(read,('roots.json','labels.json','fit.json'))
    roots={r['id']:r for r in registry['roots']};rows=labels['rows']; folds=fit['folds']
    supported=sum(f['supported'] for f in folds.values()); informative=len(fit['strict_improvements_vs_both'])
    action='达到追加独立采样的登记门槛。' if fit['larger_independent_collection'] else '没有达到追加独立采样的登记门槛，保留父包。'
    out=['# 载货采集员的真实原版终局标签','',f"**{action}** 在四张已曝光的训练地形上，八个预定时窗中有七个发现合格选择。两种部署都通过本席公开入口自由发令。全骨干在七处均把焦点工送到比父包离己方基地更远的格子，父包在其中一处会当拍到库。六处只改变焦点工自身的命令，一处同时改变三个单位。按整图留一拟合，{supported}/4 折同时具有两类训练标签。学得的选择器在 {informative}/4 张留图同时严格超过固定父包和仅截距模型。回仓特征模型在七次留出预测中选全骨干 {fit['pooled']['learned']['student_choices']} 次，地形等权的平均终局分差相对父包变化 {fit['terrain_equal']['learned']['mean_gain_vs_parent']['margin']:+.2f}。下文的留图只对本次拟合未见。[冻结根与完整合法顺序](roots.json) · [拟合与逐根选择](fit.json)", '', '| 训练图、席位和决策步 | 携货 | 父包 / 全骨干落点距己方基地 | 终局分差，父包 / 全骨干 | 官方分差变化 | 8 拍两队入库净差变化 | 24 拍两队入库净差变化 | 至终局两队树收入差变化 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        p,s=(r['arms'][k] for k in ('parent','student'));who=r['seat'];f=r['focal']
        dh=lambda phase:s['horizons'][phase]['deposit_margin']-p['horizons'][phase]['deposit_margin']
        def tree(a,side):return sum(a['horizons']['terminal']['by_physical_seat'][side].get(k,0.) for k in ('ground_rent','cargo_tax'))
        tree_gain=(tree(s,who)-tree(s,1-who))-(tree(p,who)-tree(p,1-who));out.append(f"| {r['seed']} / {who} / {r['t']} | {f['cargo']:.2f} | {f['parent_distance']} / {f['student_distance']} | {p['margin']:+,.2f} / {s['margin']:+,.2f} | {r['delta']['margin']:+,.2f} | {dh('after_8'):+,.2f} | {dh('after_24'):+,.2f} | {tree_gain:+,.2f} |")
    out+=['','表中变化均按全骨干减父包。入库净差先取本队采集员正常入库额减对手正常入库额，树木收入合计地面吸收与货税。整队实际命令共同决定各支结果。每个动作只获得一次原版自然续弈，分叉后的随机性与局面演化都会影响终分。双方各自的现金、入库、树收入、队伍资产及焦点工回款可逐根复核。[官方终分与双席现金账](labels.json)','', '## 按地形留一的真实二选一训练','', '每折以其他地图上实测终局分差较高的动作作为目标。两个可见特征分别是携货加权的两落点距仓差，以及携货加权的当拍是否到库差。含偏置的 sigmoid 神经元以等地形权重拟合，两维权重的正负均由训练数据确定。仅截距对照使用相同损失、正则化和迭代预算；胜负或经济结果都不进入推理输入。每个留出根只在两条有真实回报的整队令之间选择。', '', '| 决策方法 | 得到预测的根 | 选全骨干 | 真实胜 / 平 / 负 | 地形等权平均终局分差 | 相对父包 |','|---|---:|---:|---:|---:|---:|']
    for k,name in [('parent','固定父包'),('student','固定全骨干'),('intercept_only','仅截距'),('learned','回仓特征神经元')]:
        a,b=fit['pooled'][k],fit['terrain_equal'][k]
        if a['scored']:
            out.append(f"| {name} | {a['scored']}/7 | {a['student_choices']} | {a['wins']} / {a['draws']} / {a['losses']} | {b['mean']['margin']:+,.2f} | {b['mean_gain_vs_parent']['margin']:+,.2f} |")
        else:out.append(f'| {name} | 0/7 | — | — | — | — |')
    out+=['','| 留出地形 | 留出根数 | 特征模型选全骨干 | 特征模型相对父包的平均终差变化 | 仅截距相对父包的平均终差变化 |','|---|---:|---:|---:|---:|']
    for terrain,f in sorted(folds.items(),key=lambda i:i[1]['held_predictions'][0]['seed']):
        a=f['outcomes']['learned'];b=f['outcomes']['intercept_only'];seed=f['held_predictions'][0]['seed']
        if f['supported']:out.append(f"| {seed} | {a['scored']} | {a['student_choices']} | {a['mean_gain_vs_parent']['margin']:+,.2f} | {b['mean_gain_vs_parent']['margin']:+,.2f} |")
        else:out.append(f"| {seed} | {len(f['held_predictions'])} | — | 训练方向不足 | 训练方向不足 |")
    out+=['', '## 新增证据与原版时钟','', '原有九个采用同一首胜包 `8da4b465` 续策的根落在 t4–138，其中原训练图上的四个都在前 13 拍。既有方向试验在训练图上没有能据真实两臂终分选择权重步长的根。另一部署 `4e3701f3` 只在 t128–159 替换权重，因此本次固定晚段仍会真实变令的 `6c548995`。四图来自原有十三张训练图，按种子升序取头四张，源历史席位依序为 0、1、0、1。`621472121` 席 0 的 t210–237 窗无合格令，其余七处均在签收结果前锁定。[旧方向试验](../terminal_direction/README.md) · [本次登记](protocol.json)', '', f"每张图仅从开局预热一次原版。每处先核两包真实自由命令，再冻结原版当拍的自然命令，以及包括搜索成本与随机源在内的完整根后状态。每臂从同一快照独立恢复。根步以后均由 `8da4b465` 续弈至正式终局，原版依自然变化的对局逐拍响应。14 条按官方规则结算的开发后缀、双方命令带、真实原版时钟与现金账存于 `native/`。四图共 {labels['unique_native_calls']:,} 次原版真实公开调用，多候选零深回退 {labels['unique_native_multicandidate_zero_depth']} 次。每图两臂在同一核串行，核 0 和核 1 各处理两图。[采集程序](collect.py) · [独立结算](settle.py) · [冻结拟合程序](fit.py)"]
    immediate=[r for r in rows if roots[r['id']]['immediate_base']]
    assert len(immediate)==1
    r=immediate[0]; p,s=(r['arms'][k] for k in ('parent','student'))
    focal8=[a['horizons']['after_8']['focal_deposits'] for a in (p,s)]
    focal24=[a['horizons']['after_24']['focal_deposits'] for a in (p,s)]
    out.insert(out.index('## 按地形留一的真实二选一训练'),f"唯一有当拍直接回仓机会的是 {r['seed']} 席 {r['seat']} 第 {r['t']} 拍，焦点工携货 {r['focal']['cargo']:.2f}。父令与全骨干令下，这名工人在随后 8 拍分别实际入库 {focal8[0]:.2f}、{focal8[1]:.2f}，24 拍分别入库 {focal24[0]:.2f}、{focal24[1]:.2f}。全骨干令相对父令的终局分差变化为 {r['delta']['margin']:+.2f}。七根中全骨干终差改善 {sum(r['delta']['margin']>0 for r in rows)} 次、恶化 {sum(r['delta']['margin']<0 for r in rows)} 次。\n")
    calls=[]
    for path in (HERE/'native').rglob('*.json.gz'):
        if path.name in ('shared.json.gz','parent.json.gz','student.json.gz'):
            with gzip.open(path,'rt') as stream:calls.extend(json.load(stream)['clock']['calls'])
    assert len(calls)==labels['unique_native_calls']
    out+=['',f"这些原版调用合计占用常规公开入口墙钟 {sum(c['public_wall_seconds'] for c in calls):.2f} 秒、进程 CPU {sum(c['process_cpu_seconds'] for c in calls):.2f} 秒，其中 {sum(c['max_depth']==8 for c in calls):,} 次到深度 8，最长公开调用 {max(c['public_wall_seconds'] for c in calls):.3f} 秒。独立核算的首次尝试在第一张图的第二根前由公开哈希拦下。原因是官方裁判把工人和树的编号计数保存在进程全局，嵌套复演另一条支线会改变外层裁判的后续编号。修订后的核算先走完整条历史前缀并保存各根，再逐支重新复演。原版比赛、选根及拟合规则均沿用预登记记录。"]
    (HERE/'README.md').write_text('\n'.join(out).replace('预算；','预算。')+'\n')

if __name__=='__main__':report()
