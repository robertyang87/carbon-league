"""Replay all eight previously submitted normal-policy attempts and explain their physical cash accounts."""
from collections import Counter
from pathlib import Path
from statistics import mean
import json
import sys
import zipfile

HERE=Path(__file__).resolve().parent
HOSTED=HERE/'hosted'
sys.path.insert(0,str(HERE.parent))
import economy as e

RUN=44625133102
ARMS=('sampled_parent','sampled_child')
SEEDS=(621758450,621758451)
SIGNS=('own_delivery','ground_rent','cargo_tax','tree_collision_cash','base_collision_cash','plant','seize','hire_collector','hire_planter')


def receipt(row,side):
    who=row['pre']['workers'];cmd=row['commands_by_seat'][side]
    return dict(pre_cash=row['pre']['cash'][side],trees=row['pre']['trees_by_seat'][side],
                carried=row['pre']['carried_by_seat'][side],post_cash=row['post_cash'][side],
                commands=cmd,workers={u:q for u,q in who.items() if q[3]==side},
                signed_income=row['signed_cash_by_seat'][side])


def match_pivot(g):
    """If the learner finished ahead in actual cash, choose its last crossing into a lasting lead."""
    rows=g['transitions']; path=g['cash_margin_by_state']; seat=g['seat']
    if g['official_margin'] is None or not rows:return None
    if g['official_margin']>0 and path[-1]>0:
        start=next((t for t in range(len(path)-2,-1,-1) if path[t]<=0),0)
        label='进入保持到终局的现金领先'
    else:
        start=e.swing_windows(path)['loss_1'][0] if len(path)>1 else 0
        label='本方现金分差单拍跌幅最大'
    q=rows[start]
    opponent=1-seat
    deliveries=[(side,*detail[:2],amount) for kind,side,amount,*detail in q['cash_events'] if kind=='own_delivery']
    back={}
    for n in (3,5,10):
        pos=max(0,start-n)
        back[n]=dict(state=pos,physical_states=[receipt(rows[pos],s) for s in (0,1)],
                     flows=e.profile(seat,rows,pos,start))
    return dict(criterion=label,step=start,pre_margin=path[start],post_margin=path[start+1],
                events=q['cash_events'],collisions=q['collisions'],tree_changes=q['tree_changes'],
                deposits=deliveries,own=receipt(q,seat),opponent=receipt(q,opponent),before=back)


def summarize_cases(doc):
    records=doc['games']
    e.require(doc['workflow_run']==RUN and doc['records_reconciled'] and len(records)==8,'independent eight-game settlement missing')
    expected={(s,t,a) for s in SEEDS for t in (0,1) for a in ARMS}
    table={(g['seed'],g['seat'],g['arm']):g for g in records}
    e.require(set(table)==expected and len(doc['archives'])==4,'wrong fixture matrix')
    cfg=e.config();games={}
    for entry in doc['archives']:
        path=HOSTED/f"{entry['id']}.zip"
        e.require(e.sha(path.read_bytes())==entry['sha256'],'ZIP digest mismatch')
        with zipfile.ZipFile(path) as archive:
            for arm in ARMS:
                key=entry['seed'],entry['seat'],arm
                e.require(key in expected and key not in games,'duplicated game file')
                games[key]=e.replay(table[key],archive,cfg)
    pairs=[e.comparison(games[s,t,ARMS[1]],games[s,t,ARMS[0]]) for s in SEEDS for t in (0,1)]
    summary=e.summarize(pairs)
    summary['by_map']={s:e.summarize([x for x in pairs if x['seed']==s])['all'] for s in SEEDS}
    if doc['results']['child_minus_parent_mean_margin'] is not None:
        e.check_equal(summary['all']['mean_over_all_planned'],doc['results']['child_minus_parent_mean_margin'],'pair diff')
    for game in games.values():game['pivotal_command']=match_pivot(game)
    return games,pairs,summary


def complete_averages(games):
    result={}
    for arm in ARMS:
        group=[g for g in games.values() if g['arm']==arm and g['official_margin'] is not None]
        if group:
            result[arm]=dict(n=len(group),cash_margin=mean(g['official_margin'] for g in group),
                accounts={owner:{k:mean(g['signed_account_by_seat'][g['seat'] if owner=='learner' else 1-g['seat']].get(k,0) for g in group) for k in e.KIND}
                          for owner in ('learner','examiner')})
    return result


def write_report(games,pairs,m,means):
    valid=[p for p in pairs if p['numeric']]
    lines=['# 两张新开发图的原版对局收入复盘','',
           '8 个注册尝试包含两张地图、双方换席、同一采样方式的父网和训练网。独立结算器保留全部胜负及平台失格。下表分差均为本方官方终分减 Master3s2 官方终分。每笔收支从两方实际命令独立重放，逐拍对上官方现金。','',
           '| 地图 | 席 | 父网分差 | 训练网分差 | 差值 | 训练网终分 / 原版终分 |','|---|---:|---:|---:|---:|---:|']
    def fmt(x):return '缺官方数值' if x is None else f'{x:+,.2f}'
    for row in pairs:
        s,t=row['seed'],row['seat'];p,c=[games[s,t,a] for a in ARMS];scores=c['official_scores_by_seat']
        ls='—' if scores is None else f'{scores[t]:,.2f} / {scores[1-t]:,.2f}'
        lines.append(f"| {s} | {t} | {fmt(p['official_margin'])} | {fmt(c['official_margin'])} | {fmt(row.get('official_margin_delta'))} | {ls} |")
    lines+=['','[官方逐局结算](settlement.json) · [两席逐拍行动与收入](economy_turns.json.gz)','',
            '## 两臂配对收支','',
            f"相同地图、相同物理席分别配对。{len(valid)}/4 组有完整官方分差。全部预定配对的平均变化为 {fmt(m['all'].get('mean_over_all_planned'))}。"]
    for key,tab in (('地形',m['by_map']),('席位',m['by_learner_seat'])):
        lines+=['',key+'拆分为'+'、'.join(f"{s} 的 {fmt(r.get('mean_over_all_planned'))}" for s,r in tab.items())+'。']
    if valid:
        lines+=['','| 现金来源 | 本方变化 | 原版变化 | 净分差变化 |','|---|---:|---:|---:|']
        for k,n in zip(SIGNS,('普通交仓','树的地面吸碳','树向敌工收取的货税','碰撞直接入树账','碰撞直接入基地账','种植支出','夺树支出','招采集工支出','招种树工支出')):
            lines.append('| '+n+' | '+' | '.join(fmt(m['all'][side][k]) for side in ('learner_finances','opponent_finances','margin_components'))+' |')
        lines+=['','成本以负数入账。正变化表示该项收入增加或支出减少。剩余项目和舍入都保存在逐局账，每个配对的全部来源之和等于官方分差变化。原版在两臂所走轨迹、自然抽样和受墙钟制约的搜索均可变化。']
    lines+=['','## 逐局转折','']
    for row in pairs:
        s,t=row['seed'],row['seat'];g=games[s,t,ARMS[1]];pivot=g['pivotal_command']
        if not pivot:lines.append(f'- {s} 席 {t} 的可观测命令与异常终点留在逐拍附件。');continue
        start=pivot['step'];op=1-t
        texts=[]
        for n in (3,5,10):
            flows=pivot['before'][n]['flows']['by_physical_seat']
            texts.append(f"前 {n} 拍 {flows[t].get('own_delivery',0):,.1f}/{flows[op].get('own_delivery',0):,.1f}")
        deps='、'.join(f"{'本方' if owner==t else '原版'} {uid} 在 {cell} 交仓 {amt:,.2f}" for owner,uid,cell,amt in pivot['deposits'])
        other=[]
        for side,uid,cell,amt in pivot['deposits']:
            history=[]
            for n in (3,5,10):
                state=pivot['before'][n]['physical_states'][side]
                worker=state['workers'].get(uid)
                history.append('尚未出场' if not worker else f"{worker[1]:.1f}货/距基地{worker[4]}格/{state['commands'].get(uid,'NONE')}")
            other.append(f"{uid} 在此前 3/5/10 拍各为 "+'，'.join(history))
        if not deps:
            ev=sorted((v for v in pivot['events'] if v[0]!='rounding'),key=lambda v:abs(v[2]),reverse=True)[:3]
            deps='、'.join(f"{'本方' if x[1]==t else '原版'} {x[0]} {x[2]:+,.2f}" + (f"（{x[3]}）" if len(x)>3 else '') for x in ev)
        phrase='；'.join(other)
        lines.append(f"- **{s} 席 {t}**，{pivot['criterion']}发生于第 {start} 拍，分差由 {fmt(pivot['pre_margin'])} 变为 {fmt(pivot['post_margin'])}。{deps or '此次没有普通交仓'}。按本方/原版列普通回款，"+'，'.join(texts)+'。'+(phrase+'。' if phrase else ''))
    lines+=['','逐局也保存了最大正负单拍、最大正负十拍，以及此前 3、5、10 拍的双方现金来源、实际命令、在途货、碰撞与树权变化。末 30 拍新树投资和实际兑现收入随局归档。成对对局的收入差是已发生的账目；判定某次出令造成防守或封锁效果，还需要同一状态下可执行替代令的续弈对照。[完整窗口与逐局账](economy.json.gz)']
    (HOSTED/'economy.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    source=HOSTED/'settlement.json';doc=json.loads(source.read_bytes())
    games,pairs,summary=summarize_cases(doc);means=complete_averages(games)
    details=[dict(seed=g['seed'],seat=g['seat'],arm=g['arm'],transitions=g.pop('transitions')) for g in games.values()]
    turns=HOSTED/'economy_turns.json.gz';turn_sha=e.write(turns,dict(workflow=RUN,games=details),True)
    data=dict(workflow=RUN,settlement_sha256=e.sha(source.read_bytes()),cash_replayer_sha256=e.sha(Path(e.__file__).read_bytes()),
              analysis_sha256=e.sha(Path(__file__).read_bytes()),turns_sha256=turn_sha,physical_games=list(games.values()),
              paired_comparisons=pairs,summary=summary,absolute_complete_game_means=means,scope='Two fresh development maps, separate stochastic games. Realized cash accounting; action-specific causal effect is not identified.')
    e.write(HOSTED/'economy.json.gz',data,True);write_report(games,pairs,summary,means)
    print(summary['all']['numeric'],summary['all'].get('mean_over_all_planned'),turn_sha)
