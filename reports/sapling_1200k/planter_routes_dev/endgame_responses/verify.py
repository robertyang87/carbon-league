"""Independently verify the realized branches, their funding and actual deposits."""
from collections import Counter
import gzip,json,statistics,sys
from pathlib import Path
from select_roots import HERE,BASE,ROOT,B,tape,read,write
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'root_recovery_branches'))
from all_top5.economy_extract import CashEvents,FastCarbon,KEYS,POS
from branches import begin,clean,digest,public,utility
import live

def money(p):return [round(float(v),2) for v in p]
def track(r,s):
    return None if r is None else dict(cell=r[0],cargo=round(r[1],3),role=r[2])
def ledger(raw,root,tail):
    env=begin(raw);sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    s=raw['candidate_seat'];t=root['t'];side=1-s;opuid=root.get('enemy_worker')
    for cmd in raw['actions'][:t]:
        env.step(cmd);sim.step(*cmd)
    assert env.states[0].observation.step==t and list(sim.cash)==[p[0] for p in env.states[0].observation.players]
    start=list(sim.cash);sim.cash=CashEvents(sim.cash);gross=[Counter(),Counter()];trajectory=[];receipts=[];seen=[]
    for j,a in enumerate(tail,t):
        if env.done:raise ValueError('extra actions')
        pre=env.states[0].observation.players
        for me in (0,1):
            if set(a[me])-set(pre[me][1])-set(pre[me][2]):raise ValueError('ghost worker on native continuation')
        sim.cash.events.clear();sim.step(*a);env.step(a);post=env.states[0].observation.players
        if money(sim.cash)!=[p[0] for p in post]:raise ValueError('cash-state mismatch')
        for label,seat,v,*rest in sim.cash.events:
            gross[seat][KEYS[POS[label]]]+=v
            if label in ('own_delivery','opponent_worker_delivery_to_this_base'):
                receipts.append(dict(t=j+1,seat=seat,worker=rest[0],amount=round(v,4),kind=label))
        trajectory.append(dict(at=j+1,cash=money(sim.cash),margin=round(sim.cash[s]-sim.cash[side],2),our_target=track(post[s][2].get(root['worker']),s),
            enemy_target=track(post[side][2].get(opuid),side) if opuid else None,
            focal_order=a[s].get(root['worker'],'NONE') if root['worker'] in pre[s][2] else None,
            enemy_target_order=a[side].get(opuid,'NONE') if opuid and opuid in pre[side][2] else None))
        seen.append(a[side])
    if not env.done or [x.status for x in env.states]!=['DONE','DONE']:raise ValueError('truncated')
    scores=[x.reward for x in env.states];margin=round(scores[s]-scores[side],2)
    for i in (0,1):
        if abs(float(sim.cash[i])-start[i]-sum(gross[i].values()))>.011:raise ValueError('funding not closed')
    return dict(score_by_physical_seat=scores,margin=margin,u=utility(scores,s),wdl='W' if margin>0 else 'L' if margin<0 else 'D',
        cash_at_root=start,actual_cash_at_end=money(sim.cash),gross=[{k:round(p[k],4) for k in KEYS} for p in gross],
        deposits=receipts,path=trajectory,enemy_command_digests=[digest(json.dumps(a,sort_keys=True).encode()) for a in seen],
        focal_own_deposits=round(sum(r['amount'] for r in receipts if r['seat']==s and r['worker']==root['worker']),3),
        focal_enemy_deposits=round(sum(r['amount'] for r in receipts if r['seat']==side and r['worker']==opuid),3) if opuid else None)
def main():
    doc=read(HERE/'roots.json');plan=read(HERE/'protocol.json');raw=json.loads(gzip.decompress(tape.read_bytes()))
    assert digest(tape.read_bytes())==doc['raw_sha256']; results={}
    for k,r in doc['selected'].items():
        hist=ledger(raw,r,raw['actions'][r['t']:])
        assert hist['score_by_physical_seat']==raw['official_score_by_seat']; pairs=[]
        for i in range(4):
            where=HERE/'arms'/f'{k}_{i}';arms={};rand=read(where/'live_shared_init.json')
            for arm in ('parent','alternative'):
                outcome=read(where/f'live_{arm}.json');actfile=where/outcome['action_tape'];assert digest(actfile.read_bytes())==outcome['action_sha256']
                assert outcome['pack_sha256']==plan['candidate_sha256']==raw['pack_sha256'] and outcome['source_sha256']==live.MASTER_SHA
                assert outcome['shared_init_sha256']==digest((where/'live_shared_init.json').read_bytes()) and outcome['protocol_sha256']==digest((where/'live_protocol.json').read_bytes())
                acts=json.load(gzip.open(actfile));data=ledger(raw,r,acts)
                assert data['score_by_physical_seat']==outcome['score'] and outcome['search_calls']==len(acts)-1
                d1=clean(acts[0][1]);histact=clean(raw['actions'][r['t']][1]);diff={u for u in histact.keys()|d1.keys() if histact.get(u)!=d1.get(u)}
                assert clean(acts[0][0])==clean(raw['actions'][r['t']][0]) and diff==({r['worker']} if arm=='alternative' else set())
                if arm=='alternative':assert d1.get(r['worker'],'NONE')==r['replacement']
                data.update(search_calls=outcome['search_calls'],completed_search=outcome['search_completed'],single=outcome['search_single'],aborted=outcome['search_aborted'])
                arms[arm]=data
            old,new=arms['parent'],arms['alternative'];delta_by_source={key:round((new['gross'][1][key]-new['gross'][0][key])-(old['gross'][1][key]-old['gross'][0][key]),3) for key in KEYS}
            d=round(new['margin']-old['margin'],2)
            if abs(sum(delta_by_source.values())-d)>.02:raise ValueError('margin decomposition')
            diverge=[j for j,(a,b) in enumerate(zip(old['enemy_command_digests'],new['enemy_command_digests']),r['t']) if a!=b]
            pairs.append(dict(replicate=i,stream_sha256=digest(json.dumps(rand['mt_keys']).encode()),arms=arms,delta_margin=d,delta_u=new['u']-old['u'],delta_by_source=delta_by_source,
                              opponent_joint_order_different_at=diverge))
        assert len(set(x['stream_sha256'] for x in pairs))==4
        results[k]=dict(root=r,historical=hist,pairs=pairs,deltas=[p['delta_margin'] for p in pairs],
                        mean_delta=round(statistics.mean(p['delta_margin'] for p in pairs),3),mean_delta_u=round(statistics.mean(p['delta_u'] for p in pairs),6))
    write(HERE/'verification.json',results)
    heads=['# 距离终局 7–9 步，提前回款或拦路的真实响应','',
       '开发图 621456920，候选 311631ce 的席 1 原局在 t292 领先 451.50，最终落后 13.26。下面两步由各步实际公开观测、已发己方指令、距离、货量和友方占位选出，先锁根再算分支；该局本身因接近翻盘而事后入选。母局没有保存考官随机状态，两臂在每对中从相同的新随机流续弈。','',
       '| 决策 | 新采样 | 原令分差 | 改令分差 | 变化 | 胜负变化 | 自方关注工回库 原→改 | 敌方关注工回库 原→改 |','|---|---:|---:|---:|---:|---|---:|---:|']
    for family,v in results.items():
        for p in v['pairs']:
            a,b=[p['arms'][x] for x in ('parent','alternative')];enet='—' if a['focal_enemy_deposits'] is None else f"{a['focal_enemy_deposits']:.1f} → {b['focal_enemy_deposits']:.1f}"
            heads.append(f"| {family} t{v['root']['t']} | {p['replicate']+1} | {a['margin']:+.2f} | {b['margin']:+.2f} | {p['delta_margin']:+.2f} | {a['wdl']} → {b['wdl']} | {a['focal_own_deposits']:.1f} → {b['focal_own_deposits']:.1f} | {enet} |")
    for k,v in results.items():
        ps=v['pairs'];r=v['root'];heads+=['',f"## {'提前回库' if k=='deliver' else '卡对手回款线'}",'',
           f"t{r['t']} 我方 {r['worker']} 在 {r['origin']} 格载 {r['load']}，原令 {r['old']}，改为 {r['replacement']} 到 {r['target_cell']}。当时本方现金 {r['root_cash'][0]:.2f}，对手 {r['root_cash'][1]:.2f}，可推进 {r['remain']} 步。",
           f"四次配对中有 {sum(p['delta_margin']>0 for p in ps)} 次分差改善，{sum(p['delta_margin']<0 for p in ps)} 次变差，平均改变 {v['mean_delta']:+.2f}，平均 ΔU {v['mean_delta_u']:+.4f}。原令重新采样的终局分差跨度 {min(p['arms']['parent']['margin'] for p in ps):+.2f} 至 {max(p['arms']['parent']['margin'] for p in ps):+.2f}。"]
        if k=='deny':heads.append(f"敌工 {r['enemy_worker']} 当时在 {r['enemy_origin']} 格载 {r['enemy_cargo']}，回 Master3s2 自己的基地有两个等短下一格 {r['enemy_shortest_destinations']}。两种选择下均让对手完整搜索。")
        heads.append('从根到终局，改令相对原令的**双方现金差**平均变化：'+ '，'.join(f"{label} {statistics.mean(p['delta_by_source'][key] for p in ps):+.2f}" for key,label in [('bank','常规交付'),('tree','树息'),('plant','种树支出'),('seize','抢树支出'),('recruit','招工支出')])+'。逐步现金、两名工人位置和所有实际交割明细见账本。')
        if k=='deliver':
            focal=lambda a:[(e['t'],round(e['amount'],2)) for e in a['deposits'] if e['worker']==r['worker'] and e['seat']==1]
            old_orders={tuple(focal(p['arms']['parent'])) for p in ps};new_orders={tuple(focal(p['arms']['alternative'])) for p in ps}
            if len(old_orders)==len(new_orders)==1:
                render=lambda z:'、'.join(f"t{t} 入账 {v:.2f}" for t,v in z)
                heads.append(f"四对中该工的交付明细一致。原走法在 {render(next(iter(old_orders)))}；提前一步走，{render(next(iter(new_orders)))}。提早交第一箱后还能跑第二次，额外收益仍不足以翻盘。")
        else:
            actual=ps[0]['arms']['parent']['path'][0]['enemy_target']['cell']
            heads.append(f"锁根时存在两个等短去向。真实同时根指令把敌方货工送到 {actual} 格，我方卡的 {r['target_cell']} 格落空。之后完整重新决策，敌方目标货工在四个拦路分支里全部成功交付。结合两臂现金账，预判时应同时评估另一路的可拦截性，以及为转身补招和争树付出的现金。")
    heads+=['','每条分支从相同真实共同根发出，根步对手沿用同时发出的母局指令。自下一步起，两边按各自新观测重新决策；考官保持原版搜索和温度 1。16 条动作带分别用官方解释器从真实开局重放至 DONE，并与独立现金账逐步对齐。四对采样仍只对应一张事后选出的地图；结论限于这两个局部动作。','',
       '[冻结的方法](protocol.json) · [全部合格根及选中根](roots.json) · [逐步核验和现金来源](verification.json) · [16 条分支从开局逐步完整运行候选的动作复核](actor_recheck.json) · [全历史复核实现](check_actor.py) · [每对两条动作带](arms/) · [冻结候选原局](../candidate/official_replay.json)']
    (HERE/'README.md').write_text('\n'.join(heads)+'\n')
if __name__=='__main__':main()
