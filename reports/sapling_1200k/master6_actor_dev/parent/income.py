"""Reconcile complete scored action tapes to per-turn cash sources and identify each seat's heaviest drawdown."""
from pathlib import Path
import sys,hashlib,gzip,json
from collections import Counter
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'root_recovery_branches'))
from all_top5.economy_extract import FastCarbon,CashEvents,KEYS,POS
from branches import begin
from tree_win_gate import validate_examiner_configuration
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tables(records):
    sig=lambda a,b:[round(sum(row['books'][s][k] for row in records[a:b]),4) for s in (0,1) for k in KEYS]
    def period(a,b):
        d=sig(a,b);return dict(start=a,end=b,physical_seat=[{k:round(v,3) for k,v in zip(KEYS,d[i*len(KEYS):])} for i in (0,1)])
    bounds=[0]+[t for t in [60,150,240,269] if t<len(records)]+[len(records)]
    return [period(a,b) for a,b in zip(bounds,bounds[1:])],period
def report(out):
    run=json.loads((out/'results.json').read_bytes());reg=json.loads((out/'registration.json').read_bytes()) if (out/'registration.json').exists() else json.loads((HERE/'registration.json').read_bytes())
    off=json.loads((out/'official_replay.json').read_bytes())['runs'][out.name]['games']
    assert run['status']=='complete' and len(off)==len(run['games'])==2;games=[];scores=[]
    title={'parent':'单网 cash_v3 父代','hybrid_parent':'固定 75% it50＋25% cash_v3 双网父代','candidate':'轻量更新的 Sapling 单网候选','hybrid_candidate':'使用新版 Sapling 的 75% it50＋25% Sapling 双网候选'}.get(out.name,out.name)
    lines=['# '+title+'对原版 Master3s2 的开发成绩','',f"地图 {reg['seed']}，两边换席。Master3s2 保留全套原生搜索和温度 1。以下两场只覆盖同一张开发图。",'','| 本方席 | 本方分 | 考官分 | 分差 | 回合 | 实搜/搜索入口 | 回库 本方/考官 | 树收入 本方/考官 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    aug=[]
    for r,o in zip(run['games'],off):
        s=r['candidate_seat'];op=1-s;p=out/r['raw_file'];raw=json.load(gzip.open(p));assert sha(p)==r['raw_sha256']==o['transcript_sha256'] and raw['seed']==reg['seed'] and r['valid'] and not r['errors']
        assert raw['initial_terrain_u8_sha256']==r['initial_terrain_u8_sha256']==o['map_sha256']==reg.get('expected_terrain',reg.get('initial_terrain_u8_sha256'))
        validate_examiner_configuration(raw['configuration'])
        env=begin(raw);sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash);events=[];rootcash=list(sim.cash);deposits=[];balances=[rootcash]
        for t,joint in enumerate(raw['actions']):
            sim.cash.events.clear();sim.step(*joint);env.step(joint);balances.append(list(sim.cash));books=[{k:0. for k in KEYS},{k:0. for k in KEYS}]
            for k,pl,x,*detail in sim.cash.events:
                books[pl][KEYS[POS[k]]]+=x
                if k in ('own_delivery','opponent_worker_delivery_to_this_base'):deposits.append([t+1,pl,detail[0],round(x,5),k])
            cash=[z[0] for z in env.states[0].observation.players];assert all(abs(c-v)<.011 for c,v in zip(cash,sim.cash))
            if any(abs(cash[pl]-balances[-2][pl]-sum(books[pl].values()))>.011 for pl in (0,1)):raise ValueError(('unbalanced',t))
            events.append(dict(t=t+1,cash=cash,books=[{k:round(v,6) for k,v in book.items()} for book in books]))
        assert env.done and [x.status for x in env.states]==o['statuses']==['DONE','DONE'] and [x.reward for x in env.states]==o['official_reward_by_seat']==r['official_score_by_seat']
        phases,period=tables(events);tot=period(0,len(events))['physical_seat'];m=[v[s]-v[op] for v in balances]
        w=min(10,len(events));a=min(range(len(events)-w+1),key=lambda i:(m[i+w]-m[i],i))
        worst=dict(**period(a,a+w),margin_start=round(m[a],3),margin_end=round(m[a+w],3),margin_change=round(m[a+w]-m[a],3),prior_cash={str(max(0,a-i)):balances[max(0,a-i)] for i in (10,5,3,0)})
        score=r['official_score_by_seat'];lines.append(f"| {s} | {score[s]:.2f} | {score[op]:.2f} | {r['margin']:+.2f} | {len(events)} | {r['search_return_types'].get('searched',0)}/{r['search_invocations']} | {tot[s]['bank']:.2f}/{tot[op]['bank']:.2f} | {tot[s]['tree']:.2f}/{tot[op]['tree']:.2f} |")
        aug.append(f"席 {s} 相对分差下降最多的 {w} 步为 t{a}–{a+w}，从 {m[a]:+.2f} 变为 {m[a+w]:+.2f}。期间正常回库本方 {worst['physical_seat'][s]['bank']:.2f}、考官 {worst['physical_seat'][op]['bank']:.2f}，树息本方 {worst['physical_seat'][s]['tree']:.2f}、考官 {worst['physical_seat'][op]['tree']:.2f}。")
        games.append(dict(seed=raw['seed'],seat=s,score_by_seat=score,result=r['result'],margin=r['margin'],official_map_sha256=o['map_sha256'],
                  pack_sha256=raw['pack_sha256'],logs_sha256=sha(p),terminal=env.states[0].observation.step,per_game_totals=tot,phases=phases,
                  largest_10_turn_margin_decline=worst,receipts=deposits,events=events))
    mean=sum(r['margin'] for r in run['games'])/2;wdl=Counter(r['result'] for r in run['games'])
    lines+=['',f"共 {wdl['win']} 胜、{wdl['draw']} 平、{wdl['loss']} 负，平均分差 {mean:+.2f}。各自轨迹由同图同时根和考官真实采样形成，单图不能估计跨图胜率的不确定度。",'',*aug,'',
        '每一回合的现金与官方解释器逐席核到分，收入与支出分解另外合计对账。平局与落败均按真实结果统计。',
        '','[逐席赛果与原生搜索调用](results.json) · [官方回放验证](official_replay.json) · [逐回合现金、交付及分段来源](income.json.gz)']
    if out.name=='candidate':
        control=json.loads(gzip.decompress((HERE/'income.json.gz').read_bytes()))['games'];assert len(control)==len(games)
        lines+=['','## 相对同图单网父代','',
            '| 本方席 | 父代分差 | 候选分差 | 分差改变 | 本方入库改变 | 考官入库改变 | 本方树息改变 | 考官树息改变 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for old,new in zip(control,games):
            s=new['seat'];op=1-s;assert old['seed']==new['seed'] and old['seat']==s and old['official_map_sha256']==new['official_map_sha256']
            diff=lambda i,k:new['per_game_totals'][i][k]-old['per_game_totals'][i][k]
            lines.append(f"| {s} | {old['margin']:+.2f} | {new['margin']:+.2f} | {new['margin']-old['margin']:+.2f} | {diff(s,'bank'):+.2f} | {diff(op,'bank'):+.2f} | {diff(s,'tree'):+.2f} | {diff(op,'tree'):+.2f} |")
        lines+=['','两臂使用相同权重规格、int8、style 2、贪心部署包装与相同物理地图。考官为独立真实采样，同图两席的分差改变用于开发诊断。[单网父代逐步实绩](../parent/README.md)']
    blob=(json.dumps({'schema':'single-map-economy-v1','seats_are_physical':True,'games':games},ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    (out/'income.json.gz').write_bytes(gzip.compress(blob,mtime=0));(out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]),flush=True)
if __name__=='__main__':report(Path(sys.argv[1]).resolve())
