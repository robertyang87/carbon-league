"""Publish every verified game's turn evidence and map/seat/strength-level denominators."""
import argparse
import csv
import gzip
import io
import json
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
from defense_run import HERE,VERSION,atomic,compact

METRICS=('ticks','recruits','collision_deaths','mixed_deaths','same_side_deaths','collector_deaths','planter_deaths','deaths_empty_at_enemy_base',
         'mixed_collision_squares','positive_victims_in_mixed_collision','loaded_decisions','loaded_stay_under_threat_with_safe_move','chose_geometrically_safe_move',
         'acquisitions_all_methods','collector_seizure_lots','collector_lots_positive_cash','acquired_collector','collector_stolen_banked','collector_stolen_tax_paid_to_enemy_trees',
         'collector_stolen_carried_at_terminal','collector_stolen_opponent_worker','collector_stolen_opponent_cash','collector_stolen_ground_or_destroyed',
         'acquired_tree','acquired_base','mixed_cargo_left_on_ground','mixed_cargo_destroyed_by_cell_cap','enemy_base_refund_at_contact',
         'all_enemy_base_donations','enemy_trees_seized','trees_lost_to_enemy','games_with_collector_take','games_with_collector_take_banked')


def ranking_lookup():
    provider=json.loads(gzip.decompress((HERE.parent/'latest5_inventory_sources.json.gz').read_bytes()))
    table=provider['leaderboard'];records={}
    for v in table.get('strategies',[])+table.get('extra',[]):
        if v.get('board')=='heavy' and isinstance(v.get('board_rank'),int):records[v['id']]=v
    names=defaultdict(list)
    for v in records.values():names[v['name']].append(v)
    def lookup(ident,label):
        v=records.get(ident)
        if v is None and len(names[label])==1:v=names[label][0]
        if v is None:return 'unknown'
        rank=v['board_rank']
        return '1–5' if rank<=5 else '6–20' if rank<=20 else '21–50' if rank<=50 else '51+'
    return lookup,provider['as_of_board_utc']


def aggregate(man,root):
    top={p['id']:p['name'] for p in man['top5']}
    separate=defaultdict(lambda:defaultdict(Counter));rows=[];pool=[];cases=defaultdict(list)
    zero_captures=0;zero_banked=0
    artifacts=[v for v in man['games'] if v['frame_verified']]
    tier,asof=ranking_lookup()
    for g in artifacts:
        packet=json.loads(gzip.decompress((root/'defense'/(g['id']+'.json.gz')).read_bytes()))
        if (packet['sha256']!=g['sha256'] or packet['algorithm']!=VERSION or packet['id']!=g['id']
            or packet['side_names']!=g['seat_names'] or packet['score']!=g['score_by_seat']):
            raise ValueError(f"receipt identity changed: {g['id']}")
        packet['cohort']=g['cohort']
        pool.append(compact(packet))
        zero_captures+=all(z.get('acquired_collector',0)==0 for z in packet['seat_totals'])
        zero_banked+=all(z.get('collector_stolen_banked',0)==0 for z in packet['seat_totals'])
        for s in (0,1):
            z=Counter(packet['seat_totals'][s]);lots=[e for e in packet['lots'] if e['beneficiary']==s and e['method']=='collector']
            z.update(collector_seizure_lots=len(lots),collector_lots_positive_cash=sum(e['fate'].get('banked',0)>0 for e in lots),
                     games_with_collector_take=bool(lots),games_with_collector_take_banked=any(e['fate'].get('banked',0)>0 for e in lots))
            score,gid=g['score_by_seat'],g['id']
            result='win' if score[s]>score[1-s] else 'loss' if score[s]<score[1-s] else 'draw'
            opp_tier=tier(g['seat_strategy_ids'][1-s],g['seat_names'][1-s]);player,opp=g['seat_names'][s],g['seat_names'][1-s]
            focal=g['seat_strategy_ids'][s] in top or player in top.values()
            if focal and g['cohort']=='mutual':opp_tier='1–5'
            z['appearances']=1
            for group,key in [('all','both seats'),('cohort_all_sides',g['cohort'])]:
                separate[group][key].update(z)
            if focal:
                for group,key in [('top5','all'),('player',player),('top5_outcome',result),('top5_seat',str(s)),
                                  ('top5_opponent_rank_tier',opp_tier),('top5_opponent',opp),
                                  ('player_by_outcome',player+' / '+result),('player_by_seat',player+' / '+str(s))]:
                    separate[group][key].update(z)
            rows.append([gid,s,player,opp,opp_tier,result,'1' if focal else '0',g['cohort'],g['sha256'],*[round(z[k],5) for k in METRICS]])
        for index,e in enumerate(packet['lots']):
            if e['method']!='collector':continue
            row={'id':g['id'],'s':e['beneficiary'],'player':g['seat_names'][e['beneficiary']],'opp':g['seat_names'][1-e['beneficiary']],
                 'cohort':g['cohort'],'t':e['t'],'victim':e['victim'],'hunter':e['hunter'],'pos':e['pos'],'mass':e['mass'],'fate':e['fate'],'last':e['last']}
            for label,score in [('banked',e['fate'].get('banked',0)),('lost_to_tax',e['fate'].get('tax_paid_to_enemy_trees',0)),
                                ('recaptured',e['fate'].get('opponent_worker',0)),('never_banked',e['mass'] if e['fate'].get('banked',0)==0 else 0)]:
                if score>0:cases[label].append((score,row))
        for t,s,v,w,p,m in packet['enemy_base_refunds']:
            cases['refunded'].append((m,{'id':g['id'],'s':s,'player':g['seat_names'][s],'opp':g['seat_names'][1-s],
                                         'cohort':g['cohort'],'t':t,'victim':v,'hunter':w,'pos':p,'mass':m}))
        for m,t,s,v,p,q in packet['top_safe_escapes']:
            cases['escape'].append((m,{'id':g['id'],'s':s,'player':g['seat_names'][s],'opp':g['seat_names'][1-s],
                                       'cohort':g['cohort'],'t':t+1,'protected_worker':v,'origin':p,'dest':q,'mass':m}))
    files={}
    dst=HERE/'defense_traces.jsonl.gz';b=gzip.compress(b'\n'.join(pool)+b'\n',compresslevel=6,mtime=0);atomic(dst,b);files[dst.name]=sha256(b).hexdigest()
    text=io.StringIO();tsv=csv.writer(text,delimiter='\t',lineterminator='\n')
    tsv.writerow(['id','seat','player','opponent','opponent_heavy_rank_tier','result','top5','cohort','raw_sha256',*METRICS]);tsv.writerows(rows)
    dst=HERE/'defense_games.tsv';b=text.getvalue().encode();atomic(dst,b);files[dst.name]=sha256(b).hexdigest()
    picked={}
    lookup={g['id']:g for g in artifacts}
    for label,raw in cases.items():
        seen=set();chosen=[]
        for _,r in sorted(raw,key=lambda z:-z[0]):
            if r['id'] in seen:continue
            seen.add(r['id']);chosen.append(r)
            if len(chosen)==3:break
        for r in chosen:
            g=lookup[r['id']]
            game=json.loads(gzip.decompress((root/g['path']).read_bytes()));frames=game['steps'];s=r['s'];trace=[]
            for dt in (-10,-5,-3,-2,-1,0,1,3,5):
                t=r['t']+dt
                if t<0 or t>=len(frames):continue
                ps=frames[t][0]['observation']['players'];act=[z['action'] or {} for z in frames[t+1]] if t+1<len(frames) else [{},{}]
                if 'protected_worker' in r:
                    uid=r['protected_worker'];victim=ps[s][2].get(uid);foes=ps[1-s][2]
                    candidates=[(DISTANCE(victim[0],u[0]),a,u) for a,u in foes.items() if victim and u[2]=='COLLECTOR' and u[1]<=victim[1]]
                    nearest=min(candidates,default=None)
                    trace.append({'relative':dt,'protected':victim,'move':act[s].get(uid),'nearest_light_enemy':nearest})
                else:
                    h,v=r['hunter'],r['victim']
                    trace.append({'relative':dt,'hunter':ps[s][2].get(h),'victim':ps[1-s][2].get(v),
                                  'hunter_next':act[s].get(h),'victim_next':act[1-s].get(v),'cash':[a[0] for a in ps]})
            r['trace']=trace
        picked[label]=chosen
    return {'corpus':len(man['games']),'analyzed':len(artifacts),'source_revision':man['revision'],'rank_time':asof,
            'mismatches_or_score_only':[g['id'] for g in man['games'] if not g['frame_verified']],
            'zero_collector_take_both_seats':zero_captures,'zero_banked_stolen_both_seats':zero_banked,'by':separate,'cases':picked,'artifacts':files,'algorithm_sha256':VERSION,
            'crosscheck':json.loads((HERE/'defense_crosscheck.json').read_bytes())}


def DISTANCE(p,q):
    x,y=abs(p%15-q%15),abs(p//15-q//15)
    return min(x,15-x)+min(y,15-y)


def write_summary(r):
    total=r['by']['all']['both seats'];games=r['analyzed'];n=r['by']['top5']['all']
    lines=['# 全库头部对局的防守、截获与兑现','',
           f"在 {games:,} 场有回放的比赛中逐步还原两席，对应 {2*games:,} 份逐局逐席记录。采集工实际从对手取得 {total['collector_seizure_lots']:,} 笔、{total['acquired_collector']:,.2f} 碳，其中 {total['collector_lots_positive_cash']:,} 笔至少部分入账，实收现金 {total['collector_stolen_banked']:,.2f}。另有 {total['enemy_base_refund_at_contact']:,.2f} 碳虽然争到碰撞胜者手里，却在同一步交回受害者基地，不计入实得。",'',
           '两帧公共单位与记录的共同指令足以重建招工、移动、撞击、押货胜者、基地去向和树易主。每步均检查下一帧所有幸存者身份、位置和类型；战利品在后续遭敌树吸收时按工人当步受税率分摊。仅在审计碰撞地面容量时读回真实地面存量。26 场另以完整引擎检查，每笔取得与最终去向逐项核合。跨阵营接手开始新一笔，同一份货被多次抢回会进入多次周转统计。','',
           '重载为持货≥50。若原地可能被较轻或等重的敌工触及，且存在非敌基地、没有这种敌工一步可达的格，则记一次可避威胁。安全格只就这一类对手成立，未涵盖队友占位、敌树税或现金时限。以图为单位的逐笔货单、死亡数、零事件和源回放身份都保存在文末数据文件。','',
           '| 视角 | 逐席样本 | 对手货实得 | 终局前存入 | 敌树抽成 | 被敌工再夺 | 混战死亡 | 在可避威胁中选择安全格 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for typ in ('cohort_all_sides','player','top5_outcome','top5_seat','top5_opponent_rank_tier'):
        for lab,z in sorted(r['by'][typ].items()):
            title={'cohort_all_sides':'双方样本','player':'选手','top5_outcome':'头部结果','top5_seat':'头部席','top5_opponent_rank_tier':'头部对手重榜名次'}[typ]
            lines.append(f"| {title} {lab} | {z['appearances']:,} | {z['acquired_collector']:,.1f} | {z['collector_stolen_banked']:,.1f} | {z['collector_stolen_tax_paid_to_enemy_trees']:,.1f} | {z['collector_stolen_opponent_worker']:,.1f} | {z['mixed_deaths']:,} | {z['chose_geometrically_safe_move']:,}/{z['loaded_stay_under_threat_with_safe_move']:,} |")
    lines+=['','## 每局能兑现多少','',
            f"两席均无实际截获的比赛有 {r['zero_collector_take_both_seats']} 场，两席均未将战利品部分或全部变现的有 {r['zero_banked_stolen_both_seats']} 场。按逐席共 {total['appearances']-total['games_with_collector_take']:,} 份零截获记录。已取到的货中，终局仍携 {total['collector_stolen_carried_at_terminal']:,.2f}，途中缴给敌树 {total['collector_stolen_tax_paid_to_enemy_trees']:,.2f}，被对手工人夺去 {total['collector_stolen_opponent_worker']:,.2f}，进入对手现金 {total['collector_stolen_opponent_cash']:,.2f}。留在工人身上的碳没有进入上述现金收入。",'',
            '| 选手 | 每座位局取得敌货 | 每座位局进己方现金 | 货量兑现率 | 重载可避时走安全格 |','|---|---:|---:|---:|---:|']
    for lab,z in sorted(r['by']['player'].items()):
        lines.append(f"| {lab} | {z['acquired_collector']/z['appearances']:.2f} | {z['collector_stolen_banked']/z['appearances']:.2f} | {z['collector_stolen_banked']/z['acquired_collector']:.1%} | {z['chose_geometrically_safe_move']/z['loaded_stay_under_threat_with_safe_move']:.1%} |")
    lines+=['','对手名次匹配固定时刻重榜的策略 ID；缺 ID 时只接受唯一名称匹配。unknown 含未能作唯一重榜映射的历史策略，不将它们推断为弱者。胜负组是事后分组，表内为行为总量，不能把强手战胜弱手所积累的收入直接解释成某一个动作的因果效果。','',
            '## 关键实例','']
    labels={'banked':'最高落袋','lost_to_tax':'途中被敌树抽走最多','recaptured':'途中被对手抢回最多','never_banked':'抢到后最终未存入','refunded':'碰撞胜出却在受害方基地退款','escape':'避开重货威胁'}
    for code,rows in r['cases'].items():
        z=rows[0];f=z.get('fate',{})
        lines+=[f"- {labels[code]}。{z['id'][:8]} 第 {z['t']} 帧，席 {z['s']} 的 {z['player']} 对 {z['opp']}，相关货量 {z['mass']:,.2f}。"+(f"实收入账 {f.get('banked',0):,.2f}，敌树抽成 {f.get('tax_paid_to_enemy_trees',0):,.2f}，再次转给敌工 {f.get('opponent_worker',0):,.2f}。" if f else '')]
    lines+=['','上述各类最多选三张不同地图，逐例在 JSON 中附接触前 10、5、3、2、1 步和之后的双方位置、货量及行动。生存期间重载工选择几何安全格不能证明无此动作必败。双方最终输赢见逐席表。','',
            '[逐局逐席 TSV](defense_games.tsv) · [全量逐局逐笔，含零事件](defense_traces.jsonl.gz) · [分组、每名对手及案例轨迹](defense_summary.json) · [完整引擎交叉核验](defense_crosscheck.json)','']
    atomic(HERE/'defense.md','\n'.join(lines).encode())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    man=json.loads((HERE/'manifest.json').read_bytes())
    if man['state'] not in ('complete','partial','validating_archives'):raise SystemExit('source fetch still active')
    r=aggregate(man,a.cache)
    atomic(HERE/'defense_summary.json',compact(r));write_summary(r)
    print('published:',r['analyzed'],'/',r['corpus'],r['artifacts'])
