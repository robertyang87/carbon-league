"""After all sixteen predetermined development seats finish, trace the closest official loss."""
import gzip
import json
from pathlib import Path
import sys

from analyze import ACTOR, HERE, ROOT, FRESH, PACK, account, compressed, save, read, sha


def distance(a,b,sz):
    ar,ac=divmod(a,sz);br,bc=divmod(b,sz)
    return min(abs(ar-br),sz-abs(ar-br))+min(abs(ac-bc),sz-abs(ac-bc))


def run():
    allgames=read(HERE/'evidence.json.gz')
    assert allgames['new_complete'] and allgames['registered_new']['all']['n']==16
    pool=[g for g in allgames['games'] if g['source']==FRESH and g['result']=='loss']
    rule='Among all 16 signed paired games, largest negative official candidate margin. Ties by lower seed then seat.'
    if not pool:
        save('near_miss.json.gz',compressed({'rule':rule,'observed_losses':0}))
        return
    src=min(pool,key=lambda g:(-g['official_margin'],g['seed'],g['seat']))
    s=src['seat'];g=account.replay(ACTOR/FRESH,src['seed'],s,'candidate',PACK)
    assert g['raw_sha256']==src['raw_sha256'] and g['official_margin']==src['official_margin']<0
    raw=read(ROOT/g['raw']);sim=account.lots.FastCarbon(raw['game_configuration']).reset(seed=g['seed'])
    for pair in raw['actions']:sim.step(*pair)
    final=sim.full_observation()['players'];n=g['turns'];sz=raw['game_configuration']['size']
    assert [p[0] for p in final]==g['final_cash']
    lead=g['cash_lead']
    holds=[]
    for side in (0,1):
        rows=[]
        for uid,(pos,load,kind) in sorted(final[side][2].items()):
            if load<=0:continue
            bases=list(final[side][1].values())
            def one_frame(time):
                f=g['_frames'][time];w=f['pre_players'][side][2].get(uid)
                if w is None:return {'t':time,'present':False}
                return {'t':time,'present':True,'cell':w[0],'cargo':w[1],'kind':w[2],
                        'nearest_base_distance':min(distance(w[0],b,sz) for b in bases),
                        'submitted_action':f['joint'][side].get(uid,'NONE')}
            rows.append({'uid':uid,'kind':kind,'cargo_at_cutoff':load,'cell':pos,
                         'nearest_own_base_distance':min(distance(pos,b,sz) for b in bases),
                         'trajectory':[one_frame(t) for t in range(max(0,n-10),n)]})
        holds.append({'seat':side,'total_unbanked':round(sum(r['cargo_at_cutoff'] for r in rows),8),
                      'workers':rows,'trees_live_at_cutoff':len(final[side][3])})
    periods={}
    for h in (30,10,5,3,1):
        start=max(0,n-h)
        periods[str(h)]={'start':start,'end':n,'cash_lead_before':round(lead[start],2),'cash_lead_after':round(lead[-1],2),
                         'cash_difference_change':round(lead[-1]-lead[start],2),
                         'ledger_own_minus_enemy':account.split_delta({'candidate':g},start,n)['candidate']}
    top=int(max(range(n+1),key=lambda t:lead[t]))
    last_nonnegative=max((i for i,v in enumerate(lead) if v>=0),default=None)
    e={'rule':rule,'observed_losses':len(pool),'source_sha256':sha(Path(__file__)),
       'all_16_verified_sha256':sha(HERE/'evidence.json.gz'),'receipt_sha256':src['receipt_sha256'],
       'raw_sha256':g['raw_sha256'],'seed':g['seed'],'seat':s,'official_margin':g['official_margin'],'game':g['raw'],
       'official_scores':g['official_scores'],'actual_extinctions':g['extinct_by_physical_seat'],'complete_transitions':n,
       'whole_game_accounts_physical':g['accounts_physical'],'largest_ten_step_cash_turns':src['windows'],
       'terminal_inventory':holds,'last_cash_windows':periods,'best_observed_cash_lead':{'t':top,'value':round(lead[top],2)},
       'last_state_with_nonnegative_cash_lead':last_nonnegative}
    save('near_miss.json.gz',compressed(e))
    def row_amount(side,key):
        return g['accounts_physical'][side]['cash'].get(key,0.)
    def diff(key):
        return row_amount(s,key)-row_amount(1-s,key)
    text=['# 新开发场中分差最接近零的败局','',
          f'共复核 16 场，按官方负分差从最接近零排列。本次选中 {g["seed"]} 席 {s}，官方得分 {g["official_scores"][s]:,.2f} 对 {g["official_scores"][1-s]:,.2f}，分差 **{g["official_margin"]:+,.2f}**。比赛实际走了 {n} 步。全场正常回库差 {diff("own_delivery"):+,.2f}，树吸收差 {diff("tree_absorption"):+,.2f}，种树付款差 {diff("plant"):+,.2f}。','',
          f'己方现金领先的最高点在 t{top}，为 {lead[top]:+,.2f}。最后一次现金持平或领先在 t{last_nonnegative}。','',
          '| 截止前步数 | 窗口起点分差 | 到终点现金差变化 | 正常回库差 | 树吸收差 | 种树付款差 |','|---:|---:|---:|---:|---:|---:|']
    for h,v in periods.items():
        a=v['ledger_own_minus_enemy'];text.append(f'| {h} | {v["cash_lead_before"]:+,.2f} | {v["cash_difference_change"]:+,.2f} | {a.get("own_delivery",0):+,.2f} | {a.get("tree_absorption",0):+,.2f} | {a.get("plant",0):+,.2f} |')
    text+=['','**终局库存。**']
    for side in (s,1-s):
        r=holds[side];details='，'.join(f'{v["uid"]} 有 {v["cargo_at_cutoff"]:,.2f}，距最近己库 {v["nearest_own_base_distance"]} 格' for v in sorted(r['workers'],key=lambda r:-r['cargo_at_cutoff']))
        text.append(f'- 席 {side} 的工人留有 {r["total_unbanked"]:,.2f}，仍有 {r["trees_live_at_cutoff"]} 棵树。'+(details+'。' if details else ''))
    text+=['','以上金额是正式截止时的实际库存。每个载货工最后十步及双方最大十步现金转折前的具体命令均已留存，需要真实改令后的对手续局才能判断能否把库存转为胜分。[逐工轨迹及逐项终账](near_miss.json.gz)']
    save('NEAR_MISS.md',('\n'.join(text)+'\n').encode())
    main=HERE/'README.md';output=main.read_text();note='[新开发16场中最接近翻盘的败局及在途货](NEAR_MISS.md)'
    if note not in output:save('README.md',(output+'\n'+note+'\n').encode())
    print(src['seed'],s,src['official_margin'],holds[s]['total_unbanked'],holds[1-s]['total_unbanked'],sha(HERE/'near_miss.json.gz'),flush=True)


if __name__=='__main__':
    run()
