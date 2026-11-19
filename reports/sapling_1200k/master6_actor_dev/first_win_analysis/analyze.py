"""Verify every signed c5ad game and explain cash turns with the actual same-map return seat."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ACTOR / 'blended_actor_arena'))
import scorecard as account
from shared import read, sha
from tree_win_gate import validate_examiner_configuration

PRIOR = ('actor_partner_dev/native/candidate', 'actor_partner_dev/replication/games/candidate',
         'blended_actor_arena/games/parent')
FRESH = 'first_win_dev/games/candidate'
PACK = 'c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3'
ORIGINAL = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
ENGINE = '57f9a594cce4f2ec9e3573d8d9d26ca1'


def compressed(v):
    return gzip.compress(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),mtime=0)


def save(file, b):
    p = HERE / file
    if p.exists() and p.read_bytes() == b: return
    tmp = p.with_name(p.name+'.'+str(os.getpid())+'.partial')
    tmp.write_bytes(b);os.replace(tmp,p)


def extract_turn(game, name, info):
    s = game['seat']
    frames = game['_frames']
    start,end = info['start'],info['end']
    chosen = frames[max(0,start-10):end]
    events = [f for f in chosen if f['t']>=start]
    deposits = []
    for frame in events:
        for evt in frame['cash_events']:
            kind,side,amount,*detail=evt
            if kind=='own_delivery':
                deposits.append({'t':frame['t'],'seat':side,'amount':amount,'worker':detail[0]})
    # Link the large actual receipts to the same individual in all antecedent states.
    major = {}
    by = sorted(deposits,key=lambda v:(-v['amount'],v['t'],v['worker']))
    for side in (s,1-s):
        if not (receipts:=[r for r in by if r['seat']==side]): continue
        receipt=receipts[0];uid=receipt['worker'];trace=[]
        for k in sorted({max(0,start-n) for n in (0,3,5,10)}|{receipt['t']}):
            f=frames[k];p=f['pre_players'][side]
            trace.append({'t':k,'was_active':uid in p[2],'worker_before':p[2].get(uid),
                          'actual_command':f['joint'][side].get(uid,'NONE') if uid in p[2] else None})
        major[str(side)]={**receipt,'antecedent':trace}
    return {'name':name,'start':start,'end':end,
            'lead_at_start':round(game['cash_lead'][start],2),'lead_at_end':round(game['cash_lead'][end],2),
            'difference':round(game['cash_lead'][end]-game['cash_lead'][start],2),
            'own_minus_opponent_money':account.split_delta({'me':game},start,end)['me'],
            'largest_credited_deposit_each_physical_seat':major,'antecedent_observation_times':sorted({max(0,start-n) for n in (3,5,10)}),
            'frames':chosen}


def one(folder, rec_file):
    signed=read(rec_file);row=signed['game'];s=row['candidate_seat'];seed=row['seed']
    binding=signed['binding']; assert (binding.get('original_sha256') or binding.get('original'))==ORIGINAL
    tape=read(folder/row['raw_file']);assert tape['examiner_md5']==ENGINE
    validate_examiner_configuration(tape['configuration'])
    assert row['valid'] and not row['errors'] and not row['audit_messages']
    assert [p[0] for p in tape['search_calls']]==list(range(row['actual_transitions']))
    calls=Counter(x[1] for x in tape['search_calls']);assert calls==row['search_return_types'] and calls['searched']>0
    g=account.replay(folder,seed,s,'candidate',PACK)
    chosen={k:extract_turn(g,k,info) for k,info in g['cash_turns']['10'].items()}
    leads=g['cash_lead']
    if g['result']=='win' and any(x<=0 for x in leads):
        final_cross=1+max(i for i,v in enumerate(leads) if v<=0)
        if final_cross<len(leads):
            chosen['held_positive_until_cutoff']=extract_turn(g,'held_positive_until_cutoff',{'start':max(0,final_cross-1),'end':final_cross})
    game = {k:g[k] for k in ('seed','seat','map','raw','raw_sha256','pack_sha256','official_scores','official_margin','turns',
                               'result','extinct_by_physical_seat','accounts_physical')}
    game.update(receipt_sha256=sha(rec_file),source=str(folder.relative_to(ACTOR)),windows=chosen,
                search_calls=dict(calls),search_depth=row['search'].get('depth'),examiner_changed=row['search'].get('changed'),
                examiner_aborted=row['search'].get('aborted'),physical_fog_view_in_windows=True)
    return game


def panel(games):
    def summ(g):
        return {'n':len(g),'wdl':[sum(r['result']==k for r in g) for k in ('win','draw','loss')],
                'mean_official_margin':round(mean(r['official_margin'] for r in g),3) if g else None}
    return {'all':summ(games),'by_physical_seat':{s:summ([g for g in games if g['seat']==s]) for s in (0,1)},
            'terrains':len(set(g['map'] for g in games))}


def money(a, side, name):
    return a['accounts_physical'][side]['cash'].get(name,0.)


def markdown(data):
    old=data['historical']['all'];new=data['registered_new']['all']
    lines=['# 同一 Sapling 候选对原版 Master3s2 的胜局与换席复盘','',
        '使用冻结候选 `c5ad3f43` 和原版 `65f268b2`。每条已签动作带均经官方解释器复算终分，再用独立现金流水逐笔核对树息、回款、抢货及招工开支。按原候选实际物理席记账。','',
        f'前三批含 {old["n"]} 场、{old["wdl"][0]} 胜 {old["wdl"][1]} 平 {old["wdl"][2]} 负，平均官方分差 {old["mean_official_margin"]:+,.2f}。新注册 8 图双席，当前已有 {new["n"]}/16 场通过核验，{new["wdl"][0]} 胜 {new["wdl"][1]} 平 {new["wdl"][2]} 负。开发图胜场和分差均单列，正式 100 图计数为零。','',
        '| 来源 | 地图 | 席 | 官方分差 | 原版实际搜索 / 决策 | 回库存入差 | 树吸收差 |','|---|---|---:|---:|---:|---:|---:|']
    games=data['games'];pairs={(g['map'],g['seat']):g for g in games}
    for g in games:
        s=g['seat'];de=money(g,s,'own_delivery')-money(g,1-s,'own_delivery');tr=money(g,s,'tree_absorption')-money(g,1-s,'tree_absorption')
        lines.append(f'| {"新开发" if g["source"]==FRESH else "历史"} | {g["seed"]} | {s} | {g["official_margin"]:+,.2f} | {g["search_calls"].get("searched",0)} / {g["turns"]} | {de:+,.2f} | {tr:+,.2f} |')
    lines += ['','## 胜局和同图换席','']
    for g in games:
        if g['result']!='win':continue
        s=g['seat'];opp=pairs.get((g['map'],1-s))
        suffix=(f'换席后分差 {opp["official_margin"]:+,.2f}。' if opp else '同图换席尚待签收。')
        lines.append(f'**地图 {g["seed"]}，席 {s}。** 终分 {g["official_scores"][s]:,.2f} 对 {g["official_scores"][1-s]:,.2f}，分差 {g["official_margin"]:+,.2f}。'+suffix)
        lines.append('')
        for k,title in ((g,'胜局'),(opp,'换席局')):
            if not k:continue
            for axis in ('gain','loss'):
                w=k['windows'].get(axis)
                if not w:continue
                ds=w['largest_credited_deposit_each_physical_seat']; strongest=sorted(w['own_minus_opponent_money'].items(),key=lambda e:-abs(e[1]))[:2]
                facts='、'.join(n+' '+format(float(v),'+,.2f') for n,v in strongest)
                actions=[]
                for side,dep in ds.items():
                    trace=' → '.join('t'+str(l['t'])+' '+(str(l['worker_before'][0])+'格/'+str(l['worker_before'][1])+'货/'+str(l['actual_command']) if l['was_active'] else '未出场') for l in dep['antecedent'])
                    actions.append(f'席 {side} 工 {dep["worker"]} 在 t{dep["t"]} 到账 {dep["amount"]:,.2f}，先前为 {trace}')
                lines+= [f'- {title}最大的十步{"正摆动" if axis=="gain" else "负摆动"}在 t{w["start"]}–{w["end"]}，现金差变化 {w["difference"]:+,.2f}，其中 {facts}。'+'；'.join(actions)+'。']
            lines.append('')
    lines+=['回溯窗口以事后最大十步现金摆动确定。两席是两场真实独立响应的对局，早期命令、对手出令和搜索墙钟均可不同。流水能定位哪笔钱在何时到账，单场观察不能证明替代动作的胜率。每个窗口保存起点前 3、5、10 步以及连续实际双方命令、可见碳、现金、位置、货量和冲突。[逐局轨迹与身份](evidence.json.gz) · [核验程序](analyze.py)']
    return '\n'.join(lines)+'\n'


def run(follow):
    plan=read(ACTOR/'first_win_dev/protocol.json')
    assert plan['candidate']['sha256']==sha(ACTOR/'actor_partner_dev/candidate.py')==PACK
    assert plan['examiner']['sha256']==ORIGINAL
    assert plan['planned_games']==16 and len(plan['seeds'])==8 and plan['seats']==[0,1]
    expected={(s, k) for s in plan['seeds'] for k in plan['seats']}
    by_path={}
    last_count=-1
    while True:
        for folder in PRIOR+(FRESH,):
            folder_path=ACTOR/folder
            receipts=sorted(folder_path.glob('receipt_*.json'))
            if folder in PRIOR: assert len(receipts)==4
            for receipt in receipts:
                if receipt not in by_path:
                    g=one(folder_path,receipt)
                    if folder==FRESH:assert(g['seed'],g['seat']) in expected
                    by_path[receipt]=g
                    print(folder,g['seed'],g['seat'],g['result'],g['official_margin'],flush=True)
        allgames=sorted(by_path.values(), key=lambda g:((PRIOR+(FRESH,)).index(g['source']),g['seed'],g['seat']))
        assert len({(g['map'],g['seat']) for g in allgames})==len(allgames)==len(set(g['raw_sha256'] for g in allgames))
        old=[g for g in allgames if g['source']!=FRESH];new=[g for g in allgames if g['source']==FRESH]
        assert len(old)==12 and len({g['map'] for g in old})==6
        if len(allgames)!=last_count:
            data={'candidate':PACK,'original':ORIGINAL,'analysis_sha256':sha(Path(__file__)),
                  'ledger_sha256':sha(Path(account.__file__)), 'historical':panel(old),'registered_new':panel(new),
                  'planned_new_games':16,'new_complete':len(new)==16,'formal_games':0,'games':allgames}
            save('evidence.json.gz',compressed(data));save('README.md',markdown(data).encode());last_count=len(allgames)
        if not follow or len(new)==16:return
        time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--follow',action='store_true')
    run(parser.parse_args().follow)
