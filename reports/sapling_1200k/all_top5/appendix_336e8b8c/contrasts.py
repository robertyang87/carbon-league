"""Study the new Master6 loss and largest same-seat external score swing from frozen tape."""
import gzip
import json
import sys
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
from appendix_cbd6d545.contrasts import assets, cash_window, collector_trace, first_run, load, roster
from collect import BATCHES, REV
import fetch


def write(name,doc):
    (HERE/name).write_text(json.dumps(doc,ensure_ascii=False,indent=1,allow_nan=False)+'\n')


def short(info,seat):
    sc=info['score_by_seat']
    return {'id':info['id'],'seat':seat,'who':info['seat_names'][seat],
            'opponent':info['seat_names'][1-seat],'scores':sc,'margin':round(sc[seat]-sc[1-seat],2)}


def inputs():
    archives={g['id']:(g,TOP/b) for b in BATCHES for g in fetch.load(TOP/b/'manifest.json')['games']}
    batch=fetch.load(HERE/'manifest.json')
    seen=[]
    for n in batch['games']:
        s=next(i for i,name in enumerate(n['seat_strategy_ids']) if name)
        if 'claude-AlphaCarbon-Master6' in n['seat_names']:
            s=n['seat_names'].index('claude-AlphaCarbon-Master6')
        r={'new':short(n,s)}
        earlier=[g for g,_ in archives.values() if set(g['seat_names'])==set(n['seat_names'])
                 and g['card']['created']<n['card']['created']]
        same=[g for g in earlier if g['seat_names']==n['seat_names']]
        r.update(exact_pair_prior_count=len(earlier),same_seats_prior_count=len(same))
        if earlier:
            o=max(same or earlier,key=lambda g:(g['card']['created'],g['id']))
            s2=o['seat_names'].index(n['seat_names'][s])
            r['prior']=short(o,s2)
            r['margin_change']=round(r['new']['margin']-r['prior']['margin'],2)
        seen.append(r)
    archives.update({g['id']:(g,HERE) for g in batch['games']})
    return archives,seen


def detail(entry,archives):
    g,folder=archives[entry['id']]
    tape=next(x for x in load(folder/'turn_cash.json.gz')['games'] if x['id']==g['id'])
    path=fetch.CACHE/g['path']
    if fetch.sha(path.read_bytes())!=tape['replay_sha256'] or tape['replay_sha256']!=g['sha256']:
        raise ValueError('tape identity')
    env=load(path);fs=env['steps']; s=entry['seat'];e=len(fs)-1
    lead=[round(roster(f,s)[0]-roster(f,1-s)[0],2) for f in fs]
    if abs(lead[-1]-entry['margin'])>.011: raise ValueError('official margin')
    final_sign=1 if lead[-1]>0 else -1
    crossing=max(t for t,x in enumerate(lead) if x*final_sign<=0)+1
    def window(left,right):
        return {'t':[left,right],'cash':[cash_window(tape,left,right,k) for k in (0,1)],
                'receipts':[(r[0],k,uid,v) for r in tape['turns'][left:right] for k,uid,v in r[3]]}
    a={**entry,'source_batch':folder.name,'source_sha256':g['sha256'],'final_lead_crossing':crossing,
       'opening':[collector_trace(fs,tape,k) for k in (0,1)],'whole':window(0,e)['cash'],
       'to_crossing':window(max(0,crossing-10),crossing),'final_60':window(max(0,e-60),e),
       'crossing_frames':[{'t':t,'seats':[assets(fs,t,k) for k in (0,1)]} for t in
                          sorted({max(0,crossing-d) for d in (10,5,3,1,0)})],
       'end_states':[[roster(fs[e],k)[0],len(roster(fs[e],k)[2]),len(roster(fs[e],k)[3])] for k in (0,1)]}
    return a,fs,tape


def analyze_pair(x,archives):
    p,fs1,t1=detail(x['new'],archives); q,fs2,t2=detail(x['prior'],archives)
    s=p['seat'];v=q['seat']
    delta=[round((roster(a,s)[0]-roster(a,1-s)[0])-(roster(b,v)[0]-roster(b,1-v)[0]),2) for a,b in zip(fs1,fs2)]
    direction=1 if delta[-1]>0 else -1
    sustained=[first_run([j*direction for j in delta],threshold) for threshold in (100,500,1000)]
    return {'new':p,'previous':q,'difference_final':x['margin_change'],'sustained_margins':dict(zip((100,500,1000),sustained)),
            'separation_100':None if sustained[0] is None else
                 {'t':sustained[0],'by_game_and_physical_seat':[[cash_window(t,max(0,sustained[0]-10),sustained[0],k)
                       for k in (0,1)] for t in (t1,t2)]}}


def report(all_rows,chosen):
    out=['# 同名对手两次交手，Master6 的收入在哪里分开','',
        '每场先找相同两名策略最近一次较早比赛；同物理席优先。表内列全十场，逐步展开本批 Master6 的唯一败局，以及其他同席对外场中相对上次分差变动最大者。时点 t 是动作已结算的帧，记录的命令在 t 到 t+1 执行。所有示例来自不同物理地图。','',
        '| 本批 | 焦点 / 对手 | 本批分差 | 旧场 / 旧场数 | 旧分差 | 两局分差变化 |','|---|---|---:|---|---:|---:|']
    for x in all_rows:
        n=x['new'];q=x.get('prior')
        out.append(f"| {n['id'][:8]} | {n['who']} / {n['opponent']} | {n['margin']:+,.2f} | "+(f"{q['id'][:8]} / {x['exact_pair_prior_count']} | {q['margin']:+,.2f} | {x['margin_change']:+,.2f} |" if q else '首次 / 0 | — | — |'))
    for pair in chosen:
        n=pair['new'];q=pair['previous']; s=n['seat']
        out.extend(['',f"## {n['opponent']} 连续两次在席 {1-s} 对 Master6",'',
              f"两场分差变化 {pair['difference_final']:+,.2f}。沿最终差异方向，连续十帧差异至少 100、500、1,000 分的首次时点依次为 "+'、'.join('未到' if t is None else f't{t}' for t in pair['sustained_margins'].values())+'。对照按真实现金，时点由赛后确定。','',
              '| 对局 / 选手 | 首次回库 | 首工出现→消失 / 自己实收 | 全局回库 / 树收入 | 末60步回库 / 树收入 | 招工及种夺支出 |','|---|---|---|---:|---:|---:|'])
        for g in (n,q):
            for k,op in enumerate(g['opening']):
                c=op['first_collector'];b=op['first_bank'];full=g['whole'][k];late=g['final_60']['cash'][k]
                desc='无' if not c else f"{c['uid']} t{c['t']}→"+('终局' if c['removed_at'] is None else f"t{c['removed_at']}")+f"，{c['cash_delivered']:.2f}"
                bank='无' if not b else f"t{b['t']}，{b['uid']}，{b['amount']:.2f}"
                out.append(f"| {g['id'][:8]} / {g['who'] if k==g['seat'] else g['opponent']} | {bank} | {desc} | {full['bank']:,.2f} / {full['tree']:,.2f} | {late['bank']:,.2f} / {late['tree']:,.2f} | {-sum(full[i] for i in ('recruit','plant','seize')):,.2f} |")
        g=n;w=n['to_crossing'];c=w['cash'];winner=g['who'] if g['margin']>0 else g['opponent']
        before=[round(g['scores'][p]-sum(g['final_60']['cash'][p].values()),2) for p in (0,1)]
        out.extend(['',f"当前局 {winner} 从 t{g['final_lead_crossing']} 起一直现金领先。此前十步双方（席0 / 席1）正常回库 {c[0]['bank']:,.2f} / {c[1]['bank']:,.2f}，树收入 {c[0]['tree']:,.2f} / {c[1]['tree']:,.2f}。末60步起点双方现金 {before[0]:,.2f} / {before[1]:,.2f}。窗口中以下是真实工号和已发命令。",'','| 帧 | 席 | 现金 | 采集 / 种植工 / 树 | 最高载货采集工（坐标，实发动作） |','|---:|---:|---:|---:|---|'])
        for fr in g['crossing_frames']:
            for k,a in enumerate(fr['seats']):
                ws=sorted((w for w in a['workers'] if w['role']=='COLLECTOR'),key=lambda w:(-w['cargo'],w['uid']))
                w=ws[0] if ws else None
                out.append(f"| {fr['t']} | {k} | {a['cash']:,.2f} | {a['collectors']} / {a['planters']} / {a['trees']} | "+(f"{w['uid']} {w['cargo']:,.2f}，{tuple(w['xy'])}，{w['command']} |" if w else '无 |'))
        rece=g['to_crossing']['receipts']
        out.extend(['','这十步每笔真实回库为 '+('、'.join(f't{t} 席{k} {uid} {val:.2f}' for t,k,uid,val in rece) or '零笔')+'。'])
    out.extend(['','工人相撞的在途货不重复计进现金，树税已包括在树收入。上述对照说明两张图已实现的收支和领先时间，未控制地图、对手温度与决策反馈。附表保留所有工人的位置、命令、首工死亡与后续招工、逐笔回库，可回原件按席复核。[逐帧明细](contrasts.json) · [本批所有现金流](turn_cash.json.gz)',''])
    return '\n'.join(out)


def map_candidate():
    v=fetch.load(HERE/'validation.json'); m=fetch.load(HERE/'manifest.json')
    pack=(HERE/'validation_maps.jsonl.gz').read_bytes();plain=gzip.decompress(pack);rows=[json.loads(l) for l in plain.splitlines()]
    ids=sorted(r['id'] for r in rows)
    ids_sha=fetch.sha('\n'.join(ids).encode())
    if set(ids)!={g['id'] for g in m['games']} or any(x!=1475 for x in v['union_distinct'].values()):raise ValueError('previous overlap or missing entity')
    for a in m['new_archives']:
        p=fetch.ROOT/a['archive']
        if fetch.sha(p.read_bytes())!=a['sha256']:raise ValueError('wrong archive')
        members={g['archive_location']['member']:g['sha256'] for g in m['games']}
        with tarfile.open(p) as f:
            if sorted(f.getnames())!=sorted(members):raise ValueError('archive membership')
            for k,sha in members.items():
                if fetch.sha(f.extractfile(k).read())!=sha:raise ValueError('archive replay changed')
    return dict(revision=REV,previous_count=1465,union_distinct=v['union_distinct'],
        source_for_map_identity=[str((HERE/'validation_maps.jsonl.gz').relative_to(TOP.parent)), len(ids),fetch.sha(plain),ids_sha],
        packed_sha256=fetch.sha(pack),official_checked_members=len(ids),active_map_identity_untouched=True)


def main():
    maps,comparisons=inputs()
    r=[x for x in comparisons if x['new']['who']=='claude-AlphaCarbon-Master6' and x.get('same_seats_prior_count')]
    los=[x for x in r if x['new']['margin']<0]
    if len(los)!=1:raise ValueError('exactly one new Master6 loss expected')
    others=[x for x in r if x not in los]
    pairs=[analyze_pair(x,maps) for x in (los[0],max(others,key=lambda x:abs(x['margin_change'])))]
    write('contrasts.json',dict(comparisons=comparisons,details=pairs));(HERE/'contrasts.md').write_text(report(comparisons,pairs))
    write('map_registry_candidate.json',map_candidate())


if __name__=='__main__':main()
