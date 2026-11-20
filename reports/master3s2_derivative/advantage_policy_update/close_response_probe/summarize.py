"""Summarize joint issuance, package-specific menus and worker marginals by recorded source root."""
import gzip
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent

def compare_607(root):
    dev=HERE.parents[1]/'v3_release/development'
    t,s=root['step'],root['seat']
    case=json.loads((dev/f'return_turns/621758607_{s}_{t}_result.json').read_bytes())
    arc=json.loads((dev/'hosted/settlement.json').read_bytes())['archives']
    arc=next(a for a in arc if (a['seed'],a['seat'])==(621758607,s))
    raw=dev/f"hosted/{arc['id']}.zip"
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==arc['sha256']
    with zipfile.ZipFile(raw) as z:
        f=json.loads(gzip.decompress(z.read('challenger/observations.jsonl.gz')).splitlines()[t])
    assert f['public_observation_sha256'][s]==root['input_sha256']
    def dist(pos):
        return min(sum(min(abs(i-j),15-abs(i-j)) for i,j in zip(divmod(pos,15),divmod(h,15))) for h in homes)
    focus=case['focus']
    homes=list(f['observations'][s]['players'][s][1].values())
    pos=focus['own_cell']; y,x=divmod(pos,15)
    holder,op=focus['blocker'],focus['uid']
    report=dict(homes=homes,available=299-t,distance=dist(pos),root_holder=holder,cargo=focus['own_cargo'],enemy=op,arms={})
    for arm,direction in [('parent','base_move'),('block','new_move')]:
        cmd=focus[direction]
        cell=15*((y+{'UP':-1,'DOWN':1}.get(cmd,0))%15)+(x+{'RIGHT':1,'LEFT':-1}.get(cmd,0))%15
        row=case['observed'][arm]; ledger=case['arms'][arm]
        slack=298-t-dist(cell);end=row['timeline'][-1]
        report['arms'][arm]=dict(move=cmd,cell=cell,remaining=298-t,distance_after=dist(cell),
            turns_to_spare=slack,geometrically_reachable=slack>=0,worker_at_terminal=end['blocker'],
            realized_own_holder_banked=row['focused_deposits'].get(holder,0),enemy_actual_root_order=row['root_opponent'].get(op,'NONE'),
            realized_enemy_target_banked=row['focused_deposits'].get(op,0),
            own_team_banked=ledger['terminal'][s]['components']['own_delivery'],
            enemy_team_banked=ledger['terminal'][1-s]['components']['own_delivery'],official_margin=ledger['margin'])
    return report

def main():
    evidence=HERE/'observations.json.gz'
    data=json.loads(gzip.decompress(evidence.read_bytes()))
    allrows=[]
    for r in data['rows']:
        best=r['best']; ds=r['draws'][:-1];rep=r['draws'][-1]; cap={}
        assert len(ds)==8 and rep['key']==ds[0]['key']
        summary={k:r[k] for k in ('seed','seat','step','dataset','step_fit_schedule','continuation','margins','best','changed_units')}
        for name in ('v2','child'):
            runs=[d['arms'][name] for d in ds]
            focus={}
            for u in r['changed_units']:
                cmd=r['full_orders'][best].get(u,'NONE'); ns=r['priors'][name][u]
                focus[u]=dict(role=ns['role'],cargo=ns.get('cargo'),better_branch_direction=cmd,prior=ns['probs'][cmd],go_home=ns.get('home'),
                        menu_hits=sum(v['units'][u]['menu_matches'][best] for v in runs),
                        issued_hits=sum(v['units'][u]['issued']==cmd for v in runs))
            primary=runs[0];other=rep['arms'][name]
            cap[name]=dict(focus=focus,better_full_in_menu=sum(v['menu_matches'][best] for v in runs),
                           better_full_issued=sum(v['issued_matches'][best] for v in runs),
                           either_full_in_menu=sum(any(v['menu_matches']) for v in runs),either_full_issued=sum(any(v['issued_matches']) for v in runs),
                           registered_menu_matches=primary['menu_matches'],registered_issued_matches=primary['issued_matches'],
                           registered_A0=primary['menu'][0],registered_cmd=primary['cmd'],
                           primary_repeats=dict(menu=other['menu']==primary['menu'],issued=other['cmd']==primary['cmd'],
                                                equal_depths=other['depth']==primary['depth']))
        changed=Counter()
        exposed_better_direction_up=0
        for u in r['changed_units']:
            if cap['child']['focus'][u]['prior']>cap['v2']['focus'][u]['prior']:
                exposed_better_direction_up+=1
        for d in ds:
            a,b=[d['arms'][n]['cmd'] for n in ('v2','child')]
            changed.update(u for u in set(a)|set(b) if a.get(u,'NONE')!=b.get(u,'NONE'))
        def paired(p,q):return sum(p(a['arms']['v2'])==q(a['arms']['child']) for a in ds)
        summary.update(arms=cap,better_worker_priors_up=exposed_better_direction_up,changed_issued_by_unit=changed,
            differing_full_issued=len(ds)-paired(lambda a:a['cmd'],lambda a:a['cmd']),
            equal_indexed_depths=paired(lambda a:a['depth'],lambda a:a['depth']),
            equal_total_work=paired(lambda a:(len(a['menu']),sorted(a['depth'].values()),a['clock'].get('rolls')),lambda a:(len(a['menu']),sorted(a['depth'].values()),a['clock'].get('rolls'))))
        allrows.append(summary)
    fixed=HERE/'fixed_work.json.gz'
    control=json.loads(gzip.decompress(fixed.read_bytes()))
    assert control['native_evidence_sha256']==hashlib.sha256(evidence.read_bytes()).hexdigest()
    for q,root,source in zip(allrows,control['rows'],data['rows'],strict=True):
        assert root['report']['registry']==[q['seed'],q['seat'],q['step']]
        q['matched']=root['report']
        if q['seed']==621758607: q['recruitment_or_return_geometry']=compare_607(source)
    out=dict(fixed_work_sha256=hashlib.sha256(fixed.read_bytes()).hexdigest(),evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),distinct_maps=len({r['seed'] for r in allrows}),common_draws_per_root=8,groups=allrows)
    (HERE/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for x in allrows:
        print(x['seed'],x['seat'],x['step'],x['dataset'],x['best'],x['margins'], 'issued-change',x['differing_full_issued'],'equal-work',x['equal_indexed_depths'])
        print('menu/issued',[(a['better_full_in_menu'],a['better_full_issued'],a['either_full_issued'],a['primary_repeats']) for a in x['arms'].values()])
        print([(u,[(round(a['focus'][u]['prior'],4),a['focus'][u]['issued_hits'],a['focus'][u]['go_home']) for a in x['arms'].values()]) for u in x['changed_units']])
if __name__=='__main__':main()
