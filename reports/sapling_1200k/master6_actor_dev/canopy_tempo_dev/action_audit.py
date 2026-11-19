"""Follow frozen proposals through every issued v1 joint command, including four held development games."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean

import policy
from study import ROOT, HERE, NATIVE, native_game, read, save, collect, elite_list, public_turns

def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def sources():
    for i in range(1, 6):
        path = NATIVE / ('source_receipt.json' if i==1 else f'batches/intake_{i:02}/source_receipt.json')
        for row in read(path)['games']:
            yield 'training_source', row
    directory = HERE.parent / 'symmetry_dev/parent'
    for cert in read(directory/'official_replay.json')['runs']['parent']['games']:
        path = directory/f"raw_{cert['seed']}_{cert['seat']}.json.gz"
        yield 'held_development', {
            'raw_path': str(path.relative_to(ROOT)), 'raw_sha256': cert['transcript_sha256'],
            'seed':cert['seed'], 'seat':cert['seat'], 'map_sha256':cert['map_sha256'],
            'scores':cert['official_reward_by_seat']
        }


def scan(source, weights, gates):
    counts = Counter()
    examples = []
    actions = []
    max_mid = None
    nearest = None
    seat = source['seat']
    for views,joint,after,cfg in native_game(source):
        o=views[seat]; command=joint[seat]; t=o['step']; stage=policy.stage(t)
        st=policy.context(o,cfg,command)
        counts['joint_turns']+=1
        for u,(p,_,role) in st['own'][2].items():
            if role!='PLANTER':
                continue
            x=policy.proposal(st,u)
            if x is None:
                continue
            prob=policy.probability(weights,x); lo,hi=policy.bounds(st,u)
            held=st['dest'][u]==p
            counts[stage+'_clean']+=1
            if held: counts[stage+'_paid']+=1
            if stage=='middle':
                if max_mid is None or prob>max_mid['p']:
                    max_mid=dict(t=t,uid=u,p=prob,actual_command=command.get(u,'NONE'),quote=st['price'])
                if prob>=.5 and not held:
                    counts['middle_high_p_move']+=1
                if prob>=.5 and lo>st['price']+.02 and not held:
                    counts['middle_before_gate']+=1
            if stage=='late' and held:
                counts['late_paid_p_below_half']+=int(prob<.5)
                counts['late_paid_deterministic_upper']+=int(hi is not None)
                counts['late_paid_below_price_upper']+=int(hi is not None and hi+.02<st['price'])
                counts['final_clean_paid']+=int(t==cfg['episodeSteps']-2)
                row=dict(t=t,uid=u,pos=p,prob=prob,quoted_price=st['price'],remaining=cfg['episodeSteps']-1-t,
                         upper=hi,order=command.get(u,'NONE'))
                if nearest is None or row['remaining']<nearest['remaining']:
                    nearest=row
            candidate=policy.order(st,u,weights,gates)
            if candidate is not None:
                actions.append({'t':t,'uid':u,'v1':command.get(u,'NONE'),'proposal':candidate})
        if t==cfg['episodeSteps']-2:
            before={k for pl in o['players'] for k in pl[3]}
            for uid,(pos,_,role) in st['own'][2].items():
                if (role=='PLANTER' and command.get(uid,'NONE')=='NONE' and any(
                    trpos==pos and tree_id not in before for tree_id,(trpos,_) in after['players'][seat][3].items()
                )):
                    examples.append(dict(t=t,uid=uid,pos=pos,price=st['price'],clean=policy.proposal(st,uid) is not None,
                                         actual_net=-st['price']))
    return {'seed':source['seed'],'seat':seat,'source_sha256':source['raw_sha256'],'terrain':source['map_sha256'],
            'counts':dict(counts),'highest_middle':max_mid,'nearest_clean_late_purchase':nearest,
            'final_step_purchases':examples,'raw_proposals_before_suffix_check':actions}


def calibration(weights):
    buckets={}; high=None
    for mid,part,g in elite_list():
        if part!='val':
            continue
        _,rows=collect(public_turns(g),mid)
        for r in rows:
            if policy.stage(r['t'])!='middle':
                continue
            p=policy.probability(weights,r['x'])
            if high is None or p>high['p']:
                high={**{k:r[k] for k in ('map','seat','t','uid','hold','gross','price')},'p':p}
            bucket=min(9,int(p*10))/10
            group=buckets.setdefault(bucket,{'n':0,'p':0.,'actual':0,'gross_sum':0.,'losses':0,'maps':set()})
            group['n']+=1;group['p']+=p;group['actual']+=int(r['hold']);group['maps'].add(mid)
            if r['hold']:
                net=r['gross']-r['price']
                group['gross_sum']+=net;group['losses']+=int(net<0)
    return {str(k):{'opportunities':v['n'],'maps':len(v['maps']),'mean_p':v['p']/v['n'],
                    'purchase_rate':v['actual']/v['n'],'paid':v['actual'],'losses_among_paid':v['losses'],
                    'mean_direct_net_among_paid':v['gross_sum']/v['actual'] if v['actual'] else None}
            for k,v in sorted(buckets.items())},high


def main():
    weights=read(HERE/'weights.json');study=read(HERE/'study.json')
    if study['features_sha256']!=digest(HERE/'policy.py') or study['protocol_sha256']!=digest(HERE/'protocol.json'):
        raise ValueError('source or declaration changed')
    by_set={'training_source':[],'held_development':[]}
    for group,source in sources():
        by_set[group].append(scan(source,weights,study['gates']))
    if set(r['terrain'] for r in by_set['training_source']) & set(r['terrain'] for r in by_set['held_development']):
        raise ValueError('transfer terrains overlap native assessment terrains')
    results={}
    for group,rows in by_set.items():
        count=Counter()
        for r in rows:
            count.update(r['counts'])
        proposals=sum(len(r['raw_proposals_before_suffix_check']) for r in rows)
        results[group]=dict(games=len(rows),distinct_terrains=len({r['terrain'] for r in rows}),
                            counts=dict(count),upper_bound_on_changed_joint_commands=proposals,rows=rows)
    bins,high=calibration(weights)
    save(HERE/'action_audit.json',dict(cases=results,held_middle_probability_bins=bins,highest_held_middle=high,
         identity={p:digest(HERE/p) for p in ('study.json','native_audit.json','weights.json','policy.py','protocol.json','action_audit.py')},
         interpretation='Full original v1 action tapes are authoritative. A zero proposal count proves zero emitted changes at entry to the forced suffix check. When positive, this count is only an upper bound and grants no deployment approval. No new native game.'))
    for k,v in results.items():print(k,v['counts'],v['upper_bound_on_changed_joint_commands'])
    print(bins,high,flush=True)


if __name__=='__main__':
    main()
