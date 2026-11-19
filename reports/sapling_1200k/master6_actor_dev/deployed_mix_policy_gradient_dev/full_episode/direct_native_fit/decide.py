"""Apply the frozen two-map decision mechanically to signed, complete original-game returns."""
import itertools,json,shutil,statistics
from pathlib import Path
from prepare import HERE,ROOT,center_path,fp,P,make,sha,f
def decision(rows):
    assert {(r['sign'],r['seed'],r['seat']) for r in rows}==set(itertools.product(('plus','minus'),P['seeds'],(0,1)))
    pairs=[]; lookup={(r['sign'],r['seed'],r['seat']):r for r in rows};table={}
    for s in P['seeds']:
        for seat in (0,1):
            a,b=(lookup[k,s,seat] for k in ('plus','minus'))
            pairs.append(dict(seed=s,seat=seat,delta_plus_minus=a['margin']-b['margin'],plus=a['margin'],minus=b['margin']))
    for sign in ('plus','minus'):
        rr=[v for v in rows if v['sign']==sign]
        table[sign]=dict(wins=sum(v['result']=='win' for v in rr),draws=sum(v['result']=='draw' for v in rr),
                         losses=sum(v['result']=='loss' for v in rr),margin=statistics.mean(v['margin'] for v in rr),
                         own_extinctions=sum(v['extinction'][v['seat']] for v in rr),maps={})
        for s in P['seeds']:
            ss=[r for r in rr if r['seed']==s]
            table[sign]['maps'][s]=dict(wins=sum(r['result']=='win' for r in ss),margin=statistics.mean(r['margin'] for r in ss))
    a,b=table['plus'],table['minus'];w=a['wins']-b['wins'];diff=a['margin']-b['margin']
    sg=(1 if w>0 else -1) if w else (1 if diff>0 else -1 if diff<0 else 0)
    label='plus' if sg==1 else 'minus' if sg==-1 else None
    advantages={s:sg*(a['maps'][s]['margin']-b['maps'][s]['margin']) for s in P['seeds']}
    checks={'nonzero_choice':sg!=0,'no_map_loses_wins':all(sg*(a['maps'][s]['wins']-b['maps'][s]['wins'])>=0 for s in P['seeds']),
            'both_map_margins_positive':min(advantages.values())>0,'at_least_three_seat_pairs_positive':sum(sg*z['delta_plus_minus']>0 for z in pairs)>=3,
            'no_more_extinction':sg*(a['own_extinctions']-b['own_extinctions'])<=0,'tied_wins_each_map_at_least_250':bool(w) or min(advantages.values())>=250}
    factor=sg*(1. if w else .5) if all(checks.values()) else 0.
    return dict(inferred_direction=label,wins_differ=bool(w),signed_plus_minus_mean_margin=diff,pairs=pairs,by_sign=table,
                provisional_map_margin_advantage=advantages,checks=checks,radius_fraction_retained=factor)
def apply():
    if (HERE/'selection.json').exists():
        previous=f.read(HERE/'selection.json')
        assert previous['protocol']==sha(HERE/'protocol.json') and previous['score_sha256']==sha(HERE/'scores.json')
        assert sha(ROOT/previous['selected_pack_file'])==previous['selected_pack_sha256'] and sha(ROOT/previous['selected_weights_file'])==previous['selected_weights_sha256']
        return
    b=f.read(HERE/'binding.json');rows=f.read(HERE/'scores.json');data=decision(rows)
    assert len(rows)==8
    for r in rows:
        p=HERE/'games'/r['driver_label']/f"raw_{r['seed']}_{r['seat']}.json.gz"
        wall=p.with_name(p.name.replace('raw_','wall_'));d=f.read(p);cl=f.read(wall)
        assert sha(p)==r['raw_sha256'] and sha(wall)==r['wall_sha256'] and d['pack_sha256']==b['signs'][r['sign']]['pack']
        assert r['steps']==len(cl['calls'])==len(d['actions'])==r['actual_opponent_calls'] and d['configuration']['_S_ABORT']==.58 and d['configuration']['_S_ENGINE'] and d['configuration']['_TEMP']==1 and d['configuration']['_SEED'] is None and r['status']==['DONE','DONE']
    retained=data['radius_fraction_retained'];radius=b['trials'][-1]['radius']*retained
    if not retained:
        out={'classification':'unchanged_center','played_on_these_eight':False,'weight':P['center_npz'],'pack':P['center']}
    elif abs(retained)==1:
        sign='plus' if retained>0 else 'minus';out={'classification':'tested_endpoint','played_on_these_eight':True,'pack':{'path':str((HERE/(sign+'.py')).relative_to(ROOT)),'sha256':b['signs'][sign]['pack']},'weight':{'path':str((HERE/(sign+'.npz')).relative_to(ROOT)),'sha256':b['signs'][sign]['npz']}}
    else:
        info=make(radius,'retained_half',HERE);out={'classification':'unplayed_shrunk_training_candidate','played_on_these_eight':False,'pack':{'path':str((HERE/'retained_half.py').relative_to(ROOT)),'sha256':info['pack']},'weight':{'path':str((HERE/'retained_half.npz').relative_to(ROOT)),'sha256':info['npz']}}
    f.put(HERE/'selection.json',dict(replaces_center=bool(retained),selected_pack_file=out['pack']['path'],selected_pack_sha256=out['pack']['sha256'],selected_weights_file=out['weight']['path'],selected_weights_sha256=out['weight']['sha256'],protocol=sha(HERE/'protocol.json'),bindings=sha(HERE/'binding.json'),source=sha(Path(__file__)),score_sha256=sha(HERE/'scores.json'),
            retained=out,retained_signed_radius=radius,**data))
    print(out['classification'],data['by_sign'],data['checks'],flush=True)
if __name__=='__main__':apply()
