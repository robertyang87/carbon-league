"""Apply real trained weights to actual v2 menus without creating a submission from an unqualified head."""
import time
from pathlib import Path
import numpy as np
import probe

HERE=probe.HERE
learn=probe.D/"return_response_learning"
MODEL=probe.load("rr_trained",learn/"model.py")
PAIR=probe.D/"action_rank_data"


def inspect(players,seat,step,commands,w):
    feat=[MODEL.rr_features(players,seat,step,cmd) for cmd in commands]
    assert feat[0][1]==feat[1][1]
    pred=[MODEL.rr_predict(x,w['enhanced']) for x,_ in feat]
    scores=MODEL.rr_scores(players,seat,step,commands,w)
    assert all(np.isfinite(scores))
    changed=[]
    own=players[seat]
    for u in sorted(set(commands[0])|set(commands[1])):
        a,b=(cmd.get(u,"NONE") for cmd in commands)
        if a==b:continue
        changed.append(dict(id=u,asset=own[2].get(u) or ["BASE",own[1].get(u)],first=a,second=b))
    contributions=[]
    for i,(uid,cargo) in enumerate(feat[0][1]):
        columns=np.nonzero(feat[0][0][i]!=feat[1][0][i])[0].tolist()
        marginal=cargo*float(pred[1][i]-pred[0][i])
        if columns:contributions.append(dict(id=uid,cargo=cargo,changed_columns=columns,delta_expected_deposit=marginal))
    assert np.isclose(sum(c['delta_expected_deposit'] for c in contributions),scores[1]-scores[0])
    return dict(features_identical=not contributions,orders_changed=changed,contributors=contributions,
                bank=scores.tolist(),raw_difference=float(scores[1]-scores[0]),correction=float((scores[1]-scores[0])*w['pair_scale']))

def main():
    held=probe.read(learn/"held.json");w=probe.read(learn/"weights.json");identity=probe.sha((learn/"weights.json").read_bytes())
    assert held['weight_sha']==identity
    eps_line, = [line for line in probe.PARENT.read_text().splitlines() if line.startswith('_S_EPS =')]
    eps=float(eps_line.partition('=')[2].partition('#')[0])
    native=probe.read(HERE/"native_roots.json.gz");rows=[]
    for root in native['rows']:
        q=root['record'];row={k:root[k] for k in ('map_seed','seat','t')};row['status']=q['reason']
        if not q['eligible']:rows.append(row);continue
        a,b=q['arms'];t=time.perf_counter()
        info=inspect(root['players'],root['seat'],root['t'],[a['command'],b['command']],w)
        info['audit_inference_ms']=(time.perf_counter()-t)*1000
        info.update(native_gap=a['value']-b['value'],native_indices=[a['index'],b['index']],
                    raw_second_would_win=bool(b['value']+info['correction']>a['value']),
                    native_epsilon_preserved_second_would_win=bool(b['value']+info['correction']>a['value']+(0. if a['index'] else eps)))
        rows.append(dict(row,comparison=info))
    paired=[]
    plan=probe.read(learn/'protocol.json')
    for trained in held['pairs']:
        if trained['status']!='complete':continue
        seed=trained['seed'];ticket=next(c for c in plan['pairs'] if c['seed']==seed and c['split']=='held')
        root=probe.read(PAIR/'held'/f"{seed}_{ticket['seat']}_{ticket['step']}_root.json.gz")
        info=inspect(root['public_root']['players'],ticket['seat'],ticket['step'],[a['order'] for a in root['arms']],w)
        assert np.allclose(info['bank'],trained['cargo_scores'])
        paired.append(dict(seed=seed,comparison=info,true_first_minus_second_margin=trained['margin_delta'],first_minus_second_native=trained['native_delta']))
    result=dict(candidate_generated=False,trained_weight_sha256=identity,gate=held['qualified_for_development'],native_roots=rows,
                paired_holdout=paired,qualified_checks=held['checks'])
    probe.save(HERE/'diagnosis.json',result)
    for z in paired:
        v=z['comparison'];print(z['seed'],v['features_identical'],v['orders_changed'])
    for x in rows:
        if 'comparison' in x:
            q=x['comparison'];print(x['map_seed'],x['seat'],x['t'],q['native_gap'],q['raw_difference'],q['correction'],q['raw_second_would_win'])

if __name__=="__main__":main()
