"""Capture only self-simulated H5 leaf contexts. Opponent actions after/at root are never inputs."""
from pathlib import Path
import sys,json,inspect,time,hashlib
from collections import defaultdict,Counter
import numpy as np
HERE=Path(__file__).resolve().parent;J=HERE.parent;sys.path.insert(0,str(J))
from setup import ROOT,GROUP,ACTOR,bridge
from audit import module
read=bridge.load;sha=bridge.sha
SRC=ACTOR/'joint_response_critic_dev/signed_roots.json.gz'
def sources():
    signed=read(SRC);cards=[];held={r['terrain'] for r in read(J/'protocol.json')['sources'] if r['split']=='held'}
    for r in signed['rows']:
        for a in r['arms'].values():
            assert sha(ROOT/a['action_file'])==a['action_sha256'] and sha(ROOT/a['receipt'])==a['receipt_sha256']
        cards.append(dict(case=r['case'],family=r['family'],seed=r['seed'],seat=r['seat'],t=r['t'],terrain=r['terrain'],role=r['role'],source=r['source'],sha=r['source_sha256'],cmd=[r['parent'],r['alternative']],scores=[r['parent_scores'],r['alternative_scores']],use='held' if r['terrain'] in held else 'fit'))
    for family in ('early_region','third_route'):
        for q in read(GROUP/family/'response.json'):
            fold=GROUP/family/f"seat{q['seat']}";p=read(fold/'live_protocol.json');src=p['record']['source'];raw=read(ROOT/src);s=q['seat'];arms=['parent','third']
            cards.append(dict(case=str(fold.relative_to(ROOT)),family=family,seed=q['seed'],seat=s,t=q['step'],terrain=raw['initial_terrain_u8_sha256'],role='planter',source=src,sha=p['record']['source_sha256'],cmd=[read(fold/f'live_{a}_actions.json.gz')[0][s] for a in arms],scores=[q['arms'][a]['scores'] for a in arms],use='diagnostic_extra'))
    # On disjoint optional policy-switching records, terminal effects also include FUTURE policy changes.
    # A first-command-only model can be evaluated descriptively, never trained on that conflated target.
    T=GROUP/'teacher_window_dev'
    for q in read(T/'protocol.json')['cases']:
        s=q['seat'];rr=[read(T/f"{q['seed']}_{s}_{k}.json.gz") for k in ('parent','teacher')]
        cards.append(dict(case='teacher8_'+str(q['seed']),family='original_teacher_next8',seed=q['seed'],seat=s,t=150,terrain=q['terrain'],role='team',source=q['source'],sha=q['source_sha'],cmd=[r['tape'][0][s] for r in rr],scores=[r['scores'] for r in rr],use='diagnostic_regime'))
    base=ACTOR/'partner_phase_switch_dev';ind={r['id']:r for r in read(base/'roots.json')['roots']}
    for q in read(base/'responses/train_labels.json')['pairs']:
        r=ind[q['id']];s=r['seat'];arms=[read(base/'responses'/q[a]['raw']) for a in ('it50','it20')]
        cards.append(dict(case='switch_'+q['id']+'_'+q['stream'],family='whole_policy_switch',seed=r['seed'],seat=s,t=r['t'],terrain=r['map'],role='team',source=r['raw'],sha=r['raw_sha256'],cmd=[v['joint0'][s] for v in arms],scores=[v['official_scores'] for v in arms],use='diagnostic_regime'))
    return cards
def main():
    cards=sources();group=defaultdict(list)
    for i,c in enumerate(cards):group[c['source']].append((i,c))
    freeze=dict(baseline=sha(J/'candidate.py'),network=sha(J/'shared.npz'),signed=sha(SRC),fit_cases=[c['case'] for c in cards if c['use']=='fit'],
           held_cases=[c['case'] for c in cards if c['use']=='held'],additional_only=[c['case'] for c in cards if c['use'].startswith('diagnostic')],
           scope='108 registered once-only alternative commands with original both arms to terminal. Retain historical four held terrains, no additional one-action or whole-policy cases used to fit. No 342,343,347 eval. Freeze network policy trunk and policy head, search, rollout responses, and utility head; train only the seventeen affine economic-value residual readout coefficients. One fixed regularized optimizer on paired terminal margin and preference sign, equal terrains, no second setting; compare exact incumbent deployed H5 rank, delta error and terminal selection. PASS requires > incumbent AND >50% sign matches and strictly better map-mean margin difference MAE on 28 held pairs; also report the 16 historical primary planter roots. Otherwise stop, never open a match.',
           optimizer='Adam 80 full-batch updates lr 0.015, delta init zero, L2 0.15 mean(delta^2). SmoothL1 on paired predicted terminal margin/1000 (Huber threshold 1); 0.2 logistic ordering loss on paired margin/300 clamped to ±15. Train absolute margin differences normalized across physical maps. Baseline absolute-only regularization anchored at old leaf; all other weights frozen.')
    bridge.save(HERE/'protocol.json',freeze)
    m=module(J/'candidate.py','frozen_search');text=inspect.getsource(m._joint_search)
    a='focal=float(_score(cash[s]-cash[op],v[2*j+s]))'
    text=text.replace(a,"capture(d,i,np.stack([c[2*j+s],c[2*j+op]]),float(cash[s]-cash[op]))\n                "+a)
    text=text.replace('a.update(done=True,score=float(_score(margin,None,True)))','ended[i]=float(margin);a.update(done=True,score=float(_score(margin,None,True)))')
    ended={};observed={}
    def capture(depth,arm,ctx,margin):observed[(depth,arm)]=(ctx.copy(),margin)
    ns=dict(vars(m),capture=capture,ended=ended);exec(text,ns);g=ns['_joint_search']
    logs=[]
    for src,ls in group.items():
        path=ROOT/src;raw=read(path);assert sha(path)==ls[0][1]['sha'];s=raw['candidate_seat'];sim=bridge.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);views=[m.F.Feat(j) for j in (0,1)]
        needed=defaultdict(list)
        for i,c in ls:needed[c['t']].append((i,c))
        for t in range(max(needed)+1):
            obs=sim.obs(s);views[s].observe(obs);ox=views[1-s].observe(m.public_enemy(obs,views[s].est))
            for i,c in needed[t]:
                assert c['seat']==s and c['terrain']==raw['initial_terrain_u8_sha256'] and len(c['cmd'])==2
                observed.clear();ended.clear();pred,rounds,_=g(sim.config,obs,views,ox,c['cmd'],time.perf_counter()+6)
                assert len(rounds)==3
                ctx=[];cash=[];fixed=[]
                for arm in (0,1):
                    if (5,arm) in observed:
                        h,d=observed[(5,arm)];ctx.append(h.tolist());cash.append(d);fixed.append(None)
                    else:
                        ctx.append(np.zeros((2,16)).tolist());cash.append(ended[arm]);fixed.append(rounds[-1]['arms'][arm])
                logs.append(dict(**{k:v for k,v in c.items() if k not in ('cmd','scores')},source_sha=c['sha'],context=ctx,cash=cash,fixed=fixed,baseline=rounds[-1],margins=[a[s]-a[1-s] for a in c['scores']],number_changed=sum(c['cmd'][0].get(u)!=c['cmd'][1].get(u) for u in c['cmd'][0].keys()|c['cmd'][1].keys())))
            sim.step(*raw['actions'][t])
    assert len(logs)==len(cards)
    bridge.save(HERE/'features.json.gz',dict(receipt=sha(HERE/'protocol.json'),builder=sha(__file__),rows=logs))
if __name__=='__main__':main()
