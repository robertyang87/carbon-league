"""Apply one publicly selected collector's route with all other recorded orders fixed."""
import copy
import gzip
import importlib.util
import json
import random
from statistics import mean
import zipfile

from policy import least_tax, select_carrier
from register import HERE, HELD, PARENT, protocol, sha, enforce


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


fish=load('fixed_fish_return_rule',PARENT/'check.py')
rule=fish.rule
BEGIN=290


def engines(cfg,terrain=None):
    random.seed(cfg['randomSeed']);rule.np.random.seed(cfg['randomSeed']);env=rule.make('carbon',configuration=cfg)
    if terrain is not None:env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    e=copy.deepcopy(env.states[0].observation.full_carbon)
    sim=rule.econ.FastCarbon(cfg).reset(full_carbon=e);sim.cash=rule.econ.CashEvents(sim.cash)
    return env,sim


def advance(env,sim,pair):
    sim.cash.events.clear();sim.step(*pair);env.step(pair,[])
    enforce(fish.view(env)==fish.fast(sim),'Official simulator and cash ledger differ')


def branches(env,sim,cfg,all_orders,seat,uid,original_scores,expected_views=None,original_fish=False):
    root_bank=list(sim.cash);env.steps[:-1]=[None]*BEGIN;env.logs=[]
    frozen=rule.official_state.take(env,[5,5]);out={}
    for name in ('recorded','least_tax'):
        rule.official_state.restore(env,frozen);alt=copy.deepcopy(sim);deposits=0.;stopped=uid is None;trace=[];changes=0
        for t in range(BEGIN,len(all_orders)):
            if env.done:break
            p=alt.obs(seat)['players'];raw=all_orders[t]
            pair=[{k:v for k,v in row.items() if k in p[s][1] or k in p[s][2]} for s,row in enumerate(raw)]
            missing=[sorted(set(row)-set(pair[s])) for s,row in enumerate(raw)]
            old=pair[seat].get(uid,'NONE')
            if uid is not None and uid not in p[seat][2]:stopped=True
            if name=='least_tax' and not stopped:
                cmd=least_tax(p,seat,uid,pair[seat],t,cfg)
                enforce(cmd==least_tax(p[::-1],1-seat,uid,pair[seat],t,cfg),'Seat convention changed the route')
                if original_fish:
                    enforce(fish.least_tax(p,pair[seat],t,cfg)==cmd,'Changed original fish action at '+str(t))
                if cmd=='NONE':pair[seat].pop(uid,None)
                else:pair[seat][uid]=cmd
                changes+=int(old!=cmd)
            orig_before=p[seat][2].get(uid);advance(env,alt,pair)
            d=sum(e[2] for e in alt.cash.events if e[0]=='own_delivery' and e[1]==seat and e[3]==uid and e[2]>0)
            deposits+=d
            if d>0 or uid not in alt.obs(seat)['players'][seat][2]:stopped=True
            trace.append(dict(t=t,action=pair[seat].get(uid,'NONE'),recorded_action=old,before=orig_before,
                              after=alt.obs(seat)['players'][seat][2].get(uid),deposited=d,cash=list(alt.cash),skipped_absent_ids=missing))
            if name=='recorded' and expected_views:
                enforce(fish.view(env)==fish.archived(expected_views[t+1]),'Fish baseline path differs')
        scores=[r.reward for r in env.states]
        enforce(env.done and scores==original_scores if name=='recorded' else env.done,'Wrong termination or baseline result')
        out[name]=dict(scores=scores,cash=list(alt.cash),margin=round(scores[seat]-scores[1-seat],2),
                       picked_deposits=deposits,physical_changes=changes,trace=trace,steps=len(trace))
    return dict(incoming_cash=root_bank,branches=out,
                own_cash_change=round(out['least_tax']['cash'][seat]-out['recorded']['cash'][seat],2),
                foe_cash_change=round(out['least_tax']['cash'][1-seat]-out['recorded']['cash'][1-seat],2),
                official_margin_change=round(out['least_tax']['margin']-out['recorded']['margin'],2),
                win_change=int(out['least_tax']['margin']>0)-int(out['recorded']['margin']>0))


def fish_case():
    ref=json.loads((PARENT/'outcomes.json').read_bytes());raw=fish.TAPE.read_bytes()
    enforce(sha(raw)==ref['source_sha256'],'Different fish source')
    game=json.loads(gzip.decompress(raw));cfg=game['configuration'];frames=game['steps'];orders=[[c['action'] or {} for c in f] for f in frames[1:]]
    env,sim=engines(cfg,frames[0][0]['observation']['full_carbon'])
    for t in range(BEGIN):
        enforce(fish.view(env)==fish.archived(frames[t]),'Wrong fish prefix');advance(env,sim,orders[t])
    result=branches(env,sim,cfg,orders,0,fish.CARRIER,game['rewards'],frames,True)
    refs={r['policy']:r for r in ref['comparison'] if r['t']==BEGIN}
    for name,x in result['branches'].items():
        old=refs[name];enforce(old['scores']==x['scores'] and old['margin']==x['margin'] and old['bank']==x['picked_deposits'],'Fish scores differ')
        for a,b in zip(x['trace'],old['trace'],strict=True):
            enforce(all(a[k]==b[k] for k in ('t','action','before','after','deposited','cash')),'Fish trajectory differs')
    return result


def one(game,archive,config):
    seat=game['seat'];seed=game['seed'];arm=game['arm'];cfg=dict(config,randomSeed=seed)
    env,sim=engines(cfg)
    enforce(sha(bytes(env.states[0].observation.full_carbon))==game['terrain'],'Wrong physical test map')
    reg=json.loads(archive.read(arm+'/registration.json'))
    enforce(reg['files']['candidate']['sha256']==PLAN['v4'] and reg['files']['opponent']['sha256']==PLAN['opponent'],'Different policy pair')
    tape=archive.read(arm+'/turns.jsonl'); receipt=archive.read(arm+'/receipt.json')
    enforce(sha(receipt)==game['receipt_sha256'] and sha(tape)==json.loads(receipt)['turns_sha256'],'Wrong action tape')
    turns=[json.loads(l) for l in tape.splitlines()];orders=[[c['response']['action'] for c in row['calls']] for row in turns]
    views=[json.loads(l) for l in gzip.decompress(archive.read(arm+'/observations.jsonl.gz')).splitlines()]
    for t in range(BEGIN+1):
        observed=views[t]['observations'][seat];q=sim.obs(seat)
        enforce(observed['step']==t==env.states[0].observation.step and observed['player']==seat and
                all(q[k]==observed[k] for k in ('players','trees','carbon')) and
                rule.digest(observed)==turns[t]['public_observation_sha256'][seat], 'Historical public state differs')
        if t!=BEGIN:advance(env,sim,orders[t])
    p=views[BEGIN]['observations'][seat]['players'];uid=select_carrier(p,seat,BEGIN,cfg)
    enforce(uid==select_carrier(p[::-1],1-seat,BEGIN,cfg),'Selection under inverted seat differs')
    result=branches(env,sim,cfg,orders,seat,uid,game['official_scores_by_seat'])
    return dict(seed=seed,seat=seat,terrain=game['terrain'],uid=uid,root_unit=p[seat][2].get(uid),**result)


if __name__=='__main__':
    plan_raw=(HERE/'protocol.json').read_bytes();PLAN=json.loads(plan_raw);enforce(PLAN==protocol(),'Probe differs from freeze')
    fish_result=fish_case()
    settlement=json.loads((HELD/'settlement.json').read_bytes());cfg=rule.econ.config()
    gs=[]
    for ticket in PLAN['archive_cards']:
        a=HELD/(str(ticket['id'])+'.zip');enforce(sha(a.read_bytes())==ticket['sha256'],'Different ZIP')
        with zipfile.ZipFile(a) as z:
            g=next(g for g in settlement['games'] if g['arm']=='challenger' and (g['seed'],g['seat'])==(ticket['seed'],ticket['seat']))
            gs.append(one(g,z,cfg))
    enforce(len(gs)==8,'All eight are required')
    by_map={s:mean(g['official_margin_change'] for g in gs if g['seed']==s) for s in sorted({g['seed'] for g in gs})}
    data=dict(protocol_sha256=sha(plan_raw),fish_parity=dict(root=BEGIN,uid=fish.CARRIER,action_cash_and_positions_match=True,
               official_margin_change=fish_result['official_margin_change']),games=gs,by_map=by_map,
               mean_margin_change=mean(by_map.values()),modified_units=sum(g['branches']['least_tax']['physical_changes']>0 for g in gs),
               modified_wins=sum(g['branches']['least_tax']['margin']>0 for g in gs),baseline_wins=sum(g['branches']['recorded']['margin']>0 for g in gs))
    raw=gzip.compress((json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0)
    path=HERE/'outcomes.json.gz';enforce(not path.exists() or path.read_bytes()==raw,'Refusing conflicting branch evidence')
    path.write_bytes(raw);print(data['mean_margin_change'],data['by_map'],data['modified_wins'],data['baseline_wins'])
