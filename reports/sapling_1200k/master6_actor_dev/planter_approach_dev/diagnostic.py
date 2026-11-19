"""Post-screen census of five frozen-opponent transitions; retain separate per-worker outcomes and veto mechanisms."""
from collections import Counter
import copy
import inspect
import numpy as np
from screen import HERE,ROOT,read,retain,sha,live,clean,FastCarbon,CashEvents


def attach_trace(m):
    src=inspect.getsource(m._approach_decode)
    line='            if slot>=10 or x["tok_kind"][slot]!=1 or not legal[order] or combined[action]-combined[order]>1.5:'
    if src.count(line)!=1:raise ValueError('unknown decision instrumentation')
    src=src.replace(line,'''            _DIAG.append({"requested":int(order),"original":int(action),"legal":bool(legal[order]),"logit_gap":float(combined[action]-combined[order])})
'''+line)
    exec(src,m.__dict__)


def veto(state,ticket,pair,m):
    if ticket is None:return None
    uid,cell=ticket['uid'],ticket['cell']
    unit=state.own[2].get(uid)
    if not unit:return 'worker_gone'
    distance=int(m._ROUTE_DIST[unit[0],cell]);age=state.step-ticket['start']
    market=state.market
    if age>4 or distance>4-age:return 'deadline'
    val=market.features(cell)
    if val is None:return 'no_buyable_site'
    f,price=val
    tree=market.opp_trees.get(cell)
    if (tree[0] if tree else None)!=ticket['tree']:return 'target_identity_changed'
    if f[1]*50<distance+10:return 'insufficient_tenure'
    if not tree and f[4]<.5:return 'insufficient_recent_carbon'
    if any(m._ROUTE_DIST[cell,w[0]]<2 for w in state.enemy[2].values()):return 'enemy_can_contest_target'
    if state.quote(cell,uid,pair,distance) is None:return 'unfunded_after_other_requests'
    if m.route_value(m._APP_MODEL,f)[2]<8:return 'lower_forecast_net'
    return None


def project(m,sim,raw,s,root,variant):
    uid=root['uid'];cell=root['cell'];t0=root['t'];log=[];paid=0.;stop=None;seen=False
    for t in range(t0,t0+5):
        obs=sim.obs(s);pair=raw['actions'][t];joint=[dict(a) for a in pair]
        if variant:
            ticket=copy.deepcopy(m._APP.ticket)
            m._DIAG=[]
            joint[s]=clean(m.agent(obs,raw['game_configuration']))
            event=m._APP.last
            if t==t0 and (not event.get('changed') or event.get('issued')!=root['issued'] or event.get('target')!=cell):
                raise ValueError('shadow launch differs from frozen deployment')
            reason=event['reason'];detail=None
            if reason in ('expired_or_lost','reprice_or_threat_cancel'):
                # Parent joint equals actual issued joint at a veto.
                detail=veto(m._APP,ticket,joint[s],m)
            elif reason=='no_legal_safe_prefix':
                checks=m._DIAG
                if not checks:detail='no_safe_progress_move'
                elif not any(x['legal'] for x in checks):detail='real_prefix_illegal'
                elif not any(x['legal'] and x['logit_gap']<=1.5 for x in checks):detail='network_disagrees_over_1.5'
                else:detail='joint_action_changes_or_collision'
            if ticket and m._APP.ticket is None and reason!='pay' and stop is None:stop=detail or reason
            diag=copy.deepcopy(m._DIAG)
        else:reason='original_tape';detail=None;diag=[]
        original_opp=set(raw['actions'][t][1-s]);live_opp=set(obs['players'][1-s][2])|set(obs['players'][1-s][1])
        sim.cash.events.clear();sim.step(*joint)
        cash=[list(e) for e in sim.cash.events if e[0] in ('plant','seize') and e[1]==s and e[3]==uid and e[4]==cell]
        paid+=sum(-e[2] for e in cash)
        unit=sim.obs(s)['players'][s][2].get(uid)
        seen|=bool(unit and unit[0]==cell)
        log.append(dict(t=t,cmd=joint[s].get(uid,'NONE'),cell=unit[0] if unit else None,reason=reason,veto=detail,
                        prefix_checks=diag,cash=cash,unavailable_rival_commands=sorted(original_opp-live_opp),
                        rival_live_at_start=sorted(obs['players'][1-s][2])))
    return dict(reached=seen,paid=paid,stopped=stop,steps=log)


def main():
    screen=read(HERE/'screen.json.gz');plan=read(HERE/'protocol.json')
    results=[]
    for game,sg in zip(plan['public_source_games'],screen['games']):
        raw=read(ROOT/game['raw_path']);s=game['seat'];cfg=raw['game_configuration']
        m=live.module(HERE/'agent.py','approach_census_actor');attach_trace(m)
        targets={r['t']:r for r in sg['offers'] if r['changed']}
        sim=FastCarbon(cfg).reset(seed=game['seed']);sim.cash=CashEvents(sim.cash)
        n=0
        for t,pair in enumerate(raw['actions']):
            obs=sim.obs(s)
            if t==0:m._DIAG=[];m.agent(obs,cfg)
            else:
                if t in targets:
                    root=targets[t]
                    f,controller=m._FEAT,m._APP
                    m._FEAT=copy.deepcopy(f);m._APP=copy.deepcopy(controller)
                    aa=project(m,copy.deepcopy(sim),raw,s,root,True)
                    m._FEAT=f;m._APP=controller
                    bb=project(m,copy.deepcopy(sim),raw,s,root,False)
                    extra=0
                    for x,y in zip(aa['steps'],bb['steps']):
                        x['additional_rival_units_without_recorded_order']=sorted(set(x['rival_live_at_start'])-set(y['rival_live_at_start'])-set(raw['actions'][x['t']][1-s]))
                        del x['rival_live_at_start'];del y['rival_live_at_start']
                    if t==sg['first']['t']:
                        for name,v in [('parent',bb),('approach',aa)]:
                            old=sg['fixed_opponent_tape'][name]
                            if old['cash_paid_at_target']!=v['paid'] or old['earned_target_arrival']!=v['reached']:
                                raise ValueError('original primary mechanism changed')
                    results.append(dict(seed=game['seed'],seat=s,t=t,uid=root['uid'],target=root['cell'],distance=root['distance'],
                                        initial_net=root['net'],parent=bb,approach=aa))
                    n+=1
                m._FEAT.observe(obs);m._APP.observe(obs,cfg)
            sim.step(*pair)
        if n!=len(targets):raise ValueError('lost or duplicated root')
        print(game['seed'],s,n,flush=True)
    retain(HERE/'diagnostic.json.gz',dict(screen_sha256=sha(HERE/'screen.json.gz'),plan_sha256=sha(HERE/'diagnostic_plan.json'),
                                         script_sha256=sha(HERE/'diagnostic.py'),roots=results))
    return results


if __name__=='__main__':main()
