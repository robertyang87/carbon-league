"""Inspect the candidate's close loss, with all single-unit changes on the last simultaneous turn."""
import collections,gzip,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import tree_audited_game as arena
import official_game_replay as official
from turning_point_ledger import CashEvents

def freeze_point(sim,seat):
    def side(s):
        collectors=[v for v in sim.workers.values() if v.player==s and v.is_collector]
        return {'cash':sim.cash[s],'trees':sum(tr.player==s for tr in sim.trees.values()),
                'collectors':len(collectors),'cargo_total':round(sum(w.carbon for w in collectors),3),
                'max_loaded':[[w.id,w.pos,round(w.carbon,2)] for w in sorted(collectors,key=lambda w:-w.carbon)[:3]]}
    return [side(seat),side(1-seat)]
def payoff(sim,pre,seat):
    post=sim.get_state();lost=[arena.exam._extinct(post,i) for i in (0,1)]
    score=[sim.step_count-arena.exam.EPISODE_STEPS-1 if lost[i] else (pre if any(lost) else post)['cash'][i] for i in (0,1)]
    return [round(score[seat]-score[1-seat],2),score]

def inspect():
    run=HERE/'candidate';r=json.loads((run/'results.json').read_bytes());g=r['games'][1];seat=g['candidate_seat']
    if seat!=1 or g['actual_transitions']!=299:raise ValueError('wrong trace')
    raw=json.loads(gzip.decompress((run/g['raw_file']).read_bytes()));sim=arena.exam.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    sim.cash=CashEvents(sim.cash);trace={};turns=[]
    for t,actions in enumerate(raw['actions']):
        if t in [250,270,280,285,288,290,292,294,295,296,297,298]:trace[t]=freeze_point(sim,seat)
        if t==298:initial=sim.get_state()
        before=list(sim.cash);sim.cash.events=[];sim.step(*actions)
        if t>=278:turns.append({'from':t,'before':before,'after':list(sim.cash),'events':[e for e in sim.cash.events if e[0]!='rounding']})
    trace[299]=freeze_point(sim,seat);actual=g['margin']
    if abs(payoff(sim,initial,seat)[0]-actual)>.001:raise ValueError('score mismatch')
    original=raw['actions'][-1];alternative=[];rec=arena.exam.FastCarbon(raw['game_configuration'])
    entities={**{uid:(None,'UP','RIGHT','DOWN','LEFT') for uid,w in sim.set_state(initial).workers.items() if w.player==seat},
              **{uid:(None,'RECCOLLECTOR','RECPLANTER') for uid,_ in sim.bases[seat]}}
    for uid,domain in entities.items():
        for action in domain:
            if action==original[seat].get(uid):continue
            cmds=[dict(a) for a in original]
            if action is None:cmds[seat].pop(uid,None)
            else:cmds[seat][uid]=action
            rec.set_state(initial);rec.step(*cmds);m,sc=payoff(rec,initial,seat)
            alternative.append({'uid':uid,'original':original[seat].get(uid),'alternative':action,'margin':m,'score':sc})
    alternative.sort(key=lambda x:-x['margin']);best=alternative[0]
    cmds=[dict(a) for a in original]
    if best['alternative'] is None:cmds[seat].pop(best['uid'],None)
    else:cmds[seat][best['uid']]=best['alternative']
    np.random.seed(raw['seed']);env=official.make('carbon',configuration={**raw['game_configuration'],'randomSeed':raw['seed']})
    for cmd in raw['actions'][:-1]:env.step(cmd)
    env.step(cmds)
    if not env.done or [s.reward for s in env.states]!=best['score']:raise ValueError('best branch official disagreement')
    out={'seat':seat,'seed':raw['seed'],'observed_margin':actual,'observed_states_our_then_opponent':trace,'cash_events_last20':turns,
         'counterfactual':'Post-hoc exhaustive changes to one final command against the actually simultaneous recorded enemy actions. The original game remains a loss. Unseen opponent move not supplied to live policies. No intervening policy response required for a single final transition.',
         'tested_commands':len(alternative),'wins_if_changed':sum(b['margin']>0 for b in alternative),'best_officially_replayed':best,'alternatives':alternative}
    (HERE/'close_loss.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':inspect()
