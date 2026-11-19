"""Read remaining time and actual chosen defender's arrival against each issued warning."""
from collections import Counter
from pathlib import Path
import json
from scan import HERE,ROOT,FastCarbon,DIST,read,sha,sources


def main():
    manifest={(s['seed'],s['seat']):s for group,s in sources()}
    outputs={}
    for pool,panel in read(HERE/'warning_transfer.json')['panels'].items():
        rows=[]
        for sample in panel['games']:
            event=sample['emitted']
            if not event:continue
            source=manifest[sample['seed'],sample['seat']]
            tape=read(ROOT/source['raw_path']);s=source['seat'];t0=event['root']
            sim=FastCarbon(tape['game_configuration']).reset(seed=source['seed'])
            for t,j in enumerate(tape['actions'][:t0]):
                sim.step(*j)
            tree=sim.trees[event['tid']]
            root_players=sim._players_obs()
            flags={}
            for arm in ('parent','defense'):
                g=FastCarbon(sim.config);g.set_state(sim.get_state())
                path=[];first_loss=None; focal=event['uid']
                updates={t:(before,after) for t,before,after,reason in event['outcomes'][arm]['moves']}
                for t in range(t0,t0+5):
                    orders=[dict(c) for c in tape['actions'][t]]
                    if t in updates:
                        before,after=updates[t]
                        if orders[s].get(focal,'NONE')!=before:raise ValueError('wrong archived order')
                        orders[s].pop(focal,None)
                        if after!='NONE':orders[s][focal]=after
                    g.step(*orders)
                    now=g.trees.get(event['tid'])
                    held=now is not None and now.player==s
                    if not held and first_loss is None:first_loss=t+1
                    path.append([t+1,None if focal not in g.workers else DIST[g.workers[focal].pos][tree.pos],held])
                if event['outcomes'][arm]['at_original_seizure']['cash']!=list(g.cash):raise ValueError('wrong source replay')
                flags[arm]={'first_loss_post':first_loss,'distance_and_owner_by_post':path}
            live=flags['parent']['first_loss_post']
            if bool(live)!=bool(event['actually_seized_in_5']):raise ValueError('expiry or wrong label')
            options=sorted([DIST[w[0]][tree.pos],w[1],uid] for uid,w in root_players[s][2].items() if w[2]=='COLLECTOR')
            selected_dist=DIST[root_players[s][2][event['uid']][0]][tree.pos]
            rows.append(dict(seed=sample['seed'],seat=s,root=t0,tid=event['tid'],chosen_distance=selected_dist,
                             threats_at_root=sorted(DIST[w[0]][tree.pos] for w in root_players[1-s][2].values() if w[2]=='PLANTER'),
                             possible_defenders=options,actual_steps_to_loss=live-t0 if live else None,branches=flags))
        outputs[pool]=rows
    (HERE/'arrival.json').write_text(json.dumps({'transfer_sha256':sha(HERE/'warning_transfer.json'),'panels':outputs},indent=2)+'\n')


if __name__=='__main__':main()
