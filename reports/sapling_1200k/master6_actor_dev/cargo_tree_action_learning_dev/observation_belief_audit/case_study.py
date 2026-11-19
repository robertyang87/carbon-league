"""Reconstruct individual choices with observed/hidden status, both estimator values and prior actions."""
from audit import HERE,FastCarbon,modules,view,cards,load,save
from information import instrument
import numpy as np

def main():
    cases=load(HERE/'results.json')['counterexamples']
    samples=[];f,m=modules();bm=instrument(m)
    for card,cfg,init,tape,term in cards():
        sel=[c for c in cases if c['seed']==card['seed'] and c['registered_seat']==card['seat'] and c['track']==card['arm']]
        for c in sel:
            s=c['pov'];sim=FastCarbon(cfg).reset(seed=card['seed'],full_carbon=init);feat=f.Feat(s);tr=m.BeliefTracker(bm,s)
            for t in range(c['step']+1):
                obs=sim.obs(s);x,maps,vis,known=view(f,m,feat,tr,obs,t)
                if t==c['step']:
                    units=obs['players'][s][2];p,cargo,role=units[c['uid']];facts=[]
                    for k,d in enumerate(f.DEST[p]):
                        cells=[int(d)] if role=='COLLECTOR' else list(map(int,f.DEST[d,1:]))
                        v=[]
                        for pos in cells:
                            last=int(feat.last_step[pos])
                            v.append(dict(cell=pos,truth=float(sim.carbon[pos]),sap=float(maps[2,pos]),master=float(maps[3,pos]),
                                visible=bool(vis[pos]),sap_last_seen=last,master_known=bool(known[pos]),master_first_known_reason=str(tr.bel.reason[pos]) or 'unknown',
                                physical_mirror_last_seen=[int(feat.last_step[z]) for z in f.MIRROR[pos]]))
                        facts.append(dict(direction=c['directions'][k],feasible=c['options'][k],cells=v))
                    samples.append(dict(**c,worker_position=p,cargo=cargo,options_cells=facts,prior_own_commands=[tape[j][s].get(c['uid'],'NONE') for j in range(max(0,t-10),t)],both_workers=obs['players'][0][2]|obs['players'][1][2]))
                    break
                tr.set_actions(obs['players'],tape[t][s]);sim.step(*tape[t])
    assert len(samples)==4
    save(HERE/'case_studies.json',samples)

if __name__=='__main__':main()
