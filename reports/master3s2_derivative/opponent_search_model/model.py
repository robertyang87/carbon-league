"""Compare original opposing joint commands on a common independent self prediction. No current orders are inputs."""
from time import perf_counter
import numpy as np


def infer(model, observation, estimate, previous_players, config):
    if set(observation)!={"step","player","players","trees","carbon","remainingOverageTime"}:raise ValueError("Expected a seat's public request")
    me=observation["player"];enemy=1-me;t=observation["step"];players=observation["players"]
    all_known=[1.]*model.N;seed=int(1000003*(t+1)+17*me)%(2**31)
    net=model._net();ts=perf_counter()
    def state(ps,who,carbon,t,prev):
        return model._quant(model.encode(ps,t,who,carbon,all_known,prev)).astype(np.float32)
    xe=state(players,enemy,estimate,t,previous_players)
    ew,eb,_=net(xe[None],model._SID)
    ew,eb=model._opponent_root_readout(xe[None],ew,eb,players,enemy);ew,eb=ew[0],eb[0]
    def predict(who,lw,lb,sub):
        return model._sample(players,who,lw,lb,np.random.RandomState(sub),t)
    first=predict(enemy,ew,eb,seed+7);t0=perf_counter()
    # Predict an independent self move from the same belief. Neither the actual actor command
    # nor any of its search candidates can affect which enemy alternative wins.
    xs=state(players,me,estimate,t,previous_players);sw,sb,_=net(xs[None],model._SID)
    fixed=predict(me,sw[0],sb[0],seed+707);t1=perf_counter()
    menu=[first]
    for j in (1,2,3):
        a=predict(enemy,ew,eb,seed+7+j*31)
        if a not in menu:menu.append(a)
    def action_score(action):
        total=0.
        for uid,w in players[enemy][2].items():total+=float(ew[w[0],model.WIDX[action.get(uid,"NONE")]])
        for uid,pos in players[enemy][1].items():total+=float(eb[pos,model.BIDX[action.get(uid,"NONE")]])
        return total
    prior=[action_score(a) for a in menu];t2=perf_counter()
    root=dict(players=players,trees=observation["trees"],step=t,player=me,carbon=list(map(float,estimate)),
              remainingOverageTime=observation["remainingOverageTime"])
    X=[];sc=[];live=[];values=[None]*len(menu)
    for j,a in enumerate(menu):
        pair=[None,None];pair[me]=fixed;pair[enemy]=a
        successor=model._turn(root,config,pair);pp=successor["players"];step=successor["step"]
        # The experiment uses t<=296, but keep the exact official extinction convention for early termination.
        dead=[not v[2] and not v[3] and (not v[1] or v[0]<min(config["recCollectorCost"],config["recPlanterCost"])) for v in pp]
        if any(dead) or step>=config["episodeSteps"]-1:
            terminal=[step-config["episodeSteps"]-1 if dead[q] else players[q][0] if any(dead) else pp[q][0] for q in (0,1)]
            values[j]=float(model._terminal(terminal[enemy]-terminal[me]))
        else:
            X.append(state(pp,enemy,successor["carbon"],step,players));sc.append(model._scalars(pp,step,enemy));live.append(j)
    t3=perf_counter()
    if X:
        val=net(np.stack(X),model._SID,np.stack(sc))[2]
        for j,v in zip(live,val,strict=True):values[j]=float(v)
    t4=perf_counter()
    if not all(np.isfinite(v) for v in prior+values):raise ValueError("Nonfinite original inference")
    return dict(menu=menu,opponent_values=values,native_logit_sums=prior,common_self_model_order=fixed,batch_rows=len(X),terminal_rows=len(menu)-len(X),
                timings=dict(original_opponent_encode_forward_draw=t0-ts,common_self_prediction=t1-t0,additional_orders=t2-t1,
                             simulated_transitions_and_encode=t3-t2,batched_opponent_critic=t4-t3,increment=t4-t0))
