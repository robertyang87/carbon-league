import copy,json,sys
from carbon_deep_events import analyze_events, DEFAULT_RULES

def state(step, own, enemy, actions=None, ownbase=None, enemybase=None, trees=None, status='ACTIVE'):
    obs={'step':step,'players':[[80,ownbase or {},own,(trees or [{},{}])[0]],[80,enemybase or {},enemy,(trees or [{},{}])[1]]]}
    return [{'action':(actions or [{},{}])[0],'status':status,'observation':obs},
            {'action':(actions or [{},{}])[1],'status':status,'observation':{'player':1}}]

def result(a,b, **overrides):
    return analyze_events([a,b],0,configuration={**DEFAULT_RULES,**overrides})

def ok(r):
    assert r['metrics']['verified_transitions']==1, r['anomalies']
    assert r['metrics']['skipped_transitions']==0
    json.dumps(r,allow_nan=False)
    return r['metrics']

cases=[]
r=result(state(0,{'a':[0,100,'COLLECTOR']},{'b':[2,10,'COLLECTOR']}),
         state(1,{}, {'b':[1,110,'COLLECTOR']},[{'a':'RIGHT'},{'b':'LEFT'}]))
m=ok(r); assert m['collision_own_cargo_to_opponent']==100;cases.append('opponent_capture')
r=result(state(0,{'a':[4,100,'COLLECTOR']},{},ownbase={'base':5}),
         state(1,{'a':[5,0,'COLLECTOR']},{},[{'a':'RIGHT'},{}],ownbase={'base':5}))
m=ok(r);assert m['delivered_carbon']==100;cases.append('ordinary_delivery')
r=result(state(0,{'a':[4,30,'COLLECTOR'],'b':[6,10,'COLLECTOR']},{},ownbase={'base':5}),
         state(1,{'b':[5,0,'COLLECTOR']},{},[{'a':'RIGHT','b':'LEFT'},{}],ownbase={'base':5}))
m=ok(r);assert m['delivered_carbon']==40 and m['collision_own_cargo_retained']==30 and m['collision_own_cargo_to_opponent']==0;cases.append('same_team_capture_then_delivery')
r=result(state(0,{'a':[4,100,'COLLECTOR']},{},ownbase={'base':5}),
         state(1,{'birth':[5,0,'COLLECTOR']},{},[{'a':'RIGHT','base':'RECCOLLECTOR'},{}],ownbase={'base':5}))
m=ok(r);assert m['delivered_carbon']==100 and m['collision_own_cargo_retained']==100 and m['recruits']==1
assert next(e for e in r['events'] if e['kind']=='delivery')['worker_id']=='birth';cases.append('newborn_capture_then_delivery')
r=result(state(0,{'a':[4,0,'COLLECTOR']},{},enemybase={'base':5}),
         state(1,{},{},[{'a':'RIGHT'},{}],enemybase={'base':5}))
m=ok(r);assert m['enemy_base_elimination_count']==1 and m['collision_worker_deaths']==0;cases.append('enemy_base_elimination')
r=result(state(0,{'a':[4,100,'COLLECTOR']},{},ownbase={'base':5}),
         state(1,{'a':[5,0,'COLLECTOR']},{},[{'a':'RIGHT'},{}],ownbase={'base':5}),moveCost=.1,smeltCost=5)
m=ok(r);assert m['delivery_net_cash']==85;cases.append('move_and_smelt_cost')
r=result(state(0,{'a':[4,100,'COLLECTOR']},{}),
         state(1,{'a':[5,60,'COLLECTOR']},{},[{'a':'RIGHT'},{}]))
m=ok(r);assert m['delivery_count']==0 and m['collision_count']==0;cases.append('cargo_drop_is_not_delivery')
r=result(state(0,{'a':[0,100,'COLLECTOR']},{'b':[2,100,'COLLECTOR']}),
         state(1,{},{},[{'a':'RIGHT'},{'b':'LEFT'}]))
m=ok(r);assert m['collision_own_cargo_to_ground']==100;cases.append('tied_collision_ground')
r=result(state(0,{'a':[0,100,'COLLECTOR']},{'b':[2,100,'COLLECTOR']},trees=[{'t':[1,10]},{}]),
         state(1,{},{},[{'a':'RIGHT'},{'b':'LEFT'}],trees=[{'t':[1,11]},{}]))
m=ok(r);assert m['collision_own_cargo_retained']==100 and m['collision_enemy_cargo_captured']==100;cases.append('tied_collision_tree')
for status in ('ERROR','INVALID','TIMEOUT'):
 r=result(state(0,{'a':[4,100,'COLLECTOR']},{}),state(1,{},{},status=status))
 assert r['metrics']['skipped_transitions']==1 and not r['events'];cases.append('reject_'+status)
r=result(state(0,{'a':[4,100,'COLLECTOR']},{}),state(1,{},{}))
assert r['metrics']['skipped_transitions']==1 and not r['events'];cases.append('unexplained_death_excluded')
r=result(state(0,{'a':[4,100,'COLLECTOR']},{}),state(2,{'a':[4,100,'COLLECTOR']},{}))
assert r['metrics']['skipped_transitions']==1 and not r['events'];cases.append('step_gap_excluded')
a=state(99,{'a':[4,100,'COLLECTOR']},{},ownbase={'base':5});b=state(100,{'a':[5,0,'COLLECTOR']},{},[{'a':'RIGHT'},{}],ownbase={'base':5})
r=result(a,b);ok(r);assert r['windows'][0]['metrics']['delivery_count']==1 and r['windows'][1]['metrics']['delivery_count']==0;cases.append('source_step_window')
a[0]['observation']['full_carbon']=[float('nan')]*225;b[0]['observation']['carbon']='do not inspect';r2=result(a,b)
assert r==r2;cases.append('hidden_map_ignored')
print(json.dumps({'passed':len(cases),'cases':cases},indent=2))
