"""Independently replay the archived prefix and four actual command tapes with both rule engines."""
import copy
import gzip
import json
import random
import sys
from register import HERE, RAW, ROOT, FREIGHT, IDS, sha, save, require
from run import advance, load, physical


def audit():
    sys.path.insert(0,str(ROOT/'code/env_src/carbon_challenge-master'))
    engine=load('verify_response_actions',FREIGHT)
    freeze=json.loads((HERE/'protocol.json').read_bytes());lock=sha((HERE/'protocol.json').read_bytes())
    result=json.loads((HERE/'results.json').read_bytes());require(result['protocol_sha256']==lock,'Different frozen registration')
    game=json.loads(gzip.decompress(RAW.read_bytes()));frames=game['steps'];cfg=game['configuration']
    random.seed(cfg['randomSeed']);engine.np.random.seed(cfg['randomSeed'])
    env=engine.make('carbon',configuration=cfg);terrain=frames[0][0]['observation']['full_carbon'];env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    sim=engine.econ.FastCarbon(cfg).reset(full_carbon=terrain);sim.cash=engine.econ.CashEvents(sim.cash)
    t0=freeze['step'];banks=[c['observation']['remainingOverageTime'] for c in frames[t0]]
    for t in range(t0):
        physical(env,sim,frames[t]);advance(env,sim,[r['action'] or {} for r in frames[t+1]])
    physical(env,sim,frames[t0]);env.steps[:-1]=[None]*t0;env.logs=[]
    public_root=engine.official_state.take(env,banks);base=copy.deepcopy(sim);checks=[];native={}
    for g in result['results']:
        raw=(HERE/g['artifact']).read_bytes();require(sha(raw)==g['sha256'],'Different action tape bytes')
        rec=json.loads(gzip.decompress(raw));k=rec['ticket']['stream'];base_actions=rec['shared_root_orders']
        if k in native: require(native[k]==(base_actions,rec['history']),'Arms did not share native root state and draws')
        else:native[k]=(base_actions,rec['history'])
        bank=engine.official_state.restore(env,public_root);ghost=copy.deepcopy(base);attribution=[0.,0.]
        for offset,r in enumerate(rec['rows']):
            t=t0+offset;sides=[engine.public(env,s,bank) for s in (0,1)]
            require(r['step']==t and r['public_sha256']==[engine.digest(v) for v in sides],'Pre-action request differs')
            require(r['orders'][1]==r['native_orders'][1],'Rival order overwritten')
            ref=copy.deepcopy(r['native_orders'][0])
            if g['arm']=='three_moves' and offset<3 and IDS[0] in sides[0]['players'][0][2]:
                cmd=freeze['v4_action_override'][offset]
                if cmd=='NONE':ref.pop(IDS[0],None)
                else:ref[IDS[0]]=cmd
            require(ref==r['orders'][0],'Unexpected extra own intervention')
            advance(env,ghost,r['orders']);require(r['cash']==list(ghost.cash),'Different resulting cash state')
            for s in (0,1):
                require(r['post_workers'][s]==ghost.obs(s)['players'][s][2].get(IDS[s]),'Different worker state')
                attribution[s]+=sum(e[2] for e in ghost.cash.events if e[0]=='own_delivery' and e[1]==s and e[3]==IDS[s])
            bank=[b-max(0,c['wall']-1) for b,c in zip(bank,r['clocks'])]
        require(env.done and len(rec['rows'])==32 and [s.reward for s in env.states]==g['score'],'Independent score differs')
        for s in (0,1):engine.econ.check_equal(attribution[s],g['focus'][s]['cash_to_own_base'],'Worker delivery')
        checks.append({'stream':k,'arm':g['arm'],'verified':len(rec['rows']),'score':g['score']})
    require(len(checks)==4 and len(native)==2,'Changed four-arm denominator')
    save(HERE/'audit.json',{'protocol_sha256':lock,'restored_prefix_transitions':t0,'verified_suffixes':checks})
if __name__=='__main__':audit()
