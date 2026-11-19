"""Check the envelope gate and clock logic with synthetic workers; no candidate or opponent is loaded."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

import live


def check():
    receipt=json.loads((live.HERE/'preflight/receipt.json').read_text())
    assert receipt['protocol_sha256']==live.sha(live.HERE/'protocol.json')
    assert receipt['action_calls']==[0,0] and receipt['strategy_initializations']==0
    assert live.eligibility(receipt['probes'],receipt['containers'])==receipt['blockers']
    assert receipt['status']=='BLOCKED_BEFORE_AGENT_LOAD' and not receipt['counted']
    probes=deepcopy(receipt['probes'])
    for i,p in enumerate(probes):
        p.update(cgroup={'cpu.max':'150000 100000','memory.max':'536870912','memory.swap.max':'0'},
                 cgroup_inode=1000+i,affinity=[0,1],interfaces=['lo'],python=[3,12,16],numpy='1.26.4',
                 torch='2.5.1+cpu',helper_sha256=live.PLAN['files']['official_helpers']['sha256'])
    assert live.eligibility(probes,receipt['containers'])==[]
    for key,val in [('cpu.max','400000 100000'),('memory.max','17179869184'),('memory.swap.max','max')]:
        altered=deepcopy(probes);altered[1]['cgroup'][key]=val
        assert live.eligibility(altered,receipt['containers'])
    assert live.allowance(0,5)==3 and live.allowance(3,5)==1
    assert live.allowance(1,0)==1 and live.allowance(2,0.25)==1.25
    assert live.failures([{'status':'ok'},{'status':'ok'}]) is None
    for seat in (0,1):
        calls=[{'status':'ok'},{'status':'ok'}];calls[seat]['status']='timeout'
        assert live.failures(calls)==dict(status='forfeit',counted=True,winner_seat=1-seat,failed_seat=seat,reason='timeout')
    assert live.failures([{'status':'error'},{'status':'timeout'}])['status']=='bilateral_failure'
    env,cfg=live.official()
    for s in (0,1):
        o=live.observe(env,s,5)
        assert set(o)=={'step','player','carbon','players','trees','remainingOverageTime'} and 'randomSeed' not in cfg
        assert len(o['carbon'])==225 and o['player']==s and cfg['actTimeout']==1
        invalid={'status':'ok','response':{'action':{'enemy-unit':'UP'},'step':0}}
        live.check_action(invalid,o);assert invalid['status']=='error'
    return {'observed_preflight_reproduced':True,'synthetic_valid_cgroup_accepted':True,
            'cpu_memory_swap_mutations_rejected':True,'allowance_boundaries_and_forfeits_verified':True,
            'opening_official_terrain_and_both_seat_public_fields_verified':True,'unknown_unit_rejected':True}


async def exchange_test():
    responses=[]
    for delay,limit in [(0,1),(.18,.06)]:
        src='import sys,json,time; print("{}",flush=True); sys.stdin.readline(); time.sleep('+str(delay)+'); print("{}",flush=True)'
        child=await asyncio.create_subprocess_exec(sys.executable,'-u','-c',src,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
        try:
            assert (await live.exchange(child,timeout=2))['status']=='ok'
            responses.append((await live.exchange(child,{'ping':True},limit))['status'])
        finally:
            if child.returncode is None:child.kill()
            await child.wait()
    assert responses==['ok','timeout']
    return responses


if __name__=='__main__':
    result=check();result['synthetic_round_trip_and_delayed_reply']=asyncio.run(exchange_test())
    result['actual_agent_calls']=0
    print(json.dumps(result,indent=2))
