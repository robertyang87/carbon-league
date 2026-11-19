"""Verify prior and new source matches, physical CPU crossover, both engines, and one-step causality scope."""
import gzip
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('opening_pair_native',HERE/'native.py')
arena=importlib.util.module_from_spec(spec);spec.loader.exec_module(arena)
from shared import ROOT,read,require,save,sha
from clock import verify_completed


def run():
    plan=read(HERE/'protocol.json');b=read(HERE/'games/binding.json');oldb=read(arena.CTRL/'binding.json');cmp=read(HERE/'comparison.json');gate=read(arena.NEW/'normal_entry.json.gz')
    require(sha(HERE/'protocol.json')==b['registered_sha256'] and gate['decision']=='PASS' and gate['nonopening_differences_vs_frozen']==0,'gate identity')
    require({name:sha(ROOT/name) for name in b['checked_driver_sha256']}==b['checked_driver_sha256'],'driver changed after binding')
    newtest=arena.driver(arena.OUT,arena.pack());rows=[]
    twin=arena.module(arena.ACTOR/'endgame_queue_dev/study.py','opening_transition_comparison')
    require(len(cmp['pairs'])==len(plan['historical_baseline'])==4 and len(list((arena.OUT/'candidate').glob('raw_*.json.gz')))==4,'exact sample count')
    for src,rec in zip(plan['historical_baseline'],cmp['pairs']):
        s=src['seed'];seat=src['seat'];require((s,seat)==(rec['seed'],rec['seat']),'map/seat order')
        old=arena.historical(src,oldb);new=read(arena.OUT/'candidate'/f'receipt_{s}_{seat}.json');row=new['game'];dst=arena.OUT/'candidate'/row['raw_file']
        verify_completed(newtest,('candidate',s,seat,src['cpu'],src['terrain'],b));require(new['official']==newtest.official.check(arena.OUT/'candidate',row),'official source or status')
        raw=[read(ROOT/src['files']['raw']['path']),read(dst)]
        require(raw[0]['initial_full_carbon']==raw[1]['initial_full_carbon'] and raw[0]['initial_terrain_u8_sha256']==src['terrain']==raw[1]['initial_terrain_u8_sha256'],'physical board')
        for name,rr,proof in zip(('parent','candidate'),raw,(old,new)):
            require(rr['official_score_by_seat']==proof['official']['official_reward_by_seat'] and rr['result']==rec['arms'][name]['result'] and rr['margin']==rec['arms'][name]['official_margin'],'score identity')
        orders=[g['actions'][0] for g in raw];own=[twin.clean(o[seat]) for o in orders];enemy=[twin.clean(o[1-seat]) for o in orders]
        require(own[0]!=own[1],'adapter failed to change real opening on this map/seat')
        cells={}
        for i,a in enumerate(('historical','new')):
            for j,o in enumerate(('historical','new')):
                sim=twin.FastCarbon(raw[0]['game_configuration']).reset(seed=s);sim.cash=twin.CashEvents(list(sim.cash));pair=[{},{}];pair[seat]=own[i];pair[1-seat]=enemy[j]
                cells[a+'/'+o]=twin.twin_step(sim,pair)
        d=[twin.difference(cells['historical/'+o],cells['new/'+o],seat) for o in ('historical','new')]
        rows.append(dict(seed=s,seat=seat,cpu=src['cpu'],historical_and_new_contemporaneous_enemy_orders_identical=enemy[0]==enemy[1],
            first_orders_by_arm=orders,first_resolved_state_own_effect_under_each_actual_enemy=d,one_step_official_cases=cells,
            terminal_margin_delta=rec['delta_official_margin'],historical_raw_sha256=src['files']['raw']['sha256'],new_raw_sha256=sha(dst)))
    save(HERE/'same_opening.json.gz',{'cases':rows,'historical_natural_rng_unpaired':True,'full_official_game_difference_is_descriptive':True})
    differences=[r['terminal_margin_delta'] for r in rows]
    require(abs(statistics.mean(differences)-cmp['paired_delta'])<1e-7,'paired difference')
    return rows


def plan_identity(k):
    return read(HERE/'protocol.json')[k]


def finalize():
    run()
    for p in ['parity.json.gz','economy.json.gz','RESULTS.md']:require((HERE/p).exists(),'waiting for '+p)
    parity=read(HERE/'parity.json.gz');econ=read(HERE/'economy.json.gz');check=read(HERE/'comparison.json')
    actual=sum(r['arms']['candidate']['steps'] for r in check['pairs'])
    require(parity['exact_receipts']==4 and parity['complete_natural_candidate_turns']==actual and parity['post_opening_turns']==actual-4 and parity['parent_sha256']==plan_identity('baseline_sha256') and parity['candidate_sha256']==plan_identity('candidate_pack_sha256'),'incorrect matched trajectory provenance')
    require(econ['new_original_games']==4 and econ['formal_independent_games']==0 and econ['input_hashes']['comparison']==sha(HERE/'comparison.json') and abs(econ['paired_delta']-check['paired_delta'])<1e-7,'cash panel linkage')
    files=[p for p in sorted(HERE.rglob('*')) if p.is_file() and not p.name.startswith('.') and p.name not in ['verification.json'] and '__pycache__' not in p.parts
           and p.suffix not in ('.log','.tmp','.part')]
    cmp=read(HERE/'comparison.json')
    result={'files':{str(p.relative_to(HERE)):sha(p) for p in files},'files_excluding_receipt':len(files),'bytes_excluding_receipt':sum(p.stat().st_size for p in files),
        'same_history_nonopening_change_count':parity['post_opening_parent_candidate_differences'],'disabled_control_changes':parity['disabled_control_differences'],
        'new_games':4,'prior_games':4,'independent_formal_games':0,'new_steps':sum(r['arms']['candidate']['steps'] for r in cmp['pairs']),
        'old_steps':sum(r['arms']['parent']['steps'] for r in cmp['pairs']),'official_paired_margin_difference':cmp['paired_delta']}
    save(HERE/'verification.json',result);print(len(files),sha(HERE/'verification.json'))


if __name__=='__main__':
    run() if '--same-opening' in sys.argv else finalize()
