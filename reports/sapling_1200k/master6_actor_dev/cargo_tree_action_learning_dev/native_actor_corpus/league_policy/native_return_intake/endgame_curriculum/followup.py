"""Follow each frozen public root down its single recorded native continuation."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter, process_time
from tempfile import TemporaryDirectory
import zipfile
import roots as selection

HERE,ROOT = selection.HERE,selection.ROOT
ROOTS_SHA = '56fa0afac4a99339bbdf98ed229debb64c9566ad64dc8e8046eee22a26cacc17'
ROOTS_COMMIT = 'd298557b91d1f242dd87b1d6410dc57374ac1a7e'


def outcomes():
    raw = (HERE/'roots.json').read_bytes()
    selection.need(selection.digest(raw)==ROOTS_SHA and raw==subprocess.check_output(['git','-C',str(ROOT),'show',
        ROOTS_COMMIT+':'+str((HERE/'roots.json').relative_to(ROOT))]), 'Unfrozen roots')
    card = json.loads(raw)
    selection.need(card['selection_script_sha256']==selection.digest((HERE/'roots.py').read_bytes()),'Selection code changed')
    source=selection.signed()
    input_raw=(selection.INTAKE/'hosted/training_manifest.json').read_bytes()
    selection.need(selection.digest(input_raw)==source['training_manifest_sha256'],'Signed action probabilities changed')
    manifest=json.loads(input_raw)
    episodes={(e['seed'],e['seat'],e['arm']):e for e in manifest['episodes']}
    refs={(r['seed'],r['seat']):r for r in selection.inventory()}
    results=[]
    for ticket in card['tickets']:
        root=ticket['root']
        if root is None:
            continue
        seed,seat,arm,t=root['seed'],root['seat'],root['arm'],root['step']
        ref=refs[seed,seat]
        ep=episodes[seed,seat,arm]
        selection.need(ep['episode_verified'] and ep['registration_sha256']==root['registration_sha256'] and
                       ep['archive_sha256']==root['archive_sha256'], 'Unbound archived episode')
        start, cpu = perf_counter(),process_time()
        with selection.open_episode(ref,arm) as (env,live,tape,reg,reg_bytes):
            selection.need(selection.digest(reg_bytes)==root['registration_sha256'], 'Wrong registration')
            bank=selection.restore(root,ref,live,env,tape)
            restore_wall,restore_cpu=perf_counter()-start,process_time()-cpu
            events={e['step']:e for e in ep['events'] if e['step']>=t}
            calls=[]
            checkpoints={}
            for frame in tape:
                q=json.loads(frame)
                s=q['step']
                observations=[live.observe(env,p,bank[p]) for p in (0,1)]
                hashes=[selection.digest(live.encoded(v)) for v in observations]
                selection.need(q['public_observation_sha256']==hashes and s==env.states[0].observation.step,
                               'Continuation diverged from physical tape')
                if s==t or s in (t+3,t+5,t+10,275,290):
                    checkpoints[s]=selection.economics(observations[seat],seat)
                if s in events:
                    e=events[s];issued=q['calls'][seat]['response']['action']
                    selection.need(e['public_observation']==observations[seat] and e['public_observation_sha256']==hashes[seat]
                                   and e['options'][e['choice']]==issued and e['issued']==issued, 'Invalid root-menu history')
                    calls.append(dict(step=s,public_sha256=hashes[seat],supported=e['supported'],issued_choice=e['choice'],
                                      probabilities=e['probabilities'],sampled_uid=e.get('uid'),sampled_role=e.get('role')))
                selection.advance(env,q)
                bank=q['overage_remaining_seconds']
            score=[s.reward for s in env.states]
            end=selection.economics(live.observe(env,seat,bank[seat]),seat)
            selection.need(env.done and score==ep['official_scores_by_seat'] and
                           {e['step'] for e in calls}==set(events),'Incorrect terminal continuation')
            results.append(dict(seed=seed,seat=seat,arm=arm,step=t,stratum=root['stratum'],root_state_sha256=root['state_sha256'],
                        root_public_verified_both_seats=True,root_counters_verified=True,terminal_step=env.states[0].observation.step,
                        tape_sha256=ep['tape_sha256'],receipt_sha256=ep['receipt_sha256'],official_scores_by_seat=score,
                        margin=ep['margin'],wdl=ep['wdl'],cash_gain_by_seat=[round(end['cash'][p]-root['cash'][p],6) for p in (0,1)],
                        realized_cash_gap_change=round(end['cash_gap']-root['cash_gap'],6),terminal_economy=end,
                        milestones=checkpoints,scheduled_inclusive=len(calls),supported_inclusive=sum(e['supported'] for e in calls),
                        supported_strictly_after=sum(e['supported'] and e['step']>t for e in calls),events=calls,
                        prefix_restore_wall_s=round(restore_wall,6),prefix_restore_cpu_s=round(restore_cpu,6)))
    output=dict(schema='sapling-frozen-endgame-historical-continuations/1',roots_commit=ROOTS_COMMIT,roots_sha256=ROOTS_SHA,
        training_manifest_sha256=source['training_manifest_sha256'],map_seat_denominator=len(card['tickets']),selected=len(results),
        historical_wins=sum(r['wdl']==1 for r in results),historical_draws=sum(r['wdl']==.5 for r in results),
        historical_losses=sum(r['wdl']==0 for r in results),new_native_games=0,
        roots_with_supported_later_draw=sum(r['supported_strictly_after']>0 for r in results),
        prefix_restore_measurement='One serial CPU worker; includes archive integrity, official initialization and actions 0..t-1 with both public hashes and rule ID counters. Excludes neural/clock-state reconstruction. Times are observational.',
        suffix_fidelity='Original _act updates its hidden-carbon tracker from public observations and the actual returned actions (st[\'tr\'].step and set_actions) and keeps the previous players in st[\'prev\']. Replay those executed historical commands exactly for a future controller warm-start. The official tape does not include original private OS-seeded RandomState or the four moving _S_COST timings used to size the native search. A cold suffix with new entropy and fresh timing explicitly estimates the warm-started continuation policy; exact original process-state replication would require a live snapshot or fresh full-game comparator. These 10 historical continuations present only the actually played actions.',
        results=results)
    # Timestamped wall measurements may vary on independent re-execution. Preserve the original receipt.
    path=HERE/'followup.json'
    if path.exists():
        original=json.loads(path.read_bytes())
        for doc in (original,output):
            for row in doc['results']:
                row.pop('prefix_restore_wall_s')
                row.pop('prefix_restore_cpu_s')
        selection.need(original==output,'Re-executed prefix or outcome differs')
    else:
        selection.save(path,output)
    print(path.name,selection.digest(path.read_bytes()),len(results),output['roots_with_supported_later_draw'],flush=True)


if __name__=='__main__':
    outcomes()
