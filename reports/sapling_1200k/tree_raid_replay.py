"""Compare one-step hybrid actions at historical roots facing recorded original Master3s2.

Neither counterfactual arm drives the replay. Any follow-up dependence of the
opponent on our different move is unknown, so this measures no win-rate change.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import tree_audited_game as audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = ("tree_exam_it50", "tree_exam_v4r", "tree_exam_basecal", "tree_exam_win_cash")


def targets(obs, cfg, F, model):
    if not 50 <= obs['step'] < 250:
        return False
    if obs['players'][1-obs['player']][0] < cfg['seizeCost']:
        return False
    return any(model.tree_raid_rows(obs, F, cfg, approaches=b) for b in (False, True))


def evaluate_game(mod, row, raw, verify_commands=False):
    seed, seat = row['seed'], row['candidate_seat']
    random.seed(7919*seed+104729*seat+31)
    sim = audit.exam.FastCarbon(raw['game_configuration']).reset(seed=seed)
    F = sys.modules['cnet2_feat']
    tally = Counter()
    cases = []
    for t, actions in enumerate(raw['actions']):
        obs = sim.obs(seat)
        if verify_commands or not t or targets(obs, sim.config, F, mod):
            derived = mod.agent(obs, sim.config)
            if verify_commands:
                if derived != actions[seat]:
                    raise ValueError('actual candidate command mismatch')
                tally['verified_candidate_calls'] += 1
            tally['eligible_roots'] += bool(mod._RAID_TARGETS or mod._RAID_APPROACH)
            if mod._RAID_LAST:
                # Reconstruct the identical mixed parent on that same root.
                # Feat.observe has already consumed this frame; avoid observing twice.
                changes = list(mod._RAID_LAST)
                feats_used = mod._RAID_REPLAY_X
                original = mod._FEAT.to_actions(mod._raid_decode(feats_used, mod._FEAT.aux))
                if derived == original:
                    raise ValueError('reported modification did not change commands')
                snapshot = sim.get_state()
                targets_at_root = []
                for slot, old, new, to, stage, target, bonus in changes:
                    world = int(F.CANON[target]) if seat else target
                    tid = next(tid for tid,(cell,_) in obs['players'][seat][3].items() if world == cell)
                    foes = obs['players'][1-seat][2]
                    entrants = [uid for uid,(pos,_,role) in foes.items() if role=='PLANTER' and
                                F.DEST[pos, F.MOVES.index(actions[1-seat].get(uid) or 'NONE')] == world]
                    targets_at_root.append(dict(tree_id=tid,stage=stage,guard=mod._FEAT.own_uid[slot],
                            opponent_attempts_destination=bool(entrants),logit_bonus=bonus))
                for label, action in (('parent',original),('candidate',derived)):
                    branch = list(actions)
                    branch[seat] = action
                    sim.step(*branch)
                    for case in targets_at_root:
                        tree = sim.trees.get(case['tree_id'])
                        own = case['guard'] in sim.workers
                        case[label] = dict(guard_alive=own,tree_owner=(None if tree is None else tree.player),
                            guard_covers_tree_next=bool(own and tree and
                            F.DIST[sim.workers[case['guard']].pos,tree.pos] <= 1))
                    sim.set_state(snapshot)
                tally['changed_roots'] += 1
                cases.append(dict(step=t,cases=targets_at_root))
        else:
            mod._FEAT.observe(obs)
        sim.step(*actions)
    if audit.prior.hashed(sim.get_state()) != row['final_state_sha256']:
        raise ValueError('replay terminal state hash mismatch')
    return dict(seed=seed,seat=seat,tally=dict(tally),raw_sha256=row['raw_sha256'],cases=cases)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--pack',required=True,type=Path)
    cli.add_argument('--out',required=True,type=Path)
    cli.add_argument('--candidate-exam', action='store_true', help='verify every frozen candidate command in its actual exam')
    args=cli.parse_args()
    mod=audit.prior.shared.module(args.pack,'tree_raid_probe_pack')
    run_mixed=mod._mixed_actions
    def capture(x, aux):
        mod._RAID_REPLAY_X=x
        return run_mixed(x,aux)
    mod._mixed_actions=capture
    rows=[]
    for source in (('tree_exam_tree_raid',) if args.candidate_exam else SOURCES):
        d=HERE/source
        for row in json.loads((d/'results.json').read_text())['games']:
            blob=(d/row['raw_file']).read_bytes()
            if hashlib.sha256(blob).hexdigest()!=row['raw_sha256'] or not row['valid']:
                raise ValueError('unaudited replay')
            raw=json.loads(gzip.decompress(blob))
            result=evaluate_game(mod,row,raw,verify_commands=args.candidate_exam)
            result['behavior_source']=source
            rows.append(result)
            print(source,result["seed"],result["seat"],result["tally"],flush=True)
    totals=Counter()
    for r in rows:
        totals.update(r['tally'])
        for turn in r['cases']:
            for c in turn['cases']:
                totals['changed_'+c['stage']]+=1
                totals['opponent_targets_tree']+=c['opponent_attempts_destination']
                totals['aligned_extra_coverage_or_save']+=bool(c['opponent_attempts_destination'] and c['candidate']['guard_alive'] and ((c['candidate']['guard_covers_tree_next'] and not c['parent']['guard_covers_tree_next']) or (c['candidate']['tree_owner']==r['seat'] and c['parent']['tree_owner']!=r['seat'])))
                totals['new_cover_next_step']+=c['candidate']['guard_covers_tree_next'] and not c['parent']['guard_covers_tree_next']
                totals['own_tree_saved_after_one_step']+=c['candidate']['tree_owner']==r['seat'] and c['parent']['tree_owner']!=r['seat']
                totals['own_guard_dead_in_candidate_only']+=not c['candidate']['guard_alive'] and c['parent']['guard_alive']
    report=dict(verify_candidate_exam=args.candidate_exam, scope='one-turn intervention at recorded actual original Master games; full pack-derived parent and modified branch face the same logged opponent command. No trajectory win rate.',
                pack_sha256=hashlib.sha256(args.pack.read_bytes()).hexdigest(),totals=dict(totals),games=rows)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(dict(totals),flush=True)

if __name__=='__main__':
    main()
