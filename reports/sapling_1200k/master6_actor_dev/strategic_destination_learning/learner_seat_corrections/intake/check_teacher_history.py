"""Exercise the original belief methods on retained training commands; never call the original policy."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
STRATEGIC=HERE.parents[1]
ROOT=STRATEGIC.parents[3]
LIVE=STRATEGIC.parent/'cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake/live_curriculum'
sys.path.insert(0,str(LIVE))
import play

SHA='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
DATA=STRATEGIC/'delivery_interception/native_response/outputs/621758460_0'


def physical_destinations(players,seat,order):
    vectors={'NONE':(0,0),'UP':(-1,0),'RIGHT':(0,1),'DOWN':(1,0),'LEFT':(0,-1)}
    expected={}
    for uid,(pos,_,_) in players[seat][2].items():
        y,x=divmod(pos,15)
        dy,dx=vectors[order.get(uid) or 'NONE']
        expected[uid]=((y+dy)%15)*15+(x+dx)%15
    return expected


def run():
    source=(ROOT/'data/master3s2_source.py').read_bytes()
    assert hashlib.sha256(source).hexdigest()==SHA
    lines=source.decode().splitlines()
    original={}
    # The signed source up to the tracker, and its own further class-alias declarations.
    exec(compile('\n'.join(lines[:502]+lines[556:560]),'original-tracker','exec'),original)
    seat=1
    tracker=original['BeliefTracker'](original['_Bel'],seat)
    idempotent=original['BeliefTracker'](original['_Bel'],seat)
    def tape(name):
        return [json.loads(line) for line in gzip.decompress((DATA/(name+'.jsonl.gz')).read_bytes()).splitlines()]
    rows=tape('prefix')+tape('A')
    terrain=json.loads((DATA/'claim.json').read_bytes())['terrain']
    env,_,live=play.official(621758460,terrain,seat)
    native=json.loads(gzip.decompress((DATA/'native_requests.json.gz').read_bytes()))
    bank=[5.,5.]; example=None; n_workers=0
    for t,r in enumerate(rows):
        public=live.observe(env,seat,bank[seat])
        assert play.sha(live.encoded(public))==r['public_before_by_seat'][seat]
        players=public['players']; own=players[seat][2]
        obs=(players,t,public['carbon'])
        assert tracker.step(*obs)==idempotent.step(*obs)
        executed=r['orders'][seat]; req=native[r['native_request_id']]
        assert executed==r['original']['order']==req['order'] and req['step']==t
        tracker.set_actions(players,executed)
        idempotent.set_actions(players,executed)
        idempotent.set_actions(players,executed)
        expected=physical_destinations(players,seat,executed)
        assert tracker.last_dest==idempotent.last_dest==expected
        n_workers+=len(expected)
        env.step(r['orders']);bank=r['bank_after']
        next_obs=live.observe(env,seat,bank[seat])
        assert play.sha(live.encoded(next_obs))==r['public_after_by_seat'][seat]
        if example is not None or t+1==len(rows):
            continue
        next_input=(next_obs['players'],t+1,next_obs['carbon'])
        for uid in sorted(set(own)-set(next_obs['players'][seat][2])):
            for alternate in ('NONE','UP','RIGHT','DOWN','LEFT'):
                if alternate==(executed.get(uid) or 'NONE'):continue
                proposed={**executed,uid:alternate}
                correct=copy.deepcopy(tracker); wrong=copy.deepcopy(tracker); corrected=copy.deepcopy(tracker)
                wrong.set_actions(players,proposed); corrected.set_actions(players,proposed)
                corrected.set_actions(players,executed)
                got=correct.step(*next_input); drift=wrong.step(*next_input); reset=corrected.step(*next_input)
                assert got==reset and play.state_hash(correct.bel)==play.state_hash(corrected.bel)
                diff=[(i,a,b) for i,(a,b) in enumerate(zip(got[0],drift[0])) if abs(a-b)>1e-8]
                if diff:
                    example=dict(step=t,physical_seat=seat,uid=uid,disappears_at=t+1,
                                 executed_original_action=executed.get(uid) or 'NONE',illustrative_unissued_action=alternate,
                                 actual_destination=expected[uid],alternative_destination=wrong.last_dest[uid],
                                 public_next_hash=r['public_after_by_seat'][seat],original_request_id=req['id'],
                                 physical_before_public_hash=r['public_before_by_seat'][seat],
                                 changed_next_estimate_cells_actual_vs_uncorrected=diff,
                                 corrected_estimate_and_internal_belief_match=True,
                                 caveat='The differing order is a deliberately injected legal direction on the recorded original path, not a measured shadow teacher proposal.')
                    break
            if example:break
    assert example and env.done and play.state_hash(tracker.bel)==play.state_hash(idempotent.bel)
    return dict(source_sha256=SHA,github_exemplar_path='sources/master3s2/master3s2.py',git_blob='9c46f6fa4ff401b05939150054c2f88f549c2f7e',
                source_revision='99ec3fed8f48158b85399f6887bf62036205dabe',
                source_facts={'actual_public_history':'574–579,625 initialize persistent seat-specific tracker and RNG; prev is the last actual observed players. 99–103 populate previous-positions feature planes 14–17. 620 and 1302 pass this previous observed board into search.',
                              'intended_action_history':'624 records post-search order; 226–231 maps every currently owned worker to its intended destination (default NONE). 223,385,402–449 use this map when a friendly worker disappears, including attribution of enemy casualties, where carbon was dropped, and cells excluded from regrowth (492–499).',
                              'independent_randomness_and_cost':'553,574–577,589–599: _SEED=None and a private RandomState sampled for bases and workers. 1267–1278,1320–1322 and 1394: search uses local tactical seeds and live measured cost for its time-budget decisions.',
                              'public_abi':'629–636 accepts only observation and configuration; no executable-order feedback argument. Exceptions can return {} before cache advancement; 618–623 can swallow search failure. Capture telemetry and reject a missed teacher transition instead of assuming {} means successful execution.'},
                conclusion={'unsynchronized_public_shadow':'Feeding only causal learner observations can leave the teacher believing that its proposed action was executed. An observed dead worker has no public next position from which to infer that last order.',
                            'valid_training_adapter':'Call one persistent, isolated original per learner seat/trajectory on every pre-action observation to get its natural proposal. Preserve it and its search telemetry. Once the actual own order is finalized, call _ST["tr"].set_actions(before_players, actual_order) on this same instance before the next public observation. Preserve _ST["prev"], live policy RNG, real search wall clocks, and other state. Assert expected intended destinations from independent physical torus arithmetic each time.',
                            'scope':'This aligns the action-conditioned belief to executed behavior at each successful step without modifying the source. It is an explicit white-box training adapter, not the public interface alone. Its labels are conditional on learner-visited histories; independent original self-play returns, entropy draws, costs, and exact repeatable choices do not follow.'},
                fixture={'seed':621758460,'actor_seat':seat,'role':'The recorded original controlled this seat throughout. No new match, rollouts or original inference.',
                         'steps':len(rows),'executed_worker_destination_checks':n_workers,
                         'identical_action_rewrite_all_steps':True,'matched_final_belief':True,
                         'data':str(DATA.relative_to(ROOT)),'source_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ('prefix.jsonl.gz','A.jsonl.gz','native_requests.json.gz')},
                         'different_action_case':example},checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    doc=run()
    (HERE/'teacher_history.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
