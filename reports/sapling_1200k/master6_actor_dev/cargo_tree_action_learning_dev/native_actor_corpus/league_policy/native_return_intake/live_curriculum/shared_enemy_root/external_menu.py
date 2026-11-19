"""Issue an independently preregistered pair of full public orders against one shared native response.

Construct SharedEnemyRoot first. Its root_menu must additionally sign the exact
normal parent post-action student and private-RNG fingerprints; produce both when
selecting the physical proposals, prior to observing either continuation.
"""
import copy
import math
import time
from shared import fixture,original_state,play,record_transition,require,runner,student_state


def check_proposal(menu,index,issued,event,student_after,rng_after):
    options=menu['options']
    require(index in (0,1) and len(options)==2 and all(isinstance(v,dict) for v in options) and options[0]!=options[1],
            'Expected two physically different preregistered joint orders')
    require(event is not None and event['options']==options and event['pick']==index and issued==options[index],
            'External callback did not issue the signed full physical order')
    require(student_after==menu['parent_after_sha256'] and rng_after==menu['parent_rng_after_sha256'],
            'External callback changed student hidden state or its natural random advancement')


def issue_external(common,index,student_action):
    """Run callback(player, own_public_obs, public_cfg, index), advancing a single joint transition.

    Returns bank, transition row, proof just like SharedEnemyRoot.issue(). The
    measured local student wall covers the entire callback, including alternate
    order construction. Besides verifying the sealed full orders, the caller is
    responsible for its own menu's legality, causal feature provenance and training eligibility.
    """
    pos,seat=common.position,common.seat
    bank=pos.restore()
    menu=copy.deepcopy(common.root_menu)
    require(menu.get('parent_after_sha256') and menu.get('parent_rng_after_sha256'),'Missing normal parent state evidence')
    need_public=common.receipt['pre_public_by_seat']
    require(fixture.public_hashes(common.live,pos.env,bank)==need_public,'Different root public observation')
    world=fixture.official_state.fingerprint(pos.env,bank)
    original_pre=pos.fingerprint['original']
    rng_pre=original_state.h(original_state.rng(pos.orig.mod))
    public=common.live.observe(pos.env,seat,bank[seat])
    wall,cpu=time.perf_counter(),time.process_time()
    measurement,event=student_action(pos.student,public,common.cfg,index)
    wall,cpu=time.perf_counter()-wall,time.process_time()-cpu
    require(isinstance(measurement,dict) and isinstance(measurement.get('order'),dict) and
            all(math.isfinite(measurement[k]) and measurement[k]>=0 for k in ('wall','cpu')),
            'Missing local callback timing or physical order')
    own=dict(measurement,callback_reported_wall=measurement['wall'],callback_reported_cpu=measurement['cpu'],wall=wall,cpu=cpu)
    student_after=student_state.fingerprint(pos.student)
    rng_after=play.student_history.behavior_rng_snapshot(pos.student)
    check_proposal(menu,index,own['order'],event,student_after,rng_after)
    require(common.root_menu==menu and rng_pre==original_state.h(original_state.rng(pos.orig.mod)) and
            original_state.fingerprint(pos.orig.mod)==original_pre and
            fixture.official_state.fingerprint(pos.env,bank)==world and fixture.public_hashes(common.live,pos.env,bank)==need_public,
            'Callback changed the signed menu, public board, bank, original or shared RNG')
    original_state.restore(pos.orig.mod,common.post)
    installed=original_state.fingerprint(pos.orig.mod)
    require(installed==common.after and student_state.fingerprint(pos.student)==student_after,
            'Genuine original post-call state or advanced student state differs')
    bank,row=record_transition(pos.env,common.live,bank,common.terrain,seat,own,
                               copy.deepcopy(common.enemy),event,common.request_id,True)
    return bank,row,dict(pre_original_fingerprint=original_pre,restored_post_original_fingerprint=installed,
                         root_event=event,issuer='published_external_menu',student_after_sha256=student_after,
                         student_rng_after_sha256=rng_after,root_own_public_sha256=need_public[seat])
