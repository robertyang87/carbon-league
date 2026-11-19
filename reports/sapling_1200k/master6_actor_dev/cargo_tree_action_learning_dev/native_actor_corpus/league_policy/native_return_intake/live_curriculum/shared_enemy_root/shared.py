"""Commit one real opponent response at a simultaneous-action root and reuse only that response."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'terminal_pairs/run.py'
spec=importlib.util.spec_from_file_location('sapling_registered_terminal_pairs',NATIVE)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
fixture,play,original_state,student_state=runner.fixture,runner.play,runner.original_state,runner.student_state
require=runner.require


def record_transition(env,live,bank,terrain,seat,own,enemy,event,request_id,shared):
    t=env.states[0].observation.step
    before=fixture.public_hashes(live,env,bank)
    orders=[None,None];orders[seat]=own['order'];orders[1-seat]=enemy['order']
    after=bank.copy()
    for side,measurement in ((seat,own),(1-seat,enemy)):
        after[side]-=max(measurement['wall']-1.,0)
    env.step(orders)
    return after,dict(step=t,public_before_by_seat=before,orders=orders,bank_after=after.copy(),
                      public_after_by_seat=fixture.public_hashes(live,env,after),
                      board_after_sha256=play.board_hash(env,terrain),student=own,original=enemy,
                      student_event=event,native_request_id=request_id,shared_root=shared)


class SharedEnemyRoot:
    """Reusable within one live process; both student arms condition on one genuine enemy action."""
    def __init__(self,position,cfg,live,seat,terrain,root_menu,request):
        self.position,self.cfg,self.live,self.seat,self.terrain=position,cfg,live,seat,terrain
        self.root_menu=deepcopy(root_menu)
        env,original,student=position.env,position.orig,position.student
        bank=position.restore()
        hashes=fixture.public_hashes(live,env,bank)
        board=play.board_hash(env,terrain)
        # The callable accepts only the original observer's observation and public rules.
        # In particular, neither student menu/order nor future under either action can reach it.
        enemy,request_id=request(live.observe(env,1-seat,bank[1-seat]),cfg)
        self.post=original_state.take(original.mod)
        self.enemy=deepcopy(enemy);self.request_id=request_id
        self.after=original_state.fingerprint(original.mod)
        require(self.after!=position.fingerprint['original'],'Native opponent did not advance')
        require(board==play.board_hash(env,terrain) and hashes==fixture.public_hashes(live,env,bank) and
                position.fingerprint['student']==student_state.fingerprint(student),'Enemy request advanced own board or student')
        self.receipt=dict(pre_fingerprints=position.fingerprint,pre_public_by_seat=hashes,pre_board_sha256=board,
                          original_observation_sha256=hashes[1-seat],post_original_fingerprint=self.after,
                          root_native_call=self.enemy,request_id=request_id)
        position.restore()

    def issue(self,index):
        """Restore the root, choose the physical student order and install the genuinely observed enemy successor."""
        pos=self.position;seat=self.seat
        bank=pos.restore()
        before_global=original_state.h(original_state.rng(pos.orig.mod))
        own,ev=runner.action(pos.student,self.live.observe(pos.env,seat,bank[seat]),self.cfg,choice=index)
        require(ev is not None and all(ev[k]==self.root_menu[k] for k in
                ('options','p','unforced_choice','slot','actions','prefix','legal','native_logits','route','uid'))
                and ev['pick']==index and own['order']==ev['options'][index],'Different student support')
        require(before_global==original_state.h(original_state.rng(pos.orig.mod)) and
                original_state.fingerprint(pos.orig.mod)==pos.fingerprint['original'],
                'Student touched the shared original/process-global random state')
        own_after=student_state.fingerprint(pos.student)
        original_state.restore(pos.orig.mod,self.post)
        actual=original_state.fingerprint(pos.orig.mod)
        require(actual==self.after and own_after==student_state.fingerprint(pos.student),
                'Post-original continuation does not match the captured root action')
        bank,row=record_transition(pos.env,self.live,bank,self.terrain,seat,own,deepcopy(self.enemy),ev,self.request_id,True)
        return bank,row,dict(pre_original_fingerprint=pos.fingerprint['original'],
                             restored_post_original_fingerprint=actual,root_event=ev)
