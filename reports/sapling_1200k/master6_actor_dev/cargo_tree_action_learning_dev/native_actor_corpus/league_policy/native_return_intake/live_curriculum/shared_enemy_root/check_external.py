"""Exercise the external callback on the exact already published t12 board, without playing another game."""
import os
from pathlib import Path
from shared import HERE,fixture,original_state,play,runner,SharedEnemyRoot,require,student_state
from external_menu import issue_external

def run():
    old=HERE/'receipt.json'
    require(fixture.hfile(old)=='eeea0a97976d0688b4c404aeabeda20e32ef31ab7a41e0f170283e025afae994','Wrong exposed source')
    src=play.read(old);seat,terrain=src['seat'],src['terrain'];roster=play.read(play.HERE/'protocol.json')['actors']
    env,cfg,live=play.official(src['seed'],terrain,seat);bank=[5.,5.]
    orig=play.audit_menu.collect.Player(play.ROOT/roster['original']['file']);student=play.audit_menu.collect.Player(play.ROOT/roster['student']['file'])
    # Warm on the public prefix and commit the archived actually issued orders to the carbon belief.
    # Original private entropy and cost meter are fresh in this adapter check, not claimed to be archived.
    for r in src['prefix']:
        require(fixture.public_hashes(live,env,bank)==r['public_before_by_seat'],'Wrong public warm-up')
        own,_=runner.action(student,live.observe(env,seat,bank[seat]),cfg)
        require(own['order']==r['orders'][seat],'Expected deterministic pre-menu prefix')
        enemy=live.observe(env,1-seat,bank[1-seat]);runner.action(orig,enemy,cfg,original=True)
        orig.mod._ST['tr'].set_actions(enemy['players'],r['orders'][1-seat]);env.step(r['orders']);bank=r['bank_after']
    require(play.board_hash(env,terrain)==src['shared']['pre_board_sha256'],'Wrong physical root')
    pos=fixture.LivePosition(env,orig,student,bank)
    cmd,event=runner.action(student,live.observe(env,seat,bank[seat]),cfg)
    require(event is not None,'Lost known support')
    opts=[cmd['order'],event['options'][1-event['pick']]]
    menu=dict(options=opts,parent_after_sha256=student_state.fingerprint(student),
              parent_rng_after_sha256=play.student_history.behavior_rng_snapshot(student))
    calls=[]
    def response(obs,config):
        out,_=runner.action(orig,obs,config,original=True);calls.append(out)
        return out,len(calls)-1
    common=SharedEnemyRoot(pos,cfg,live,seat,terrain,menu,response)
    def policy(player,obs,config,index):
        actual,native=runner.action(player,obs,config)
        require(actual['order']==opts[0] and native['options']==event['options'],'External baseline was not the canonical natural parent')
        actual['order']=opts[index]
        return actual,dict(pick=index,options=opts,native_parent_order=opts[0],native_natural_pick=native['pick'])
    rows=[];poisoned={}
    for idx in (0,1):
        bank,row,proof=issue_external(common,idx,policy)
        rows.append(dict(order=row['orders'][seat],enemy=row['orders'][1-seat],request_id=row['native_request_id'],
                         proof=proof,board_after_sha256=row['board_after_sha256']))
    require(rows[0]['order']!=rows[1]['order'] and rows[0]['enemy']==rows[1]['enemy'] and
            rows[0]['proof']['student_after_sha256']==rows[1]['proof']['student_after_sha256'],'Wrong physical alternatives')
    for issue in ('wrong_order','wrong_student_state','wrong_env_state','wrong_original_state'):
        def bad(player,obs,configuration,index):
            out,ev=policy(player,obs,configuration,index)
            if issue=='wrong_order':out['order']=opts[1-index]
            elif issue=='wrong_student_state':player.mod._behavior_rng.random()
            elif issue=='wrong_env_state':env.states[0].observation.full_carbon[0]+=.01
            else:orig.mod._S_COST[0]+=.01
            return out,ev
        try: issue_external(common,0,bad)
        except ValueError as error:poisoned[issue]=str(error)
        else:raise ValueError('Accepted invalid callback '+issue)
    pos.restore();require(len(calls)==1,'Repeated real enemy root request during callback validation')
    return dict(schema='sapling-shared-enemy-external-callback-audit/1',source_sha256=fixture.hfile(old),seed=src['seed'],seat=seat,
                implementation_sha256=fixture.hfile(HERE/'external_menu.py'),auditor_sha256=fixture.hfile(Path(__file__)),
                root_step=src['root_step'],source_board_sha256=src['shared']['pre_board_sha256'],
                strict_parent_after_state_and_rng=True,real_root_calls=len(calls),new_full_games=0,terminal_labels=0,
                arms=rows,rejected=poisoned,source_scope='Already recorded physical prefix, freshly warmed examiner private entropy and timing.')

if __name__=='__main__':
    os.sched_setaffinity(0,{2,3});out=HERE/'external_audit.json'
    require(not out.exists(),'Existing engineering receipt')
    v=run();play.save(out,v);print(fixture.hfile(out))
