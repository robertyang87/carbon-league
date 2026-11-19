"""Check same-step material attribution with the real game transition."""
import unittest
from interception_cash import changed_transfers, collision_events
from fastsim import FastCarbon
from eval_v53 import CFG


def setup_state(own, enemy, ours=64, theirs=66):
    state=FastCarbon(CFG).reset(seed=234).get_state()
    state['workers']={'player-0-worker-0':dict(player=0,pos=ours,carbon=own,type='COLLECTOR'),
                      'player-1-worker-0':dict(player=1,pos=theirs,carbon=enemy,type='COLLECTOR')}
    state['wid_counter']=[1,1]
    return state


class InterceptionAccounting(unittest.TestCase):
    mine={'player-0-worker-0':'RIGHT'}
    opponent={'player-1-worker-0':'LEFT'}

    def test_light_intercept_captures_and_then_banks(self):
        state=setup_state(0.,75.)
        counts=changed_transfers(state,CFG,{},self.mine,self.opponent,0)
        self.assertEqual(counts['additionally_captured'],75.)
        self.assertEqual(counts['own_transfer_prevented'],0.)
        sim=FastCarbon(CFG);sim.set_state(state);sim.step(self.mine,self.opponent)
        collision=collision_events(state,sim,[self.mine,self.opponent])
        self.assertEqual(collision['player-1-worker-0'][3],('worker','player-0-worker-0'))
        self.assertEqual(sim.cash[0],80.)
        sim.step({'player-0-worker-0':'LEFT'},{})
        self.assertEqual(sim.cash[0],155.)

    def test_heavy_intercept_cedes_and_evading_protects(self):
        state=setup_state(83.,0.)
        lost=changed_transfers(state,CFG,{},self.mine,self.opponent,0)
        saved=changed_transfers(state,CFG,self.mine,{},self.opponent,0)
        self.assertEqual(lost['additional_own_transfer'],83.)
        self.assertEqual(saved['own_transfer_prevented'],83.)

    def test_equal_cargo_collision_on_empty_cell_gives_no_loot(self):
        counts=changed_transfers(setup_state(75.,75.),CFG,{},self.mine,self.opponent,0)
        self.assertFalse(any(counts.values()))

    def test_own_base_credits_loot_as_cash_immediately(self):
        state=setup_state(0.,75.,ours=63,theirs=65)
        sim=FastCarbon(CFG);sim.set_state(state);sim.step(self.mine,self.opponent)
        self.assertEqual(collision_events(state,sim,[self.mine,self.opponent])['player-1-worker-0'][3],('cash',0))
        self.assertEqual(sim.cash[0],155.)


if __name__=='__main__':
    unittest.main()
