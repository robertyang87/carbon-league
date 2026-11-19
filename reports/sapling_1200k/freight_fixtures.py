"""Both-seat real simulator checks for delivery, threat, mining and economic routing."""
import argparse
from pathlib import Path

from cash_practice import audit, np
from freight_build import OUTPUT, PARENT
from it50_endhome_fixtures import FixtureNet


class ProbeNet(FixtureNet):
    def lin(self, x, key):
        if key == "whead":
            result = np.zeros(5, np.float32)
            result[self.move] = .12
            return result
        return super().lin(x, key)


def main(out):
    candidate = audit.shared.module(OUTPUT, "freight_fixture")
    parent = audit.shared.module(PARENT, "freight_ref")
    F = candidate.sys.modules["cnet2_feat"]
    cases = []
    for seat in (0, 1):
        for name in ("midgame_two_step", "second_home_last_step", "last_step_guarded_home",
                     "mine_rich_tile", "tree_toll_detour", "later_recruit", "low_load"):
            sim = audit.FastCarbon(audit.CFG).reset(seed=610881189)
            st = sim.get_state()
            own_home = st["bases"][seat][int(name=="second_home_last_step")][1]
            worker = f"player-{seat}-worker-0"
            def move(cell, *directions):
                for k in directions:
                    cell = int(F.DEST[cell,k])
                return cell
            pos = move(own_home, 2)
            if name == "midgame_two_step":
                pos = move(pos, 2)
            if name == "tree_toll_detour":
                pos = move(pos, 3)
                tree = f"player-{1-seat}-tree-0"
                st["trees"][tree] = dict(player=1-seat, pos=move(own_home, 2, 2), age=8)
                st["trees_dict"][tree] = ["retired-planter", 0.]
            st["cash"] = [80.,80.]
            st["workers"][worker] = dict(player=seat, pos=pos, carbon=12. if name=="low_load" else 150., type="COLLECTOR")
            if name == "last_step_guarded_home":
                st["workers"][f"player-{1-seat}-worker-0"]=dict(player=1-seat,pos=own_home,carbon=0.,type="COLLECTOR")
            st["carbon"][pos] = 100. if name=="mine_rich_tile" else 0.
            st["step"] = 298 if name in ("second_home_last_step", "last_step_guarded_home", "later_recruit") else 130
            sim.set_state(st)
            raw_prefer = 4 if name=="mine_rich_tile" else 1 if name=="tree_toll_detour" else 0
            normal = int(F.CANON_MOVE[raw_prefer]) if seat else raw_prefer
            nets = [ProbeNet(normal, recruit=name=="later_recruit") for _ in range(2)]
            candidate._FEAT, candidate._NETS, candidate._FREIGHT_ENABLED = None, nets, True
            pre = sim.obs(seat)
            proposed = candidate.agent(pre, sim.config)
            cmd = proposed.get(worker, "NONE")
            wanted = {"midgame_two_step":"LEFT","second_home_last_step":"LEFT","last_step_guarded_home":"NONE",
                      "mine_rich_tile":"NONE","tree_toll_detour":"LEFT","later_recruit":"NONE","low_load":"NONE"}[name]
            assert cmd == wanted, (name,seat,cmd,candidate._FREIGHT_LAST)
            assert bool(candidate._FREIGHT_LAST) == (raw_prefer != F.MOVES.index(wanted))
            assert nets[0].inputs == nets[1].inputs
            joint = [{},{}]
            joint[seat] = proposed
            logger = audit.Observer()
            logger.step(sim, joint)
            if name in ("midgame_two_step", "tree_toll_detour"):
                joint[seat] = candidate.agent(sim.obs(seat), sim.config)
                logger.step(sim, joint)
            delivered = float(logger.d[seat]["own_positive_delivery_carbon"])
            if name in ("midgame_two_step", "tree_toll_detour", "second_home_last_step"):
                assert delivered==150., (name,seat,delivered)
            elif name=="mine_rich_tile":
                assert sim.workers[worker].carbon==180. and not delivered
            elif name=="last_step_guarded_home":
                assert sim.step_count==299 and not delivered
            candidate._FREIGHT_ENABLED=False
            candidate._NETS, candidate._FEAT, parent._FEAT = None, None, None
            assert candidate.agent(pre, sim.config)==parent.agent(pre, sim.config)
            assert candidate._COUNTS==parent._COUNTS
            cases.append(dict(name=name, seat=seat, preferred=F.MOVES[raw_prefer], selected=cmd,
                              banked=delivered, parent_parity_when_disabled=True))
    audit.shared.atomic_json(out, dict(parent_sha256=audit.digest(PARENT),candidate_sha256=audit.digest(OUTPUT),cases=cases))
    print(len(cases))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    main(p.parse_args().out)
