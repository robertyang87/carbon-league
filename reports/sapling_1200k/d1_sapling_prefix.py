"""Independently teacher-force both original single policies on every alternative's executed prefix."""
import gzip
import json
from pathlib import Path
import numpy as np

from d1_sapling_build import HERE, OUTPUT
from tree_audited_game import exam, prior
from turning_point_ledger import write


def main():
    directory = HERE/"delivery_parent_dev"
    data = json.loads((directory/"results.json").read_bytes())
    pack = prior.shared.module(OUTPUT, "sapling_d1_prefix_test")
    totals = {"sampled_roots_with_alternatives":0, "alternative_actions":0, "active_slots_compared":0}
    for row in data["games"][:2]:
        raw = json.loads(gzip.decompress((directory/row["raw_file"]).read_bytes()))
        seat = row["candidate_seat"]
        sim = exam.FastCarbon(raw["game_configuration"]).reset(seed=row["seed"])
        feat = pack._P["cnet2_feat"].Feat(seat)
        observed = False
        for t, actions in enumerate(raw["actions"]):
            obs = sim.obs(seat)
            x = feat.observe(obs)
            # Open-loop histories advance every step. An individual test root is chosen deterministically.
            if t in (40,80,120,160,200,240,280):
                if pack._MODELS is None:
                    pack.agent(obs, sim.config)
                encoded = pack._prepare_root(x)
                pack._WITNESSES.clear()
                pack._LAST.update(legal_branches=0)
                base, _ = pack._decode_joint(x, feat, encoded)
                if feat.to_actions(base) != actions[seat]:
                    raise ValueError(f"bad reference A0: {seat}/{t}")
                slot = next((s for s, uid in enumerate(feat.own_uid) if uid in obs["players"][seat][2]),None)
                if slot is not None:
                    totals["sampled_roots_with_alternatives"] += 1
                    for a in range(5):
                        if a == base[slot]:
                            continue
                        forced = np.full(base.shape, -1)
                        forced[:slot], forced[slot] = base[:slot], a
                        alt, _ = pack._decode_joint(x, feat, encoded, forced)
                        if alt is None:
                            continue
                        reference = [net.act(x,feat.aux, teacher=2,forced=alt)[1] for net in pack._MODELS]
                        mixed = 0.75*reference[0]+0.25*reference[1]
                        live = np.nonzero(x["tok_mask"][:len(alt)])[0]
                        if (not np.array_equal(alt[:slot],base[:slot]) or alt[slot] != a or
                                any(s != slot and alt[s] != np.argmax(mixed[s]) for s in live)):
                            raise ValueError(f"forced/redecoded score mismatch: {seat}/{t}/{slot}/{a}")
                        totals["active_slots_compared"] += len(live)-1
                        totals["alternative_actions"] += 1
                        observed = True
            sim.step(*actions)
        if not observed:
            raise ValueError("missing seat or untested alternatives")
    write(HERE/"d1_sapling_prefix.json",dict(pack_sha256=prior.digest(OUTPUT), **totals))


if __name__ == "__main__":
    main()
