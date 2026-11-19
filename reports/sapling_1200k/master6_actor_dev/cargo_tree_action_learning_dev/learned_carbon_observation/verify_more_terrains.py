"""Compare uninstrumented physical public commands over every archived decision of the other held terrains."""
from collections import Counter
import argparse
from common import HERE, PROTOCOL, public_frames, save, sha
from verify_entry import PACKS, run_pack


def main(cpu):
    cards = [c for c in PROTOCOL["cards"] if c["split"] == "val"][1:]
    combo = ("incumbent", "student")
    results, details = [], []
    for card in cards:
        frames = [[], []]
        for observations, truth, joint, cfg in public_frames(card):
            for seat, obs in enumerate(observations):
                frames[seat].append(obs)
        per_seat = []
        for seat, observations in enumerate(frames):
            arms = {name: run_pack(PACKS[name], "timing", cpu, observations, cfg)["rows"] for name in combo}
            counts = Counter()
            for t, obs in enumerate(observations):
                parent, student = [arms[arm][t]["action"] for arm in combo]
                team = obs["players"][seat]
                roles = {**dict.fromkeys(team[1], "base"), **{u:w[2].lower() for u,w in team[2].items()}}
                assert set(student) <= roles.keys() and set(parent) <= roles.keys()
                counts["perspective_turns"] += 1
                counts["changed_turns"] += parent != student
                counts["slots"] += len(roles)
                changes = []
                for uid, role in roles.items():
                    original, alternative = parent.get(uid, "NONE"), student.get(uid, "NONE")
                    if alternative != original:
                        counts["changed_slots"] += 1
                        counts["changed_"+role] += 1
                        changes.append((uid,role,original,alternative))
                details.append(dict(seed=card["seed"],seat=seat,t=t,parent=parent,student=student,changed=changes))
            print(card["seed"],seat,counts,flush=True)
            per_seat.append(dict(seat=seat,**counts))
        results.append(dict(card=card,seats=per_seat))
        # Writes one complete terrain at a time and records its completed denominators.
        save(HERE/"other_held_action_tapes.json.gz",details)
        save(HERE/"other_held_actions.json",dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),
             pack_sha256={k:sha(PACKS[k].read_bytes()) for k in combo},records=results,
             tapes_sha256=sha((HERE/"other_held_action_tapes.json.gz").read_bytes()),
             method="All 299 normal public entry calls per separate process, each physical seat and each pack, fixed archive path. No opponent process executed. Extra terrain check requested for the downstream comparison gate."))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cpu",type=int,default=2)
    main(parser.parse_args().cpu)
