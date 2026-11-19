"""Positive control and simultaneous-response matrix at the Fable last-courier trade."""
import io
import json
import tarfile
from scan import ACTOR, BASE, COMMANDS, FastCarbon, HERE, ROOT, Counter, audit_root, fetch, gzip, summary

GID = "30590f1ec1d871b15cba118834d247d0-1"


def main():
    directory = BASE / "all_top5/appendix_b5e6a7f8"
    card = next(g for g in fetch.load(directory/"manifest.json")["games"] if g["id"] == GID)
    source = card["archive_location"]
    path = ROOT / source["archive"]
    packed = path.read_bytes() if path.is_file() else fetch.git("show", "HEAD:" + source["archive"])
    archive_info = next(a for a in fetch.load(directory/"manifest.json")["new_archives"] if a["archive"] == source["archive"])
    if fetch.sha(packed) != archive_info["sha256"]:
        raise ValueError("source archive fingerprint")
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        raw = archive.extractfile(source["member"]).read()
    if fetch.sha(raw) != card["sha256"]:
        raise ValueError("source replay fingerprint")
    game = json.loads(gzip.decompress(raw))
    frames, cfg = game["steps"],game["configuration"]
    sim = FastCarbon(cfg).reset(seed=card["seed"])
    sim.load_observation(frames[0][0]["observation"])
    for t in range(90):
        sim.step(*[r["action"] or {} for r in frames[t+1]])
    commands = [r["action"] or {} for r in frames[91]]
    pre = sim.obs(0)["players"]
    if pre != frames[90][0]["observation"]["players"]:
        raise ValueError("incorrect pre-action board")
    case = audit_root(sim,pre,commands,cfg,1,90,Counter())
    if not case or case["actual_victim_survived"] or case["observed_contact_hunters"] != ["player-0-worker-5"]:
        raise ValueError("missed true last-courier neutralization")
    snap = sim.get_state()
    probe = FastCarbon(cfg)
    matrix = []
    for own_move in COMMANDS:
        for their_move in COMMANDS:
            altered = [dict(x) for x in commands]
            for seat, wid, action in [(0,"player-0-worker-5",own_move),(1,"player-1-worker-34",their_move)]:
                if action == "NONE":
                    altered[seat].pop(wid,None)
                else:
                    altered[seat][wid] = action
            probe.set_state(snap)
            probe.step(*altered)
            result = probe._players_obs()
            if (own_move,their_move) == ("LEFT","NONE") and result != frames[91][0]["observation"]["players"]:
                raise ValueError("logged joint command does not reproduce the game")
            matrix.append({"master6_worker5":own_move,"fable_worker34":their_move,"master6_worker5_alive":
                           "player-0-worker-5" in result[0][2],"fable_worker34_alive":"player-1-worker-34" in result[1][2],
                           "rival_after":summary(result,1,cfg)})
    fetch.dump(HERE/"ladder_control.json",{"gid":GID,"raw_sha256":card["sha256"],"t":90,"observed":case,"matrix":matrix,
                                          "later_observed_final_rewards":game["rewards"],"later_last_rival_collector_t":max(
                                              t for t,frame in enumerate(frames) if any(w[2]=="COLLECTOR"
                                              for w in frame[0]["observation"]["players"][1][2].values()))})


if __name__ == "__main__":
    main()
