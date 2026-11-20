"""Extract observed policy histories using only the acting side's own carbon and pre-state."""
from collections import Counter
import gzip
import json
import zipfile

import numpy as np

from register import H, ROOT, ORIGINAL, get, sha
from model import module, parent, parse, weight_blob


def ladder(card):
    raw = (ROOT / card["source"]).read_bytes()
    if sha(raw) != card["sha256"]:
        raise ValueError("Ladder source hash")
    g = json.loads(gzip.decompress(raw))
    if g["statuses"] != ["DONE","DONE"] or not np.isfinite(g["rewards"]).all():
        raise ValueError("Missing official reward")
    frames = g["steps"]
    initial = frames[0][0]["observation"]["full_carbon"]   # inventory identity only
    if sha(bytes(initial)) != card["terrain"]:
        raise ValueError("Ladder terrain hash")
    seat = card["seats"][0]
    def records():
        for t,frame in enumerate(frames[:-1]):
            board = frame[0]["observation"]
            personal = frame[seat]["observation"]
            if board["step"] != t or personal["player"] != seat:
                raise ValueError("Incorrect replay chronology")
            yield t,board["players"],personal["carbon"],frames[t+1][seat]["action"] or {}
    return g["rewards"], [(seat,records())]


def hosted(card):
    source = ROOT / card["source"]
    if sha(source.read_bytes()) != card["sha256"]:
        raise ValueError("Hosted ZIP hash")
    arm = card["repeat"] + "/"
    with zipfile.ZipFile(source) as z:
        reg = z.read(arm+"registration.json")
        receipt = json.loads(z.read(arm+"receipt.json"))
        turns = z.read(arm+"turns.jsonl")
        if sha(reg) != receipt["protocol_sha256"] or sha(turns) != receipt["turns_sha256"]:
            raise ValueError("Hosted archive stream identity")
        reg = json.loads(reg)
        if not (reg["seed"] == card["seed"] and reg["candidate_seat"] == card["candidate"] and
                reg["physical_map_sha256"] == card["terrain"] and
                reg["files"]["candidate"]["sha256"] == card["own"] and
                reg["files"]["opponent"]["sha256"] == ORIGINAL):
            raise ValueError("Unexpected hosted match")
        if receipt["status"] != "complete" or not receipt["counted"]:
            raise ValueError("Hosted official score qualification")
        frames = gzip.decompress(z.read(arm+"observations.jsonl.gz")).splitlines()
    calls = list(map(json.loads,turns.splitlines()))
    assert len(frames) == len(calls) == receipt["official_transitions"]
    def records(seat):
        for t,(line,turn) in enumerate(zip(frames,calls)):
            frame = json.loads(line)
            obs = frame["observations"][seat]
            checksum = sha((json.dumps(obs,separators=(",",":"),allow_nan=False)+"\n").encode())
            assert t == frame["step"] == obs["step"] == turn["step"] and obs["player"] == seat
            assert checksum == frame["public_observation_sha256"][seat] == turn["public_observation_sha256"][seat]
            assert "full_carbon" not in obs and turn["calls"][seat]["status"] == "ok"
            yield t,obs["players"],obs["carbon"],turn["calls"][seat]["response"]["action"]
    return receipt["official_scores_by_seat"], [(s,records(s)) for s in card["seats"]]


def extract(split):
    protocol = get(H/"protocol.json")
    m = module(parent(),tap=True)
    core = m._TorchNet(m._POLICY,parse(weight_blob(parent())))
    rows, xs, ss, allh, test_x = [], [], [], [], []
    full = [1.] * m.N
    def flush():
        if xs:
            allh.append(core(np.stack(xs),m._SID))
            xs.clear()
    for index,card in enumerate(protocol["tickets"]):
        if card["split"] != split:
            continue
        scores, streams = (hosted if card["domain"]=="h2h" else ladder)(card)
        for seat,stream in streams:
            tracker = m.BeliefTracker(m._Bel,seat)
            previous = None
            margin = scores[seat] - scores[1-seat]
            for t,players,carbon,order in stream:
                est, known = tracker.step(players,t,carbon)
                if t in protocol["steps"]:
                    x = m._quant(m.encode(players,t,seat,est,full,previous)).astype(np.float32)
                    s = m._scalars(players,t,seat)
                    if len(test_x)<8 and t in [6,81,156,291]:
                        test_x.append((x,s))
                    xs.append(x)
                    ss.append(s)
                    rows.append(dict(terrain=card["terrain"],game=index,domain=card["domain"],seat=seat,
                                     step=t,margin=round(margin,2),truth=m._terminal(margin),
                                     cash_terminal=m._terminal(players[seat][0]-players[1-seat][0]),
                                     observation_unknown=int(np.sum(np.asarray(known)<1))))
                    if len(xs)==8:
                        flush()
                tracker.set_actions(players,order)
                previous=players
    flush()
    for z in (allh,ss,rows):
        assert len(z)>0
    groups = {}
    for r in rows:
        games=groups.setdefault(r["terrain"],{})
        seats=games.setdefault(r["game"],Counter())
        seats[r["seat"]]+=1
    def weight(r):
        domain_games=groups[r["terrain"]]
        acts=domain_games[r["game"]]
        return 1/(len(groups)*len(domain_games)*len(acts)*acts[r["seat"]])
    weights=np.array([weight(r) for r in rows],np.float32)
    assert np.isclose(weights.sum(),1)
    return np.concatenate(allh),np.stack(ss),rows,weights,test_x
