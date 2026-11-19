"""Produce the two-step single-file agent; verify public-callback outputs on both native sites."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from funnel import HERE, ROOT, FastCarbon, load, save
from fastsim import DEFAULT_CONFIG


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(path):
    base=HERE.parent/'planter_capture_full_dev/candidate.py'
    if sha(base)!=load(HERE/'site_protocol.json')['source_hash']:
        raise ValueError("parent changed")
    geom=(HERE.parent/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
    scorer=(HERE.parent/'tree_ground_advantage_dev/decision.py').read_text().split('def consider(')[0]
    route=(HERE/'route.py').read_text()
    for text,line in [(scorer,'from guard_tree import _D,_TO,_DIRECTION'),(route,'from decision import _D, _DIRECTION, _TO, predicted, quote, values')]:
        if line not in text:raise ValueError("upstream import changed")
    scorer=scorer.replace('from guard_tree import _D,_TO,_DIRECTION','')
    route=route.replace('from decision import _D, _DIRECTION, _TO, predicted, quote, values','')
    wrapper='\n_SITE_IMPL = {}\nexec('+repr(geom+scorer+route)+', _SITE_IMPL)\n'
    wrapper+='_SITE_WEIGHTS = '+repr(load(HERE.parent/'tree_ground_advantage_dev/weights.json')['coef'])+'\n'
    wrapper+='_SITE_DEFAULT_CFG = '+repr(DEFAULT_CONFIG)+'\n'
    wrapper+='''
_SITE_GAMES = {}
_site_parent_agent = agent

def agent(observation, configuration=None):
    original = _site_parent_agent(observation, configuration)
    seat, t = observation["player"], observation["step"]
    if t == 0 or seat not in _SITE_GAMES:
        _SITE_GAMES[seat] = ({}, None)
    memory, chosen = _SITE_GAMES[seat]
    if chosen and t > chosen["t"] + 1:
        return original
    cfg = _SITE_DEFAULT_CFG if configuration is None else configuration
    _SITE_IMPL["update_memory"](observation, memory)
    if chosen:
        return _SITE_IMPL["pay"](observation, memory, original, cfg, chosen, _SITE_WEIGHTS)[0]
    chosen = _SITE_IMPL["approach"](observation, memory, original, cfg, _SITE_WEIGHTS)
    if chosen:
        _SITE_GAMES[seat] = memory, chosen
        return dict(original, **{chosen["worker"]:chosen["alternative"]})
    return original
'''
    path.write_bytes(base.read_bytes()+wrapper.encode())
    if path.stat().st_size>10_000_000:raise ValueError("above submission limit")


def parity(path):
    rows=[]
    for q in load(HERE/'selected_sites.json'):
        raw=load(ROOT/q['record']['source']);s=q['seat'];cfg=raw['game_configuration']
        sim=FastCarbon(cfg).reset(seed=q['seed']);t=q['root_t']
        spec=importlib.util.spec_from_file_location("solo_site_package",path)
        deployed=importlib.util.module_from_spec(spec);spec.loader.exec_module(deployed)
        for h,pair in enumerate(raw['actions'][:t]):
            got=deployed.agent(sim.obs(s),cfg)
            if got!=pair[s]:raise ValueError(f"pre-root mismatch {q['seed']} t={h}")
            sim.step(*pair)
        p=HERE/f"seat{s}";rec=load(p/'live_approach.json');tape=load(p/rec['action_tape'])
        if sha(p/rec['action_tape'])!=rec['action_sha256']:raise ValueError('native archive mismatch')
        for i in range(3):
            got=deployed.agent(sim.obs(s),cfg)
            if got!=tape[i][s]:raise ValueError(f"package vs live candidate mismatch {q['seed']} t={t+i}")
            sim.step(*tape[i])
        rows.append(dict(seed=q['seed'],seat=s,unchanged_prefix_turns=t,matched_intervention_and_later_turns=[t,t+1,t+2]))
    save(HERE/'site_package_parity.json',dict(size=path.stat().st_size,sha256=sha(path),matches=rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        out=a.out or Path(td)/'candidate.py';build(out);parity(out)
