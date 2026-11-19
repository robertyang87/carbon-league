"""Generate an economic joint-order catalog from archived public histories before reading the current action."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
LEAGUE = INTAKE.parent
ROOT = next(p for p in HERE.parents if (p/'.git').exists())
sys.path.insert(0,str(LEAGUE/'natural_original'))
sys.path.insert(0,str(INTAKE/'live_curriculum'))
import audit_menu
import student_history

PROTOCOL = '31faf8bfcac071dc22a20e569cccdc9a4720f89185d3b72d84a00a8204fec79b'
CLASSES = ('recruit','plant_or_capture','freight','ordinary')
STEPS = tuple(range(12,277,24))


def need(flag,reason):
    if not flag: raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_native(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),default=json_native,allow_nan=False).encode()


def publish(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()==raw: return
    tmp=path.with_suffix('.tmp.'+str(os.getpid()))
    tmp.write_bytes(raw);os.replace(tmp,path)


def inventory():
    registration=(HERE/'protocol.json').read_bytes();need(sha(registration)==PROTOCOL,'changed registration')
    p=json.loads(registration);e=p['evidence'];manifest=(INTAKE/'hosted/training_manifest.json').read_bytes()
    need(sha(manifest)==e['manifest_sha256'] and tuple(e['steps'])==STEPS and len(e['archives'])==4,'wrong source plan')
    sources={(z['seed'],z['seat'],z['arm']):z for z in json.loads(manifest)['episodes']}
    games=[]
    for ref in e['archives']:
        key=ref['seed'],ref['seat'],ref['arm'];ep=sources[key]
        need(all(ep[k]==v for k,v in ref.items()) and ep['episode_verified'] and ep['arm']=='draw_a','different training episode')
        games.append(ep)
    need({(g['seed'],g['seat']) for g in games}=={(s,p) for s in e['seeds'] for p in e['seats']},'repeat or missing trajectory')
    return p,games


@contextmanager
def trajectory(ep,pack,expected_sha):
    archive=ROOT/ep['archive_path'];need(sha(archive.read_bytes())==ep['archive_sha256'],'modified ZIP')
    with zipfile.ZipFile(archive) as z, TemporaryDirectory() as dirname:
        reg=z.read(ep['registration_member']);receipt=z.read(ep['receipt_member'])
        need(sha(reg)==ep['registration_sha256'] and sha(receipt)==ep['receipt_sha256'],'changed receipt or registration')
        r=json.loads(reg)
        need(r['candidate_seat']==ep['seat'] and r['seed']==ep['seed'] and
             r['files']['candidate']['sha256']==expected_sha==ep['policy_sha256'] and
             r['physical_map_sha256']==ep['terrain'] and json.loads(receipt)['turns_sha256']==ep['tape_sha256'],'identity mismatch')
        regpath=Path(dirname)/'registration.json';regpath.write_bytes(reg)
        live=audit_menu.open_live(regpath);env,cfg=live.official()
        with z.open(ep['turns_member']) as tape:
            yield live,env,cfg,tape,audit_menu.collect.Player(pack)


def key(cmd):
    return tuple(sorted((u,a) for u,a in cmd.items() if a!='NONE'))


def evaluate_menu(built,recorded,issued):
    need(key(built['baseline'])==key(recorded['options'][0]) and
         issued==recorded['options'][recorded['choice']],'native A0 or actual action differs')
    marginal={}
    for item in built['orders']:
        k=key(item['command']);q=float(item['probability'])
        need(k not in marginal and math.isfinite(q) and 0<q<=1,'duplicate or invalid physical probability')
        marginal[k]=q
    need(math.isclose(sum(marginal.values()),1.,abs_tol=1.e-10) and
         math.isclose(marginal[key(built['baseline'])],built['baseline_probability'],abs_tol=1.e-10),
         'physical mass mismatch')
    old={key(c) for c in recorded['options']};seen=key(issued)
    novel_by_class = {name:sum(key(w["command"]) not in old for w in info["witnesses"])
                      for name,info in built["classes"].items()}
    return dict(novel_witnesses_by_class=novel_by_class,original_supported=recorded['supported'],original_options=recorded['options'],
                original_probabilities=recorded['probabilities'],actual_old_choice=recorded['choice'],
                issued=issued,actual_probability_in_catalog=marginal.get(seen,0.),
                original_support_size=len(old),catalog_support_size=len(marginal),
                novel_commands=len(marginal.keys()-old),novel_probability=sum(marginal[k] for k in marginal.keys()-old),
                old_orders_outside_catalog=len(old-marginal.keys()))


def run_game(p,ep):
    import proposals
    pack=LEAGUE/p['behavior']['package'];expected=p['behavior']['sha256']
    need(sha(pack.read_bytes())==expected,'wrong active trained network')
    seat=ep['seat'];events=iter(ep['events']);rows=[];bank=[5.,5.];digest=hashlib.sha256();n=0
    with trajectory(ep,pack,expected) as (live,env,cfg,tape,actor):
        while not env.done:
            t=env.states[0].observation.step
            need(t==n,'public history gap')
            obs=[live.observe(env,side,bank[side]) for side in (0,1)]
            hashes=[sha(live.encoded(x)) for x in obs]
            with audit_menu.collect.attached(actor.modules):
                if t==0:
                    # Initializes the exact deployed network, normal route and own opening observation.
                    actor.mod.agent(obs[seat],cfg)
                else:
                    actor.mod._obs_key=t,seat
                    x=actor.mod._FEAT.observe(obs[seat])
                if t in STEPS:
                    history=student_history.snapshot(actor)
                    built=proposals.build(actor.mod,x,actor.mod._FEAT.aux,obs[seat])
                    need(history==student_history.snapshot(actor),'building proposals mutated public history')
                    # Seal the candidate set before the current recorded command or old menu is read.
                    sealed=canonical(built)
                    built=json.loads(sealed)
            line=tape.readline();need(line,'missing observed action');digest.update(line);record=json.loads(line)
            need(record['step']==t and record['public_observation_sha256']==hashes and
                 all(c['status']=='ok' and c['response']['step']==t for c in record['calls']),'wrong observed prefix')
            actual=[c['response']['action'] for c in record['calls']]
            if t in STEPS:
                old=next(events);need(old['step']==t and old['public_observation_sha256']==hashes[seat],'old menu source mismatch')
                observed=evaluate_menu(built,old,actual[seat])
                rows.append(dict(seed=ep['seed'],seat=seat,arm=ep['arm'],step=t,terrain=ep['terrain'],
                                 history_sha256=history,public_observation_sha256=hashes[seat],catalog_sha256=sha(sealed),
                                 catalog=built,comparison_after_freeze=observed))
            # Only source orders are ever submitted; proposals are data, never applied here.
            env.step(actual,[]);bank=record['overage_remaining_seconds'];n+=1
        need(not tape.read() and digest.hexdigest()==ep['tape_sha256'] and n==ep['official_transitions'] and
             [s.reward for s in env.states]==ep['official_scores_by_seat'] and
             [r['step'] for r in rows]==list(STEPS) and next(events,None) is None,'incomplete source replay')
    identity={k:ep[k] for k in ('seed','seat','arm','terrain','archive_sha256','registration_sha256','receipt_sha256','tape_sha256')}
    return dict(**identity,transitions=n,rows=rows)


def summarize(rows):
    def mean(xs):
        return sum(xs)/len(xs) if xs else None
    group={}
    for cls in CLASSES:
        entries=[r['catalog']['classes'][cls] for r in rows]
        tags=Counter()
        for entry in entries:
            tags.update({tag for w in entry['witnesses'] for tag in w['tags']})
        novel=[r['comparison_after_freeze']['novel_witnesses_by_class'][cls] for r in rows]
        group[cls]=dict(novel_witnesses_total=sum(novel),roots_with_witness_outside_signed_support=sum(v>0 for v in novel),eligible_roots=sum(e['eligible_count']>0 for e in entries),
                        proposed_before_validation=sum(e['eligible_count'] for e in entries),
                        attempted=sum(len(e['attempted']) for e in entries),
                        validated_roots=sum(bool(e['witnesses']) for e in entries),
                        retained_witnesses=sum(len(e['witnesses']) for e in entries),
                        fallback_roots=sum(bool(e['fallback']) for e in entries),roots_per_witness_tag=dict(sorted(tags.items())))
    cs=[r['comparison_after_freeze'] for r in rows]
    return dict(roots=len(rows),classes=group,native_binary_supported=sum(c['original_supported'] for c in cs),
                catalog_support_histogram=dict(Counter(c['catalog_support_size'] for c in cs)),
                mean_support_size=mean([c['catalog_support_size'] for c in cs]),original_mean_support_size=mean([c['original_support_size'] for c in cs]),
                mean_A0_probability=mean([r['catalog']['baseline_probability'] for r in rows]),
                roots_with_novel_orders=sum(c['novel_commands']>0 for c in cs),
                distinct_within_root_novel_total=sum(c['novel_commands'] for c in cs),
                mean_novel_mass=mean([c['novel_probability'] for c in cs]),
                original_orders_outside_catalog=sum(c['old_orders_outside_catalog'] for c in cs),
                observed_order_outside_catalog=sum(c['actual_probability_in_catalog']==0 for c in cs))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('ticket','merge','all'))
    parser.add_argument('--seed',type=int);parser.add_argument('--seat',type=int)
    parser.add_argument('--shards',type=Path,required=True)
    args=parser.parse_args();p,games=inventory()
    code_sha={name:sha((HERE/name).read_bytes()) for name in ('proposals.py','audit.py','protocol.json')}
    def path(g):return args.shards/(str(g['seed'])+'_'+str(g['seat'])+'.json.gz')
    if args.mode!='merge':
        work=games if args.mode=='all' else [g for g in games if (g['seed'],g['seat'])==(args.seed,args.seat)]
        need(bool(work),'unregistered ticket')
        for g in work:
            if not path(g).exists():
                audit=run_game(p,g);audit['code_sha256']=code_sha
                publish(path(g),gzip.compress(canonical(audit),mtime=0))
            print(g['seed'],g['seat'],flush=True)
    if args.mode=='ticket':return
    items=[json.loads(gzip.decompress(path(g).read_bytes())) for g in games]
    for e,i in zip(games,items):
        need(i['code_sha256']==code_sha and all(i[k]==e[k] for k in ('seed','seat','arm','terrain','tape_sha256','archive_sha256'))
             and len(i['rows'])==12,'stale or incomplete shard')
    rows=[r for i in items for r in i['rows']];need(len(rows)==48==len({(r['seed'],r['seat'],r['step']) for r in rows}),'incomplete full census')
    roots=gzip.compress(canonical(dict(schema='sapling-public-catalog-roots/1',protocol_sha256=PROTOCOL,
               labels=p['labels'],tickets=items)),mtime=0)
    summary=dict(schema='sapling-public-catalog-coverage/1',code_sha256=code_sha,protocol_sha256=PROTOCOL,
                 source_training_manifest_sha256=p['evidence']['manifest_sha256'],package_sha256=p['behavior']['sha256'],
                 roots_file_sha256=sha(roots),joint_commands_executed_for_counterfactuals=0,new_games=0,labels=p['labels'],
                 public_requests_verified=sum(2*i['transitions'] for i in items),all=summarize(rows),
                 by_ticket=[dict(seed=i['seed'],seat=i['seat'],**summarize(i['rows'])) for i in items],
                 by_observer_seat=[dict(seat=s,**summarize([r for r in rows if r['seat']==s])) for s in (0,1)])
    publish(HERE/'roots.json.gz',roots);publish(HERE/'coverage.json',json.dumps(summary,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n')


if __name__=='__main__':
    main()
