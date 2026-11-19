"""Snapshot a warmed 851c NumPy actor for sequential branches in the same process.

The state object is in-memory and belongs to one actor. fingerprint() contains
model decision state only, excluding the harness's elapsed-time counters.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import student_history as history
import play

SOURCE_SHA=history.SHA
MODULE_FIELDS=('_COUNTS','_chosen_route','_obs_key','_capture','_forced')
STATIC_POLICY_FIELDS={'cfg','w','NB','L','TA','ta','TNH','rel_cell','tbias','wm','C','G','NH','gih','ghh'}
SOURCE_GLOBALS=('_RETURN_THETA','_HYBRID','_OPTS','_SAMPLE_STEPS','_SRC_SHA256','_PACK_SRC_SHA256')


def _module(player):
    mod=getattr(player,'mod',player)
    if history.package_sha(mod.__file__)!=SOURCE_SHA:
        raise ValueError('Snapshot requires the registered 851c actor')
    if mod._capture is not None or mod._forced is not None:
        raise ValueError('Capture state only between complete public calls')
    if mod._FEAT is None or mod._NETS is None or len(mod._NETS)!=2:
        raise ValueError('Warm both student networks and public feature history first')
    if not isinstance(mod._behavior_rng,np.random.Generator):
        raise ValueError('Finish any active RNG hook before taking a snapshot')
    return mod


def _static_bound(mod):
    attrs=[]
    for model in mod._NETS:
        keys=set(vars(model))-{'trunk'}
        if not keys<=STATIC_POLICY_FIELDS or not {'w','wm','cfg','gih','ghh'}<=keys:
            raise ValueError('Unexpected mutable model field(s) '+str(keys-STATIC_POLICY_FIELDS))
        attrs.append({k:getattr(model,k) for k in keys})
    module_inputs={k:getattr(mod,k) for k in SOURCE_GLOBALS}
    # Weights and transforms are large but fit the training box. The digest is
    # deliberately recomputed rather than treating object identity as content.
    return history.state_hash((module_inputs,attrs))


def _data(mod):
    return dict(features=vars(mod._FEAT),
                module={k:getattr(mod,k) for k in MODULE_FIELDS},
                rng=mod._behavior_rng.bit_generator.state,
                trunks=[getattr(net,'trunk',None) for net in mod._NETS])


def fingerprint(player):
    mod=_module(player)
    return history.state_hash(dict(source=SOURCE_SHA,static=_static_bound(mod),dynamic=_data(mod)))


def snapshot(player):
    mod=_module(player)
    return dict(owner=mod,feature_type=type(mod._FEAT),static=_static_bound(mod),
                policy_instances=tuple(mod._NETS),rng_type=type(mod._behavior_rng.bit_generator),
                dynamic=copy.deepcopy(_data(mod)),
                harness=(list(player.wall),float(player.cpu)) if hasattr(player,'mod') else None)


def restore(player,state):
    mod=getattr(player,'mod',player)
    if state['owner'] is not mod or tuple(mod._NETS)!=state['policy_instances']:
        raise ValueError('Restore requires the original in-process policy instances')
    if _static_bound(mod)!=state['static']:
        raise ValueError('Immutable source, weights, or weight transforms drifted')
    d=copy.deepcopy(state['dynamic'])
    for attr,value in d['module'].items():
        setattr(mod,attr,value)
    if type(mod._FEAT) is not state['feature_type']:
        mod._FEAT=state['feature_type'].__new__(state['feature_type'])
    vars(mod._FEAT).clear()
    vars(mod._FEAT).update(d['features'])
    for policy,trunk in zip(mod._NETS,d['trunks']):
        if trunk is None:
            vars(policy).pop('trunk',None)
        else:
            policy.trunk=trunk
    desired=state['rng_type']
    if not isinstance(mod._behavior_rng,np.random.Generator) or type(mod._behavior_rng.bit_generator) is not desired:
        mod._behavior_rng=np.random.Generator(desired(0))
    mod._behavior_rng.bit_generator.state=d['rng']
    if hasattr(player,'mod') and state['harness'] is not None:
        player.wall[:]=state['harness'][0]
        player.cpu=state['harness'][1]
    if history.state_hash(_data(mod))!=history.state_hash(state['dynamic']):
        raise ValueError('Student state restoration differed')


class _Draw:
    def __init__(self,rng,forced):
        self.rng=rng;self.forced=forced;self.event=None

    def random(self):
        local=sys._getframe(1).f_locals
        if self.event is not None:
            raise ValueError('Multiple natural sampling requests')
        u=float(self.rng.random())
        p=list(map(float,local['probs']))
        options=[dict(local['base_cmd']),dict(local['alt_cmd'])]
        slot=int(local['slot']);moves=[int(local['best']),int(local['second'])]
        legal=local['trace'][slot][2].tolist()
        if not (len(p)==2 and all(0<x<1 for x in p) and abs(sum(p)-1)<1e-12 and
                options[0]!=options[1] and moves[0]!=moves[1] and all(legal[m] for m in moves)):
            raise ValueError('Unusable physical joint action menu')
        natural=int(u>=p[0]);picked=natural if self.forced is None else self.forced
        self.event=dict(options=options,p=p,pick=picked,unforced_choice=natural,
                        forced=self.forced is not None,logp=math.log(p[picked]),slot=slot,
                        actions=moves,prefix=local['baseline'][:slot].tolist(),legal=legal,
                        native_logits=local['logits'].tolist(),route=bool(local['route']),
                        uid=str(sys._getframe(1).f_globals['_FEAT'].own_uid[slot]))
        return u if self.forced is None else (0.0 if picked==0 else p[0])


def act_with_choice(player,obs,cfg,choice=None):
    """Run one warmed public call; both interventions advance the genuine student RNG once."""
    if choice not in (0,1,None):
        raise ValueError('Invalid menu selection')
    if obs['step']==0:
        raise ValueError('Use the ordinary public entry for initial cold start')
    mod=_module(player)
    stream=mod._behavior_rng
    tap=_Draw(stream,choice)
    mod._behavior_rng=tap
    try:
        order=player.act(obs,cfg)
    finally:
        mod._behavior_rng=stream
    event=tap.event
    if (choice is not None and event is None) or (event is not None and order!=event['options'][event['pick']]):
        raise ValueError('Requested physical action was not available or issued')
    return order,event


def source_audit():
    """Read-only counterfactual calls on one historical public observation; no new game/suffix."""
    import ast
    import tempfile
    import zipfile
    doc=play.read(HERE.parent.parent/'hosted/training_manifest.json')
    ep=next(e for e in doc['episodes'] if e['seed']==621758460 and e['seat']==0 and e['arm']=='draw_a')
    pack=history.PACK
    actor=play.audit_menu.collect.Player(pack)
    with zipfile.ZipFile(play.ROOT/ep['archive_path']) as z,tempfile.TemporaryDirectory() as tmp:
        reg=Path(tmp)/'registration.json';reg.write_bytes(z.read(ep['registration_member']))
        live=play.audit_menu.open_live(reg);env,cfg=live.official()
        bank=[5.,5.]
        for raw in z.read(ep['turns_member']).splitlines():
            frame=json.loads(raw)
            t=frame['step'];obs=live.observe(env,0,bank[0])
            if play.sha(live.encoded(obs))!=frame['public_observation_sha256'][0]:
                raise ValueError('fixture public observation')
            if t==12:break
            actor.act(obs,cfg)
            bank=frame['overage_remaining_seconds'];env.step([v['response']['action'] for v in frame['calls']])
    # Only this offline identity test uses a known generator. Live training must take the genuine OS-seeded state.
    actor.mod._behavior_rng=np.random.default_rng(751901)
    mod=actor.mod;snap=snapshot(actor);before=fingerprint(actor)
    start_rng=history.state_hash(mod._behavior_rng.bit_generator.state)
    a,ea=act_with_choice(actor,obs,cfg,0);first=fingerprint(actor)
    after_a=history.state_hash(mod._behavior_rng.bit_generator.state)
    if first==before or after_a==start_rng or not ea['forced']:
        raise ValueError('No physical state transition in fixture')
    mod._obs_key=(-17,-17);mod._COUNTS.clear();mod._chosen_route=not mod._chosen_route
    mod._FEAT.est.fill(1234);mod._NETS[0].trunk.fill(-91);mod._NETS[1].trunk.fill(-17)
    mod._capture={'poison':True};mod._forced=(7,0);mod._behavior_rng=None
    restore(actor,snap)
    if fingerprint(actor)!=before:raise ValueError('Restore after deliberate state corruption')
    b,eb=act_with_choice(actor,obs,cfg,1);second=fingerprint(actor)
    if ea['options']!=eb['options'] or ea['p']!=eb['p'] or a==b or first!=second or after_a!=history.state_hash(mod._behavior_rng.bit_generator.state):
        raise ValueError('Both requested commands and common stream must be real')
    restore(actor,snap);normal1=actor.act(obs,cfg);post=fingerprint(actor);restore(actor,snap)
    normal2,en=act_with_choice(actor,obs,cfg);post_tapped=fingerprint(actor)
    if normal1!=normal2 or post_tapped!=post or en['forced'] or en['unforced_choice']!=en['pick']:
        raise ValueError('Observer changed unmodified public choice/state')
    restore(actor,snap)
    # Frozen policy identity fails closed when a derived/static tensor is changed.
    test=mod._NETS[0].w['U.bias'];original=test[0].copy();test[0]+=1
    try:
        try:restore(actor,snap)
        except ValueError:guard=True
        else:raise ValueError('Immutable tensor mutation passed')
    finally:
        test[0]=original
    restore(actor,snap)
    tree=ast.parse(Path(mod.__file__).read_bytes())
    embedded=mod._SRC
    writes=sorted({x for node in ast.walk(tree) if isinstance(node,ast.Global) for x in node.names})
    class_writes={}
    for name,text in embedded.items():
        source=ast.parse(text)
        for cl in (x for x in source.body if isinstance(x,ast.ClassDef)):
            for func in (x for x in cl.body if isinstance(x,ast.FunctionDef)):
                assigned=sorted({x.attr for x in ast.walk(func) if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='self' and isinstance(x.ctx,ast.Store)})
                if assigned:class_writes[f'{name}.{cl.name}.{func.name}']=assigned
    return dict(schema='native-851c-student-same-process-state/1',source_sha256=SOURCE_SHA,
                implementation_sha256=play.sha(Path(__file__).read_bytes()),model_static_digest=snap['static'],
                module_global_declarations=writes,package_embedded_sources=mod._SRC_SHA256,
                embedded_self_attr_writes=class_writes,student_dynamic_feature_fields=sorted(snap['dynamic']['features']),
                saved_module_fields=MODULE_FIELDS,variable_policy_caches=['expert.trunk','student.trunk'],
                read_only_network_attributes=[sorted(vars(policy).keys()-{'trunk'}) for policy in mod._NETS],
                restored_harness_fields=['wall','cpu'],fixture=dict(seed=ep['seed'],seat=0,step=12,
                    training_archive_sha256=ep['archive_sha256'],public_observation_sha256=frame['public_observation_sha256'][0],
                    diagnostic_student_generator_seed=751901,new_physical_games=0,actual_examiner_calls=0,
                    before=before,after_same_prefix=first,natural_equal_with_and_without_tap=True,immutable_mutation_rejected=guard,
                    actual_joint_orders=[a,b],probabilities=ea['p'],same_parent_rng_state_before=True,
                    after_single_draw_both_sha256=after_a,forced_commands_distinct=True,
                    restored_after_poisoned_feature_history_joint_cache_and_rng=True),
                scope='In-process restore is restricted to the exact warmed actor instance, on public call boundaries. Reuses immutable compiled code, feature geometry tables and network tensors; fingerprints the actual two dequantized networks and derivatives. Copies mutable memory and genuine bit-generator state. The test fork selects one registered real archived root and exercises the actual deployment twice, without a new physical game or any claim about the examiner, the environment, or wins.')


if __name__=='__main__':
    result=source_audit();dest=HERE/'student_state_audit.json';play.save(dest,result)
    print(json.dumps(result['fixture'],ensure_ascii=False))
