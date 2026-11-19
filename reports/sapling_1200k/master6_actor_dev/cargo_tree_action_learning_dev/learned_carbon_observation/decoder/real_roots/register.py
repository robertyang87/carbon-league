"""Register the first actual decoder-path worker differences on the two preregistered histories."""
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np

HERE=Path(__file__).resolve().parent
CARBON=HERE.parents[1]
sys.path.insert(0,str(CARBON))
from common import ROOT, F, read, save, sha
sys.path.insert(0,str(CARBON/"active_roots"))
from select_roots import streaming_archive, begin, public, clean, digest

ORDER=("NONE","UP","RIGHT","DOWN","LEFT")


def released(path,signature):
    raw=path.read_bytes()
    assert sha(raw)==signature
    m=types.ModuleType("public_"+signature[:12]);m.__file__=str(path)
    exec(compile(raw,str(path),"exec"),m.__dict__)
    # Preserve the public return value from the unmodified decode. Only remember
    # the consumed x/aux to independently replay both action maps' legal prefixes.
    normal=m._mixed_actions
    def remember(x,aux):
        y=normal(x,aux)
        m.last_inputs=(x,aux,y.copy())
        return y
    m._mixed_actions=remember
    return m


def legality(m,command):
    x,aux,y=m.last_inputs
    assert m._FEAT.to_actions(y)==command
    prefix=F.Prefix(x,aux)
    logs=[]
    for slot in np.flatnonzero(x["tok_mask"][:F.OWN]):
        uid=m._FEAT.own_uid[slot]
        _,_,leg=prefix.feats(slot)
        cmd=int(y[slot])
        assert leg[cmd]
        logs.append(dict(slot=int(slot),uid=uid,canonical_cmd=cmd,physical_cmd=command.get(uid,"NONE"),
                         legal=leg.astype(bool).tolist(),cash=float(prefix.cash),recruit_slots=int(prefix.slots)))
        prefix.advance(slot,cmd)
    return dict(slots=logs,friendly_conflicts=sorted(m._return_conflicts(x,y)))


def main():
    plan=read(HERE/"protocol.json")
    policies=plan["policies"]
    val=[c for c in read(CARBON/"protocol.json")["cards"] if c["split"]=="val"]
    historic=read(HERE.parent/"held_turns.json.gz")
    fixed={(r["seed"],r["seat"],r["t"]):r["commands"] for r in historic}
    roots=[]
    for card,seat in zip(val[:2],(1,0),strict=True):
        m=released(CARBON/"candidate.py",policies["parent_encoder_only"])
        n=released(HERE.parent/"candidate.py",policies["alternative_encoder_decoder"])
        source=ROOT/card["path"]
        assert sha(source.read_bytes())==card["sha256"]
        meta,tape=streaming_archive(source)
        assert meta["initial_terrain_u8_sha256"]==card["terrain"]
        env=begin(meta);cfg={**meta["game_configuration"],"randomSeed":None}
        seq=[];prior=[];result=None
        for t in range(299):
            obs=public(env,seat)
            assert set(obs)=={"step","player","players","carbon","trees","remainingOverageTime"} and obs["step"]==t
            osig=digest({k:v for k,v in obs.items() if k!="remainingOverageTime"})
            p,a=m.agent(obs,cfg),n.agent(obs,cfg)
            reference=fixed[card["seed"],seat,t]
            assert (p,a)==(reference["encoder_student"],reference["decoder_student"])
            workers=obs["players"][seat][2]
            changed=sorted(u for u in workers if p.get(u,"NONE")!=a.get(u,"NONE"))
            seq.append(dict(t=t,public_obs_sha256=osig,parent=p,student=a))
            if changed:
                plegal=legality(m,p); alegal=legality(n,a)
                # Root enemy is inspected only after first emitted worker change has been fixed.
                enemy=clean(next(tape)[1-seat])
                changed_detail=[]
                for u in changed:
                    pos,load,role=workers[u]
                    cmds=[arm.get(u,"NONE") for arm in (p,a)]
                    changed_detail.append(dict(uid=u,role=role,cargo=float(load),start_cell=pos,physical_commands=cmds,
                        destinations=[int(F.DEST[pos,ORDER.index(c)]) for c in cmds]))
                case_id=f"{card['seed']}_{seat}"
                witness=HERE/(case_id+"_prefix.json.gz")
                save(witness,dict(physical_fixed_history=prior,both_normal_commands_at_each_public_state=seq,
                                  root_public_observation=obs,public_configuration=cfg))
                result=dict(id=case_id,seed=card["seed"],seat=seat,t=t,terrain=card["terrain"],source=card["path"],
                    source_sha256=card["sha256"],prefix_sha256=digest(prior),public_obs_sha256=osig,kind="actual_worker_change",
                    prefix_file=witness.name,prefix_file_sha256=sha(witness.read_bytes()),seq_sha256=digest(seq),
                    parent=p,student=a,root_enemy=enemy,physical_changed_workers=changed_detail,
                    all_changed_uids=sorted(u for u in p.keys()|a.keys() if p.get(u)!=a.get(u)),
                    legal=dict(parent=plegal,student=alegal),earlier_worker_differences=0,checked_public_steps=len(seq),
                    identity_sha256=digest([card["terrain"],seat,t,osig,[p,a],enemy]))
                break
            if t<298:
                pair=next(tape);prior.append(pair);env.step(pair)
        tape.close()
        roots.append(result if result else dict(seed=card["seed"],seat=seat,terrain=card["terrain"],
            source=card["path"],source_sha256=card["sha256"],kind="no_worker_difference",checked_public_steps=len(seq),seq_sha256=digest(seq)))
    save(HERE/"roots.json",dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),policies=policies,cases=roots,
        held_turns_sha256=sha((HERE.parent/"held_turns.json.gz").read_bytes()),method="Chronological literal normal public entries on each separate physical history, first worker difference per registered map/seat. The same fixed trajectory was previously used for held activity. New outcomes absent."))

if __name__=="__main__":main()
