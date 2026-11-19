"""Exercise exact admission and full action/ablation proofs on two existing development trajectories."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
sys.path.insert(0,str(BASE))
import compound_formal as formal
import terminal_compound_formal as terminal
import tree_win_gate as gate
from turning_point_ledger import compact


def rejected(fn):
    try:fn()
    except (ValueError,FileNotFoundError) as exc:return str(exc)
    raise AssertionError("unexpected admission")


def run():
    actors=gate.ROOT/formal.ACTORS
    pack=actors/'terminal_return_fit_v1.py';weights=actors/'master6_actor_v1.npz'
    data=HERE/'dev_learned';games=json.loads((data/'results.json').read_bytes())['games']
    out=HERE/'formal_gate_evidence';out.mkdir(exist_ok=True)
    checks={}
    checks['rebuild_npz_pack']=gate.prove_candidate(weights,pack)
    assert checks['rebuild_npz_pack']==(terminal.TRAINED,formal.TERMINAL)
    checks['wrong_npz']=rejected(lambda:gate.prove_candidate(actors/'cash_v3.npz',pack))
    checks['reject_manifest_as_weights']=rejected(lambda:gate.prove_candidate(HERE/'learned_checkpoint.json',pack))
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        altered=folder/'changed.py';altered.write_bytes(pack.read_bytes()+b'\n')
        checks['changed_candidate']=rejected(lambda:formal.prove(weights,altered,formal.TERMINAL))
        with patch.dict(terminal.RECEIPTS,{altered:"0"*64}):
            checks['changed_training_input']=rejected(lambda:gate.prove_candidate(weights,pack))
        witnesses=[]
        for g in games:
            seed,seat=g['seed'],g['candidate_seat']
            if seed!=621461220 or seat not in (0,1):
                raise ValueError("this probe accepts only the pre-existing first development pair")
            row=copy.deepcopy(g);p=data/row['raw_file'];raw=json.loads(gzip.decompress(p.read_bytes()))
            assert gate.digest(p)==row['raw_sha256'] and raw['pack_sha256']==formal.TERMINAL
            original=row['raw_sha256']
            if seat==0:
                # Exercise the formal entry point on scratch bytes. Only the original
                # run's manifest identifier is replaced with the actual rebuilt NPZ ID.
                raw['weight_identity']=terminal.TRAINED
                tape=folder/row['raw_file'];tape.write_bytes(gzip.compress(compact(raw),mtime=0))
                row['raw_sha256']=gate.digest(tape)
                checks['wrong_envelope']=rejected(lambda:gate.verify_raw(folder,row,formal.TERMINAL,formal.STUDENT,formal=True))
                income=gate.verify_raw(folder,row,formal.TERMINAL,terminal.TRAINED,formal=True,income=True,record_compound=True)
                assert income['official_margin']==row['margin']
                proof=row['compound_policy_audit']
            else:
                proof=formal.verify_actions(folder,row,raw,formal.TERMINAL,record=True)
            assert (proof['verified_roots']==row['actual_transitions']==299 and proof['student_changes']>0
                    and proof['drop_sapling_changed_roots']>=proof['student_changes'] and proof['residual_changed_roots']>0)
            witness=folder/proof['evidence_file']
            steps=json.loads(gzip.decompress(witness.read_bytes()))['turns']
            assert sum(z['full']!=z['drop_sapling'] for z in steps)==proof['drop_sapling_changed_roots']
            assert sum(z['full']!=z['expert_normalized'] for z in steps)==proof['normalized_expert_changed_roots']
            assert sum(z['first_same_prefix_sapling'] is not None for z in steps)==proof['student_changes']
            public=out/witness.name
            if public.exists() and public.read_bytes()!=witness.read_bytes():
                raise ValueError('existing regression evidence differs')
            if not public.exists():shutil.copyfile(witness,public)
            witnesses.append({'seat':seat,'original_development_tape_sha256':original,'scratch_npz_envelope':seat==0,'proof':proof})
            if seat==1:
                broken=copy.deepcopy(raw)
                bid=next(iter(broken['actions'][0][seat]))
                broken['actions'][0][seat][bid]='RECPLANTER' if broken['actions'][0][seat][bid]=='RECCOLLECTOR' else 'RECCOLLECTOR'
                checks['changed_executed_command']=rejected(lambda:formal.verify_actions(folder,row,broken,formal.TERMINAL))
                # Preserve binding fields and rehash the edited artifact; reconstructed contents must reject it.
                doc=json.loads(gzip.decompress(witness.read_bytes()))
                doc['turns'][0]['drop_sapling']={'forged':'LEFT'}
                witness.write_bytes(gzip.compress(compact(doc),mtime=0))
                checks['forged_no_sapling_counterfactual']=rejected(lambda:formal.verify_actions(folder,row,raw,formal.TERMINAL))
        checks['development_games']=witnesses
    info={'scope':'Existing map 621461220 only, both physical seats; no formal map draw, game, or score claim. '
                  'Seat0 uses a temporary formal NPZ identity envelope; seat1 retains the original archive hash.',
          'new_verifier_sha256':gate.digest(Path(terminal.__file__)), 'gate_sha256':gate.digest(Path(gate.__file__)),
          'checks':checks}
    path=out/'tests.json'
    path.write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(witnesses),flush=True)


def sampled_prefixes():
    from tree_audited_game import exam
    folder=HERE/'dev_learned';rows=json.loads((folder/'results.json').read_bytes())['games']
    F,policies=terminal.networks();records=[]
    for row in rows:
        if row['seed']!=621461220:raise ValueError("sample restricted to archived development map")
        path=folder/row['raw_file'];raw=json.loads(gzip.decompress(path.read_bytes()))
        if gate.digest(path)!=row['raw_sha256'] or raw['pack_sha256']!=formal.TERMINAL:raise ValueError("unbound sample")
        seat=row['candidate_seat'];f=F.Feat(seat)
        sim=exam.FastCarbon(raw['game_configuration']).reset(seed=row['seed'])
        for t,joint in enumerate(raw['actions']):
            obs=sim.obs(seat)
            if t in (0,64,128,192,256,298):
                r=terminal.RootDecoder(obs,f,F,policies)
                y,guard,base=r.choice(.75,.25);off,og,_=r.choice(.75,0.);full=f.to_actions(y)
                if full!=joint[seat]:raise ValueError("production action mismatch on fixed real root")
                records.append({'seat':seat,'t':t,'full':full,'drop_sapling':f.to_actions(off),'vetoes':[guard,og]})
            else:f.observe(obs)
            sim.step(*joint)
    if len(records)!=12:raise ValueError("missing sampled real root")
    out=HERE/'formal_gate_evidence'/'sampled_prefixes.json'
    out.parent.mkdir(exist_ok=True);out.write_text(json.dumps({'sampled_real_roots':records,
        'verifier_sha256':gate.digest(Path(terminal.__file__))},indent=2)+'\n')
    print("12 actual root actions reproduced",flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--smoke']:sampled_prefixes()
    elif len(sys.argv)==1:run()
    else:raise SystemExit("expected --smoke or no argument")
