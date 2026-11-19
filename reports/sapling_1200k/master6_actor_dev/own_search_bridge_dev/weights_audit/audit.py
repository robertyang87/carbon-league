"""Inventory usable actors and recognize opponent tensors by their decoded bytes; local Git objects only."""
import ast
import base64
from collections import Counter
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PREFIX = 'reports/sapling_1200k/master6_actor_dev/'
BEFORE = 'b2069da2f42a18c3e0e0296d2e16e45b3681d7cc'
STAGE = 'd944673f8c6487cb4cd9ff10a001cc61735d157c'
END = '3d0e0537fc94332a58636ddddf88ecfbdb99758b'
ASSETS = {
    'master3s2': 'sources/master3s2/master3s2.py', 'apex_fix3': 'code/apexcarbon_v1_fix3.py',
    'apex_fix2': 'code/apexcarbon_v1_fix2.py', 'apex_fix1': 'code/apexcarbon_v1_fixed.py',
    'apex_v2': 'code/apexcarbon_v2.py', 'echo7_apex_v1': 'code/apexcarbon_v4tta_search.py',
    'echo3': 'code/packs/echo3_int8.py', 'echo4_it20': 'code/packs/echo4_int8.py',
    'echo5_it40': 'code/packs/ppou2_m_it40.py', 'echo6': 'sources/echo6/echo4_search_int8_d1_homefix.py',
    'it45': 'code/packs/ppou2_m_it45.py', 'it50': 'code/packs/ppou2_m_it50.py',
    'return_base': 'data/weights/sapling_1200k/terminal_return_fit_v1.py',
    'master6_sapling_npz': 'data/weights/sapling_1200k/master6_actor_v1.npz',
    'cash_sapling_v3': 'data/weights/sapling_1200k/cash_v3.npz',
    **{k: PREFIX + v for k, v in {
        'echo4_sapling': 'actor_partner_dev/candidate.py',
        'encoder_update': 'blended_actor_dev/serving/candidate.py', 'unmixed_training_control': 'blended_actor_dev/serving/control.py',
        'encoder_update_npz': 'blended_actor_dev/candidate.npz', 'unmixed_training_control_npz': 'blended_actor_dev/control.npz',
        'terminal_update': 'blended_policy_return_dev/candidate.py', 'terminal_update_npz': 'blended_policy_return_dev/candidate.npz',
        'response_critic': 'joint_response_critic_dev/candidate.py', 'step0_delta': 'blended_actor_dev/opening_training/opening_weights.npz',
    }.items()}}


def h(b):
    return hashlib.sha256(b).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def blob(path):
    return git('show', END + ':' + path)


def npz(data):
    with np.load(BytesIO(data), allow_pickle=False) as a:
        params = {k: [a[k].dtype.str, list(a[k].shape)] for k in a.files if a[k].dtype.kind in 'ufbi'}
        raw_meta = a['meta'].tolist() if 'meta' in a else None
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
        if isinstance(meta, dict):
            core = {k: meta[k] for k in ('cfg','it','model_name','preset','params') if k in meta}
            core.update(original_meta_keys=sorted(meta), full_meta_sha256=h(str(a['meta'].item()).encode()))
        else:
            core = meta
        return {'npz_sha256': h(data), 'tensor_count': len(params),
                'tensor_elements': sum(int(np.prod(s)) for _,s in params.values()), 'meta': core,
                'shape_and_dtype_sha256': h(dump(params)), 'examples': list(params.items())[:4]}


def dump(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':')).encode()


def pack(data):
    wanted = {'_WEIGHTS','_VALUE','_B1','_Q1','_M1','_W64','_QMETA','_WMETA','CONF'}
    constants = {}
    for node in ast.parse(data).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            k = node.targets[0].id
            if k in wanted:
                try: constants[k]=ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
    v = {}
    for name in ('_WEIGHTS','_VALUE'):
        if name in constants:
            v[name] = npz(base64.b64decode(constants[name]))
    for keys, label in [(('_B1','_Q1','_M1'),'actor'), (('_W64','_QMETA','_WMETA'),'student')]:
        if not all(k in constants for k in keys):
            continue
        raw_meta = constants[keys[2]]
        meta = json.loads(raw_meta) if isinstance(raw_meta,str) else raw_meta
        if not isinstance(meta,dict): meta={'stored_value':meta}
        subset = {k:meta[k] for k in ('cfg','it','model_name','preset','params') if k in meta}
        v[label] = {'stored_actor_with_metadata_sha256': h(dump({k:constants[k] for k in keys})),
                    'encoded_payload_bytes_sha256': h(base64.b64decode(constants[keys[0]])),
                    'meta': subset, 'all_meta_keys': sorted(meta), 'meta_sha256': h(raw_meta.encode() if isinstance(raw_meta,str) else dump(raw_meta))}
    if 'CONF' in constants:
        v['actor_configuration'] = {k: constants['CONF'][k] for k in ('SEARCH','TEACHER','TEACHER_LATE','CMP_H','CMP_TAU','MODE','GUARD') if k in constants['CONF']}
    return v


def audit():
    git('merge-base','--is-ancestor',BEFORE,END)
    allfiles={}
    for line in git('ls-tree','-rl',END).decode().splitlines():
        hdr,p=line.split('\t',1)
        _,_,oid,size=hdr.split()
        allfiles[p]=(oid,int(size))
    diff=set(git('diff','--name-only',BEFORE,END).decode().splitlines())
    asset={}
    for name,path in ASSETS.items():
        data=blob(path)
        slots = {'npz':npz(data)} if path.endswith('.npz') else pack(data)
        assert slots,name
        asset[name]={'path':path,'sha256':h(data),'git_blob':allfiles[path][0],'bytes':len(data),
                     'added_or_changed':path in diff, **slots}
    orig=asset['master3s2']
    for item in asset.values():
        if '_WEIGHTS' in item:
            item['exact_native_actor_and_small_value_npz']=all(item[k]['npz_sha256']==orig[k]['npz_sha256'] for k in ('_WEIGHTS','_VALUE'))
    prior=json.loads(blob(PREFIX+'actor_partner_dev/evidence.json'))
    for name,key in [('echo4_it20','echo4'),('echo6','echo6'),('echo5_it40','it40_echo5'),('it50','it50')]:
        assert asset[name]['actor']['stored_actor_with_metadata_sha256']==prior['assets'][key]['stored_actor_with_metadata_sha256']
    it55='b42a6c3438530d9cfebe27abb96d81c6'
    named=[p for p in allfiles if re.search(r'(?:it55|ckpt[_-]?55|m55_e4)',p,re.I)]
    same_size=[p for p,(_,n) in allfiles.items() if n==5747509]
    strong_ref=[p for p in allfiles if 'krypton' in p.lower() and Path(p).suffix.lower() in ('.py','.npz','.pt','.bin','.pth','.onnx','.safetensors')]
    external=[]
    for i in (1,2):
        folder=Path('/workspace')/('sapling-win-ppo-%03d'%i)
        if folder.is_dir():
            path=folder/'ckpt_4.npz'
            d=path.read_bytes()
            same_root=[q for q,(_,n) in allfiles.items() if q.endswith('.npz') and n==len(d) and h(blob(q))==h(d)]
            def local_file(p):return {'path':str(p),'sha256':h(p.read_bytes())}
            external.append({'folder':str(folder),'same_hash_tracked_npz':same_root,'checkpoint':npz(d),
                            'same_ckpt_last_bytes': d==(folder/'ckpt_last.npz').read_bytes(),
                            'pilot_identity':local_file(folder/'pilot_identity.json'),'state_last':local_file(folder/'state_last.pt'),
                            'source_parent_sha256':json.loads((folder/'pilot_identity.json').read_bytes())['parent_sha256']})
    ir=PREFIX+'blended_actor_dev/opening_training/release.json'
    rel=json.loads(blob(ir))
    local=Path('/workspace/sapling-opening-trained-agent.py')
    lc={'release':rel,'release_file_sha256':h(blob(ir)),'local_confirmed':local.exists() and h(local.read_bytes())==rel['materialized_sha256']}
    if lc['local_confirmed']:lc['parsed']=pack(local.read_bytes())
    receipt={}
    for s,path in {
        'prior_dossier': 'actor_partner_dev/evidence.json', 'different_historical_4':'actor_partner_dev/comparison.json',
        'fresh_dual_arms_8':'actor_partner_dev/replication/comparison.json','fresh_blend_8':'blended_actor_arena/comparison.json',
        'fresh_opening_4_reuse_parent_4':'opening_pair_dev/comparison.json','opening_same_history':'opening_pair_dev/parity.json.gz',
        'proxy_terminal_update_only':'blended_policy_return_dev/result.json','response_critic_gate':'joint_response_critic_dev/verdict.json'
    }.items():
        d=blob(PREFIX+path)
        j=json.loads(gzip.decompress(d) if path.endswith('.gz') else d)
        r={k:j[k] for k in ('arms','by_arm','paired_mean_margin_difference','paired_delta','completed_games','independent_maps','completed_native_games','observed_t0_changes','candidate_vs_actual_differences','post_opening_parent_candidate_differences','decision','classification','pass') if k in j}
        receipt[s]={'path':PREFIX+path,'sha256':h(d),'summary':r}
    return {'freeze':END,'stage':STAGE,'after_stage_commit_count':int(git('rev-list','--count',STAGE+'..'+END)),
            'post_stage_changed_program_or_weight':git('diff','--name-only',STAGE,END,'--','code','sources','data/weights').decode().splitlines(),'since':BEFORE,'git_commits':int(git('rev-list','--count',BEFORE+'..'+END)),'n_files':len(allfiles),
            'changed_code_sources_data_weights':[p for p in sorted(diff) if p.startswith(('code/','sources/','data/weights/'))],
            'changed_large_or_executable_model_candidates':[{'path':p,'bytes':n,'git_blob':oid} for p,(oid,n) in allfiles.items()
                if p in diff and p.endswith(('.py','.npz','.pt','.pth','.onnx','.safetensors','.bin')) and n>300000],
            'it55_registration_md5':it55,'all_paths_containing_it55_signature':named,'all_files_with_exact_registered_size':same_size,
            'trees_named_krypton_matching_model_or_code':strong_ref,'tested_weight_sources':asset,
            'local_sapling_ppo':external,'opening_materialization':lc,'match_sources':receipt}


if __name__=='__main__':
    r=audit()
    (HERE/'identity.json').write_text(json.dumps(r,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
