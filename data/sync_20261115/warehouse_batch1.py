#!/usr/bin/env python3
"""Batch-1 warehouse: LOCAL + L20 staging + local katago items -> carbon-league main.
Dedupe by git blob sha against current main tree; 422 re-parent retry; read-back verify."""
import sys, os, json, base64, hashlib, glob
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
BASE = '/home/hatch/workspace'
S = f'{BASE}/carbon-incoming/sync_20261115'
LO = f'{S}/LOCAL'
L = f'{S}/L20'
H = f'{L}/from_l20_host/ppou2'
V = f'{L}/from_vm_workspace/ppou2'
P = f'{L}/from_l20_host/probe_master2_repro'
D = f'{L}/from_l20_host/value-diag'

pairs = [
    (f'{LO}/war_plan_20261114.md', 'reports/WAR_PLAN_20261114.md'),
    (f'{LO}/katago_PREREG.md', 'reports/KATAGO_PROBE_PREREG_20261114.md'),
    (f'{LO}/PREREG_VALUE.md', 'reports/PREREG_VALUE_DIAG_20261114.md'),
    (f'{LO}/RECEIPT_AB.md', 'reports/VALUE_DIAG_RECEIPT_20261114.md'),
    (f'{LO}/katago3-exam.sh', 'tools/katago3/katago3-exam.sh'),
    (f'{LO}/smoke.sh', 'tools/katago3/smoke.sh'),
    (f'{LO}/snaploop.sh', 'tools/katago3/snaploop.sh'),
    (f'{LO}/analyze_katago3.py', 'tools/katago3/analyze_katago3.py'),
    (f'{LO}/N2B_RECEIPT.md', 'reports/N2B_RECEIPT_20261114.md'),
    (f'{LO}/gh_api.py', 'tools/gh_api.py'),
    (f'{LO}/warehouse_pull0045_csplant.py', 'tools/warehouse_pull0045_csplant.py'),
    (f'{LO}/LOG_20261114.md', 'reports/war_room_log/LOG_20261114.md'),
    (f'{LO}/LOG_20261115_snapshot.md', 'reports/war_room_log/LOG_20261115_snapshot_0245.md'),
    (f'{LO}/bridge_echo3bridge.json', 'reports/BRIDGE_ECHO3_ECHO2_20261115.json'),
]
for n in ['launch_ppou2_v4.py', 'check_anchor_equal.py', 'check_arms.py',
          'remote_anchor_check.py', 'remote_edit_kl.py', 'remote_round_check.py',
          'remote_serial_check.py', 'remote_verify_arm.py']:
    pairs.append((f'{H}/{n}', f'tools/ppou2/{n}'))
pairs += [
    (f'{V}/FILING.md', 'reports/PPOU2_FILING_20261114.md'),
    (f'{V}/SERIAL_PLAN.md', 'reports/PPOU2_SERIAL_PLAN_20261115.md'),
    (f'{V}/CHECKPOINT_PROCEDURE.md', 'reports/PPOU2_CHECKPOINT_PROCEDURE.md'),
    (f'{V}/anchor_watch_result.txt', 'reports/PPOU2_ANCHOR_WATCH_RESULT_20261114.txt'),
]
for n in ['anchor_watch.py', 'anchor_watch.sh', 'first_round_watch.py', 'first_round_watch.sh',
          'first_round_watch_m.py', 'launch_ppou2.py', 'launch_ppou2_v2.py', 'launch_ppou2_v3.py',
          'serial_driver.py', 'run_step0.sh', 'summarize_step0.py', 'watchdog_serial_driver.sh']:
    pairs.append((f'{V}/{n}', f'tools/ppou2/{n}'))
for n in ['README.md', 'ppou2_m.log', 'ppou2_m_config.json', 'ppou2_m_effective_params.json',
          'ppou2_m_league.json']:
    pairs.append((f'{V}/oom_evidence_20261115/{n}', f'reports/PPOU2_OOM_EVIDENCE_20261115/{n}'))
pairs += [
    (f'{P}/PREREG.md', 'reports/PROBE_MASTER2_REPRO_PREREG_20261114.md'),
    (f'{P}/data/bigeval.jsonl', 'data/probe_master2_repro/bigeval.jsonl'),
    (f'{D}/out/results.json', 'reports/VALUE_DIAG_RESULTS_20261114.json'),
    (f'{D}/out/realized_u_provenance.json', 'reports/VALUE_DIAG_PROVENANCE_20261114.json'),
    (f'{D}/realized_u.py', 'tools/value_diag/realized_u.py'),
    (f'{BASE}/katago-probe/impl/cnet2_search_vleaf.py', 'code/exp/cnet2_search_vleaf.py'),
    (f'{BASE}/katago-probe/impl/RECEIPT_GATE.md', 'reports/KATAGO_GATE_RECEIPT_20261114.md'),
    (f'{BASE}/katago-probe/impl/RECEIPT_CONSTRUCTION.md', 'reports/KATAGO_CONSTRUCTION_RECEIPT_20261114.md'),
    (f'{BASE}/katago-probe/impl/gate_driver.py', 'tools/katago/gate_driver.py'),
    (f'{BASE}/carbon-live/katago3/katago_vleaf_value.py', 'code/exp/vleaf_value_pack.py'),
    (f'{BASE}/carbon-live/katago3/katago_vleaf_const.py', 'code/exp/vleaf_const_pack.py'),
]
for p in sorted(glob.glob(f'{P}/data/bigeval_live/*.jsonl')):
    pairs.append((p, 'data/probe_master2_repro/bigeval_live/' + os.path.basename(p)))
for pat in ['run_step*.sh', 'step*.log', 'step*_done.txt']:
    for p in sorted(glob.glob(f'{P}/{pat}')):
        pairs.append((p, 'data/probe_master2_repro/run/' + os.path.basename(p)))

def blob_sha(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()

items = []
for src, dst in pairs:
    data = open(src, 'rb').read()
    items.append({'src': src, 'dst': dst, 'data': data, 'sha': blob_sha(data)})
print(f'ITEMS {len(items)} TOTAL_BYTES {sum(len(i["data"]) for i in items)}', flush=True)

def head_and_tree():
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    tree_sha = commit['tree']['sha']
    tree = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/trees/{tree_sha}?recursive=1')
    m = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}
    return head, tree_sha, m

head, tree_sha, cur = head_and_tree()
print('HEAD', head, flush=True)
todo = [i for i in items if cur.get(i['dst']) != i['sha']]
print(f'TODO {len(todo)} SKIP_IDENTICAL {len(items) - len(todo)}', flush=True)

for i in todo:
    for attempt in range(2):
        try:
            r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                           {'content': base64.b64encode(i['data']).decode(), 'encoding': 'base64'})
            break
        except Exception as e:
            if attempt == 1:
                raise
            print('blob retry', i['dst'], e, flush=True)
    assert r['sha'] == i['sha'], (i['dst'], r['sha'], i['sha'])
    print('blob ok', i['dst'], flush=True)

newhead = None
for attempt in range(4):
    entries = [{'path': i['dst'], 'mode': '100644', 'type': 'blob', 'sha': i['sha']} for i in todo]
    nt = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees', {'base_tree': tree_sha, 'tree': entries})
    nc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                    {'message': 'sync 2026-11-15 batch1: ppou2 code+filing+serial plan, katago vleaf+packs+gate receipts, step0/probe/value-diag results, war plan update (war room)',
                     'tree': nt['sha'], 'parents': [head]})
    try:
        gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main', {'sha': nc['sha']})
        newhead = nc['sha']
        break
    except Exception as e:
        print('patch failed, re-parenting:', e, flush=True)
        head, tree_sha, cur = head_and_tree()
        todo = [i for i in todo if cur.get(i['dst']) != i['sha']]
        if not todo:
            newhead = head
            break
if newhead is None:
    raise SystemExit('FAILED: ref patch')
print('NEW_HEAD', newhead, flush=True)

commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{newhead}')
tree = gh_api.req('GET', f"/repos/{OWNER}/{REPO}/git/trees/{commit['tree']['sha']}?recursive=1")
m = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}
bad = [i['dst'] for i in items if m.get(i['dst']) != i['sha']]
print('READBACK', 'ALL_OK' if not bad else 'BAD:' + ','.join(bad), flush=True)
