#!/usr/bin/env python3
"""Recover publicly viewable Carbon League replays; standard library only.

Examples (run from the repository root):
  python reports/master_family/deep_review/recover_public.py \
    --matches reports/master_family/deep_review/inputs/official_matches.json \
    --repo . --ref 37965ea5 --out run-cache/replays --insecure
  python reports/master_family/deep_review/recover_public.py \
    --source-manifest reports/master_family/deep_review/inputs/replay_sources.json --repo . --ref 37965ea5 \
    --out run-cache/replays --insecure

The second mode uses archived URLs and checks the canonical replay JSON SHA256.
No login/cookies/admin APIs, JavaScript execution, git mutation, or submissions.
Existing canonical replay filenames anywhere in the pinned Git tree are skipped.
A cache is used only after seat/reward/step validation and any archived hash check.
Public match IDs bind the URL and archived score row. The embedded engine UUID is
recorded separately; it is not incorrectly equated with the platform match ID.
"""
import argparse
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import ssl
import subprocess
import time
import threading
import urllib.parse
import urllib.request

NAME = re.compile(r'^([0-9a-fA-F]{32})-(\d+)\.json(?:\.gz)?$')
KEY = re.compile(r'^[0-9a-f]{32}-[1-9]\d*$')
CANONICAL = 'json.dumps(ensure_ascii=False,separators=(comma,colon),allow_nan=False); UTF-8'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def unpack(data):
    for _ in range(3):
        if data[:2] != b'\x1f\x8b':
            return data
        data = gzip.decompress(data)
    if data[:2] == b'\x1f\x8b':
        raise ValueError('too many gzip layers')
    return data


def extract(data):
    decoded = unpack(data)
    text = decoded.decode('utf-8')
    if text.lstrip().startswith('{'):
        obj = json.loads(text)
        env = obj.get('environment', obj)
        return env, decoded, 'json'
    needle = 'window.kaggle ='
    pos = text.index(needle) + len(needle)
    obj, _ = json.JSONDecoder().raw_decode(text[pos:].lstrip())
    return obj['environment'], decoded, 'public_spectator_html'


def walk_matches(obj):
    if isinstance(obj, list):
        for value in obj:
            yield from walk_matches(value)
    elif isinstance(obj, dict):
        if all(k in obj for k in ('id', 'game', 'name_a', 'name_b', 'scores', 'order')):
            yield obj
        for value in obj.values():
            if isinstance(value, (list, dict)):
                yield from walk_matches(value)


def key_of(row):
    key = str(row['id']).lower() + '-' + str(row['game'])
    if not KEY.fullmatch(key):
        raise ValueError('invalid match key: ' + key)
    return key


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def identity(row):
    order = row['order']
    if sorted(order) != ['a', 'b'] or any(not number(row['scores'].get(k)) for k in ('a', 'b')):
        raise ValueError('bad order or scores')
    return tuple((row['name_' + p], row['scores'][p]) for p in order)


def validate(env, row):
    key = key_of(row)
    expected = identity(row)
    rewards = env['rewards']
    if len(rewards) != 2:
        raise ValueError('requires two rewards')
    for seat, (_, score) in enumerate(expected):
        if not number(rewards[seat]) or abs(rewards[seat] - score) > 0.011:
            raise ValueError(f'{key}: seat {seat} reward mismatch')
    steps = env['steps']
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError('empty/incomplete replay')
    for index, pair in enumerate(steps):
        if len(pair) != 2:
            raise ValueError('requires two agents at every state')
        ob = pair[0]['observation']
        if ob['step'] != index or len(ob['players']) != 2:
            raise ValueError('non-contiguous steps or malformed players')
        if ob.get('player', 0) != 0 or pair[1]['observation'].get('player', 1) != 1:
            raise ValueError('observation seat mismatch')
    if len(steps) > env['configuration']['episodeSteps']:
        raise ValueError('steps exceed episode length')
    if len(env['statuses']) != 2 or any(s not in ('DONE', 'INVALID', 'TIMEOUT', 'ERROR') for s in env['statuses']):
        raise ValueError('nonterminal statuses')
    for seat in (0, 1):
        value = steps[-1][seat]['reward']
        if not number(value) or abs(value - rewards[seat]) > 0.011:
            raise ValueError('last state rewards disagree')
    return {'scores_checked': True, 'both_seats_checked': True, 'steps_checked': True,
            'step_count': len(steps), 'last_step': len(steps) - 1,
            'environment_uuid': env['id'], 'rewards_by_seat': rewards,
            'statuses': env['statuses'], 'order': row['order'],
            'seat_model_names': [item[0] for item in expected],
            'random_seed': env['configuration'].get('randomSeed')}


def repository_index(repo, ref):
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', ref], text=True).strip()
    output = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', revision], text=True)
    index = {}
    for line in output.splitlines():
        attrs, path = line.split('\t', 1)
        mode, kind, sha = attrs.split()
        match = NAME.fullmatch(path.rsplit('/', 1)[-1])
        if kind == 'blob' and match:
            key = match[1].lower() + '-' + str(int(match[2]))
            index.setdefault(key, []).append({'path': path, 'git_blob': sha})
    return revision, index


def load_records(path):
    obj = json.loads(Path(path).read_text())
    records = obj.get('records', []) if isinstance(obj, dict) else obj
    return {r['key']: r for r in records if isinstance(r, dict) and 'key' in r}


def checked_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('expected a public HTTP(S) URL without credentials')
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--matches', type=Path, action='append', default=[], help='Archived score-list JSON; repeatable; nested lists/dicts supported')
    parser.add_argument('--source-manifest', type=Path, help='Trusted archived sources JSON; URL and replay_json_sha256 are enforced')
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--ref', default='37965ea5', help='Pinned repository revision for whole-tree skip index')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-cache', type=Path, action='append', default=[])
    parser.add_argument('--base-url', default='https://117.72.210.134')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=35)
    parser.add_argument('--min-interval', type=float, default=2.0, help='Global minimum seconds between HTTP starts (default caps this process at 30/min)')
    parser.add_argument('--insecure', action='store_true', help='Allow the public IP service certificate mismatch; never sends authentication')
    parser.add_argument('--limit', type=int, help='Diagnostic limit on missing keys; not a complete recovery')
    args = parser.parse_args()
    if not args.matches and not args.source_manifest:
        parser.error('supply --matches or --source-manifest')
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be 1..16')
    if args.min_interval < 0:
        parser.error('--min-interval must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    revision, index = repository_index(args.repo, args.ref)
    source_records = load_records(args.source_manifest) if args.source_manifest else {}
    rows = {}; inputs = []
    for path in args.matches + ([args.source_manifest] if args.source_manifest else []):
        data = path.read_bytes(); inputs.append({'path': str(path), 'sha256': digest(data), 'bytes': len(data)})
        for row in walk_matches(json.loads(data)):
            if row.get('state', 'done') != 'done':
                continue
            key = key_of(row)
            if key in rows and identity(rows[key]) != identity(row):
                raise ValueError('conflicting score/name/seat identity for ' + key)
            identity(row); rows[key] = row
    if not rows:
        raise ValueError('no completed scored matches found')
    if args.source_manifest:
        for key in rows:
            rec = source_records.get(key)
            if key not in index and (not rec or not rec.get('url') or not rec.get('replay_json_sha256')):
                raise ValueError('trusted URL and replay_json_sha256 required for ' + key)
    cache_dirs = [args.out] + args.reuse_cache
    cache_sources = {}
    for directory in cache_dirs:
        for name in ('sources.json', 'public_sources.json'):
            path = directory / name
            if path.exists():
                for key, value in load_records(path).items():
                    if value.get('ok'):
                        cache_sources.setdefault(key, value)
    records = []
    for key in sorted(rows.keys() & index.keys()):
        records.append({'key': key, 'ok': True, 'status': 'repository_present', 'repository_paths': index[key], 'source_match': rows[key], 'validation': 'filename/key indexed; full replay validation belongs to downstream analysis'})
    missing = sorted(rows.keys() - index.keys())
    if args.limit is not None:
        missing = missing[:args.limit]
    started = time.time()
    context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
    request_lock = threading.Lock()
    next_request_at = [0.0]

    def job(key):
        row = rows[key]; expected = source_records.get(key, {}); error_cache = []
        url = checked_url(expected.get('url') or args.base_url.rstrip('/') + f"/replay/{row['id']}/{row['game']}")
        try:
            for directory in cache_dirs:
                path = directory / (key + '.json.gz')
                if not path.exists():
                    continue
                try:
                    packed = path.read_bytes(); env, decoded, kind = extract(packed); valid = validate(env, row)
                    data = canonical(env)
                    if expected and digest(data) != expected['replay_json_sha256']:
                        raise ValueError('cached replay SHA256 differs from trusted manifest')
                    prev = cache_sources.get(key, {})
                    if prev.get('gzip_sha256') and digest(packed) != prev['gzip_sha256']:
                        raise ValueError('cached gzip SHA256 differs from cache provenance')
                    if prev.get('replay_json_sha256') and digest(data) != prev['replay_json_sha256']:
                        raise ValueError('cached replay SHA256 differs from cache provenance')
                    if directory != args.out:
                        target = args.out / path.name; tmp = target.with_suffix('.tmp'); tmp.write_bytes(packed); tmp.replace(target)
                    return dict(prev, key=key, ok=True, status='cache_valid', cache_path=str(args.out / path.name), source_match=row,
                                url=prev.get('url', url), replay_json_sha256=digest(data), replay_json_bytes=len(data),
                                gzip_sha256=digest(packed), gzip_bytes=len(packed), validation=valid,
                                trusted_manifest_checked=bool(expected))
                except Exception as error:
                    error_cache.append(str(path) + ': ' + repr(error))
            request = urllib.request.Request(url, headers={'User-Agent': 'CarbonLeague-public-replay-audit/2.0', 'Accept': 'text/html,application/json', 'Accept-Encoding': 'gzip'})
            with request_lock:
                delay = next_request_at[0] - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                next_request_at[0] = time.monotonic() + args.min_interval
            with urllib.request.urlopen(request, context=context, timeout=args.timeout) as response:
                status = response.status; final = response.url; wire = response.read(); headers = dict(response.headers)
            if status != 200 or urllib.parse.urlsplit(final).path != urllib.parse.urlsplit(url).path:
                raise ValueError('unexpected status or redirected path')
            env, decoded, kind = extract(wire); valid = validate(env, row); data = canonical(env)
            if expected and digest(data) != expected['replay_json_sha256']:
                raise ValueError('downloaded canonical replay SHA256 differs from trusted manifest')
            packed = gzip.compress(data, compresslevel=9, mtime=0); target = args.out / (key + '.json.gz')
            tmp = target.with_suffix('.tmp'); tmp.write_bytes(packed); tmp.replace(target)
            return {'key': key, 'ok': True, 'status': 'downloaded', 'url': url, 'final_url': final, 'http_status': status,
                    'downloaded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'response_content_type': headers.get('Content-Type'), 'response_content_encoding': headers.get('Content-Encoding'),
                    'wire_bytes': len(wire), 'wire_sha256': digest(wire), 'decoded_source_sha256': digest(decoded), 'source_format': kind,
                    'replay_json_bytes': len(data), 'replay_json_sha256': digest(data),
                    'gzip_bytes': len(packed), 'gzip_sha256': digest(packed), 'cache_path': str(target),
                    'source_match': row, 'validation': valid, 'trusted_manifest_checked': bool(expected),
                    'tls_certificate_verification': not args.insecure, 'rejected_caches': error_cache}
        except Exception as error:
            return {'key': key, 'ok': False, 'status': 'failed', 'url': url, 'error': repr(error), 'source_match': row, 'rejected_caches': error_cache}

    def save():
        ids = [(r['key'], r['validation']['environment_uuid']) for r in records if r.get('ok') and isinstance(r.get('validation'), dict)]
        duplicated = len(ids) != len({u for _, u in ids})
        obj = {'schema': 'carbon-public-recovery-v2', 'revision': revision, 'inputs': inputs,
               'target_scored_games': len(rows), 'repository_present': sum(r['status'] == 'repository_present' for r in records),
               'missing_target_games': len(rows.keys() - index.keys()), 'selected_missing_games': len(missing),
               'public_recovered': sum(r['ok'] and r['status'] != 'repository_present' for r in records),
               'failed': sum(not r['ok'] for r in records), 'completed_records': len(records),
               'elapsed_seconds': round(time.time()-started, 3), 'canonical_json_format': CANONICAL,
               'duplicate_environment_uuid': duplicated, 'id_binding': 'exact spectator URL plus archived score-table ID/game; engine UUID is a separate identity',
               'records': sorted(records, key=lambda r:r['key'])}
        target = args.out / 'public_sources.json'; tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n'); tmp.replace(target)
        return duplicated
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in concurrent.futures.as_completed([pool.submit(job, key) for key in missing]):
            value = result.result(); records.append(value); save()
            print(json.dumps({'key': value['key'], 'status': value['status'], 'error': value.get('error'), 'finished_missing': len(records)-(len(rows.keys() & index.keys())), 'selected_missing': len(missing)}, ensure_ascii=False), flush=True)
    duplicate = save()
    return 1 if duplicate or any(not r['ok'] for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
