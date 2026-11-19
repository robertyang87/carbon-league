"""Audit search evidence for a raw original-examiner game without importing a policy."""
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path


ORIGINAL = 'data/master3s2_source.py'
MD5 = '57f9a594cce4f2ec9e3573d8d9d26ca1'
SHA256 = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
KINDS = {'searched', 'single', 'depth_zero_return_base'}


def _hex(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=4)
def _original(path, version):
    data = path.read_bytes()
    return hashlib.md5(data).hexdigest(), _hex(data)


def _read(path, repo, evidence, role):
    b = path.read_bytes()
    evidence[role] = {'path': str(path.relative_to(repo)), 'sha256': _hex(b)}
    return json.loads(gzip.decompress(b) if path.suffix == '.gz' else b)


def _receipt(raw, path, repo, evidence, issues):
    seed, seat = raw.get('seed'), raw.get('candidate_seat')
    tag = f'{seed}_{seat}'
    link = evidence['raw']['sha256']
    single = path.with_name(f'receipt_{tag}.json')
    batch = path.with_name('results.json')
    native = path.with_name(f'native_seat{seat}.json')
    header, source_sha = {}, None
    if single.is_file():
        obj = _read(single, repo, evidence, 'native_receipt')
        row = obj.get('game', {})
        binding = obj.get('binding', {})
        source_sha = binding.get('original', binding.get('original_sha256'))
    elif batch.is_file():
        obj = _read(batch, repo, evidence, 'native_receipt')
        rows = [r for r in obj.get('games', []) if r.get('raw_sha256') == link
                and r.get('seed') == seed and r.get('candidate_seat') == seat]
        if len(rows) != 1:
            issues.append('batch_raw_link_missing_or_ambiguous')
            return {}, {}, None
        row = rows[0]
        header = obj.get('header', {})
        source_sha = header.get('source_sha256', header.get('sources_sha256', {})).get(ORIGINAL)
    elif native.is_file():
        row = _read(native, repo, evidence, 'native_receipt')
        ticket = path.parent.parent.parent.parent / 'protocol.json'
        official = path.with_name(f'official_seat{seat}.json')
        if ticket.is_file() and official.is_file():
            plan = _read(ticket, repo, evidence, 'experiment_protocol')
            proof = _read(official, repo, evidence, 'official_receipt')
            if (proof.get('raw_sha256') == link
                    and proof.get('native_receipt_sha256') == evidence['native_receipt']['sha256']
                    and proof.get('protocol_sha256') == evidence['experiment_protocol']['sha256']
                    and proof.get('official', {}).get('transcript_sha256') == link
                    and proof['official'].get('seed') == seed and proof['official'].get('seat') == seat):
                source_sha = plan.get('examiner_source_sha256')
            else:
                issues.append('experiment_protocol_link_mismatch')
    else:
        issues.append('native_receipt_missing')
        return {}, {}, None
    if (row.get('seed'), row.get('candidate_seat'), row.get('raw_file'), row.get('raw_sha256')) != (
            seed, seat, path.name, link):
        issues.append('native_receipt_raw_identity_mismatch')
        return {}, {}, None
    return row, header, source_sha


def _duration(x):
    return type(x) in (float, int) and math.isfinite(x) and x >= 0


def _times(samples, threshold):
    if not isinstance(samples, list) or not all(_duration(x) for x in samples):
        return None
    return {'observations': len(samples), 'sum_seconds': sum(samples), 'max_seconds': max(samples, default=None),
            'over_native_budget': sum(x > threshold for x in samples) if _duration(threshold) else None,
            'over_one_second': sum(x > 1 for x in samples)}


def _wall(raw, path, repo, evidence, issues, row):
    tag = f"{raw.get('seed')}_{raw.get('candidate_seat')}"
    wall = path.with_name(f'wall_{tag}.json.gz')
    if not wall.is_file():
        return {}, []
    data = _read(wall, repo, evidence, 'public_clock')
    identity = data.get('identity', {})
    calls = data.get('calls', [])
    events = raw.get('search_calls', [])
    if (data.get('schema') != 'blended-original-clock/1' or data.get('raw_sha256') != evidence['raw']['sha256']
            or identity.get('seed') != raw.get('seed') or identity.get('seat') != raw.get('candidate_seat')
            or identity.get('terrain') != raw.get('initial_terrain_u8_sha256')
            or len(calls) != len(events) or len(calls) != len(raw.get('actions', []))
            or data.get('step') != len(calls)
            or any(not isinstance(c, dict) or c.get('step') != i or [i, c.get('kind')] != events[i]
                   or c.get('kind') not in KINDS for i, c in enumerate(calls))):
        issues.append('public_clock_raw_identity_or_step_mismatch')
        return {}, []
    for c in calls:
        d, k, n = c.get('max_depth'), c.get('kind'), c.get('native_seconds')
        if (type(d) is not int or d < 0 or d > min(8, 299-c['step'])
                or (d > 0) != (k == 'searched') or type(c.get('native_aborted')) is not bool
                or not _duration(c.get('public_wall_seconds')) or not _duration(c.get('process_cpu_seconds'))
                or (n is not None if k == 'single' else not _duration(n) or n > c['public_wall_seconds'] + 1e-6)
                or (k == 'depth_zero_return_base' and not c['native_aborted'])):
            issues.append('invalid_per_call_native_clock')
            return {}, []
    chain = bytes(32)
    for commands in raw['actions']:
        encoded = json.dumps(commands, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
        chain = hashlib.sha256(chain + encoded).digest()
    stats_keys = ('depth', 'cands', 'rolls', 'time', 'aborted', 'changed')
    if chain.hex() != data.get('actions_chain_sha256'):
        issues.append('public_clock_action_chain_mismatch')
        return {}, []
    if row and ({k: row.get('search', {}).get(k) for k in stats_keys} != data.get('native_stats')
                or row.get('final_state_sha256') != data.get('state_sha256')):
        issues.append('public_clock_disagrees_with_native_receipt')
        return {}, []
    segments = data.get('segments', [])
    first = 0
    hosts = []
    for seg in segments:
        if (seg.get('first_step') != first or type(seg.get('end_step')) is not int
                or seg['end_step'] <= first or seg.get('host', {}).get('affinity') != [identity.get('cpu')]):
            break
        first = seg['end_step']
        host = seg['host']
        hosts.append({k: host.get(k) for k in ('hostname', 'pid', 'cpu_model', 'affinity', 'cgroup', 'cpu_max')})
    if first != len(calls) or len(hosts) != len(segments):
        issues.append('incomplete_clock_host_segments')
        hosts = []
    return {'calls': calls}, hosts


def clock(raw, path, repo):
    """Return JSON evidence for this path. No score, winner or depth threshold selects records."""
    repo = Path(repo).resolve()
    path = repo / Path(path)
    evidence, issues = {}, []
    if _read(path, repo, evidence, 'raw') != raw:
        raise ValueError('raw argument differs from the recorded file')
    seat = raw.get('candidate_seat')
    physical = 1-seat if type(seat) is int and seat in (0, 1) else None
    if physical is None:
        issues.append('unknown_physical_examiner_seat')
    row, header, source_sha = _receipt(raw, path, repo, evidence, issues)
    full, hosts = _wall(raw, path, repo, evidence, issues, row)
    observed_md5, original_md5, original_sha = raw.get('examiner_md5'), None, None
    original = repo / ORIGINAL
    if original.is_file():
        stat = original.stat()
        original_md5, original_sha = _original(original, (stat.st_ino, stat.st_mtime_ns, stat.st_size))
        evidence['canonical_source'] = {'path': ORIGINAL, 'sha256': original_sha}
    observed_sha = raw.get('examiner_sha256')
    wrong = ((observed_md5 is not None and observed_md5 != MD5)
             or any(x is not None and x != SHA256 for x in (observed_sha, source_sha, original_sha))
             or (original_md5 is not None and original_md5 != MD5))
    if wrong:
        level = 'conflicting_identity'
        issues.append(level)
    elif observed_md5 == MD5 or observed_sha == SHA256:
        level = 'observed_raw_identity'
    elif source_sha == SHA256:
        level = 'attributed_by_bound_receipt'
    else:
        level = 'unknown'
    identity = {'level': level, 'candidate_seat': seat, 'physical_seat': physical,
                'raw_md5': observed_md5, 'raw_sha256': observed_sha, 'receipt_source_sha256': source_sha,
                'expected_md5': MD5, 'expected_sha256': SHA256, 'canonical_source_md5': original_md5,
                'canonical_source_sha256': original_sha}
    cfg = raw.get('configuration', {})
    budget = cfg.get('_S_ABORT')
    complete_config = all(k in cfg for k in ('_TEMP', '_SEED', '_S_ABORT', '_S_ENGINE'))
    configuration = {'temperature': cfg.get('_TEMP'), 'native_budget_seconds': budget,
                     'seed_recorded': '_SEED' in cfg, 'seed_value': cfg.get('_SEED'),
                     'search_engine': cfg.get('_S_ENGINE'), 'maximum_depth': cfg.get('_S_D2'),
                     'natural_t1_058': (type(cfg['_TEMP']) in (int, float) and cfg['_TEMP'] == 1
                                        and cfg['_SEED'] is None and budget == .58
                                        and cfg['_S_ENGINE'] is True) if complete_config else None}
    actions = raw.get('actions', [])
    events = raw.get('search_calls')
    pairs = [(e[0], e[1]) for e in events or [] if isinstance(e, list)
             and len(e) == 2 and type(e[0]) is int and isinstance(e[1], str)]
    counts = dict(Counter(k for _, k in pairs))
    coverage = (len(pairs) == len(events) and [s for s, _ in pairs] == list(range(len(actions)))
                if isinstance(events, list) else None)
    if coverage is False or set(counts) - KINDS:
        issues.append('missing_or_unclassified_search_calls')
    stats = row.get('search', {})
    if row and (row.get('search_return_types') != counts or row.get('search_invocations') != len(pairs)):
        issues.append('native_receipt_return_counts_disagree')
    calls = full.get('calls')
    hist = stats.get('depth')
    if calls is not None:
        hist_seen = dict(Counter(str(c['max_depth']) for c in calls if c['kind'] == 'searched'))
        if hist is not None and hist_seen != hist:
            issues.append('searched_depth_histogram_mismatch')
    wall_by_seat = row.get('call_seconds_by_seat', [])
    wall = wall_by_seat[physical] if physical is not None and len(wall_by_seat) == 2 else None
    search = {'tier': ('public_clock_per_call' if full else 'native_aggregate_and_return_events'
                       if stats and events is not None else 'return_events' if events is not None else 'unknown'),
              'action_pairs': len(actions), 'official_transitions_reported': raw.get('steps'),
              'call_count': len(events) if isinstance(events, list) else None,
              'every_action_logged': coverage, 'return_counts': counts,
              'searched_completed_depth_histogram': hist, 'native_aborted_including_partial': stats.get('aborted'),
              'zero_depth_returns': counts.get('depth_zero_return_base', 0) if coverage else None,
              'single_candidate_returns': counts.get('single', 0) if coverage else None,
              'searched_native_seconds': _times(stats.get('time'), budget),
              'public_elapsed_seconds_aggregate': wall,
              'all_public_wall_seconds': _times([c['public_wall_seconds'] for c in calls], budget) if full else None,
              'all_public_cpu_seconds': _times([c['process_cpu_seconds'] for c in calls], budget) if full else None,
              'all_observed_native_seconds': _times([c['native_seconds'] for c in calls
                                                     if c['native_seconds'] is not None], budget) if full else None,
              'errors_by_step_and_physical_seat': raw.get('errors'),
              'audit_messages': raw.get('audit_messages'), 'receipt_audit_messages': row.get('audit_messages')}
    # These native runners time both agents on the same worker. Its cgroup is the
    # outer process budget. Neither the affinity mask nor cpu.max observes two isolated seats.
    quota = header.get('cpu_quota')
    resources = {'tier': ('worker_cgroup_and_affinity' if hosts or quota is not None else
                          'worker_affinity_only' if row.get('cpu_affinity') else 'unknown'),
                 'official_same_host_quota': None, 'per_seat_resource_evidence': 'unknown',
                 'runner_cpu_max': quota, 'runner_cpu_affinity': row.get('cpu_affinity'),
                 'worker_segments': hosts}
    return {'identity': identity, 'configuration': configuration, 'search': search,
            'resources': resources, 'evidence': evidence, 'issues': issues}
