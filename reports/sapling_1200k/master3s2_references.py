"""Audit archived complete-Master3s2 reference series. Uses Git objects only.

In carbon-league, fetch the named archive if it is missing, then run:
    git fetch --depth=1 --no-tags origin refs/heads/sync/h29
    python reports/sapling_1200k/master3s2_references.py > /tmp/master3s2_references.json
"""
import hashlib
import json
from collections import Counter
import statistics
import subprocess

MAIN = '6712c5d53d028545f0444ae8f75f80c313c7b10b'
ARCHIVE = 'd599ec679718dd4407a718aa6ad810e35e2d3983'
DEX = 'synced/h29/bigeval_live/'
SERIES = [
    ('it50', 'abD_it50_3s2.jsonl', 300, 'code/packs/ppou2_m_it50.py'),
    ('Echo4', 'f2_echo4_3s2.jsonl', 500, 'code/packs/echo4_int8.py'),
    ('Echo6', 'f1_comp_3s2.jsonl', 500, 'sources/echo6/echo4_search_int8_d1_homefix.py'),
]


def content(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'])


def summary(rows):
    margins = [r['margin'] for r in rows]
    return {'games': len(rows), 'wins': sum(v > 0 for v in margins),
            'draws': sum(v == 0 for v in margins), 'losses': sum(v < 0 for v in margins),
            'mean_margin': statistics.fmean(margins)}


def verify(arm, name, count, candidate):
    path = DEX + name
    raw = content(ARCHIVE, path)
    agent = content(MAIN, candidate)
    digest = hashlib.md5(agent).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == count
    assert {r['seed'] for r in rows} == set(range(20291101, 20291101 + count))
    for r in rows:
        assert r['side'] == r['seed'] % 2 and r['ah'] == digest
        assert r['bh'] == '8d209ddb46faafae151aae16b7e17d55'
        assert r['hh'] == '354bcd1bc2dc8bc188c813f53d28c893'
        assert r['err'] is None and r['serr'] == r['berr'] == 0
        assert abs(r['sa'] - r['sb'] - r['margin']) < 1e-7 and r['win'] == (r['margin'] > 0)
    return {'arm': arm, 'candidate_path': candidate, 'candidate_md5': digest,
            'candidate_sha256': hashlib.sha256(agent).hexdigest(), 'archive_path': path,
            'archive_sha256': hashlib.sha256(raw).hexdigest(), **summary(rows),
            'by_seat': [summary([r for r in rows if r['side'] == s]) for s in (0, 1)],
            'reported_harness_errors': 0,
            'observed_opponent_stats_keys': sorted({k for r in rows for k in r.get('bst', {})})}


def recent_opponent_timing():
    example = []
    for arm in ('v2', 'v3'):
        path = f'reports/sapling_1200k/cash_exam_{arm}.json'
        raw = content(MAIN, path)
        report = json.loads(raw)
        assert report['status'] == 'complete'
        assert report['examiner_md5'] == '57f9a594cce4f2ec9e3573d8d9d26ca1'
        opponent_calls = 0
        opponent_seconds = 0
        searches = Counter()
        depths = Counter()
        for row in report['games']:
            timing = row['call_seconds_by_seat'][1 - row['candidate_seat']]
            stats = row['search']
            assert not any(row['agent_errors_by_seat'])
            assert stats['searched'] + stats['single'] == timing['n']
            assert sum(stats['depth'].values()) == stats['searched']
            opponent_calls += timing['n']
            opponent_seconds += timing['sum']
            depths.update(stats['depth'])
            for k in ('searched', 'single', 'aborted', 'changed'):
                searches[k] += stats[k]
        example.append({'arm': arm, 'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'games': len(report['games']), 'opponent_calls': opponent_calls,
                        'opponent_mean_call_ms': 1000 * opponent_seconds / opponent_calls,
                        'reported_search_counts': dict(searches), 'completed_depths': dict(depths)})
    return example


if __name__ == '__main__':
    runs = [verify(*s) for s in SERIES]
    direct = hashlib.md5(content(MAIN, 'data/master3s2_source.py')).hexdigest()
    assert direct == '57f9a594cce4f2ec9e3573d8d9d26ca1'
    print(json.dumps({'candidate_revision': MAIN, 'archive_revision': ARCHIVE,
                      'registered_opponent_source_md5': direct,
                      'historical_opponent_wrapper_md5': '8d209ddb46faafae151aae16b7e17d55',
                      'later_registered_opponent_wrapper_md5': 'b4cde262695d38270e16bde33875a443',
                      'historical_identity_document': 'reports/RECORD_COMPOSITE_SUBMISSION_DEVIATION_20261116.md',
                      'selection': 'Previously used fixed maps, one game per distinct map, seat by parity.',
                      'analysis': runs, 'direct_source_timing_on_later_Sapling_research_games': recent_opponent_timing()}, indent=2))
