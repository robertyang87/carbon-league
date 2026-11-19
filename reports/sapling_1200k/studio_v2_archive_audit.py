"""Recompute the Studio V2 raw-score readout and exercise pairing on a separate 72 archive."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tarfile
import tempfile

from studio_paired_scores import compare, read_arm, require, SOURCE, V2, SQRT

REV = '3adddd64'
DIR = 'reports/'
FULL = 'd715bd21a176f5a236b74aebff77eb9e'


def git_file(ref, filename):
    return subprocess.check_output(['git','show',f'{ref}:{filename}'])


def arm_manifest(tar, tmp, machine, role, members):
    shards = []
    for member in members:
        live = Path(member)
        summary = str(live.parent.parent/'bigeval.jsonl')
        shard = {}
        for prefix, name in (('',member),('summary_',summary)):
            data = tar.extractfile(name).read()
            path = tmp/f'{role}-{len(shards)}-{prefix}rows.jsonl'
            path.write_bytes(data)
            shard[prefix+'path'] = path.name
            shard[prefix+'sha256'] = hashlib.sha256(data).hexdigest()
        shards.append(shard)
    manifest = tmp/f'{role}.json'
    manifest.write_text(json.dumps({'machine':machine, 'examiner_source_md5':SOURCE,
                                    'provenance':f'{machine} exam author and archive in carbon-league', 'shards':shards}))
    return manifest


def analyze(rows):
    def mean(key):
        return statistics.fmean(r[key] for r in rows)
    require(all(r['sa'] >= 0 or float(r['sa']).is_integer() and -300 <= r['sa'] <= -2 for r in rows),
            'negative score does not fit the registered extinction rule')
    require(all(r['sb'] >= 0 for r in rows), 'the expected Studio raw archive has no opponent extinction')
    dead = [int(r['sa'] + 301) for r in rows if r['sa'] < 0]
    return {'n':len(rows), 'wins':sum(r['win'] for r in rows), 'draws':sum(r['margin']==0 for r in rows),
            'official_margin_mean':mean('margin'), 'official_margin_se':statistics.stdev(r['margin'] for r in rows)/len(rows)**.5,
            'official_own_score_mean':mean('sa'), 'official_opponent_score_mean':mean('sb'),
            'own_diagnostic_deposit_mean':mean('dep'), 'opponent_diagnostic_deposit_mean':mean('odep'),
            'rows_with_positive_diagnostic_dep':sum(r['dep']>0 for r in rows),
            'own_tree_income_mean':mean('tinc'), 'opponent_tree_income_mean':mean('otinc'), 'mean_u':mean('u'),
            'own_dep_plus_tree_minus_plant_and_seize_spend':mean('dep')+mean('tinc')-mean('pspend')-mean('sspend'),
            'opponent_same_ledger_proxy':mean('odep')+mean('otinc')-mean('opspend')-mean('osspend'),
            'explicit_joint_transition_rows':sum(r.get('actual_transitions') is not None for r in rows),
            'negative_own_scores':len(dead),'nonnegative_scores_both':len(rows)-len(dead),
            'inferred_extinction_transition_range':[min(dead),max(dead)] if dead else None,
            'inferred_mean_transitions_if_registered_scoring_and_nonnegative_cash':statistics.fmean(dead+[299]*(len(rows)-len(dead)))}


def main(with_sample):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        filename = 'sapling_exam_v2_200_results.tgz'
        raw = git_file(REV, DIR+filename)
        archive_sha = hashlib.sha256(raw).hexdigest()
        require(archive_sha == '2a1332179eb34152e5f67ae2d57c805ecebbb795cb520a75bd1ba656cd99008b', 'archive changed')
        target = tmp/filename
        target.write_bytes(raw)
        with tarfile.open(target) as t:
            members = sorted(n for n in t.getnames() if '/bigeval_live/' in n and n.endswith('.jsonl'))
            require(len(members)==10, 'expected ten live shards')
            candidate = t.extractfile('top5_h2h_v2_agent.py').read()
            require(hashlib.md5(candidate).hexdigest()==V2, 'V2 packaged bytes mismatch')
            manifest = arm_manifest(t,tmp,'Studio','v2',members)
        rows, evidence = read_arm(manifest,'Studio',V2,set(range(20291101,20291301)))
        out = {'revision':REV, 'archive':DIR+filename, 'archive_sha256':archive_sha,
               'status':'V2 audited. The source archive contains no Studio sqrt raw game rows.',
               'identities':evidence, 'overall':analyze(list(rows.values())),
               'by_seat':{s:analyze([r for r in rows.values() if r['side']==s]) for s in (0,1)},
               'missing_for_pairing':'Studio sqrt f92b1823, 200 live rows on the same 20291101..20291300 maps, plus origin and shard hashes.',
               'duration_note':'No raw transition count. On the registered 300-state rules, a negative official score equals transitions-301; games with two nonnegative scores are horizon finishes assuming nonnegative survivor cash.',
               'ledger_note':'Rounded live fields dep+tinc-pspend-sspend omit recruitment and other terms. Official scores use sa/sb.'}
        if with_sample:
            filename = 'sapling_exam_r2_results.tgz'
            raw = git_file('83f54e2d',DIR+filename)
            target = tmp/filename
            target.write_bytes(raw)
            with tarfile.open(target) as t:
                manifests = [arm_manifest(t,tmp,'spot-72',name,
                             [f'results/{name}/s0/data/bigeval_live/sapling_{name}_s0.jsonl']) for name in ('sqrt','full')]
            trial = compare(*manifests,20291101,100,'spot-72',SQRT,FULL)
            out['pairing_function_test_only']={'machine':'spot-72','pair':'full minus sqrt, shard s0 only',
                                              'source_sha256':hashlib.sha256(raw).hexdigest(), 'maps':trial['count'],
                                              'paired_official_margin':trial['overall']['fields']['margin'],
                                              'paired_official_own_score':trial['overall']['fields']['own_score'],
                                              'paired_official_opponent_score':trial['overall']['fields']['opponent_score']}
        return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-sample', action='store_true', help='include an expressly separate spot-72 pairing test')
    print(json.dumps(main(parser.parse_args().with_sample),indent=2,allow_nan=False))
