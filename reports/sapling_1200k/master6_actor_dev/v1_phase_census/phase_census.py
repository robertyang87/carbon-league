"""Trace when the same frozen Sapling policy lost ground across its archived original games."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent/'tree_response_training/ten_map_economy'))
from census import game_row

REV = 'd578532d7'
PACK = '9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0'
EXAM = '57f9a594cce4f2ec9e3573d8d9d26ca1'


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def load_at(path):
    return json.loads(git('show', REV + ':' + str(path)))


def map_mean(rows, fn):
    maps = defaultdict(list)
    for r in rows:
        maps[r['map_sha256']].append(fn(r))
    return mean(mean(vals) for vals in maps.values())


def main():
    files = git('ls-tree','--name-only','-r',REV,'reports/sapling_1200k').decode().splitlines()
    rows = {}
    for file in sorted(f for f in files if f.endswith('/results.json')):
        result = load_at(file)
        if not isinstance(result,dict) or result.get('header',{}).get('pack_sha256') != PACK:
            continue
        if result.get('status') != 'complete' or result['header']['examiner_md5'] != EXAM:
            raise ValueError('incomplete or wrong original in this roster')
        base = Path(file).parent
        for r in result['games']:
            if not r.get('valid') or r['errors'] or r['audit_messages'] or r['search']['searched'] < 1:
                raise ValueError('invalid game or search')
            raw = base/r['raw_file']
            tape = gzip.decompress((ROOT/raw).read_bytes())
            d = json.loads(tape)
            if (d['pack_sha256'] != PACK or d['examiner_md5'] != EXAM or not d['configuration']['_S_ENGINE']
                    or d['configuration']['_TEMP'] != 1 or d['configuration']['_SEED'] is not None):
                raise ValueError('wrong policy configuration')
            if git('hash-object',str(ROOT/raw)).strip() != git('rev-parse',REV+':'+str(raw)).strip():
                raise ValueError('working transcript differs from frozen revision')
            key = r['actions_sha256']
            if key in rows:
                if rows[key]['official_margin'] != round(r['margin'],2):raise ValueError('duplicate differs')
                rows[key]['sources'].append(str(raw))
                continue
            cert=dict(seed=r['seed'],seat=r['candidate_seat'],map_sha256=d['initial_terrain_u8_sha256'],
                      steps=r['actual_transitions'],transcript_sha256=r['raw_sha256'],
                      official_reward_by_seat=r['official_score_by_seat'])
            observed=game_row(ROOT/raw,cert)
            if observed['map_sha256'] != hashlib.sha256(bytes(map(int,d['initial_full_carbon']))).hexdigest():
                raise ValueError('wrong terrain')
            for point in observed['phases']:
                snap = r['snapshots'].get(str(point['post']))
                if snap and abs(point['cash_lead'] - (snap['candidate']['cash']-snap['opponent']['cash'])) > .02:
                    raise ValueError('independent recorded snapshot disagrees')
            observed['sources']=[str(raw)]
            observed['search_calls']=r['search_invocations']
            observed['original_searched']=r['search']['searched']
            rows[key] = observed
    batch = list(rows.values())
    maps={r['map_sha256'] for r in batch}
    if any({x['seat'] for x in batch if x['map_sha256']==m}!={0,1} for m in maps):
        raise ValueError('missing a physical seat for a map')
    summary = []
    for i in range(6):
        phases = [r['phases'][i] for r in batch]
        p = phases[0]
        out={k:map_mean(batch, lambda r:r['phases'][i]['signed_cash_components'][k])
             for k in p['signed_cash_components']}
        summary.append(dict(state=p['post'], leading=sum(x['cash_lead']>0 for x in phases),
                            mean_cash_lead=map_mean(batch,lambda r:r['phases'][i]['cash_lead']),components=out))
    dump=dict(source_commit=git('rev-parse',REV).decode().strip(), pack_sha256=PACK, examiner_md5=EXAM,
              games=len(batch),terrains=len(maps),wins=sum(r['official_margin']>0 for r in batch),
              draws=sum(r['official_margin']==0 for r in batch), phases=summary,
              map_equal_margin=map_mean(batch,lambda r:r['official_margin']),seat_rows=[
                  dict(seat=i,n=len([r for r in batch if r['seat']==i]),
                       mean=map_mean([r for r in batch if r['seat']==i],lambda r:r['official_margin']))
                  for i in (0,1)])
    (HERE/'phases.json').write_text(json.dumps(dump, indent=2, allow_nan=False)+'\n')
    (HERE/'games.json.gz').write_bytes(gzip.compress(json.dumps(batch,separators=(',',':'),allow_nan=False).encode(),mtime=0))


if __name__ == '__main__':
    main()
