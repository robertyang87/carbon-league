"""Verify eight full native transcripts, all root clock records, map allocation and bound artifacts."""
from collections import Counter
from pathlib import Path
import statistics
from shared import HERE,read,require,save,sha
from native import driver
from resume import verify as matches_frozen_source
from clock import verify_completed
from clock_report import summarize
from first_split import run as first_run


def main():
    matches_frozen_source();binding=read(HERE/'games/binding.json');scores=read(HERE/'comparison.json')
    g=HERE/'games';dirs=['parent','candidate'];rs=[]
    for arm in dirs:
        for prefix in ('raw_','row_','receipt_','wall_','ticket_'):
            require(len(list((g/arm).glob(prefix+'*')))==4,'unexpected or missing repeated game/partial clock')
    for mid,(seed,terrain) in enumerate(binding['boards'].items()):
        for seat in (0,1):
            for ai,arm in enumerate(dirs):
                job=(arm,int(seed),seat,(mid+seat+ai)%2,terrain,binding)
                clock=verify_completed(driver(),job)
                row=read(g/arm/f'row_{seed}_{seat}.json')
                official=driver().official.check(g/arm,row)
                require(official==read(g/arm/f'receipt_{seed}_{seat}.json')['official'],'full raw could not reproduce signed official')
                raw=read(g/arm/f'raw_{seed}_{seat}.json.gz')
                require(all(s=='DONE' for s in official['statuses']) and bool(raw['steps']==299 or any(raw['extinct_by_seat'])) and len(raw['actions'])==len(clock['calls'])==official['steps'],'partial or unaccounted temporal game')
                rs.append((arm,int(seed),seat,row['margin'],row['result'],raw['steps']))
    for arm in dirs:
        group=[r for r in rs if r[0]==arm];a=scores['by_arm'][arm]
        require(a['wins']==sum(r[4]=='win' for r in group) and a['draws']==sum(r[4]=='draw' for r in group) and a['mean_official_margin']==statistics.mean(r[3] for r in group),'WDL or group score')
    require(len({r[1] for r in rs})==2 and len(rs)==8,'physical groups')
    summarize();first_run()
    names=[p for p in sorted(HERE.rglob('*')) if p.is_file() and not p.name.startswith('.') and not any(k=='__pycache__' for k in p.parts)
           and p.name not in ['verification.json'] and p.suffix not in ('.log','.part','.tmp')]
    doc={'files':{str(p.relative_to(HERE)):sha(p) for p in names},'file_count_excluding_receipt':len(names),'bytes':sum(p.stat().st_size for p in names),
         'native_transitions':sum(r[-1] for r in rs),'new_development_maps':2,'new_formal_maps':0,'native_games':8,
         'official_replay_and_clock_seals':True,'parent_and_candidate_wins':{k:v['wins'] for k,v in scores['by_arm'].items()},
         'paired_delta':scores['paired_delta']}
    save(HERE/'verification.json',doc);print(len(names),doc['native_transitions'],scores['paired_delta'])


if __name__=='__main__':main()
