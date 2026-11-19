"""Identify cross-index duplicates and compare observed ladder maps to the declared development board."""
from concurrent.futures import ThreadPoolExecutor
from intake import HERE,ladder
from audit import sr
def check():
    prior='a5867c07b'
    names=[p for p in ladder.fetch.git('ls-tree','-r','--name-only',prior,'data').decode().splitlines() if ladder.fetch.sourcefile(p)]
    old=set()
    with ThreadPoolExecutor(4) as pool:
        for group in pool.map(lambda p:ladder.cards(prior,p),names):old.update(group)
    boards=sr.load(HERE/'clearance.json')['physical_boards']
    rows=[]
    for name in ('master5s_delta','additional_indices'):
        for g in sr.load(HERE/(name+'.json'))['games']:
            assert g['terrain'] not in boards.values() and str(g['verified']['random_seed']) not in boards
            rows.append(dict(id=g['id'],source=name,already_in_prior_union=g['id'] in old,seed=g['verified']['random_seed'],terrain=g['terrain']))
    sr.retain(HERE/'intake_check.json',dict(prior=prior,indices=len(names),games=rows))
if __name__=='__main__':check()
