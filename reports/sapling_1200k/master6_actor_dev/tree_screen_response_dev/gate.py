from screen_trial import HERE,sr
def run():
    gains=[];delays=[]
    for name in ('.','first_trigger','opposite_map'):
        pair=sr.load(HERE/name/'verified.json')
        parent,alt=pair
        seat=parent['ledger']['seat']
        def lost(a):
            return next((e['t'] for e in a['ownership_changes'] if e['owner_after']==1-seat),299)
        gains.append(alt['ledger']['margin']-parent['ledger']['margin'])
        if name!='first_trigger':
            delays.append(lost(alt)>lost(parent) and lost(parent)<299)
    parity=sr.load(HERE/'parity.json')
    allowed=all(delays) and sum(gains)>0 and all(p['edits']>0 and p['mismatch']==0 for p in parity)
    sr.retain(HERE/'gate.json',dict(gains=gains,capture_delayed=delays,eligible=allowed))
    return allowed
if __name__=='__main__':print(run())
