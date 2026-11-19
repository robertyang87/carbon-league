"""Independently settle secondary-population archived tapes with the official referee."""
from collect import HERE,ROOT,begin,read,sha,write,load_tape,clean

def main():
    results=[]
    for r in read(HERE/'rows.json.gz'):
        if r['split'] not in ('transfer','later_repeat','later_option'):continue
        raw=read(ROOT/r['source'])
        for arm_index,arm in enumerate(('parent','alternative')):
            info=r['both_receipts'][arm];tape=load_tape(info,arm)
            env=begin(raw)
            for joint in raw['actions'][:r['t']]:
                assert not env.done
                env.step(joint)
            assert clean(tape[0][r['seat']])==r['both_commands'][arm_index]
            for joint in tape:
                assert not env.done
                env.step(joint)
            status=[s.status for s in env.states];scores=[s.reward for s in env.states]
            assert env.done and status==['DONE','DONE'] and scores==r['settlements'][arm_index]['official_scores']
            results.append(dict(case=r['case'],arm=arm,arm_receipt_sha256=info['receipt_sha256'],
                           action_sha256=info['action_sha256'],scores=scores,status=status))
    write(HERE/'supplement_official.json.gz',dict(rows_sha256=sha(HERE/'rows.json.gz'),arms=results),True)
    print('Official end scores matched',len(results))
if __name__=='__main__':main()
