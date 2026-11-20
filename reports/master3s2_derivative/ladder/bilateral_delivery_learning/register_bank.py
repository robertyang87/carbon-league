"""Register the amended observable (all three receipts) without changing the original experiment."""
from register import H,sha,dump
KINDS=('own_delivery','opponent_worker_delivery_to_this_base','base_collision_cash')
if __name__=='__main__':
    doc=dict(revision='all_bank_receipts_v1',parent_protocol=sha((H/'protocol.json').read_bytes()),
        own_delivery_weight=sha((H/'weights.json').read_bytes()),original_code={n:sha((H/n).read_bytes()) for n in ['model.py','fit.py','source.py']},
        target=list(KINDS),recipient='CashEvents event[1] is the physical wallet credited; the worker whose carbon is counted need not belong to that wallet. Include three terms, exclude tree_collision_cash, taxes and ground rent.',
        reason='Already exposed t279 example demonstrates 219.24 reaching an opposing bank. This revision was requested before training this model; original narrow-target model retained. Test is on the same formerly sealed four physical test maps and now retrospective. No model tuning from test or examples.',
        fit='Same train maps, seats, steps, map weights, features, normalized objectives, optimizer, initialization, solver cap, outputs, controls, held cohort, time window and paired qualification as parent. Verify original feature/label digest and narrow target per row before replacing only Y. Keep separate counts/sums by receipt kind. Compare both networks on inclusive target.',
        branches='Compute inclusive physical-wallet targets when full window and categories recoverable. Preserve missing and partial windows. Actual 219.24 branch verification uses the three terms separately and qualifies held-opponent assumptions.',
        labels='Credits unrounded as written to cash, ending balance cents. Each root sums 10 actual transitions starting at its own action; last registered root t288.')
    print(dump(H/'bank_protocol.json',doc))
