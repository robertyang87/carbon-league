"""Regression fixtures for process ownership, incomplete source games and frozen selectors. No matches."""
from concurrent.futures import Future
import contextlib
import gzip
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import batch_cli as batch
import continue_intakes as continuation
import tree_pack_arena as arena
import tree_win_gate as gate


def supervisor_fixture(directory):
    """Run the actual supervisor in another OS process; replace only its expensive source subprocess."""
    root=Path(directory)
    continuation.CODE=root
    def state():
        done=(root/'done').exists()
        return {'pending':None if done else {'batch':'intake_02','stage':'paired_responses','live_batch_lock':False},'fit_support':done}
    def run(_cmd,**_kwargs):
        with (root/'launches').open('a') as f:
            f.write('launched\n')
        until=time.monotonic()+12
        while not (root/'release').exists() and time.monotonic()<until:
            time.sleep(.025)
        if not (root/'release').exists():
            raise RuntimeError('fixture timeout')
        (root/'done').touch()
        return type('Done',(),{'returncode':0})()
    with patch.object(continuation,'board_state',state),patch.object(continuation.subprocess,'run',run):
        continuation.supervise()


class ContinuationTests(unittest.TestCase):
    def test_duplicate_supervisor_and_restart_after_completion(self):
        ctx=multiprocessing.get_context('fork')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            owner=ctx.Process(target=supervisor_fixture,args=(tmp,));owner.start()
            try:
                for _ in range(150):
                    if (root/'launches').exists():break
                    time.sleep(.02)
                self.assertTrue((root/'launches').exists())
                contender=ctx.Process(target=supervisor_fixture,args=(tmp,));contender.start();contender.join(4)
                self.assertEqual(contender.exitcode,0)
                self.assertEqual((root/'launches').read_text(),'launched\n')
                (root/'release').touch();owner.join(4);self.assertEqual(owner.exitcode,0)
                last=ctx.Process(target=supervisor_fixture,args=(tmp,));last.start();last.join(4)
                self.assertEqual(last.exitcode,0)
                self.assertEqual((root/'launches').read_text(),'launched\n')
            finally:
                (root/'release').touch()
                owner.join(2)
                if owner.is_alive():owner.terminate();owner.join()

    def test_source_resume_runs_only_missing_seat_and_rejects_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            seed=13
            pack=root/'pack.py';pack.write_text('# dynamic fixture')
            weights=root/'weights';weights.write_bytes(b'abc')
            args=['tree_pack_arena','--pack',str(pack),'--weights',str(weights),'--pack-sha256',batch.sha(pack),
                  '--maps',str(seed),'--run',str(root/'run')]
            games=[]
            class Pool:
                def __init__(self,**kw):pass
                def __enter__(self):return self
                def __exit__(self,*exc):pass
                def submit(self,_play,job):
                    _,_,board,seat,_,out,_=job
                    self_file=Path(out)/f'raw_{board}_{seat}.json.gz'
                    self_file.write_bytes(f'{seat}'.encode())
                    games.append((board,seat))
                    row={'seed':board,'candidate_seat':seat,'raw_file':self_file.name,'raw_sha256':batch.sha(self_file),
                         'valid':True,'result':'loss','margin':-2.,'audit_messages':[]}
                    fut=Future();fut.set_result(row);return fut
            def exercise():
                with (patch.object(sys,'argv',args),patch.object(arena,'ProcessPoolExecutor',Pool),
                      patch.object(gate,'verify_raw'),patch.object(gate,'validate_game'),
                      contextlib.redirect_stdout(io.StringIO())):
                    arena.main()
            exercise()
            self.assertEqual(games,[(seed,0),(seed,1)])
            out=root/'run';result=out/'results.json';doc=json.loads(result.read_text())
            doc['status']='running';doc['games']=doc['games'][:1]
            result.write_text(json.dumps(doc));games.clear()
            with self.assertRaisesRegex(ValueError,'already played raw outcome'):exercise()
            self.assertEqual(games,[])
            (out/f'raw_{seed}_1.json.gz').unlink()
            exercise();self.assertEqual(games,[(seed,1)])
            exercise();self.assertEqual(games,[(seed,1)])

    def test_content_addressed_selectors_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'input.py';p.write_text('print(1)\n')
            with patch.object(batch,'CODE',root):
                key=batch.snapshot_selector(p);self.assertEqual(key,hashlib.sha256(p.read_bytes()).hexdigest())
                dest=root/'selector_versions'/(key+'.py.gz')
                self.assertEqual(gzip.decompress(dest.read_bytes()),p.read_bytes());self.assertEqual(batch.snapshot_selector(p),key)
                p.write_text('print(2)\n');self.assertNotEqual(batch.snapshot_selector(p),key)
                p.write_text('print(1)\n');dest.write_bytes(gzip.compress(b'invalid'))
                with self.assertRaisesRegex(ValueError,'different selector'):batch.snapshot_selector(p)


if __name__=='__main__':
    unittest.main()
