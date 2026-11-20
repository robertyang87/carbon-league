"""Inspect the complete git changed-file set for new score indexes since the numeric appendix freeze."""
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import fetch
from continue_appendix import previous_manifests
from appendix_97de1ebc.collect import inventories


def main():
    root=HERE.parent
    base=fetch.git('rev-parse','c2754b12^{commit}').decode().strip()
    head=fetch.git('rev-parse',sys.argv[1]+'^{commit}').decode().strip()
    fetch.git('merge-base','--is-ancestor',base,head)
    prior=[str(p.parent.relative_to(root)) for p in previous_manifests(root/'appendix_ea224932')]
    registrations=prior+['appendix_c2754b12']
    delta=inventories(base,head,registrations)
    out={'source_base':base,'source_tip':head,'new_count':delta['new_count'],'union_games':delta['union_games'],
         'unscored':delta['unscored_new_ids'],'changed_index_delta':delta,
         'bounded_scope':'Only scorecards through source_tip. Earlier source aliases deduplicated by game ID; later new scores require a separate appendix.'}
    # If the frozen tip contains a new distinct game, report it rather than silently treating it as covered.
    fetch.dump(HERE/'head_delta.json',out)
    print(out['new_count'],out['union_games'],out['unscored'])


if __name__=='__main__':main()
