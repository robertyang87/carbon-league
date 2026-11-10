"""Grove1's policy with the search off (opponent for data/knob_search.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knob_us_factory as _F


def make():
    return _F.make()


agent = make()          # module-level agent too (awr_gen / eval_v53 load the module fresh per game)
