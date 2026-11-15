"""Local stub for selfplay_gen STUB mode: make_search_agent is never called
when SPGEN_STUB=1 (agents are lambdas). Present only so the import succeeds."""


def make_search_agent(*a, **k):
    raise RuntimeError("stub cnet2_search: real agent unavailable locally")
