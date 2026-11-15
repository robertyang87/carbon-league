import importlib.util, sys
sys.path.insert(0, '/root/carbon/code')
import cnet2_search as S
for name in ('m3s2r1', 'echo2', 'echo_e0'):
    spec = importlib.util.spec_from_file_location(name, '/root/carbon/code/exp/%s.py' % name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ag = S.make_search_agent(m.CONF, weights=m._W)
    print('BUILT', name, type(ag).__name__)
print('REPRO-OK')
