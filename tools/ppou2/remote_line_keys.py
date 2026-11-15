import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_m2.jsonl"
d = json.loads(open(path).readline())

def walk(o, p, depth):
    if depth > 4:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            extra = ""
            if isinstance(v, (dict, list)):
                extra = " len=%d" % len(v)
            elif isinstance(v, (int, float, str)):
                extra = " =%s" % str(v)[:40]
            print("%s/%s %s%s" % (p, k, type(v).__name__, extra))
            walk(v, p + "/" + str(k), depth + 1)
    elif isinstance(o, list) and o:
        print("%s[0] %s" % (p, type(o[0]).__name__))
        walk(o[0], p + "[0]", depth + 1)

walk(d, "", 0)
