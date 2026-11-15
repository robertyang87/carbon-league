"""Remediation: recreate the 0.05 backup of ppou2_c state_last.pt.

The 0.15 edit itself succeeded and was read back (kl_coef 0.15, it 4), but
remote_edit_kl.py expects the caller to pre-create the backup file; the
path I passed did not exist, so its post-edit deep-compare crashed and no
0.05 backup was left. This script rebuilds a content-identical 0.05
backup from the current (edited) state: only kl_coef differs.
The original 0.05 file's md5 was 8c3ae29c07810b45fe6c1e2fc477c13d
(recorded at the 2026-11-15 02:40 edit); the rebuilt file will differ in
bytes (torch.save zip metadata) but is content-identical elsewhere.
"""
import hashlib
import torch

SRC = "/root/carbon/data/rl3/ppou2_c/state_last.pt"
BAK = "/root/carbon/data/rl3/ppou2_c/state_last.pt.kl0.05.bak"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def deep_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and bool((a == b).all())
    return a == b


cur = torch.load(SRC, map_location="cpu", weights_only=False)
assert cur.get("kl_coef") == 0.15 and cur.get("it") == 4, cur.get("kl_coef")
bak = dict(cur)
bak["kl_coef"] = 0.05
torch.save(bak, BAK)
re_cur = torch.load(SRC, map_location="cpu", weights_only=False)
re_bak = torch.load(BAK, map_location="cpu", weights_only=False)
assert re_bak.get("kl_coef") == 0.05 and re_bak.get("it") == 4
diff = [k for k in set(re_cur) | set(re_bak)
        if not deep_equal(re_cur.get(k), re_bak.get(k))]
print("BACKUP_OK" if diff == ["kl_coef"] else f"BACKUP_DIFF {diff}")
print("cur_md5", md5(SRC))
print("bak_md5", md5(BAK))
