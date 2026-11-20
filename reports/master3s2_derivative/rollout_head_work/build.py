"""Remove policy and value readouts that native V2 search never consumes."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
PARENT = FAMILY / 'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    raw = PARENT.read_bytes()
    assert sha(raw) == PARENT_SHA
    first = raw.index(b'class _TorchNet:')
    last = raw.index(b'\r\n_net()', raw.index(b'def _search('))
    block = raw[first:last].decode().replace('\r\n', '\n')

    def replace(old, new, count=1):
        nonlocal block
        assert block.count(old) == count, old
        block = block.replace(old, new)

    replace('def __call__(self, X, sid, sc=None):',
            'def __call__(self, X, sid, sc=None, value_stride=1, policy=True):', 2)
    replace('''            lw = (_torch.einsum('kc,ncp->npk', self.hw[0], hf) + self.hw[1]).numpy()
            lb = (_torch.einsum('kc,ncp->npk', self.hb[0], hf) + self.hb[1]).numpy()''',
            '''            lw = lb = None
            if policy:
                lw = (_torch.einsum('kc,ncp->npk', self.hw[0], hf) + self.hw[1]).numpy()
                lb = (_torch.einsum('kc,ncp->npk', self.hb[0], hf) + self.hb[1]).numpy()''')
    replace('''                s = _torch.from_numpy(sc)
                z = _torch.relu(_torch.einsum('kc,ncp->nkp', self.v1[0], hf) + self.v1[1][None, :, None])''',
            '''                s = _torch.from_numpy(sc[::value_stride])
                z = _torch.relu(_torch.einsum('kc,ncp->nkp', self.v1[0], hf[::value_stride]) + self.v1[1][None, :, None])''')
    replace('''            lw = (hf @ self.hw[0] + self.hw[1]).permute(1, 0, 2).contiguous().numpy()
            lb = (hf @ self.hb[0] + self.hb[1]).permute(1, 0, 2).contiguous().numpy()''',
            '''            lw = lb = None
            if policy:
                lw = (hf @ self.hw[0] + self.hw[1]).permute(1, 0, 2).contiguous().numpy()
                lb = (hf @ self.hb[0] + self.hb[1]).permute(1, 0, 2).contiguous().numpy()''')
    replace('''                s = _torch.from_numpy(sc)
                z = _torch.relu(hf @ self.v1[0] + self.v1[1])''',
            '''                s = _torch.from_numpy(sc[::value_stride])
                z = _torch.relu(hf[:, ::value_stride] @ self.v1[0] + self.v1[1])''')
    replace('def look(live, both):', 'def look(live, both, value_only=False):')
    replace('''                lw, lb, vv = net(X, _SID, S)
                vv = vv[::w]''',
            '''                lw, lb, vv = net(X, _SID, S, value_stride=w, policy=not value_only)''')
    replace('v, pol = look(live, both)', 'v, pol = look(live, both, value_only=(t == D2))')
    replace('v, pol = look(live, t < D2)', 'v, pol = look(live, t < D2, value_only=(t == D2))')
    edited = raw[:first] + block.replace('\n', '\r\n').encode() + raw[last:]
    dst = HERE / 'candidate.py'
    if dst.exists():
        assert dst.read_bytes() == edited
    else:
        compile(edited, str(dst), 'exec')
        dst.write_bytes(edited)
    return sha(edited)


if __name__ == '__main__':
    print(build())
