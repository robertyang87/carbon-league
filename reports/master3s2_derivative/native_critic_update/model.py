"""Only the six native value parameters change. All training features come from the parent's Torch trunk."""
import base64
import io
import re
import types

import numpy as np
import torch
from torch import nn

from register import H, PARENT, PARENT_SHA, sha

VALUE = re.compile(rb'(?m)^_VALUE = "([A-Za-z0-9+/=]+)"')
TENSORS = ["v1.weight", "v1.bias", "v2.weight", "v2.bias", "v3.weight", "v3.bias"]


def parse(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        if set(z.files) != set(TENSORS):
            raise ValueError("Invalid native value archive")
        return {name:z[name].astype(np.float32) for name in TENSORS}


def parent():
    raw = PARENT.read_bytes()
    if sha(raw) != PARENT_SHA:
        raise ValueError("Unexpected parent")
    return raw


def module(raw, tap=False):
    if tap:
        old = b"            hf = h.flatten(2)"
        if raw.count(old) != 1:
            raise ValueError("Trunk instrumentation mismatch")
        raw = raw.replace(old, old + b"\r\n            return hf.numpy()")
    m = types.ModuleType("native_value_study")
    exec(compile(raw, "native_master.py", "exec"), m.__dict__)
    torch.set_num_threads(1)
    return m


def weight_blob(raw):
    matches = VALUE.findall(raw)
    if len(matches) != 1:
        raise ValueError("Ambiguous value blob")
    return base64.b64decode(matches[0], validate=True)


def package(archive):
    raw = parent()
    a, b = VALUE.search(raw).span(1)
    return raw[:a] + base64.b64encode(archive) + raw[b:]


class Critic(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.v1 = nn.Conv1d(96,32,1)
        self.v2 = nn.Linear(69,128)
        self.v3 = nn.Linear(128,1)
        state = dict(parameters)
        state["v1.weight"] = state["v1.weight"].reshape(32,96,1)
        self.load_state_dict({k: torch.from_numpy(v) for k,v in state.items()})

    def forward(self, h, s):
        z = torch.relu(self.v1(h))
        pooled = torch.cat((z.mean(2),z.amax(2),s),1)
        return s[:,0] + self.v3(torch.relu(self.v2(pooled)))[:,0]

    def as_blob(self):
        state = {k:v.detach().numpy().astype(np.float16) for k,v in self.state_dict().items()}
        state["v1.weight"] = state["v1.weight"].reshape(32,96,1,1)
        out = io.BytesIO()
        np.savez_compressed(out, **state)
        return out.getvalue()
