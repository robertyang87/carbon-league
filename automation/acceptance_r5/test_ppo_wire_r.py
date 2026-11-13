"""Final acceptance: F01 wiring into the PPO production load points.
1) rl2_ppo_cnet2.Cnet2Adapter.make_model with a PRE-HEADS npz (was: strict load RuntimeError).
2) rl2_ppo_train._resume_sd with a pre-heads state_last.pt (net+opt) and with a new-format one."""
import os, sys
os.environ["BC_DEVICE"] = "cpu"
CODE = os.path.expanduser("/home/hatch/workspace/fix4-residual/tree")
sys.path.insert(0, CODE)
import numpy as np
import torch

from cnet2_model import CNet2, NEW_HEAD_KEYS
import rl2_ppo_cnet2 as RC
import rl2_ppo_train as RT

CFG = dict(C=32, NB=2, D=64, NH=4, L=1, E=16, G=8)
torch.manual_seed(11)
donor = CNet2(**CFG).eval()
old_sd = {k: v for k, v in donor.state_dict().items() if k not in NEW_HEAD_KEYS}
assert len(donor.state_dict()) - len(old_sd) == 6

# --- 1. adapter make_model: pre-heads npz
weights = {k: v.numpy().astype(np.float32) for k, v in old_sd.items()}
weights["meta"] = str({"cfg": CFG}).replace("'", '"')
try:
    torch.nn.Module.load_state_dict(CNet2(**CFG), old_sd)  # old behaviour control
    print("CTRL unexpected: strict load succeeded")
except RuntimeError as e:
    print("CTRL strict load of pre-heads weights fails as before:", str(e)[:80])
net = RC.Cnet2Adapter().make_model(weights)
with torch.no_grad():
    import cnet2_feat as F
    g = torch.Generator().manual_seed(1)
    B, Tn = 2, 14
    b = {"maps": torch.rand(B, F.P, 15, 15, generator=g), "tok_pos": torch.randint(0, 225, (B, Tn), generator=g),
         "tok_kind": torch.randint(0, 6, (B, Tn), generator=g), "tok_cargo": torch.rand(B, Tn, generator=g),
         "hist_cell": torch.randint(-1, 225, (B, Tn, F.H), generator=g),
         "hist_cargo": torch.rand(B, Tn, F.H, generator=g), "hist_move": torch.randint(0, 6, (B, Tn, F.H), generator=g),
         "tok_mask": torch.zeros(B, Tn, dtype=torch.bool), "scal": torch.rand(B, F.NS, generator=g),
         "teacher": torch.randint(0, F.N_TEACH, (B,), generator=g)}
    lg1, v1 = donor.policy_value(b) if hasattr(donor, "policy_value") else (None, None)
print("PASS adapter make_model pre-heads npz loads; params:", sum(p.numel() for p in net.parameters()))

# adapter round-trip with NEW-format weights must also work
w_new = {k: v.numpy().astype(np.float32) for k, v in donor.state_dict().items()}
w_new["meta"] = weights["meta"]
RC.Cnet2Adapter().make_model(w_new)
print("PASS adapter make_model new-format npz loads")

# --- 2. _resume_sd: build a genuine OLD optimizer state (ids = old param order)
old_names = [n for n, _ in donor.named_parameters() if n not in NEW_HEAD_KEYS]
class OldModel(torch.nn.Module):
    def __init__(self, donor):
        super().__init__()
        for n in old_names:
            mod = donor
            parts = n.split(".")
            for p in parts[:-1]: mod = getattr(mod, p)
            setattr(self, "_dummy_" + n.replace(".", "_"), torch.nn.Parameter(getattr(mod, parts[-1]).detach().clone()))
old_params = [p for n, p in donor.named_parameters() if n not in NEW_HEAD_KEYS]
old_opt = torch.optim.Adam(old_params, lr=3e-3)
for i, p in enumerate(old_params):
    st = old_opt.state[p]
    st["step"] = torch.tensor(5.0)
    st["exp_avg"] = torch.full_like(p, 0.01 * (i + 1))
    st["exp_avg_sq"] = torch.full_like(p, 0.02)
old_opt_sd = old_opt.state_dict()
# torch Adam state_dict keys state by integer id in old_params order -- rebuild as _resume_sd expects:
# migrate_optimizer_checked derives names from sd (the NET state dict), opt ids follow old order. OK.

net2 = CNet2(**CFG)
opt2 = torch.optim.Adam(net2.parameters(), lr=9e-9)
mode = RT._resume_sd(net2, old_sd, opt2, old_opt_sd, "net")
assert mode == "migrated", mode
# moments bitwise on a carried param + step preserved + lr from old group
pmap = dict(net2.named_parameters())
st = opt2.state[pmap[old_names[3]]]
assert torch.equal(st["exp_avg"], torch.full_like(pmap[old_names[3]], 0.01 * 4)), "exp_avg mismatch"
assert float(st["step"]) == 5.0
assert abs(opt2.param_groups[0]["lr"] - 3e-3) < 1e-12
with torch.no_grad():
    same = all(torch.equal(pmap[n], dict(donor.named_parameters())[n]) for n in old_names)
assert same, "carried weights differ from donor"
print("PASS _resume_sd pre-heads: migrated, exp_avg bitwise, step=5, lr=3e-3, weights == donor")

# --- 3. _resume_sd new-format: strict path
full_sd = donor.state_dict()
opt3 = torch.optim.Adam(CNet2(**CFG).parameters(), lr=1e-3)
new_opt_sd = opt3.state_dict()
net3 = CNet2(**CFG); opt3b = torch.optim.Adam(net3.parameters(), lr=1e-3)
mode = RT._resume_sd(net3, full_sd, opt3b, new_opt_sd, "net")
assert mode == "strict"
print("PASS _resume_sd new-format: strict path")

# --- 4. corrupt checkpoint (extra key) must raise, not migrate
bad = dict(full_sd); bad["bogus.weight"] = torch.zeros(1)
try:
    RT._resume_sd(CNet2(**CFG), bad, torch.optim.Adam(CNet2(**CFG).parameters()), new_opt_sd, "net")
    print("FAIL corrupt checkpoint accepted")
    sys.exit(1)
except RuntimeError as e:
    print("PASS _resume_sd corrupt checkpoint raises:", str(e)[:70])
print("ALL PPO WIRING CHECKS PASS")
