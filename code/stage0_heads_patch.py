"""Stage 0 (a): add E[U] / WDL / margin heads to cnet2_model.CNet2.

Applies exact-anchor edits to code/cnet2_model.py and the BC_INIT
allow-list in code/cnet2_train.py, then verifies by import + weight load.

- utility: E[U] main value head, sigmoid -> [0,1], target U = 0.5*W + 0.5*sigmoid(M/1000)
- wdl: win/draw/loss auxiliary head (3 logits)
- margin: raw-margin expectation auxiliary head, predicts M/1000 (Huber)
Existing value head (tanh(M/3000)) is untouched; old weights load with
strict=False and only the 3 new heads report as missing keys.
Labels are derived from the dataset's stored value field v = tanh(M/3000)
(see stage0_smoke_heads.py); v1 caveat: atanh saturates for |M| > ~11400.

Usage (on DGX, from code/):  python3 stage0_heads_patch.py
Exits non-zero on any anchor mismatch (fail loudly, no silent partial edit).
"""
import sys

MODEL = "cnet2_model.py"
TRAINER = "cnet2_train.py"

MODEL_ANCHOR = "        self.value = nn.Linear(D, 1)\n"
MODEL_REPL = (
    "        self.value = nn.Linear(D, 1)\n"
    "        # Stage 0 value heads (2026-11-13, CarbonFormer adoption):\n"
    "        self.utility = nn.Linear(D, 1)   # E[U] main value head (sigmoid -> [0,1])\n"
    "        self.wdl = nn.Linear(D, 3)       # win/draw/loss auxiliary head\n"
    "        self.margin = nn.Linear(D, 1)    # raw-margin expectation aux head (predicts M/1000)\n"
)

CTX_ANCHOR = '''    def ctx_from(self, pre, teacher):
        """style enters only here: trunk summary [B, C+D+64] + teach[teacher] -> ctx."""
        return Fn.relu(self.ctx(torch.cat([pre, self.teach(teacher)], -1)))
'''
CTX_REPL = CTX_ANCHOR + '''
    def heads(self, ctx):
        """Stage 0 value heads from context: (utility [B] in [0,1] = E[U],
        wdl logits [B,3] (win/draw/loss), margin_k [B] = E[margin]/1000)."""
        u = torch.sigmoid(self.utility(ctx)).squeeze(1)
        return u, self.wdl(ctx), self.margin(ctx).squeeze(1)

    def forward_heads(self, b):
        """teacher-forced forward + Stage 0 heads.
        Returns (logits, value, aux, utility, wdl, margin_k)."""
        tok, ctx, aux = self.encode(b)
        lg = self.decode(b, tok, ctx)
        v = torch.tanh(self.value(ctx)).squeeze(1)
        u, wdl, mk = self.heads(ctx)
        return lg, v, aux, u, wdl, mk
'''

TRAINER_ANCHOR = '        bad_ = [k for k in mk_.missing_keys if not k.startswith("tattn.")] + list(mk_.unexpected_keys)\n'
TRAINER_REPL = (
    '        # Stage 0 new heads (utility/wdl/margin) may be absent in old weights, like tattn.*\n'
    '        bad_ = [k for k in mk_.missing_keys if not (k.startswith("tattn.") or k.startswith(("utility.", "wdl.", "margin.")))] + list(mk_.unexpected_keys)\n'
)


def patch(path, anchor, repl, marker):
    with open(path) as f:
        src = f.read()
    if marker in src:
        print(f"{path}: already patched, skip")
        return
    n = src.count(anchor)
    assert n == 1, f"{path}: anchor count {n} (want 1)"
    with open(path, "w") as f:
        f.write(src.replace(anchor, repl))
    print(f"patched {path}")


def main():
    patch(MODEL, MODEL_ANCHOR, MODEL_REPL, "self.utility = nn.Linear(D, 1)")
    patch(MODEL, CTX_ANCHOR, CTX_REPL, "def heads(self, ctx):")
    patch(TRAINER, TRAINER_ANCHOR, TRAINER_REPL, "Stage 0 new heads")
    import torch  # noqa: F401
    import cnet2_model as M
    net = M.CNet2()
    names = [n for n, _ in net.named_parameters() if n.split(".")[0] in ("utility", "wdl", "margin")]
    print("new head params:", names)
    assert len(names) == 6, names  # weight+bias x3
    ctx = torch.zeros(2, net.cfg["D"])
    u, wdl, mk = net.heads(ctx)
    assert u.shape == (2,) and wdl.shape == (2, 3) and mk.shape == (2,), (u.shape, wdl.shape, mk.shape)
    assert bool(((u >= 0) & (u <= 1)).all()), "utility out of [0,1]"
    print("PATCH-OK heads forward: utility in [0,1], wdl [B,3], margin [B]")


if __name__ == "__main__":
    main()
