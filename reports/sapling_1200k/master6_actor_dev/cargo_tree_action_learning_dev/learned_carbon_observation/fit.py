"""Fit one estimator and its bias-only control on the preregistered terrain split."""
import time
import numpy as np
import torch
from torch import nn
from common import F, HERE, PROTOCOL, selected, sha, save
from runtime import _carbon_inputs

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(19113)
torch.use_deterministic_algorithms(True)
xs, ys, cards = [], [], []
for card in PROTOCOL["cards"]:
    if card["split"] != "train":
        continue
    start = time.monotonic()
    n = 0
    for feat, x, truth, o, joint, cfg in selected(card):
        features, eligible = _carbon_inputs(x, feat, F)
        xs.append(features[eligible])
        ys.append(truth[eligible] / 100)
        n += int(eligible.sum())
    cards.append(dict(seed=card["seed"], eligible=n))
    print("reconstructed", card["seed"], n, round(time.monotonic()-start, 2), flush=True)
features = torch.from_numpy(np.concatenate(xs))
labels = torch.from_numpy(np.concatenate(ys))
net = nn.Sequential(nn.Linear(features.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
# Start both residuals at exactly the parent map, including on cold-start boards.
nn.init.zeros_(net[-1].weight)
nn.init.zeros_(net[-1].bias)
constant = nn.Parameter(torch.zeros(1))
opts = [torch.optim.Adam(net.parameters(), lr=.002), torch.optim.Adam([constant], lr=.002)]
generator = torch.Generator().manual_seed(19113)
log = []
for epoch in range(18):
    losses = np.zeros(2)
    for indices in torch.randperm(len(labels), generator=generator).split(1024):
        batch, target = features[indices], labels[indices]
        baseline = batch[:, 0]
        for idx, delta in enumerate((net(batch).flatten(), constant)):
            loss = nn.functional.smooth_l1_loss(baseline + delta, target, beta=.03)
            opts[idx].zero_grad()
            loss.backward()
            opts[idx].step()
            losses[idx] += loss.detach().item()*len(indices)
    log.append((losses / len(labels)).tolist())
    print("epoch", epoch+1, log[-1], flush=True)
weights = [net[0].weight.T, net[0].bias, net[-1].weight.flatten(), net[-1].bias, constant]
np.savez(HERE / "weights.npz", **{k:w.detach().numpy().copy() for k,w in zip(("w","b","v","a","c"),weights)})
save(HERE/"fit.json", dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()), weights_sha256=sha((HERE/"weights.npz").read_bytes()), training_cards=cards, eligible=len(labels), features=features.shape[1], params=sum(v.numel() for v in net.parameters()), constant=constant.item(), epoch_train_network_constant=log))
