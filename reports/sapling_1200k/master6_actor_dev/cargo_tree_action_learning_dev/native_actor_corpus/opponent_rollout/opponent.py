"""NumPy-only, stateful simulated Master3s2 opponent. Create one instance per simulated physical seat."""
from copy import deepcopy

import numpy as np
from model import F, HERE, bridge, load, sha


class OriginalOpponentRollout:
    def __init__(self, seat, *, weights=None):
        if seat not in (0, 1):
            raise ValueError("expected physical player 0 or 1")
        self.seat=seat
        self.encoder=F.Feat(seat)
        self.next_step=0
        if weights is None:
            path=HERE/"model.npz"
            if sha(path)!=load(HERE/"fit.json")["weights_sha256"]:
                raise ValueError("opponent weights do not match training receipt")
            with np.load(path,allow_pickle=False) as arrays:
                weights={k:arrays[k] for k in arrays}
        self.weights=weights

    def fork(self):
        branch=OriginalOpponentRollout(self.seat,weights=self.weights)
        branch.encoder=deepcopy(self.encoder)
        branch.next_step=self.next_step
        return branch

    def _observe(self,observation):
        if observation["player"]!=self.seat or observation["step"]!=self.next_step:
            raise ValueError("use one continuous physical-seat history per rollout branch")
        # The offline simulator may also own a full physical board. Only the acting seat's
        # masked carbon and public roster reach the encoder, including at synthetic roots.
        public={k:observation[k] for k in ("step","player","players","trees","carbon")}
        self.next_step+=1
        return self.encoder.observe(public)

    def warm(self,observation):
        self._observe(observation)

    def action(self,observation):
        x=self._observe(observation)
        return bridge.decode(self.encoder,x,self.weights)[0]
