# ---------------------------------------------------------------------
# Distilled actor (Case-B stage 1).  This file is NOT standalone: it is
# concatenated by gen_distilled.py after (a) the verbatim Belief chunk
# of base_v53.py and (b) distill_features.py.  It expects: np, Belief,
# NB, DIST, featurize, and the SHIELDS flag injected by the generator.
#
# Inference protocol (shadow-v2 lessons, design section 1.7):
#   worker head: sample T=1 from the softmax, sequentially in slot
#     order, with same-step destination reservation (own base cells
#     exempt) -- argmax collapses to the stay mode and unreserved
#     sampling stacks own workers into self-collisions.
#   base head: argmax, gated by affordability / unit cap.
# Hybrid shields (only when SHIELDS is True; design section 1.1):
#   (a) collision-safety mask before sampling (rl2 danger口径),
#   (b) endgame return override from step 270,
#   (c) base-mouth integration override from step 297.
# ---------------------------------------------------------------------
import os as _os

_WPATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       "distilled_v1_weights.npz")
_MOVE = {1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
_REC = {1: "RECCOLLECTOR", 2: "RECPLANTER"}
_DELTA = {0: (0, 0), 1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
DANGER_MIN_CARGO = 15.0
ENDGAME_STEP = 270
INTEGRATE_STEP = 297


def _suf(k):
    try:
        return int(str(k).rsplit("-", 1)[1])
    except Exception:
        return 0


def _dest(pos, a):
    dr, dc = _DELTA[int(a)]
    r, c = divmod(int(pos), SIZE)
    return ((r + dr) % SIZE) * SIZE + ((c + dc) % SIZE)


def _np_forward(w, prefix, x):
    h = (x - w[prefix + "mean"]) / w[prefix + "std"]
    for i in (1, 2):
        h = h @ w[f"{prefix}W{i}"].T + w[f"{prefix}b{i}"]
        h = np.maximum(h, 0.0)
    return h @ w[prefix + "W3"].T + w[f"{prefix}b3"]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Tracker:
    """Deployment-side tracker: prev_act per worker id, the seq triple
    and steps-since-deposit -- the same quantities distill_capture
    recorded from the teacher's games (identical update rules)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_pos = {}
        self.prev_cargo = {}
        self.seq_prev_n = None
        self.seq_prev_rec = 0
        self.seq_last_rec = None
        self.seq_last_step = None
        self.last_dep_step = None

    def observe(self, obs, me, my_bases):
        step = int(obs["step"])
        workers = obs["players"][me][2]
        wids = sorted(workers.keys(), key=_suf)
        prev_act = np.full(10, 5, dtype=np.int8)
        for j, wid in enumerate(wids[:10]):
            if wid in self.prev_pos:
                a = _delta_act(self.prev_pos[wid], workers[wid][0])
                prev_act[j] = a
        for wid, w in workers.items():
            if w[2] == "COLLECTOR" and float(w[1]) == 0.0 \
                    and int(w[0]) in my_bases \
                    and self.prev_cargo.get(wid, 0.0) > 0:
                self.last_dep_step = step
        n_now = len(wids)
        if self.seq_prev_n is None or self.seq_last_step is None \
                or step != self.seq_last_step + 1:
            seq = (float(n_now), 0.0, 60.0)
        else:
            lost = max(0, self.seq_prev_n + self.seq_prev_rec - n_now)
            since = 60.0 if self.seq_last_rec is None \
                else float(step - self.seq_last_rec)
            seq = (float(self.seq_prev_n), float(lost), since)
        since_dep = 60.0 if self.last_dep_step is None \
            else float(min(step - self.last_dep_step, 60))
        return prev_act, np.asarray(seq, dtype=np.float32), since_dep

    def update(self, obs, me, n_rec):
        step = int(obs["step"])
        workers = obs["players"][me][2]
        self.seq_prev_n = len(workers)
        self.seq_prev_rec = n_rec
        self.seq_last_step = step
        if n_rec > 0:
            self.seq_last_rec = step
        self.prev_pos = {wid: int(w[0]) for wid, w in workers.items()}
        self.prev_cargo = {wid: float(w[1])
                           for wid, w in workers.items()}


def _delta_act(prev_pos, cur_pos):
    if int(prev_pos) == int(cur_pos):
        return 0
    pr, pc = divmod(int(prev_pos), SIZE)
    cr, cc = divmod(int(cur_pos), SIZE)
    dr = (cr - pr) % SIZE
    dc = (cc - pc) % SIZE
    if dr == SIZE - 1 and dc == 0:
        return 1
    if dr == 0 and dc == 1:
        return 2
    if dr == 1 and dc == 0:
        return 3
    if dr == 0 and dc == SIZE - 1:
        return 4
    return 5


class DistilledActor:
    def __init__(self, weights=None, seed=20261103):
        if weights is None:
            weights = dict(np.load(_WPATH))
        self.w = weights
        self.rng = np.random.default_rng(seed)
        self.belief = None
        self.tracker = _Tracker()
        self.last_dest = {}

    def reset(self):
        self.belief = None
        self.tracker.reset()
        self.last_dest = {}

    # -- canonical arrays in corpus layout --
    def _canonical(self, obs):
        players = obs["players"]
        cash = np.array([float(players[0][0]), float(players[1][0])])
        bases = np.full((2, 2), -1, dtype=np.int64)
        workers = np.zeros((2, 10, 3), dtype=np.float32)
        wmask = np.zeros((2, 10), dtype=bool)
        my_wids, my_bids = [], []
        trows = []
        for pid in range(2):
            pl = players[pid]
            bids = sorted(pl[1].keys(), key=_suf)
            for j, bid in enumerate(bids[:2]):
                bases[pid, j] = int(pl[1][bid])
            wids = sorted(pl[2].keys(), key=_suf)
            for j, wid in enumerate(wids[:10]):
                w = pl[2][wid]
                workers[pid, j] = (float(w[0]), float(w[1]),
                                   0.0 if w[2] == "COLLECTOR" else 1.0)
                wmask[pid, j] = True
            for tid, t in pl[3].items():
                trows.append((pid, _suf(tid), float(t[0]), float(t[1])))
            if pid == obs["player"]:
                my_wids, my_bids = wids, bids
        trows.sort()
        trees = np.zeros((30, 3), dtype=np.float32)
        tmask = np.zeros(30, dtype=bool)
        for j, (pid, _s, pos, age) in enumerate(trows[:30]):
            trees[j] = (pos, age, float(pid))
            tmask[j] = True
        return cash, bases, workers, wmask, trees, tmask, my_wids, my_bids

    def _danger_mask(self, canon, me, j, pos, cargo):
        """Shield (a): forbid landing cells a lighter enemy collector
        can reach next step (its own cell + 4-neighbours); own bases
        exempt; None if no mask applies / mask would block everything."""
        cash, bases, workers, wmask, trees, tmask = canon
        q = 1 - me
        hard = set()
        for k in range(10):
            if wmask[q, k] and int(workers[q, k, 2]) == 0 \
                    and float(workers[q, k, 1]) < cargo:
                epos = int(workers[q, k, 0])
                hard.add(epos)
                for nb in NB[epos]:
                    hard.add(int(nb))
        if not hard:
            return None
        own_bases = {int(b) for b in bases[me] if int(b) >= 0}
        allowed = np.ones(5, dtype=bool)
        for a in range(5):
            d = _dest(pos, a)
            if d in hard and d not in own_bases:
                allowed[a] = False
        if not np.any(allowed):
            return None
        return allowed

    def _home_action(self, pos, my_bases):
        """Distance-descent step toward the nearest own base; 0 (stay)
        when already on a base cell; None if no strictly-closer move."""
        if pos in my_bases:
            return 0
        cur = min(DIST[pos][b] for b in my_bases)
        best_a, best_d = None, cur
        for a in (1, 2, 3, 4):
            d = min(DIST[_dest(pos, a)][b] for b in my_bases)
            if d < best_d:
                best_a, best_d = a, d
        return best_a

    def __call__(self, obs, conf):
        step = int(obs["step"])
        me = int(obs["player"])
        players = obs["players"]
        my_bases = [int(pos) for _bid, pos in players[me][1].items()]
        en_bases = [int(pos) for _bid, pos in players[1 - me][1].items()]
        if step == 0 or self.belief is None:
            self.belief = Belief(me, my_bases + en_bases)
            self.tracker.reset()
            self.last_dest = {}
        # belief update, exactly the teacher's call (base_v53 L710)
        workers_d = {}
        trees_d = {}
        for pid in range(2):
            pl = players[pid]
            for wid, w in pl[2].items():
                workers_d[wid] = (int(w[0]), float(w[1]),
                                  "C" if w[2] == "COLLECTOR" else "P",
                                  pid)
            for tid, t in pl[3].items():
                trees_d[tid] = (int(t[0]), float(t[1]), pid)
        visible = []
        for wid, w in workers_d.items():
            if w[3] == me:
                if w[2] == "C":
                    visible.append(w[0])
                else:
                    visible.extend(NB[w[0]])
        self.belief.update(workers_d, trees_d, visible, obs["carbon"],
                           step == 0, self.last_dest)
        est = np.asarray(self.belief.estimate(), dtype=np.float32)
        known = np.asarray(self.belief.known, dtype=bool)

        prev_act, seq, since_dep = self.tracker.observe(
            obs, me, set(my_bases))
        canon = self._canonical(obs)
        cash, bases, workers, wmask, trees, tmask, my_wids, my_bids = \
            canon
        W_x, B_x = featurize(me, est, known, cash, bases, workers,
                             wmask, trees, tmask, step, prev_act, seq,
                             since_dep)
        acts = {}
        n_my = len(my_wids)
        chosen = np.zeros(n_my, dtype=np.int64)
        if n_my:
            probs = _softmax(_np_forward(self.w, "w_", W_x))
            # NaN sanitize (RL stage-2 lesson: an extreme standardized
            # feature in a rare state can poison the softmax; without
            # this the sampler raises and the whole game is lost).
            # Only states that would otherwise crash are affected.
            if not np.isfinite(probs).all():
                probs = np.nan_to_num(probs, nan=0.0, posinf=0.0,
                                      neginf=0.0)
                rs = probs.sum(axis=1, keepdims=True)
                probs = np.where(rs > 1e-12, probs / np.maximum(rs, 1e-12),
                                 1.0 / 5.0)
            own_bases = {int(b) for b in bases[me] if int(b) >= 0}
            reserved = set()
            for j in range(n_my):
                pos = int(workers[me, j, 0])
                cargo = float(workers[me, j, 1])
                mask = np.ones(5, dtype=bool)
                if SHIELDS and int(workers[me, j, 2]) == 0 \
                        and cargo >= DANGER_MIN_CARGO:
                    dm = self._danger_mask(
                        (cash, bases, workers, wmask, trees, tmask),
                        me, j, pos, cargo)
                    if dm is not None:
                        mask &= dm
                for a in range(5):
                    if mask[a] and _dest(pos, a) in reserved \
                            and _dest(pos, a) not in own_bases:
                        mask[a] = False
                if not np.any(mask):
                    mask = np.ones(5, dtype=bool)
                pr = np.where(mask, probs[j], 0.0)
                tot = pr.sum()
                pr = pr / tot if tot > 1e-12 else mask / mask.sum()
                # ARGMAX, not T=1 sampling (segment-2c A/B, seed 1000:
                # sampled play dies at cash ~7 with cargo hoarded and
                # never deposited; argmax play runs a living economy,
                # final cash 1238 and cargo cleared).  With a ~0.69
                # top-1 BC head, sampling breaks the multi-step
                # go-home sequences every few steps; the shadow-v2
                # sampling doctrine came from PPO-trained policies
                # and does not transfer to pure BC deployment.
                a = int(np.argmax(pr))
                chosen[j] = a
                reserved.add(_dest(pos, a))
            if SHIELDS:
                # (b) endgame return + (c) base-mouth integration +
                # (d-R3) cash-chain rescue: deterministic overrides,
                # lightest cargo first so the lightest lands the base
                # cell for consolidation.  Priority: (c) > (b) > (d).
                richest_loaded = None
                if float(cash[me]) < 30.0:
                    loaded = [j for j in range(n_my)
                              if int(workers[me, j, 2]) == 0
                              and float(workers[me, j, 1]) > 0]
                    if loaded:
                        richest_loaded = max(
                            loaded,
                            key=lambda j: float(workers[me, j, 1]))
                order = sorted(range(n_my),
                               key=lambda j: float(workers[me, j, 1]))
                for j in order:
                    if int(workers[me, j, 2]) != 0:
                        continue
                    cargo = float(workers[me, j, 1])
                    if cargo <= 0:
                        continue
                    pos = int(workers[me, j, 0])
                    if step >= INTEGRATE_STEP and \
                            min(DIST[pos][b] for b in my_bases) <= 2:
                        ha = self._home_action(pos, my_bases)
                        if ha is not None:
                            chosen[j] = ha
                    elif step >= ENDGAME_STEP:
                        ha = self._home_action(pos, my_bases)
                        if ha is not None:
                            chosen[j] = ha
                    elif j == richest_loaded:
                        ha = self._home_action(pos, my_bases)
                        if ha is not None:
                            chosen[j] = ha
            for j, wid in enumerate(my_wids):
                a = int(chosen[j])
                self.last_dest[wid] = _dest(int(workers[me, j, 0]), a)
                if a != 0:
                    acts[wid] = _MOVE[a]
        # base head: argmax with affordability / cap gates.
        # Shield (d-R1/R2) cash-chain rescue comes first: the states
        # it covers (0 workers / 0 collectors) are outside the
        # teacher's distribution, where the cloned base head's
        # argmax is an unguarded extrapolation (shadow-DS lock-in).
        n_rec = 0
        if len(my_bids):
            cash_now = float(cash[me])
            forced = False
            if SHIELDS and cash_now >= 30.0 and n_my < 10:
                n_coll = int(np.sum(
                    (workers[me, :n_my, 2] == 0))) if n_my else 0
                if n_my == 0 or n_coll == 0:
                    acts[my_bids[0]] = "RECCOLLECTOR"
                    cash_now -= 30.0
                    n_rec += 1
                    forced = True
            logits = _np_forward(self.w, "b_", B_x)
            for j, bid in enumerate(my_bids[:len(B_x)]):
                if forced and j == 0:
                    continue
                a = int(np.argmax(logits[j]))
                if a == 0:
                    continue
                if cash_now < 30.0 or n_my + n_rec >= 10:
                    continue
                acts[bid] = _REC[a]
                cash_now -= 30.0
                n_rec += 1
        self.tracker.update(obs, me, n_rec)
        return acts


def make_agent(seed=20261103):
    return DistilledActor(seed=seed)


_ACTOR = None


def agent(observation, configuration):
    global _ACTOR
    if observation["step"] == 0 or _ACTOR is None:
        _ACTOR = DistilledActor()
    try:
        return _ACTOR(observation, configuration)
    except Exception:
        return {}
