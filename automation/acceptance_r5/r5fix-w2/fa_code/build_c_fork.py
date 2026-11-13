"""build_c_fork: generate the 案C pricing forks from base_v53.py.

Variants (PARAMS switches price_soft / price_lethal / price_minreach):
  fork_c_pricing.py  all tiers ON   (the deliverable)
  fork_c_soft.py     soft tier only (ablation)
  fork_c_lethal.py   lethal tier (incl. tie pricing) only (ablation)
  fork_c_off.py      all tiers OFF  (parity check vs base: must reproduce
                                     base games bit-for-bit)

Every replacement is asserted to occur exactly once in the base source,
so a silent no-op edit is impossible. The watch/statistics machinery is
present in all variants (it never feeds decisions unless its tier's
switch is on), mirroring probe_v53's zero-pollution discipline.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = '''"""Carbon League strategy -- {name} (muse-spark1-3).

NOT ORIGINAL. Derived from the public strategy claude-opus-v53 by author
"Claude" (source: https://132.232.176.153/source/97898ce7047c0bb1dd7a13f02eff0dac),
local base copy base_v53.py (byte-identical to reference_claude_opus_v53.py).
Change in this version (structure case C, step 2: safety-layer pricing,
per avoid_tax_report.md section 7):
  * soft tier (DP distance-2 soft risk): fixed f prior ->
    P_hat(strike | n2 tier x enemy cargo tier x my cargo tier) * (c+30),
    buckets filled online in probe-(b) accounting; a bucket's posterior
    only takes over at n>=30, before that the current f value is the
    price (accuracy red line).
  * lethal tier: -1e6 kept only for the red zone (my cargo >= 150 and
    P_hat >= 0.5); otherwise expected price P_hat*(2c+30) (cargo swings
    to the enemy, run_ev accounting). Tie deduction scaled by P_tie with
    a unit-value floor. Endgame R<=6 scaling and the run EV branch are
    untouched.
  * minreach tier: flat avoid1/avoid2 -> P_hat(z,c)*(c+30), priors =
    the current 0.5 / 0.15.
  * pred_next strong feature: an enemy predicted to land on z raises
    that cell's P_hat floor to 0.9.
  Switches: price_soft / price_lethal / price_minreach in PARAMS
  (this file: soft={soft}, lethal={lethal}, minreach={minreach}).
The original docstring follows.
"""

'''

PBSTATE = ("self.pb1 = {}; self.pb1m = {}; self.pbt = {}; "
           "self.pb2 = {}; self.pb2m = {}; "
           "self.pb_pend1 = []; self.pb_pend2 = []; "
           "self.pb_lvl = [0, 0, 0]; self._pc = {}; ")

HELPERS = '''

def _pb_tier(x):
    # probe (b) cargo tiers: 0 / 1-80 / 81-150 / 151-250 / 251+
    if x <= 0:
        return 0
    if x <= 80:
        return 1
    if x <= 150:
        return 2
    if x <= 250:
        return 3
    return 4

'''

METHODS = '''    # ------------------------------------------------------------------
    # case C: strike-rate statistics + expectation pricing helpers.
    # Watch accounting mirrors probe (b): a pair-level opportunity is an
    # enemy collector e and my collector m with dm (m's realised final
    # destination) in N1[e pos]; strike iff e's realised cell == dm.
    # An enemy that vanished is not an observation (not counted).
    def _post(self, tab, key, mtab, mkey, prior):
        """Priced posterior: the bucket posterior (prior strength
        price_strength) takes over only at n >= price_min_n; else the
        marginal bucket under the same rule; else the pure prior.
        Returns (P_hat, n, level) with level 2=bucket, 1=marginal,
        0=prior."""
        ck = (id(tab), key, prior)
        hit = self._pc.get(ck)
        if hit is not None:
            return hit
        P = self.P
        o, s = tab.get(key, (0, 0))
        if o >= P['price_min_n']:
            r = ((s + P['price_strength'] * prior)
                 / (o + P['price_strength']), o, 2)
        elif mtab is not None:
            o2, s2 = mtab.get(mkey, (0, 0))
            if o2 >= P['price_min_n']:
                r = ((s2 + P['price_strength'] * prior)
                     / (o2 + P['price_strength']), o2, 1)
            else:
                r = (prior, o, 0)
        else:
            r = (prior, o, 0)
        self._pc[ck] = r
        self.pb_lvl[r[2]] += 1
        return r

    def _pb_resolve(self, workers, step):
        self._pc = {}
        if self.pb_pend1:
            for (ew_, dm_, ect_, mct_, kind_) in self.pb_pend1:
                w_ = workers.get(ew_)
                if w_ is None:
                    continue        # enemy gone: not an observation
                if kind_ == 0:
                    b_ = self.pb1.setdefault((ect_, mct_), [0, 0])
                    m_ = self.pb1m.setdefault(mct_, [0, 0])
                    b_[0] += 1
                    m_[0] += 1
                    if w_[0] == dm_:
                        b_[1] += 1
                        m_[1] += 1
                else:
                    b_ = self.pbt.setdefault(mct_, [0, 0])
                    b_[0] += 1
                    if w_[0] == dm_:
                        b_[1] += 1
            self.pb_pend1 = []
        if self.pb_pend2:
            keep_ = []
            for (st_, ews_, dm_, key_) in self.pb_pend2:
                hit_ = False
                for e_ in ews_:
                    w_ = workers.get(e_)
                    if w_ is not None and w_[0] == dm_:
                        hit_ = True
                        break
                if hit_ or step - st_ >= 2:
                    b_ = self.pb2.setdefault(key_, [0, 0])
                    m_ = self.pb2m.setdefault(key_[0], [0, 0])
                    b_[0] += 1
                    m_[0] += 1
                    if hit_:
                        b_[1] += 1
                        m_[1] += 1
                else:
                    keep_.append((st_, ews_, dm_, key_))
            self.pb_pend2 = keep_

'''


def sub_once(src, old, new, tag):
    n = src.count(old)
    assert n == 1, f"{tag}: anchor occurs {n}x"
    return src.replace(old, new)


def build():
    src = open(os.path.join(HERE, "base_v53.py")).read()

    # ---- PARAMS: pricing switches + constants
    src = sub_once(
        src,
        "avoid1=0.5, avoid2=0.15, sweep_w=0.3,",
        "avoid1=0.5, avoid2=0.15, price_soft=1, price_lethal=1, "
        "price_minreach=1, price_min_n=30, price_strength=30.0, "
        "price_lethal_prior=0.5, price_red=0.5, price_red_carry=150.0, "
        "price_pred_hi=0.9, price_tie_unit=30.0, price_tie_frac=0.2, "
        "sweep_w=0.3,",
        "params")

    # ---- module helper before class Bot
    src = sub_once(src, "class Bot:", HELPERS + "class Bot:", "helpers")

    # ---- Bot.__init__ state
    src = sub_once(
        src,
        "        self.run_b = [[0, 0], [0, 0], [0, 0]]\n"
        "        self.run_last = -99",
        "        self.run_b = [[0, 0], [0, 0], [0, 0]]\n"
        "        " + PBSTATE.replace("; ", ";\n        ").rstrip() + "\n"
        "        self.run_last = -99",
        "init-state")

    # ---- step-0 reset line state
    src = sub_once(
        src,
        "self.run_b = [[0, 0], [0, 0], [0, 0]]; self.run_last = -99; "
        "self.hunt_t = {};",
        "self.run_b = [[0, 0], [0, 0], [0, 0]]; self.run_last = -99; "
        + PBSTATE + "self.hunt_t = {};",
        "reset-state")

    # ---- Bot methods (before act)
    src = sub_once(
        src,
        "    def act(self, obs, conf):",
        METHODS + "    def act(self, obs, conf):",
        "methods")

    # ---- resolve watches right after the run_pend resolution
    src = sub_once(
        src,
        "        self.run_pend = {}\n"
        "        for wid, w in workers.items():\n"
        "            if w[3] == me and w[2] == 'C' and w[1] == 0 and "
        "w[0] in my_bases:",
        "        self.run_pend = {}\n"
        "        self._pb_resolve(workers, step)\n"
        "        for wid, w in workers.items():\n"
        "            if w[3] == me and w[2] == 'C' and w[1] == 0 and "
        "w[0] in my_bases:",
        "resolve-call")

    # ---- pred_hit map after pred_next construction
    src = sub_once(
        src,
        "        for k_ in [k_ for k_ in self.ehist if k_ not in workers]:\n"
        "            del self.ehist[k_]\n",
        "        for k_ in [k_ for k_ in self.ehist if k_ not in workers]:\n"
        "            del self.ehist[k_]\n"
        "        pred_hit = {}\n"
        "        for ew_, z_ in pred_next.items():\n"
        "            ec_ = workers[ew_][1]\n"
        "            if z_ not in pred_hit or ec_ < pred_hit[z_]:\n"
        "                pred_hit[z_] = ec_\n",
        "pred-hit")

    # ---- minreach tier (DP costs C1 / Cr)
    src = sub_once(
        src,
        "            if enC and c > 0:\n"
        "                a1 = P['avoid1'] * (c + 30.0); "
        "a2 = P['avoid2'] * (c + 30.0)\n"
        "                for z in range(N):\n"
        "                    if minreach[z] <= c and z not in my_base_set:\n"
        "                        C1[z] += a1\n"
        "                        Cr[z] += a2\n",
        "            if enC and c > 0:\n"
        "                if P['price_minreach']:\n"
        "                    mct_ = _pb_tier(c)\n"
        "                    ph1_ = {}; phr_ = {}\n"
        "                    for ect_ in range(5):\n"
        "                        ph1_[ect_] = self._post(self.pb1, "
        "(ect_, mct_), self.pb1m, mct_, P['avoid1'])[0]\n"
        "                        phr_[ect_] = self._post(self.pb1, "
        "(ect_, mct_), self.pb1m, mct_, P['avoid2'])[0]\n"
        "                    for z in range(N):\n"
        "                        mr_ = minreach[z]\n"
        "                        if mr_ <= c and z not in my_base_set:\n"
        "                            p1_ = ph1_[_pb_tier(mr_)]\n"
        "                            if pred_hit.get(z, INF) < c:\n"
        "                                p1_ = max(p1_, "
        "P['price_pred_hi'])\n"
        "                            C1[z] += p1_ * (c + 30.0)\n"
        "                            Cr[z] += phr_[_pb_tier(mr_)] "
        "* (c + 30.0)\n"
        "                else:\n"
        "                    a1 = P['avoid1'] * (c + 30.0); "
        "a2 = P['avoid2'] * (c + 30.0)\n"
        "                    for z in range(N):\n"
        "                        if minreach[z] <= c and "
        "z not in my_base_set:\n"
        "                            C1[z] += a1\n"
        "                            Cr[z] += a2\n",
        "minreach")

    # ---- soft tier (DP distance-2 soft risk)
    src = sub_once(
        src,
        "            # soft risk for the next two arrivals\n"
        "            if enC:\n"
        "                for z, lst in cnt2.items():\n"
        "                    n2 = sum(1 for x in lst if x <= c)\n"
        "                    if n2:\n"
        "                        pen = P['soft_w'] * (c + 30.0) * "
        "(1.0 if n2 >= 3 else (0.25 if n2 == 2 else 0.06))\n"
        "                        C0[z] += pen\n"
        "                        C1[z] += 0.5 * pen\n",
        "            # soft risk for the next two arrivals\n"
        "            if enC:\n"
        "                mct_ = _pb_tier(c)\n"
        "                for z, lst in cnt2.items():\n"
        "                    thr_ = [x for x in lst if x <= c]\n"
        "                    n2 = len(thr_)\n"
        "                    if n2:\n"
        "                        if P['price_soft']:\n"
        "                            n2t_ = 1 if n2 == 1 else "
        "(2 if n2 == 2 else 3)\n"
        "                            prior_ = P['soft_w'] * (1.0 "
        "if n2t_ == 3 else (0.25 if n2t_ == 2 else 0.06))\n"
        "                            ph_ = self._post(self.pb2, "
        "(n2t_, _pb_tier(min(thr_)), mct_), self.pb2m, n2t_, "
        "prior_)[0]\n"
        "                            if pred_hit.get(z, INF) <= c:\n"
        "                                ph_ = max(ph_, "
        "P['price_pred_hi'])\n"
        "                            pen = ph_ * (c + 30.0)\n"
        "                        else:\n"
        "                            pen = P['soft_w'] * (c + 30.0) * "
        "(1.0 if n2 >= 3 else (0.25 if n2 == 2 else 0.06))\n"
        "                        C0[z] += pen\n"
        "                        C1[z] += 0.5 * pen\n",
        "soft")

    # ---- lethal tier (action layer, plain -1e6 branch only)
    src = sub_once(
        src,
        "                        else:\n"
        "                            # lethal: ordered by how likely "
        "the hunter really goes there\n"
        "                            q5[a] -= 1e6 + (500.0 if a == 0 "
        "else 0.0)\n",
        "                        else:\n"
        "                            if P['price_lethal']:\n"
        "                                ph_ = self._post(self.pb1, "
        "(_pb_tier(m), _pb_tier(c)), self.pb1m, _pb_tier(c), "
        "P['price_lethal_prior'])[0]\n"
        "                                if pred_hit.get(z, INF) < c:\n"
        "                                    ph_ = max(ph_, "
        "P['price_pred_hi'])\n"
        "                                if c >= P['price_red_carry'] "
        "and ph_ >= P['price_red']:\n"
        "                                    q5[a] -= 1e6 + (500.0 "
        "if a == 0 else 0.0)\n"
        "                                else:\n"
        "                                    q5[a] -= ph_ * "
        "(2.0 * c + 30.0) + (ph_ * 500.0 if a == 0 else 0.0)\n"
        "                            else:\n"
        "                                # lethal: ordered by how "
        "likely the hunter really goes there\n"
        "                                q5[a] -= 1e6 + (500.0 "
        "if a == 0 else 0.0)\n",
        "lethal")

    # ---- tie pricing (same switch as lethal tier)
    src = sub_once(
        src,
        "                elif m == c:\n"
        "                    q5[a] -= tie_cost + c\n",
        "                elif m == c:\n"
        "                    if P['price_lethal']:\n"
        "                        pt_ = self._post(self.pbt, "
        "_pb_tier(c), None, None, 1.0)[0]\n"
        "                        q5[a] -= max(P['price_tie_unit'] + "
        "P['price_tie_frac'] * c, pt_ * (tie_cost + c))\n"
        "                    else:\n"
        "                        q5[a] -= tie_cost + c\n",
        "tie")

    # ---- watch recording after final destinations are set
    src = sub_once(
        src,
        "        # ---------------- recruiting "
        "------------------------------",
        "        # --- case C: record strike watches vs this turn's "
        "final destinations\n"
        "        if enC:\n"
        "            for mw_ in myC:\n"
        "                dm_ = self.last_dest.get(mw_)\n"
        "                if dm_ is None:\n"
        "                    continue\n"
        "                mc_ = workers[mw_][1]\n"
        "                if mc_ <= 0:\n"
        "                    continue\n"
        "                mct_ = _pb_tier(mc_)\n"
        "                thr2_ = []\n"
        "                emin2_ = None\n"
        "                for ew_ in enC:\n"
        "                    ep_ = workers[ew_][0]; ec_ = workers[ew_][1]\n"
        "                    d_ = DIST[ep_][dm_]\n"
        "                    if d_ <= 1:\n"
        "                        if ec_ < mc_:\n"
        "                            self.pb_pend1.append((ew_, dm_, "
        "_pb_tier(ec_), mct_, 0))\n"
        "                        elif ec_ == mc_:\n"
        "                            self.pb_pend1.append((ew_, dm_, "
        "_pb_tier(ec_), mct_, 1))\n"
        "                    if d_ <= 2 and ec_ <= mc_:\n"
        "                        thr2_.append(ew_)\n"
        "                        if emin2_ is None or ec_ < emin2_:\n"
        "                            emin2_ = ec_\n"
        "                if thr2_:\n"
        "                    n2t_ = 1 if len(thr2_) == 1 else "
        "(2 if len(thr2_) == 2 else 3)\n"
        "                    self.pb_pend2.append((step, tuple(thr2_), "
        "dm_, (n2t_, _pb_tier(emin2_), mct_)))\n"
        "\n"
        "        # ---------------- recruiting "
        "------------------------------",
        "record")

    variants = [
        ("fork_c_pricing.py", "fork-c-pricing", 1, 1, 1),
        ("fork_c_soft.py", "fork-c-soft (ablation: soft tier only)",
         1, 0, 0),
        ("fork_c_lethal.py", "fork-c-lethal (ablation: lethal tier only)",
         0, 1, 0),
        ("fork_c_off.py", "fork-c-off (ablation: all tiers off)", 0, 0, 0),
    ]
    for fname, name, s_, l_, m_ in variants:
        text = src
        if (s_, l_, m_) != (1, 1, 1):
            text = sub_once(
                text,
                "price_soft=1, price_lethal=1, price_minreach=1,",
                f"price_soft={s_}, price_lethal={l_}, "
                f"price_minreach={m_},",
                f"variant-{fname}")
        out = HEADER.format(name=name, soft=s_, lethal=l_,
                            minreach=m_) + text
        path = os.path.join(HERE, fname)
        with open(path, "w") as f:
            f.write(out)
        print("wrote", fname, len(out), "bytes")


if __name__ == "__main__":
    build()
