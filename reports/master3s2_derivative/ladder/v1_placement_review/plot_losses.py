"""Plot cumulative normal deliveries and net tree income alongside the official wallet."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "losses_nov20"))
from compare import economy


def read_curves(game):
    seat = game["seat"]
    raw = (HERE.parent / "replays" / (game["match"] + "-1.json.gz")).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == game["replay_sha256"]
    env = json.loads(gzip.decompress(raw))
    timeline, _, _, _ = economy.reconstruct(env["steps"], env["configuration"])
    rows = np.asarray(timeline)
    flows = rows[:, seat] - rows[:, 1 - seat]
    tree_cols = [economy.KEYS.index(name) for name in ("tree", "seize", "plant")]
    wallet = [step["margin"] for step in game["steps"]]
    bank = np.r_[0, flows[:, economy.KEYS.index("bank")].cumsum()]
    tree = np.r_[0, flows[:, tree_cols].sum(axis=1).cumsum()]
    assert np.allclose(wallet, np.r_[0, flows.sum(axis=1).cumsum()], atol=1e-6, rtol=0)
    assert abs(bank[-1] - game["whole"]["difference"]["bank"]) <= 0.00501
    assert abs(tree[-1] - game["whole"]["difference"]["net_tree"]) <= 0.00501
    return wallet, bank, tree


def main():
    font = next(x for x in font_manager.fontManager.ttflist if "Noto Sans CJK" in x.name)
    plt.rcParams.update({"font.family": font.name, "font.size": 10, "axes.unicode_minus": False})
    ledger = json.loads(gzip.decompress((HERE / "ledger.json.gz").read_bytes()))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.98, top=.85, bottom=.1, hspace=.37, wspace=.10)
    for ax, game in zip(axes.flat, ledger["losses"], strict=True):
        wallet, bank, tree = read_curves(game)
        ax.axhline(0, color="#949ca4", lw=.9)
        ax.plot(bank, color="#4489bb", lw=1.25, ls=(0, (4, 2)), label="归库差（累计）")
        ax.plot(tree, color="#bb640e", lw=1.5, label="树净收入差（累计）")
        ax.plot(wallet, color="#182f44", lw=1.45, label="官方钱包差")
        ax.scatter(299, wallet[-1], color="#182f44", s=14, zorder=4)
        ax.set_title(f"{game['opponent']}     {wallet[-1]:+,.2f}", loc="left", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bac3cc")
        ax.grid(axis="y", alpha=.16)
        ax.set(xlim=(0, 302), xticks=[0, 50, 100, 150, 200, 250, 299])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [2, 0, 1]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(.075, .93), frameon=False, ncol=3)
    fig.text(.085, .955, "ds-pro-v1 六场定位赛败局：分差从哪一步开始拉开", fontsize=19)
    fig.text(.5, .046, "已完成的联合行动步数", ha="center")
    fig.text(.019, .48, "分数（我方减对手）", rotation=90, va="center")
    fig.text(.085, .01, "各图使用相同比例尺。官方钱包还包括招工等收支；在途货到归库时才计入蓝线。", fontsize=9)
    fig.savefig(HERE / "loss_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
