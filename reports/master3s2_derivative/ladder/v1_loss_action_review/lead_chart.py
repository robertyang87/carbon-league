"""Display each finished losing game at the same time and score scales."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.font_manager import FontProperties
import numpy as np

HERE=Path(__file__).resolve().parent
LADDER=HERE.parent

def read(path):
    with gzip.open(path,"rt") as handle:
        return json.load(handle)

def main():
    games=read(LADDER/"v1_placement_review/ledger.json.gz")["losses"]
    games+=[read(LADDER/"placement_refresh/4c3bc230e7cc588cf3dc70dbf1dfa8ae-1.ledger.json.gz")]
    official=json.loads((LADDER/"placement_refresh/summary.json").read_text())["losses"]
    if [r["match"] for r in official]!=[r["match"] for r in games]:
        raise ValueError("Chart and official loss cohorts differ")
    cash=np.array([[step["margin"] for step in g["steps"]] for g in games])
    if cash.shape!=(7,300) or not np.allclose(cash[:,-1],[r["margin"] for r in official]):
        raise ValueError("Official terminal scores do not match")
    font=FontProperties(family="Noto Sans CJK JP")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":11})
    palette=LinearSegmentedColormap.from_list("lead",["#81251d","#e5a080","#f4f2ea","#8dc7bc","#155752"])
    extent=max(100,100*np.ceil(np.abs(cash).max()/100))
    norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
    fig,ax=plt.subplots(figsize=(12.4,5.0),layout="constrained")
    chart=ax.imshow(cash,aspect="auto",cmap=palette,norm=norm,interpolation="nearest")
    ax.set_yticks(range(len(games)),[g["opponent"] for g in games])
    ax.set_xticks([0,50,100,150,200,250,299]);ax.set_xlabel("已完成的联合行动步数")
    ax.tick_params(length=0,pad=7)
    for i in range(len(games)-1):
        ax.axhline(i+.5,color="white",lw=2)
    ax.set_title("ds-pro-v1 七场定位赛败局的逐拍现金差",loc="left",pad=14,weight="bold",fontsize=18)
    for i,series in enumerate(cash):
        ax.text(1.018,i,f"{series[-1]:+,.2f}   ·   {sum(series[1:]>0)}/299",transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=10)
    ax.text(1.018,1.04,"终局分差  ·  领先拍数",transform=ax.transAxes,fontsize=10)
    ax.spines[:].set_visible(False)
    bar=fig.colorbar(chart,ax=ax,orientation="horizontal",fraction=.10,pad=.13,shrink=.65,aspect=45)
    bar.set_label("步末现金（我方减对手），负值表示落后")
    fig.savefig(HERE/"lead_all_losses.png",dpi=150,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    main()
