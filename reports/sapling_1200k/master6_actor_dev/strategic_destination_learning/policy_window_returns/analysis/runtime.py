"""Public root features and persisted one-window option, embedded into the actual Sapling release."""
from math import tanh

def _win_input(obs, configuration):
    size=int((configuration or {}).get("size",15))
    end=int((configuration or {}).get("episodeSteps",300))-1
    t=int(obs["step"]); seat=int(obs["player"])
    def distance(a,b):
        dx,dy=abs(a%size-b%size),abs(a//size-b//size)
        return min(dx,size-dx)+min(dy,size-dy)
    def credit(player):
        bases=tuple(player[1].values())
        if not bases:
            return 0.
        total=0.
        for pos,cargo,kind in player[2].values():
            if str(kind).upper()=="COLLECTOR" and cargo>0:
                home=min(distance(pos,b) for b in bases)
                if home<=end-t:
                    total+=cargo/(1.+home/5.)
        return total
    ours,rival=(obs["players"][s] for s in (seat,1-seat))
    return (1.,(t-268)/12.,tanh((credit(ours)-credit(rival))/500.),
            tanh((ours[0]-rival[0])/1500.+(len(ours[3])-len(rival[3]))/6.))


def _win_option(obs,configuration,coef):
    return sum(a*b for a,b in zip(_win_input(obs,configuration),coef))>0.


def _win_install():
    global agent
    original=agent
    def gated(observation,configuration=None):
        global _WIN_BC_NET,_WIN_ELITE_NET,_WIN_AT
        t=int(observation["step"])
        if t==0 or _WIN_BC_NET is None:
            if _WIN_BC_NET is not None:
                _NETS[1]=_WIN_BC_NET
            _WIN_AT=None
            issued=original(observation,configuration)
            if _WIN_BC_NET is None:
                _WIN_BC_NET=_NETS[1]
                w=_dequantize(zlib.decompress(base64.b64decode(_WIN_ELITE_W)),_QMETA)
                w["meta"]=np.array(_WMETA)
                _WIN_ELITE_NET=sys.modules["cnet2_np"].Policy(w)
            return issued
        if _WIN_AT is None and t in (256,280) and _win_option(observation,configuration,_WIN_COEF):
            _WIN_AT=t
        active=(_WIN_AT is not None and _WIN_AT<=t<_WIN_AT+8)
        _NETS[1]=_WIN_ELITE_NET if active else _WIN_BC_NET
        return original(observation,configuration)
    agent=gated
