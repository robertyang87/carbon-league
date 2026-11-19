"""Pre-action, physical-board features for five simultaneously available victim destinations."""
import numpy as np

_PRESSURE_CELLS=np.arange(225)
_PRESSURE_ROW,_PRESSURE_COL=np.divmod(_PRESSURE_CELLS,15)
_PRESSURE_DEST=np.stack([_PRESSURE_CELLS,((_PRESSURE_ROW-1)%15)*15+_PRESSURE_COL,
                         _PRESSURE_ROW*15+(_PRESSURE_COL+1)%15,((_PRESSURE_ROW+1)%15)*15+_PRESSURE_COL,
                         _PRESSURE_ROW*15+(_PRESSURE_COL-1)%15],axis=1)
_dx=abs(_PRESSURE_COL[:,None]-_PRESSURE_COL);_dy=abs(_PRESSURE_ROW[:,None]-_PRESSURE_ROW)
_PRESSURE_DIST=np.minimum(_dx,15-_dx)+np.minimum(_dy,15-_dy)
_PRESSURE_FEATURES=("stay","home_progress","can_bank","escape_progress","in_hunter_reach","continue_heading",
                    "return_streak_progress","near_base_progress","heavy_home_progress","loaded_stay",
                    "late_home_progress","heavy_escape","short_route_escape")


def pressure_features(hunter,victim,load,last,streak,step,home_dist):
    """Returns (..., five actions, thirteen features), using only current and earlier public records."""
    hunter,victim,load,last,streak,step=map(np.asarray,(hunter,victim,load,last,streak,step))
    candidates=_PRESSURE_DEST[victim]
    homes=np.asarray(home_dist)
    if homes.ndim==1:
        hd=homes[victim]; next_hd=homes[candidates]
    else:
        hd=homes[np.arange(len(victim)),victim]
        next_hd=np.take_along_axis(homes,candidates,axis=1)
    progress=hd[...,None]-next_hd
    enemy_dist=_PRESSURE_DIST[hunter[...,None],candidates]
    escape=enemy_dist-_PRESSURE_DIST[hunter,victim][...,None]
    stay=np.broadcast_to(np.arange(5)==0,progress.shape)
    heavy=np.minimum(load/300.,2.)[...,None]
    near=(hd<=3)[...,None]
    return np.stack([stay,progress,next_hd==0,escape,enemy_dist<=1,last[...,None]==np.arange(5),
                     np.minimum(streak,3)[...,None]/3.*progress,near*progress,heavy*progress,
                     heavy*stay,(step[...,None]>=240)*progress,heavy*escape,near*escape],axis=-1)
