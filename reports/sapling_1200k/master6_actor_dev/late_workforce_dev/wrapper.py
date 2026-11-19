_lw_parent=agent
_lw_observer=LateRecruit()
_lw_public=None


def _lw_select(x,aux,enc,labels,branch):
    if _lw_public is None:return labels
    obs,cfg=_lw_public
    commands=_FEAT.to_actions(labels)
    proposed,event=_lw_observer.propose(obs,cfg,commands)
    if proposed==commands:return labels
    focal=_FEAT.own_uid.index(event['base'])
    if focal<10 or labels[focal]!=2:raise ValueError('different parent order or role')
    result=_lw_redecode(x,aux,enc,branch,focal)
    if result is None or np.flatnonzero(result!=labels).tolist()!=[focal] or _FEAT.to_actions(result)!=proposed:
        event['reason']='prefix_redecode_rejected';return labels
    _lw_observer.used=True
    event.update(changed=True,reason='one_hire_replaced')
    return result


def agent(observation,configuration=None):
    global _lw_public
    defaults={'size':15,'episodeSteps':300,'recCollectorCost':30,'recPlanterCost':30,'recCollectorIncreaseCost':0,'recPlanterIncreaseCost':0,'workerLimit':10,'treeLifespan':50,'plantCost':20,'plantCostInflationRatio':5,'plantCostInflationBase':1.235,'playerTreeProtectiveNumber':0,'cellAbsorptionRate':.0375,'initialCollectRate':.30,'collectDecreaseRate':0.,'moveCost':0.,'smeltCost':0.}
    cfg={k:(configuration or {}).get(k,v) for k,v in defaults.items()}
    _lw_public=(observation,cfg)
    return _lw_parent(observation,configuration)
