"""Fast vectorized joint scores on the learned, frozen actor layer at a public root."""
def _jrc_scores(features,weights):
    z=np.clip((features-weights['mu'])/weights['sd'],-8,8)
    a=np.tanh(z@weights['l1_weight'].T+weights['l1_bias'])
    b=np.tanh(a@weights['l2_weight'].T+weights['l2_bias'])
    e=np.concatenate([a,b,b@weights['aux_weight'].T+weights['aux_bias']],axis=1)
    vals=(e/weights['term_scale'])@weights['term']
    return vals[:,0]-vals[:,1]


_JRC_LAST={}
_JRC_VIEW=None
_JRC_BASE_AGENT=agent


def _jrc_attach(x,aux,enc,parent):
    obs,cfg=_JRC_VIEW;ctx=_jrc_context(obs,cfg,_FEAT,x,aux,enc[1][1])
    menu=_jrc_options(ctx,obs,cfg,_FEAT,x,aux,parent)
    scores=_jrc_scores(np.stack([m[1] for m in menu]),_JRC_WEIGHTS)
    if not np.isfinite(scores).all():raise ValueError('nonfinite joint scores')
    ix=int(np.argmax(scores));chosen=ix if scores[ix]-scores[0]>=100 else 0
    _JRC_LAST.clear();_JRC_LAST.update(role=menu[chosen][2],menu=len(menu),upgrade=float(scores[chosen]-scores[0]),
                                      parent=_FEAT.to_actions(parent),chosen=_FEAT.to_actions(menu[chosen][0]))
    return menu[chosen][0]


def agent(observation,configuration=None):
    global _JRC_VIEW
    _JRC_VIEW=(observation,{**_JRC_CFG,**(configuration or {})})
    return _JRC_BASE_AGENT(observation,configuration)
