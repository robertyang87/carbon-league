"""One-turn right to buy the landing that the frozen policy already chose."""
class _PlanterOption:
    def __init__(self):
        self.market=PlanterRouteState()
        self.pending=None
        self.last=None
        self.seat=None
        self.t=-1

    def observe(self,obs,cfg):
        seat=obs['player']
        t=obs['step']
        if seat!=self.seat or t!=self.t+1 or t==0:
            self.market=PlanterRouteState()
            self.pending=None
        self.seat,self.t=seat,t
        self.obs=obs
        self.market.observe(obs,seat,obs['carbon'],cfg,t)
        self.last={'step':t,'previous':self.pending,'reason':'no_previous','committed':False,'next':None}

    def quote(self,uid,destination,commands):
        """All cash adjustments here refer to currently visible state, with no future deposits."""
        s=self.market
        own=self.obs['players'][self.seat]
        if uid not in own[2] or own[2][uid][2]!='PLANTER':
            return None,'missing_worker'
        features=s.features(destination)
        if features is None:
            return None,'unavailable_or_expired'
        z,price=features
        tree=s.opp_trees.get(destination)
        if z[1]*50<_OPTION_CFG['minimum_remaining_tree_turns']:
            return None,'too_little_future_income'
        if any(_ROUTE_DIST[destination,int(w[0])]<=1 for w in self.obs['players'][1-self.seat][2].values()):
            return None,'opponent_can_contest_cell'
        counts={'COLLECTOR':0,'PLANTER':0}
        for player in self.obs['players']:
            for _,_,kind in player[2].values():
                counts[kind]+=1
        cfg={'recCollectorCost':30,'recCollectorIncreaseCost':0,'recPlanterCost':30,
             'recPlanterIncreaseCost':0,'treeLifespan':50,'seizeCost':20,**s.cfg}
        hires={'RECCOLLECTOR':cfg['recCollectorCost']+cfg['recCollectorIncreaseCost']*counts['COLLECTOR'],
               'RECPLANTER':cfg['recPlanterCost']+cfg['recPlanterIncreaseCost']*counts['PLANTER']}
        reserved=sum(hires.get(commands.get(b),0) for b in own[1])
        for w,(pos,_,kind) in own[2].items():
            if w==uid or kind!='PLANTER' or commands.get(w,'NONE') not in ('NONE',None):
                continue
            if pos in s.own_bases or pos in s.opp_bases or pos in s.own_trees:
                continue
            # A currently expiring enemy tree can disappear before the pay phase.
            enemy=s.opp_trees.get(pos)
            reserved += cfg['seizeCost'] if enemy and enemy[1]<cfg['treeLifespan'] else s.new_price
        if own[0]-reserved-price<min(hires.values()):
            return None,'unfunded_after_requests'
        _,_,net=route_value(_OPTION_MODEL,z)
        if net<_OPTION_CFG['minimum_first_tenure_net']:
            return None,'net_too_low'
        return {'uid':uid,'cell':int(destination),'tree':tree[0] if tree else None,'price':float(price),
                'net':net,'cash':float(own[0]),'other_reserved':float(reserved),'reserve':float(min(hires.values()))},None

    def confirm(self,ticket,commands):
        unit=self.obs['players'][self.seat][2].get(ticket['uid'])
        if not unit or unit[0]!=ticket['cell']:
            return None,'landing_or_worker_changed'
        q,reason=self.quote(ticket['uid'],ticket['cell'],commands)
        if q is None:
            return None,reason
        if q['tree']!=ticket['tree']:
            return None,'tree_identity_changed'
        if q['price']>ticket['price']*(1+_OPTION_CFG['max_price_increase_as_fraction']):
            return None,'price_jump'
        return q,None

    def propose(self,commands):
        low,high=_OPTION_CFG['proposal_window']
        if not low<=self.t<=high:
            return None
        options=[]
        action_ix={'UP':1,'RIGHT':2,'DOWN':3,'LEFT':4}
        for uid,(pos,_,kind) in self.obs['players'][self.seat][2].items():
            cmd=commands.get(uid)
            if kind!='PLANTER' or cmd not in action_ix:
                continue
            dest=int(_ROUTE_DEST[pos,action_ix[cmd]])
            q,reason=self.quote(uid,dest,commands)
            if q:
                options.append(q)
        if not options:
            return None
        best=min(options,key=lambda q:(-q['net'],q['uid'],q['cell']))
        return {k:best[k] for k in ('uid','cell','tree','price')}

    def decide(self,x,aux,enc,parent,using_residual):
        commands=_FEAT.to_actions(parent)
        ticket=self.pending
        commit=parent
        if ticket:
            q,reason=self.confirm(ticket,commands)
            self.last.update(reason=reason or 'qualified',quote=q)
            uid=ticket['uid']
            if q and commands.get(uid,'NONE') in ('NONE',None):
                self.last['reason']='parent_already_stays'
            elif q:
                slot=next((i for i in range(10) if x['tok_mask'][i] and _FEAT.own_uid[i]==uid),None)
                alternative=None if slot is None else _option_force_decode(x,aux,enc,using_residual,slot)
                if alternative is None or np.flatnonzero(alternative!=parent).tolist()!=[slot]:
                    self.last['reason']='joint_prefix_veto'
                else:
                    conflicts=_return_conflicts(x,alternative)
                    if conflicts-_return_conflicts(x,parent) or any(slot in pair for pair in conflicts):
                        self.last['reason']='occupancy_veto'
                    else:
                        commit=alternative
                        self.last.update(committed=True,reason='exercise',parent_move=commands[uid])
        self.pending=self.propose(_FEAT.to_actions(commit))
        self.last['next']=self.pending
        return commit


_OPTION=_PlanterOption()
_option_original_agent=agent

def agent(observation,configuration=None):
    _OPTION.observe(observation,configuration or {})
    return _option_original_agent(observation,configuration)
