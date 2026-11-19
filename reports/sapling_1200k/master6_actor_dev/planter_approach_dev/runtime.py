"""A single public approach option. A committed route can be abandoned at every decision."""
_MOVES_PHYS = ('NONE','UP','RIGHT','DOWN','LEFT')


class Approach:
    def __init__(self):
        self.market = PlanterRouteState()
        self.ticket = None
        self.used = False
        self.last = {}
        self.seat, self.step = None, -1

    def observe(self, obs, cfg):
        seat, t = obs['player'], obs['step']
        if self.seat != seat or t != self.step + 1 or t == 0:
            self.__init__()
        self.seat, self.step, self.obs = seat, t, obs
        self.own, self.enemy = obs['players'][seat], obs['players'][1-seat]
        self.market.observe(obs, seat, obs['carbon'], cfg, t)
        self.last = {'t':t,'reason':'no_request'}
        self.values = {}

    def safe(self, cell, uid, joint):
        if any(_ROUTE_DIST[cell,w[0]] < 2 for w in self.enemy[2].values()):
            return False
        if any(cell == _ROUTE_DEST[w[0],_MOVES_PHYS.index(joint.get(k,'NONE'))]
               for k,w in self.own[2].items() if k != uid):
            return False
        return not any(cell == base and k in joint for k,base in self.own[1].items())

    def quote(self, cell, uid, joint, distance):
        s=self.market
        if cell not in self.values:
            val=s.features(cell)
            self.values[cell] = (val,route_value(_APP_MODEL,val[0])[2]) if val else None
        pair=self.values[cell]
        if not pair:
            return None
        (features, price),net=pair
        tree=s.opp_trees.get(cell)
        if features[1]*50 < distance+10 or (not tree and features[4] < .5):
            return None
        if any(_ROUTE_DIST[cell,w[0]]<2 for w in self.enemy[2].values()):
            return None
        costs=s.cfg
        counts={kind:sum(w[2]==kind for p in self.obs['players'] for w in p[2].values())
                for kind in ('COLLECTOR','PLANTER')}
        fees={kind: costs.get('rec'+kind.title()+'Cost',30)+count*costs.get('rec'+kind.title()+'IncreaseCost',0)
              for kind,count in counts.items()}
        reserved=sum(fees[joint[k][3:]] for k in self.own[1] if joint.get(k) in ('RECCOLLECTOR','RECPLANTER'))
        for k,(at,_,kind) in self.own[2].items():
            if k==uid or kind!='PLANTER' or k in joint or at in s.own_trees or at in s.own_bases or at in s.opp_bases:
                continue
            other=s.opp_trees.get(at)
            reserved+=costs.get('seizeCost',20) if other and other[1]<costs.get('treeLifespan',50) else s.new_price
        if s.cash-price-reserved<max(30,min(fees.values())):
            return None
        return {'cell':cell,'tree':tree[0] if tree else None,'price':price,'net':float(net),
                'recent_fraction':float(features[4]),'reserved':float(reserved),'score':float(net-3*distance)}

    def next_moves(self, uid, cell, joint):
        origin=self.own[2][uid][0]
        d=int(_ROUTE_DIST[origin,cell])
        if d==0:
            return ['NONE'] if self.safe(origin,uid,joint) else []
        return [move for i,move in enumerate(_MOVES_PHYS[1:],1)
                if _ROUTE_DIST[_ROUTE_DEST[origin,i],cell]==d-1 and self.safe(int(_ROUTE_DEST[origin,i]),uid,joint)]

    def proposal(self,joint):
        if self.used or not 50<=self.step<=180:
            return None
        options=[]
        for uid,(at,_,kind) in self.own[2].items():
            parent=joint.get(uid,'NONE')
            if kind!='PLANTER' or parent=='NONE':
                continue
            candidates=[];parent_values=[0.]
            for cell0 in np.flatnonzero((_ROUTE_DIST[at]>=2)&(_ROUTE_DIST[at]<=4)):
                cell=int(cell0)
                distance=int(_ROUTE_DIST[at,cell])
                moves=self.next_moves(uid,cell,joint)
                if not moves:
                    continue
                q=self.quote(cell,uid,joint,distance)
                if not q:
                    continue
                if parent in moves:
                    parent_values.append(q['score'])
                elif q['net']>=25:
                    candidates.append({**q,'uid':uid,'origin':at,'distance':distance,'moves':moves})
            # Count an already valuable adjacent parent landing in the competition.
            nextcell=int(_ROUTE_DEST[at,_MOVES_PHYS.index(parent)])
            immediate=self.quote(nextcell,uid,joint,1)
            if immediate and self.safe(nextcell,uid,joint):
                parent_values.append(immediate['score'])
            for item in candidates:
                edge=item['score']-max(parent_values)
                if edge>=8:
                    options.append({**item,'edge':edge,'parent_best':max(parent_values)})
        return min(options,key=lambda z:(-z['edge'],-z['score'],z['uid'],z['cell'])) if options else None

    def decide(self,x,aux,enc,labels,branch):
        joint=_FEAT.to_actions(labels)
        pending=self.ticket
        if pending:
            uid,cell=pending['uid'],pending['cell']
            unit=self.own[2].get(uid)
            if not unit or unit[2]!='PLANTER' or self.step>pending['start']+4:
                self.ticket=None;self.last['reason']='expired_or_lost';return labels
            d=int(_ROUTE_DIST[unit[0],cell])
            q=self.quote(cell,uid,joint,d)
            if not q or q['tree']!=pending['tree'] or q['net']<8 or d>pending['start']+4-self.step:
                self.ticket=None;self.last['reason']='reprice_or_threat_cancel';return labels
            moves=self.next_moves(uid,cell,joint); mode='pay' if d==0 else 'continue'
        else:
            q=self.proposal(joint)
            if not q:
                return labels
            uid=q['uid'];cell=q['cell'];d=q['distance'];moves=q['moves'];mode='launch'
        parent=joint.get(uid,'NONE')
        self.last.update(mode=mode,worker=uid,target=cell,old=parent,distance=d,quote=q,reason='no_legal_safe_prefix')
        slot=_FEAT.own_uid.index(uid)
        chosen=None
        for move in ([parent] if parent in moves else [])+[m for m in moves if m!=parent]:
            if move==parent:
                chosen=(labels,move);break
            index=int(_FEAT.labels({uid:move})[slot])
            alternative=_approach_decode(x,aux,enc,branch,slot,index)
            if (alternative is not None and np.flatnonzero(alternative!=labels).tolist()==[slot] and
                    not (_return_conflicts(x,alternative)-_return_conflicts(x,labels)) and
                    not any(slot in pair for pair in _return_conflicts(x,alternative))):
                chosen=(alternative,move);break
        if chosen is None:
            self.ticket=None;return labels
        result,move=chosen
        if mode=='launch':
            self.used=True
            self.ticket={'uid':uid,'cell':cell,'tree':q['tree'],'start':self.step}
        elif mode=='pay':
            self.ticket=None
        self.last.update(reason=mode,issued=move,changed=move!=parent)
        return result


_APP=Approach()
_approach_underlying=agent


def agent(observation,configuration=None):
    cfg={k:(configuration or {})[k] for k in ('treeLifespan','episodeSteps','seizeCost','plantCost',
         'plantCostInflationRatio','plantCostInflationBase','playerTreeProtectiveNumber','recCollectorCost',
         'recPlanterCost','recCollectorIncreaseCost','recPlanterIncreaseCost') if k in (configuration or {})}
    _APP.observe(observation,cfg)
    return _approach_underlying(observation,configuration)
