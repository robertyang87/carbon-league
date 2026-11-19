class GroundSelection:
    """One new tree proposal during the middle third of the game. Record both seats separately."""
    def __init__(self):
        self.seats={}
    def act(self,obs,cfg,cmd,w):
        s=obs['player'];t=obs['step']
        if t==0 or s not in self.seats:
            self.seats[s]=dict(t=-1,mem={},used=False,selected=None)
        st=self.seats[s]
        if st['t']+1!=t:raise ValueError('nonconsecutive public history')
        st['t']=t;update_memory(obs,st['mem'])
        if st['used'] or not 100<=t<200:return cmd
        candidates=[]
        for uid,v in obs['players'][s][2].items():
            if v[2]!='PLANTER':continue
            c=consider(obs,st['mem'],cmd,uid,cfg,w)
            if c and c['accept']:candidates.append(c)
        if not candidates:return cmd
        chosen=min(candidates,key=lambda q:(-q['score'],q['worker']))
        out=dict(cmd);out.pop(chosen['worker'])
        st.update(used=True,selected=dict(t=t,**chosen))
        return out
