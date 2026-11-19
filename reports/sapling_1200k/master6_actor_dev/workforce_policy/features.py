"""Base features using current public assets, visible carbon/belief and the already issued own prefix."""
import numpy as np

_WORKFORCE_NAMES = ["bias","elapsed","cash","foe_cash","free_slots","collectors","planters","foe_collectors","foe_planters",
    "first_collector","first_planter","planters_surplus","near_cap","reserve_one_hire","reserve_two_hires","cash_if_hiring",
    "active_own_trees","active_enemy_trees","own_tree_expiring","enemy_tree_expiring","mature_tree_income_units",
    "base_occupied_after_orders","enemy_in_one","enemy_in_two","own_collectors_near","own_carriers_near","enemy_planters_near",
    "near_public_carbon","near_tree_tax","plant_quote","scheduled_planter_spend","collectors_per_planter","early_no_collector",
    "other_base_spent","threat_with_low_cash","cash_gap","enemy_hires_ahead","tree_count_deficit"]


def _workforce_features(x,aux,F,slot,prefix,prior_labels):
    pos=int(x["tok_pos"][slot]);dist=F.DIST[pos]
    m=x["maps"].reshape(-1,225).astype(np.float32)/255.
    ownc=m[8]>.5;ownp=m[9]>.5;enc=m[11]>.5;enp=m[12]>.5
    ot=m[16]>.5;et=m[17]>.5
    nc=int(np.count_nonzero(ownc));np_=int(np.count_nonzero(ownp))
    ec=int(np.count_nonzero(enc));ep=int(np.count_nonzero(enp))
    # Reconstruct role of any earlier same-turn hire (the engine allocates the funds sequentially).
    earlier=[int(y) for j,y in enumerate(prior_labels[10:slot],10) if x["tok_mask"][j]]
    nc+=earlier.count(1);np_+=earlier.count(2)
    cash=float(prefix.cash);sc=x["scal"];left=x["left"];close=dist<=3
    price=x["price"]
    # Worker-prefix stand orders that would pay for a still-standing new tree or capture are observable.
    spend=0.
    for w in np.flatnonzero(x["tok_mask"][:10]):
        if x["tok_kind"][w]!=1 or int(prior_labels[w])!=0:continue
        p=int(x["tok_pos"][w])
        if aux["en_tree"][p]:spend+=20.
        elif not (aux["own_tree"][p] or aux["own_base"][p] or aux["en_base"][p]):spend+=price
    threatened=int(np.count_nonzero((enc|enp)&(dist<=1)))
    loaded_near=sum(bool(x["tok_kind"][w]==0 and x["tok_cargo"][w]>=30 and dist[x["tok_pos"][w]]<=3) for w in np.flatnonzero(x["tok_mask"][:10]))
    v=[1.,(299-left)/170,np.log1p(max(0,cash))/7,float(sc[3]),prefix.slots/10,nc/10,np_/5,ec/10,ep/5,
       nc==0,np_==0,max(np_-1,0)/4,prefix.slots<=2,cash>=60,cash>=90,np.clip((cash-30-spend)/120,-2,3),
       ot.sum()/12,et.sum()/12,np.count_nonzero(ot&(m[18]>=.80))/6,np.count_nonzero(et&(m[18]>=.80))/6,
       np.sum((1-m[18])*ot)/10,prefix.reserved[pos]>0,threatened,np.count_nonzero((enc|enp)&(dist<=2))/4,
       np.count_nonzero(ownc&close)/5,loaded_near/4,np.count_nonzero(enp&close)/3,
       float(np.mean(aux["est"][close])),float(np.mean(aux["toll"][close]))/3,min(price/120,3),min(spend/90,3),
       nc/(1+np_)/4,left>245 and nc<2,sum(y>0 for y in earlier),int(threatened>0 and cash<90),
       float(sc[4]),(ec+ep-nc-np_)/10,(et.sum()-ot.sum())/12]
    return np.asarray(v,np.float32),dict(ncollector=nc,nplanter=np_,cash=cash,spend=spend,risk=threatened,
        occupied=bool(prefix.reserved[pos]),free=int(prefix.slots),earlier=earlier)
