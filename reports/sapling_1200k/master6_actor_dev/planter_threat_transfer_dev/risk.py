"""Map-trained one-step planter hazard; called only after A has chosen its original window."""
def bucket(distance):
    return min(int(distance),2)

def category(dc,dp,kind):
    return 6*bucket(dc)+2*bucket(dp)+(kind in (1,'seize'))

def estimate(model,dc,dp,kind):
    return model['risk'][category(dc,dp,kind)]

def pick_parent(model,changed,dc,dp,kind):
    return bool(changed and estimate(model,dc,dp,kind)>.50)
