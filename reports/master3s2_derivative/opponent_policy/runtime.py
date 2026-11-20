# BEGIN ORIGINAL OPPONENT READOUT
# Feature maps are the original predicted opponent's root input computed from our seat's belief.
_OPP_PLANES = np.array([0,2,3,4,5,6,7,8,9,10,11,12,13],dtype=int)
_OPP_STEPS = np.array([[move(p,k) for k in range(5)] for p in range(225)],dtype=int)

def _opponent_features(xo, z, positions):
    pos=np.asarray(positions,dtype=int)
    x=xo[0].reshape(54,225)
    local=x[:,pos].T
    adjacent=x[_OPP_PLANES[:,None,None],_OPP_STEPS[pos][None,:,:]]
    adjacent=adjacent.transpose(1,2,0).reshape(len(pos),-1)
    l=z[0,pos]
    l=np.clip(l-l.mean(1,keepdims=True),-12.,12.)/4.
    return np.concatenate((local,adjacent,np.tanh(l)),axis=1).astype(np.float32)

def _opponent_root_readout(xo,lwo,lbo,players,op):
    for z,positions,prefix in ((lwo,[unit[0] for unit in players[op][2].values()],"w"),
                               (lbo,list(players[op][1].values()),"b")):
        if len(positions):
            features=_opponent_features(xo,z,positions)
            z[0,positions]+=np.tanh(features/_OPP_H[prefix+"scale"])@_OPP_H[prefix+"w"]+_OPP_H[prefix+"b"]
    return lwo,lbo

# PARAMETER_TABLE
# END ORIGINAL OPPONENT READOUT
