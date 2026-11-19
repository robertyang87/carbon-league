"""Bounded residual scores for full physical orders on the existing legal Sapling catalog.

The frozen 851c student is _NETS[1], whose native GRU hidden state has 96
components. Previous assignments in the same joint order affect the subsequent
legality mask, occupancy, budget and GRU stream.
"""
import math
import numpy as np

ACTION_TYPES = 8
STUDENT_DIM = 96
PARAMETERS = ACTION_TYPES * STUDENT_DIM
MAX_SHIFT = 0.125


def command_key(order):
    return tuple(sorted((uid, move) for uid, move in order.items() if move != 'NONE'))


def require_pair(catalog, a, b):
    """Return the indices of exactly the two actually compared distinct physical orders."""
    rows = catalog['orders']
    lookup = {command_key(r['command']):i for i,r in enumerate(rows)}
    if len(lookup) != len(rows) or command_key(a) == command_key(b):
        raise ValueError('Duplicate catalog or paired commands')
    for r in rows:
        if not math.isfinite(r['probability']) or r['probability'] <= 0:
            raise ValueError('Invalid physical catalog mass')
    if not math.isclose(math.fsum(r['probability'] for r in rows), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Unnormalized catalog')
    if command_key(a) not in lookup or command_key(b) not in lookup:
        raise ValueError('Compared physical command absent from frozen catalog')
    return lookup[command_key(a)], lookup[command_key(b)]


def features(model, x, aux, commands):
    """Return [orders, 8 canonical action classes, 96 student hidden features]."""
    if model._FEAT is None or model._NETS is None or len(model._NETS) != 2:
        raise ValueError('Warm native actor required')
    # The player's private embedded module is attached only during a public
    # call; the existing Feat method retains the same signed module namespace.
    feature_namespace = type(model._FEAT).observe.__globals__
    prefix_class, own_slots = feature_namespace['Prefix'], feature_namespace['OWN']
    net = model._NETS[1]
    # Policy.encode writes a convenience buffer. Preserve the caller's preexisting
    # object/reference so merely enumerating proposals has no serving side effect.
    have_trunk = hasattr(net, 'trunk')
    old_trunk = getattr(net, 'trunk', None)
    try:
        tokens, ctx = net.encode(x, model._HYBRID['style'])
    finally:
        if have_trunk: net.trunk=old_trunk
        else: vars(net).pop('trunk',None)
    h0 = net.lin(ctx, 'h0')
    if h0.shape != (STUDENT_DIM,) or net.cfg['D'] != STUDENT_DIM:
        raise ValueError('Wrong student network or hidden-state dimension')
    slots = np.flatnonzero(x['tok_mask'][:own_slots])
    if not len(slots) or not commands:
        raise ValueError('Missing active own units or physical commands')
    result = np.zeros((len(commands), ACTION_TYPES, STUDENT_DIM), dtype=np.float64)
    for index, command in enumerate(commands):
        key = command_key(command)
        labels = model._FEAT.labels(command)
        if command_key(model._FEAT.to_actions(labels)) != key:
            raise ValueError('Unknown worker, base or physical command')
        hidden = h0.copy()
        prefix = prefix_class(x, aux)
        previous = 8
        for slot in slots:
            unit, moves, legal = prefix.feats(slot)
            move = int(labels[slot])
            if not (0 <= move < (5 if slot < 10 else 3)) or not legal[move]:
                raise ValueError('Illegal candidate on its own autoregressive prefix')
            encoded = np.concatenate([tokens[slot], ctx, unit, moves.reshape(-1)])
            hidden = net.gru(net.lin(encoded, 'din') + net.w['aemb.weight'][previous], hidden)
            action_type = move if slot < 10 else 5 + move
            result[index, action_type] += hidden
            prefix.advance(slot, move)
            previous = action_type
        result[index] /= len(slots)
        result[index] /= max(1.,float(np.linalg.norm(result[index])))
    if not np.isfinite(result).all(): raise ValueError('Nonfinite student encoding')
    return result


def zeros():
    return np.zeros((ACTION_TYPES,STUDENT_DIM),dtype=np.float64)


def probabilities(parent_probs, phi, theta):
    """Rescore the complete physical catalog; no physical order loses its support."""
    p=np.asarray(parent_probs,dtype=np.float64)
    x=np.asarray(phi,dtype=np.float64)
    w=np.asarray(theta,dtype=np.float64)
    if p.ndim!=1 or x.shape!=(len(p),ACTION_TYPES,STUDENT_DIM) or w.shape!=(ACTION_TYPES,STUDENT_DIM):
        raise ValueError('Wrong settings for the 8 by 96 student head')
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(w)) and np.all(p>0) and
            math.isclose(math.fsum(p),1.,rel_tol=0,abs_tol=1e-12)):
        raise ValueError('Invalid feature, parameter, or physical probability')
    if np.linalg.norm(w)>MAX_SHIFT+1e-12 or np.any(np.linalg.norm(x.reshape(len(p),-1),axis=1)>1+1e-12):
        raise ValueError('The registered residual norm bound was exceeded')
    delta=np.einsum('nad,ad->n',x,w)
    if not np.any(delta): return p.copy()
    if np.max(np.abs(delta))>MAX_SHIFT+1e-12:
        raise ValueError('A menu logit would move more than 0.125')
    # Reweight positive old probabilities directly instead of introducing
    # underflow in exp(log(p)): bounded scores keep the true support intact.
    adjusted=p*np.exp(delta)
    adjusted/=math.fsum(adjusted)
    if np.any(adjusted<=0) or not math.isclose(math.fsum(adjusted),1.,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Physical support changed during rescoring')
    return adjusted
