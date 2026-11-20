"""Publicly computable descriptors of rent that an existing tree has left."""

NAMES = ('age', 'remaining', 'ring_carbon', 'center_carbon', 'unknown_ring', 'conflicting_trees',
         'own_planter_distance', 'enemy_planter_distance', 'own_base_distance', 'enemy_base_distance',
         'near_enemy_collectors', 'near_enemy_cargo', 'own_tree_count', 'phase')
HORIZON = 16


def extract(bot, players, step, carbon, known):
    workers, trees, bases, _ = bot.parse(players)
    sites = {pos for pos, _, _ in trees.values()}
    planters = [[pos for pos, _, role, side in workers.values() if side == owner and role == 'P']
                for owner in (0, 1)]
    collectors = [[(pos, cargo) for pos, cargo, role, side in workers.values() if side == owner and role == 'C']
                  for owner in (0, 1)]
    counts = [len(players[owner][3]) for owner in (0, 1)]
    for tree_id, (pos, age, owner) in trees.items():
        enemy = 1-owner
        adjacent = [pos, *bot.NB8[pos]]
        close = [cargo for place, cargo in collectors[enemy] if bot.DIST[pos][place] <= 2]
        yield tree_id, owner, [
            age, max(0, min(HORIZON, 50-age, 299-step)),
            sum(carbon[place] for place in adjacent), carbon[pos],
            sum(not known[place] for place in adjacent),
            sum(place in sites for place in adjacent if place != pos),
            min((bot.DIST[pos][p] for p in planters[owner]), default=16),
            min((bot.DIST[pos][p] for p in planters[enemy]), default=16),
            min((bot.DIST[pos][p] for _, p in bases[owner]), default=16),
            min((bot.DIST[pos][p] for _, p in bases[enemy]), default=16),
            len(close), sum(close), counts[owner], step,
        ]
