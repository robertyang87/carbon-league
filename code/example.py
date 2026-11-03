import random


def agent(observation, configuration):
    """A deliberately simple public starter; replace this with your strategy."""
    me = observation['players'][observation['player']]
    cash, centers, workers = me[:3]
    actions = {}
    for identifier in workers:
        actions[identifier] = random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    for identifier in centers:
        if cash >= 100 and len(workers) < 6:
            actions[identifier] = 'RECCOLLECTOR'
    return actions
