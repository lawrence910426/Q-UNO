from Brains.Brain import Brain
import numpy as np


class DummyBrain(Brain):
    def __init__(self):
        return

    def get(self, s):
        available = Brain.get_available(s)
        if len(available) == 0:
            return None
        chosen_card = np.random.choice(available)
        if chosen_card.color is 0:
            chosen_card.color = np.random.choice(range(1, 5))
        return chosen_card

    def add_observation(self, s, a, r, s_):
        return

    def learn(self):
        return
