from Brains.Brain import Brain
import numpy as np
import copy


class DummyBrain(Brain):
    def __init__(self):
        return

    def get(self, s):
        used, hand_card, penalty = s[0], s[2], s[4]
        available = []
        for item in hand_card:
            if len(used) is not 0:
                if used[-1].valid_card(item, penalty):
                    available.append(item)
            else:
                available.append(item)
        if len(available) == 0:
            return None

        chosen_card = copy.deepcopy(np.random.choice(available))
        if chosen_card.color is 0:
            chosen_card.color = np.random.choice(range(1, 5))
        return chosen_card

    def add_observation(self, s, a, r, s_):
        return

    def learn(self):
        return
