from Brains.Brain import Brain
import numpy as np


class OffensiveBrain(Brain):
    def __init__(self):
        return

    def get(self, s):
        available = Brain.get_available(s)
        chosen_card = None
        for card in available:
            if chosen_card is None:
                chosen_card = card
            else:
                if chosen_card.value < card.value:
                    chosen_card = card
        if chosen_card is not None and chosen_card.color is 0:
            chosen_card.color = np.random.choice(range(1, 5))
        return chosen_card

    def add_observation(self, s, a, r, s_):
        return

    def learn(self):
        return


class DefensiveBrain(Brain):
    def get(self, s):
        available = Brain.get_available(s)
        chosen_card = None
        for card in available:
            if chosen_card is None:
                chosen_card = card
            else:
                if chosen_card.value > card.value:
                    chosen_card = card
        if chosen_card is not None and chosen_card.color is 0:
            chosen_card.color = np.random.choice(range(1, 5))
        return chosen_card

    def add_observation(self, s, a, r, s_):
        return

    def learn(self):
        return
