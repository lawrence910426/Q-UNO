"""
-- nn INPUT --

one hot -> [ 13 (cards) * 4 (colors) + 2(black card) ] * 2 (is multi)
numbers -> 5 (color statistics of hand card)
numbers -> 15 (value statistics of hand card)

one hot -> prev_steps * [ 4 (colors) * 2(is chosen) + 1(None) ]

one hot -> opponent cards < opponent_low_threshold
one hot -> opponent_low_threshold <= opponent cards < opponent_high_threshold
one hot -> opponent_high_threshold <= opponent cards

number -> accumulate penalty

one hot -> prev_steps * 1(is functional card)
-- nn OUTPUT

15 (cards) * 4 (colors)

"""
import numpy as np
from Enviroment.Card import Card


class NNTranslate:
    prev_steps = 15
    features = (13 * 4 + 2) * 2 + 5 + 15 + prev_steps * 9 + 3 + 1 + prev_steps
    actions = 15 * 4 + 1
    oppo_low_thre, oppo_high_thre = 3, 7

    @staticmethod
    def state_to_nn(state):
        """
        [self.used, len(self.unused), self.hand_card[player],
        len(self.hand_card[1 - player]), self.accumulate_penalty]
        """
        ans = np.zeros((NNTranslate.features, ))
        cursor = 0
        for card in state[2]:
            if card.color == 0:
                pos = 13 * 4 + 0 if card.value is 13 else 1
            else:
                pos = card.value * (card.color - 1)
            pos += (13 * 4 + 2) if ans[pos] is not 0 else 0
            ans[cursor + pos] = 1

        hand_card_color, hand_card_value = [0 for _ in range(5)], [0 for _ in range(15)]
        for card in state[2]:
            hand_card_color[card.color] += 1
            hand_card_value[card.value] += 1

        cursor = (13 * 4 + 2) * 2
        for i in range(5):
            ans[cursor] = hand_card_color[i]
            cursor += 1
        for i in range(15):
            ans[cursor] = hand_card_value[i]
            cursor += 1

        for i in range(min(NNTranslate.prev_steps, len(state[0]))):
            card = state[0][-(i + 1)]
            if card is None:
                ans[cursor + 8] = 1
            else:
                if card.value in [13, 14]:
                    ans[cursor + 4 + card.color] = 1
                else:
                    ans[cursor + card.color] = 1
            cursor += 9

        cursor = (13 * 4 + 2) * 2 + 5 + 15 + NNTranslate.prev_steps * 9
        cursor += 1 if NNTranslate.oppo_low_thre <= state[3] else 0
        cursor += 1 if NNTranslate.oppo_high_thre <= state[3] else 0
        ans[cursor] = 1

        cursor = (13 * 4 + 2) * 2 + 5 + 15 + NNTranslate.prev_steps * 9 + 3
        ans[cursor] = state[4]

        cursor = (13 * 4 + 2) * 2 + 5 + 15 + NNTranslate.prev_steps * 9 + 3 + 1
        for i in range(min(NNTranslate.prev_steps, len(state[0]))):
            card = state[0][-(i + 1)]
            ans[cursor] = 1 if card is not None and card.value >= 10 else 0
            cursor += 1
        return ans

    @staticmethod
    def nn_to_state(state):
        value = np.argmax(state)
        return Card(value / 15, value % 15) if value is 15 * 4 else None
