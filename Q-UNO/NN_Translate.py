"""
-- nn INPUT --

one hot -> [ 13 (cards) * 4 (colors) + 2(black card) ] * 2 (is multi)
numbers -> 5 (color statistics of hand card)
numbers -> 4 (colors) * 2 (is functional) + 2 (black cards)

one hot -> prev_steps * [ 4 (colors) * 2(is chosen) + 1(None) ]

number -> opponent cards
number -> accumulate penalty

one hot -> prev_steps * 1(is functional card)

numbers -> 1 (functional cards sum)


-- nn OUTPUT

15 (cards) * 4 (colors)

"""
import numpy as np
from Enviroment.Card import Card


class NNTranslate:
    prev_steps = 8
    features = (13 * 4 + 2) * 2 + 5 + (4 * 2 + 2) + prev_steps * 9 + 1 + 1 + prev_steps + 1
    actions = 15 * 4 + 1

    @staticmethod
    def state_to_nn(state):
        """
        [self.used, len(self.unused), self.hand_card[player],
        len(self.hand_card[1 - player]), self.accumulate_penalty]
        """
        ans = np.zeros(NNTranslate.features)

        # one hot -> [ 13 (cards) * 4 (colors) + 2(black card) ] * 2 (is multi)
        cursor = 0
        for card in state[2]:
            if card.color == 0:
                pos = 13 * 4 + 0 if card.value is 13 else 1
            else:
                pos = card.value + (card.color - 1) * 13
            pos += (13 * 4 + 2) if ans[pos] is not 0 else 0
            ans[cursor + pos] = 1

        # numbers -> 5 (color statistics of hand card)
        cursor = (13 * 4 + 2) * 2
        hand_card_color = [0 for _ in range(5)]
        for card in state[2]:
            hand_card_color[card.color] += 1
        for i in range(5):
            ans[cursor] = hand_card_color[i]
            cursor += 1

        # numbers -> 4 (colors) * 2 (is functional) + 2 (black cards)
        cursor = (13 * 4 + 2) * 2 + 5
        for card in state[2]:
            if card.color is 0:
                ans[cursor + 4 * 2 + 1 if card.value == 14 else 0] += 1
            else:
                ans[cursor + (card.color - 1) * 2 + 1 if card.value >= 10 else 0] += 1

        # one hot -> prev_steps * [ 4 (colors) * 2(is chosen) + 1(None) ]
        cursor = (13 * 4 + 2) * 2 + 5 + (4 * 2 + 2)
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

        # number -> opponent cards
        cursor = (13 * 4 + 2) * 2 + 5 + (4 * 2 + 2) + NNTranslate.prev_steps * 9
        ans[cursor] = state[3]

        # number -> accumulate penalty
        cursor = (13 * 4 + 2) * 2 + 5 + (4 * 2 + 2) + NNTranslate.prev_steps * 9 + 1
        ans[cursor] = state[4]

        # one hot -> prev_steps * 1(is functional card)
        cursor = (13 * 4 + 2) * 2 + 5 + (4 * 2 + 2) + NNTranslate.prev_steps * 9 + 1 + 1
        for i in range(min(NNTranslate.prev_steps, len(state[0]))):
            card = state[0][-(i + 1)]
            ans[cursor] = 1 if card is not None and card.value >= 10 else 0
            cursor += 1

        # numbers -> 1 (functional cards sum)
        cursor = (13 * 4 + 2) * 2 + 5 + (4 * 2 + 2) + NNTranslate.prev_steps * 9 + 1 + 1
        for card in state[2]:
            if card.value >= 10:
                ans[cursor] += 1
        return ans

    @staticmethod
    def nn_to_state(value):
        return None if value == 15 * 4 else Card(int(value / 15) + 1, value % 15)

    @staticmethod
    def card_to_nn(card):
        return 15 * 4 if card is None else (card.color - 1) * 15 + card.value

    @staticmethod
    def get_available_mask(available_cards):
        ans = np.zeros(NNTranslate.actions)
        ans[15 * 4] = 1
        for card in available_cards:
            if card.color == 0:
                for i in range(4):
                    ans[i * 15 + card.value] = 1
            else:
                ans[(card.color - 1) * 15 + card.value] = 1
        return ans
