"""
-- VALUES --
0-9 are the original numbers
10 is stop
11 is reverse
12 is two additional cards

13 is change color (specified color send by the color attribute)
14 is four additional cards (specified color send by the color attribute)
"""

"""
-- COLOR --
1 is red
2 is orange
3 is green
4 is blue
0 is special card which has not specify any color
"""


class Card:
    def __init__(self, color, value):
        self.color, self.value = color, value

    def __eq__(self, other):
        return self.color == other.color and self.value == other.value

    def valid_card(self, card):
        if card.value == 13 or card.value == 14:
            return True
        return card.color == self.color
