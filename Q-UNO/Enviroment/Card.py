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
0 is special card which has arbitrary number
"""


class Card:
    def __init__(self, color, value):
        self.color, self.value = color, value

    def __eq__(self, other):
        if self.value in range(13, 15) and self.value == other.value:
            return True
        return self.color == other.color and self.value == other.value

    def valid_card(self, card, accumulate_penalty):
        if accumulate_penalty > 0:
            if card.value == 14:
                return True
            elif card.value == 12:
                return card.color == self.color
            else:
                return False
        else:
            if card.value in [13, 14]:
                return True
            if card.value == self.value:
                return True
            return card.color == self.color

    def get_color_string(self):
        if self.color == 1:
            return "red"
        if self.color == 2:
            return "orange"
        if self.color == 3:
            return "green"
        if self.color == 4:
            return "blue"
        if self.color == 0:
            return "black"

    def get_value_string(self):
        if self.value in range(0, 10):
            return str(self.value)
        if self.value == 10:
            return "STP"
        if self.value == 11:
            return "REV"
        if self.value == 12:
            return "+2"
        if self.value == 13:
            return "COL"
        if self.value == 14:
            return "+4"

    def get_string(self):
        return self.get_color_string() + " " + self.get_value_string()
