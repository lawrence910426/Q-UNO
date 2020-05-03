"""
-- CONFIG --
init_hand_cards -> integer
max_plays -> integer
"""

from Enviroment.UNO_Client import UNOClient
from Enviroment.Card import Card
from Enviroment.GUI_Output import GUIOutput
import random
import threading


class UNOServer:
    def __init__(self, config):
        self.config = config
        self.clients = [None, None]
        self.unused = []
        self.used = []
        self.hand_card = {
            0: [],
            1: []
        }
        self.GUI = GUIOutput()

    def init_game(self):
        self.unused.extend([Card(color, value) for color in range(1, 5) for value in range(0, 13)])
        self.unused.extend([Card(0, 13) for _ in range(1, 5)])
        self.unused.extend([Card(color, value) for color in range(1, 5) for value in range(1, 13)])
        self.unused.extend([Card(0, 14) for _ in range(1, 5)])
        random.shuffle(self.unused)
        for i in range(2):
            self.take_cards(i, self.config["init_hand_cards"])
            self.clients[i] = UNOClient(self.hand_card[i])
        return self.clients

    def take_cards(self, player, amount):
        self.hand_card[player].extend(self.unused[0:amount])
        self.unused = self.unused[amount:]

    def get_hidden_state(self, player):
        return [self.used, len(self.unused), self.hand_card[player], len(self.hand_card[1 - player])]

    def get_state(self):
        return [self.used, self.unused, self.hand_card[0], self.hand_card[1]]

    def start_game(self):
        threading.Thread(target=self.play).start()

    def play(self):
        try:
            player, plays, player_temp, result = 0, 0, 0, 0
            s, a, r, s_ = [None, None], [None, None], [None, None], [0, 0]
            self.GUI.update(self.get_state())
            while plays < self.config["max_plays"] and len(self.unused) > 0 and result is 0:
                a[player] = card = self.clients[player].play(self.get_hidden_state(player))
                """ -- GAME LOGIC -- """
                s[player] = self.get_hidden_state(player)
                if card is None:
                    self.take_cards(player, 1)
                    player_temp = 1 - player
                else:
                    if card not in self.hand_card[player]:
                        raise Exception("Played a card which does not exist in hand card")
                    if len(self.used) != 0 and not self.used[-1].valid_card(card):
                        raise Exception("Played a invalid card")
                    if card.value in range(10, 15):
                        player_temp = player
                        if card.value is 12:
                            self.take_cards(1 - player, 2)
                        if card.value is 14:
                            self.take_cards(1 - player, 4)
                    else:
                        player_temp = 1 - player
                    self.hand_card[player].remove(card)
                    self.used.append(card)
                s_[player] = self.get_hidden_state(player)
                """ ---------------- """

                player, plays = player_temp, plays + 1
                """ result = {0: draw, 1: first hand win, 2: second hand win} """
                result = 0 if result is 0 and len(self.hand_card[0]) > 0 else 1
                result = 0 if result is 0 and len(self.hand_card[1]) > 0 else -1
                if not (plays < self.config["max_plays"] and len(self.unused) > 0 and result == 0):
                    if result == 0:
                        r[0] = len(self.hand_card[0]) - len(self.hand_card[1])
                        r[1] = len(self.hand_card[1]) - len(self.hand_card[0])
                    else:
                        r = [100 * result, 100 * result * (-1)]
                for i in range(2):
                    self.clients[i].observe(s[i], a[i], r[i], s_[i])
                self.GUI.update(self.get_state())

            self.GUI.update(result)
        except Exception as e:
            print(e)
