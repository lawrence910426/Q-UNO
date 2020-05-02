"""
-- CONFIG --
init_hand_cards -> integer
max_plays -> integer
"""

from Enviroment.UNO_Client import UNOClient
from Enviroment.Card import Card
import random
import asyncio


class UNOServer:
    def __init__(self, config):
        self.config = config
        self.clients = (None, None)
        self.unused = []
        self.used = []
        self.hand_card = {
            0: [],
            1: []
        }

    def init_game(self):
        self.unused.append([Card(color, value) for color in range(1, 5) for value in range(0, 13)])
        self.unused.append([Card(0, value) for value in range(13, 15)])
        random.shuffle(self.unused)
        for i in range(2):
            self.take_cards(i, self.config["init_hand_cards"])
            self.clients[i] = UNOClient(self.hand_card[i])
        return self.clients

    def take_cards(self, player, amount):
        self.hand_card[player] = self.unused[0:amount]
        self.unused = self.unused[amount]

    def get_state(self, player):
        return [self.used, len(self.unused), len(self.hand_card[player], self.hand_card[1 - player])]

    def start_game(self):
        async def loop():
            while True:
                try:
                    self.play()
                except Exception as e:
                    print(e)
        asyncio.run(loop())

    def play(self):
        player, plays, player_temp = 0, 0, 0
        while plays < self.config["max_plays"] and len(self.unused) > 0:
            card = self.clients[player].play(self.get_state(player))
            if card not in self.hand_card[player]:
                raise Exception("Played a card not exist on hand card")
            if len(self.used) != 0 and not self.used[-1].valid_card(card):
                raise Exception("Played a invalid card")

            """ -- GAME LOGIC -- """
            reward = len(self.hand_card[player]) - len(self.hand_card[1 - player])
            if card.value in range(10, 15):
                reward += 10
                player_temp = player
            else:
                player_temp = 1 - player
            """ ---------------- """
            former = self.get_state(player)
            self.hand_card.remove(card)
            self.used.append(card)
            latter = self.get_state(player)

            self.clients[player].observe(former, card, reward, latter)
            player, plays = player_temp, plays + 1
