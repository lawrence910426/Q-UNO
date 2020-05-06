from Enviroment.UNO_Server import UNOServer


class Versus:
    def __init__(self, first, second):
        self.first_hand, self.second_hand = first, second
        self.first, self.second = 0, 0

    def start_game(self, done):
        def first_hand_play(state):
            return self.first_hand.get(state)

        def first_hand_observe(s, a, r, s_):
            self.first_hand.add_observation(s, a, r, s_)

        def second_hand_play(state):
            return self.second_hand.get(state)

        def second_hand_observe(s, a, r, s_):
            self.second_hand.add_observation(s, a, r, s_)

        server = UNOServer({"init_hand_cards": 3, "max_plays": 1000, "enable_GUI": False})
        clients = server.init_game()
        clients[0].start_game(first_hand_observe, first_hand_play)
        clients[1].start_game(second_hand_observe, second_hand_play)
        server.start_game(done)

