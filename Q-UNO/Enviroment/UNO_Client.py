class UNOClient:
    def __init__(self, hand_card):
        self.hand_card = hand_card
        self.observe_callback, self.play_callback = None, None

    def start_game(self, observe_callback, play_callback):
        self.observe_callback = observe_callback
        self.play_callback = play_callback

    def observe(self, s, a, r, s_):
        self.observe_callback(s, a, r, s_)

    def play(self, state):
        return self.play_callback(state)
