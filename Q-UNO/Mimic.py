from Enviroment.UNO_Server import UNOServer
from Brains.Dummy_Brain import DummyBrain
import time
import threading


class Mimic:
    def __init__(self, trainer, trainee, dummy):
        self.trainer, self.trainee, self.dummy = trainer, trainee, dummy
        self.finished_games, self.conduct_games = 0, 300

    def learn(self, callback):
        def done(final_standings):
            self.finished_games += 1
            # print(self.finished_games)

        def trainer_play(state):
            return self.trainer.get(state)

        def trainer_observe(s, a, r, s_):
            self.trainee.add_observation(s, a, r, s_, 256)

        def dummy_play(state):
            return self.dummy.get(state)

        def dummy_observe(s, a, r, s_):
            self.dummy.add_observation(s, a, r, s_, 256)

        self.finished_games = 0
        for _ in range(self.conduct_games):
            server = UNOServer({"init_hand_cards": 3, "max_plays": 1000, "enable_GUI": False})
            clients = server.init_game()
            clients[0].start_game(trainer_observe, trainer_play)
            clients[1].start_game(dummy_observe, dummy_play)
            server.start_game(done)

            server = UNOServer({"init_hand_cards": 3, "max_plays": 1000, "enable_GUI": False})
            clients = server.init_game()
            clients[1].start_game(trainer_observe, trainer_play)
            clients[0].start_game(dummy_observe, dummy_play)
            server.start_game(done)

        def finalize():
            while self.finished_games is not self.conduct_games * 2:
                time.sleep(0.1)
            callback()

        threading.Thread(target=finalize).start()
