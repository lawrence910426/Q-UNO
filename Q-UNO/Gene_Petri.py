from Versus import *
import numpy as np
from Brains.Gene_Brain import GeneBrain
from Brains.Dummy_Brain import DummyBrain
from Brains.Naive_Brain import OffensiveBrain
import time
import threading
import tensorflow as tf
import datetime


class GenePetri:
    games_count = 5
    organism_amount = 5
    test_game_count = 30

    def __init__(self):
        self.session = tf.Session()
        self.rank = np.zeros(GenePetri.organism_amount)
        self.organism = [GeneBrain(self.session, i) for i in range(GenePetri.organism_amount)]
        self.steps = self.genetic_rank = self.dummy_rank = self.draw = self.conducted = 0
        self.win_rate_log = tf.summary.FileWriter(
            "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            self.session.graph)
        self.win_rate = tf.placeholder(tf.float64)
        self.win_rate_summary = tf.summary.scalar(name='win_rate', tensor=self.win_rate)
        threading.Thread(target=self.evolution).start()

    def evolution(self):
        self.steps = 0
        while True:
            self.conduct_game()
            while self.conducted is not \
                    GenePetri.games_count * (GenePetri.organism_amount - 1) * GenePetri.organism_amount:
                time.sleep(0.1)
            self.show_win_rate()
            while self.genetic_rank + self.dummy_rank + self.draw is not GenePetri.test_game_count * 2:
                print(self.genetic_rank, self.dummy_rank, self.draw)
                time.sleep(0.1)
            value = self.genetic_rank / (self.genetic_rank + self.dummy_rank + 1e-9)
            summary = self.session.run(
                self.win_rate_summary,
                feed_dict={self.win_rate: value}
            )
            print("-------------------------------")
            print("Win rate against opponent is: ", self.genetic_rank, self.dummy_rank, value, self.steps)
            self.win_rate_log.add_summary(summary, self.steps)
            print("-------------------------------")
            self.genetic_rank = self.dummy_rank = self.draw = self.conducted = 0
            self.loser_elimination()
            self.genetic_mutate()
            self.steps += 1

    def conduct_game(self):
        def done_gen(parameter):
            def done(final_standings):
                if final_standings["result"] == 1:
                    self.rank[parameter[0]] += 1
                if final_standings["result"] == 2:
                    self.rank[parameter[1]] += 1
                print(final_standings, " ", self.conducted)
                self.conducted += 1
            return done

        for i in range(GenePetri.organism_amount):
            for j in range(GenePetri.organism_amount):
                if i is not j:
                    for _ in range(GenePetri.games_count):
                        Versus(self.organism[i], self.organism[j]).start_game(done_gen((i, j)))

    def genetic_mutate(self):
        for i in range(GenePetri.organism_amount):
            self.organism[i].mutate()

    def loser_elimination(self):
        self.organism[np.argmin(self.rank)].die_off()

    def show_win_rate(self):
        self.genetic_rank = self.dummy_rank = self.draw = 0

        def done_gen(final_standings):
            self.genetic_rank += 1 if final_standings["result"] == 1 else 0
            self.dummy_rank += 1 if final_standings["result"] == 2 else 0
            self.draw += 1 if final_standings["result"] == 0 else 0

        def done_dum(final_standings):
            self.dummy_rank += 1 if final_standings["result"] == 1 else 0
            self.genetic_rank += 1 if final_standings["result"] == 2 else 0
            self.draw += 1 if final_standings["result"] == 0 else 0

        alpha_id = np.argmax(self.rank)
        for _ in range(GenePetri.test_game_count):
            Versus(self.organism[alpha_id], OffensiveBrain()).start_game(done_gen)
            Versus(OffensiveBrain(), self.organism[alpha_id]).start_game(done_dum)
