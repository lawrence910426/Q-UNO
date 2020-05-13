from Versus import *
import numpy as np
from Brains.Gene_Brain import GeneBrain
from Brains.Dummy_Brain import DummyBrain
from Brains.Naive_Brain import *
import time
import threading
import tensorflow as tf
import datetime


class GenePetri:
    games_count = 3
    organism_amount = 5
    test_game_count = 30
    Opponent = DefensiveBrain()

    def __init__(self, tag):
        self.session = tf.Session()
        self.rank = np.zeros(GenePetri.organism_amount)
        self.organism = [GeneBrain(self.session, i) for i in range(GenePetri.organism_amount)]
        self.steps = self.genetic_rank = self.dummy_rank = self.draw = self.conducted = 0
        self.win_rate_log = tf.summary.FileWriter("logs/fit/" + tag, self.session.graph)
        self.win_rate = tf.placeholder(tf.float64)
        self.win_rate_summary = tf.summary.scalar(name='win_rate_gene', tensor=self.win_rate)
        threading.Thread(target=self.evolution, daemon=True).start()

    def evolution(self):
        self.steps = 0
        while True:
            self.conduct_game()
            while self.conducted is not GenePetri.games_count * GenePetri.organism_amount * 2:
                print("Conducted games: ", self.conducted)
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
            self.fertilization()
            self.genetic_mutate()
            self.steps += 1

    def conduct_game(self):
        def done_gen(parameter):
            def done(final_standings):
                if final_standings["result"] == 1 and parameter[0] is not -1:
                    self.rank[parameter[0]] += 1
                if final_standings["result"] == 2 and parameter[1] is not -1:
                    self.rank[parameter[1]] += 1
                self.conducted += 1
            return done

        for i in range(GenePetri.organism_amount):
            for _ in range(GenePetri.games_count):
                Versus(self.organism[i], GenePetri.Opponent).start_game(done_gen((i, -1)))
                Versus(GenePetri.Opponent, self.organism[i]).start_game(done_gen((-1, i)))

    def genetic_mutate(self):
        for i in range(GenePetri.organism_amount):
            self.organism[i].mutate()

    def fertilization(self):
        alpha_id = np.argmax(self.rank)
        for i in range(GenePetri.organism_amount):
            if i is not alpha_id:
                self.organism[i].fertilization(alpha_id)

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