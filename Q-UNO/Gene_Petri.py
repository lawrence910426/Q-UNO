from Versus import *
import numpy as np
from Brains.Gene_Brain import GeneBrain
from Brains.Dummy_Brain import DummyBrain
from Brains.Naive_Brain import *
from Brains.RL_Brain import RLBrain
from Mimic import Mimic
import time
import threading
import tensorflow as tf
import pickle


class GenePetri:
    games_count = 10
    organism_amount = 3
    test_game_count = 30
    opponent = DefensiveBrain()

    def __init__(self, tag, restore=False):
        self.session = tf.Session()
        self.rank = np.zeros(GenePetri.organism_amount)
        self.inited_organism, self.tag = 0, tag
        self.steps = self.genetic_rank = self.dummy_rank = self.draw = self.conducted = 0
        self.win_rate_log = tf.summary.FileWriter("logs/fit/" + self.tag, self.session.graph)
        self.win_rate = tf.placeholder(tf.float64)
        self.win_rate_summary = tf.summary.scalar(name='win_rate', tensor=self.win_rate)

        if restore:
            with open('models/' + self.tag + "/pickle", 'rb') as file:
                manifest = pickle.load(file)
                self.organism = [RLBrain(self.session, i, restore=True,
                                         memory=manifest["data"][i]["memory"],
                                         memory_count=manifest["data"][i]["memory_count"],
                                         swap_vars=manifest["data"][i]["swap_vars"],
                                         suicide_vars=manifest["data"][i]["suicide_vars"])
                                 for i in range(GenePetri.organism_amount)]
                self.saver = tf.train.import_meta_graph('models/' + self.tag + "/model-0.meta")
                self.saver.restore(self.session, tf.train.latest_checkpoint('models/' + self.tag + "/"))
                self.steps = manifest["steps"]
                threading.Thread(target=self.evolution, daemon=True).start()
        else:
            def finish_init():
                self.inited_organism += 1
                if self.inited_organism == self.organism_amount:
                    threading.Thread(target=self.evolution, daemon=True).start()

            self.organism = [RLBrain(self.session, i) for i in range(GenePetri.organism_amount)]
            for cell in self.organism:
                Mimic(GenePetri.opponent, cell, GenePetri.opponent).learn(finish_init)
            self.saver = tf.train.Saver()
            self.save(True)

    def evolution(self):
        while True:
            self.conduct_game()
            while self.conducted is not GenePetri.games_count * GenePetri.organism_amount * 2:
                print("Conducted games: ", self.conducted)
                time.sleep(0.1)
            self.show_win_rate()
            while self.genetic_rank + self.dummy_rank + self.draw is not GenePetri.test_game_count * 2:
                print("Win rate: ", self.genetic_rank, self.dummy_rank, self.draw)
                time.sleep(0.1)
            value = self.genetic_rank / (self.genetic_rank + self.dummy_rank + 1e-9)
            summary = self.session.run(
                self.win_rate_summary,
                feed_dict={self.win_rate: value}
            )
            print("-------------------------------")
            print("Win rate against opponent is: ", value, self.genetic_rank, self.dummy_rank, self.steps)
            self.win_rate_log.add_summary(summary, self.steps)
            print("-------------------------------")
            if value is 0:
                self.genocide()
            self.genetic_rank = self.dummy_rank = self.draw = self.conducted = 0
            self.loser_elimination()
            # self.fertilization()
            # self.genetic_mutate()
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
                Versus(self.organism[i], GenePetri.opponent).start_game(done_gen((i, -1)))
                Versus(GenePetri.opponent, self.organism[i]).start_game(done_gen((-1, i)))

    def genetic_mutate(self):
        for i in range(GenePetri.organism_amount):
            self.organism[i].mutate()

    def loser_elimination(self):
        self.organism[np.argmin(self.rank)].reset()

    def genocide(self):
        for cell in self.organism:
            cell.reset()

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
            Versus(self.organism[alpha_id], GenePetri.opponent).start_game(done_gen)
            Versus(GenePetri.opponent, self.organism[alpha_id]).start_game(done_dum)

    def save(self, meta=False):
        self.saver.save(self.session,
                        'models/' + self.tag + '/model',
                        write_meta_graph=meta, global_step=self.steps)
        with open('./models/' + self.tag + "/pickle", 'wb') as file:
            pickle.dump({
                "data": [
                    {
                        "memory_count": item.memory_count,
                        "memory": item.memory,
                        "swap_vars": item.swap_vars,
                        "suicide_vars": item.suicide_vars,
                    }
                    for item in self.organism
                ],
                "steps": self.steps
            }, file)
