from Brains.Human_Brain import HumanBrain
from Brains.Dummy_Brain import DummyBrain
from Brains.RL_Brain import RLBrain
from Brains.Naive_Brain import *
from Brains.Gene_Brain import GeneBrain
from mttkinter import mtTkinter as tk
from Versus import Versus
from Petri.RL_Petri import RLPetri
from Petri.Gene_Petri import GenePetri
from Mimic import Mimic
import tensorflow as tf
from functools import partial
import sys
import signal
import datetime


win, lose = 0, 0
sess = tf.Session()
first_hand, second_hand = OffensiveBrain(), DummyBrain()


def done_gen(rev):
    def done(final_standings):
        global win, lose
        if final_standings["result"] == 1:
            if rev:
                lose += 1
            else:
                win += 1
        if final_standings["result"] == 2:
            if rev:
                win += 1
            else:
                lose += 1

        # print(final_standings)
        # print(win, lose)
        # if win + lose is 200:
        # print("----------------------------------")
        print("Win rate: ", win / (win + lose + 1e-9))
        # print("----------------------------------")
        # win = lose = 0
        run()
    return done


def run():
    Versus(first_hand, second_hand).start_game(done_gen(False))
    Versus(second_hand, first_hand).start_game(done_gen(True))


root = tk.Tk()
run()
root.mainloop()
