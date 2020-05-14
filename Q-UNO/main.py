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
first_hand, second_hand = None, None


def done(final_standings):
    print(final_standings)
    global win, lose
    if final_standings["result"] == 1:
        win += 1
    if final_standings["result"] == 2:
        lose += 1
    if win + lose is 30:
        print("----------------------------------")
        print("Win rate: ", win / (win + lose))
        print("----------------------------------")
        win = lose = 0
    run(False)


def run(new_brain):
    global first_hand, second_hand
    if new_brain:
        first_hand, second_hand = RLBrain(tf.Session(), 1), OffensiveBrain()
    Versus(first_hand, second_hand).start_game(done)


def save():
    petri.save()


root = tk.Tk()
tk.Button(root, text="save model", command=save).pack()
# first_hand, second_hand = DefensiveBrain(), OffensiveBrain()
# first_hand, second_hand = RLBrain(tf.Session(), 1), DummyBrain()
# Mimic(DefensiveBrain(), first_hand, DummyBrain()).learn(partial(run, False))

petri = GenePetri(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# petri = RLPetri(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# petri = RLPetri("RL Haha 幹", True)

root.mainloop()
