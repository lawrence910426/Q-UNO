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


def extract():
    GenePetri.Opponent = petri_rl.get_alpha()
    GenePetri(tag + " Gene")
    petri_rl.keep = False


root = tk.Tk()
tk.Button(root, text="extract RL to gene", command=extract).pack()
# first_hand, second_hand = DefensiveBrain(), OffensiveBrain()
# first_hand, second_hand = RLBrain(tf.Session(), 1), DummyBrain()
# Mimic(DefensiveBrain(), first_hand, DummyBrain()).learn(partial(run, False))

tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
petri_rl = RLPetri(tag + " RL")
# petri = RLPetri("RL Haha 幹", True)

root.mainloop()
