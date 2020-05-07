from Brains.Human_Brain import HumanBrain
from Brains.Dummy_Brain import DummyBrain
from Brains.RL_Brain import RLBrain
from Brains.Naive_Brain import *
from Brains.Gene_Brain import GeneBrain
from mttkinter import mtTkinter as tk
from Versus import Versus
from Gene_Petri import GenePetri


def done(final_standings):
    print(final_standings)
    run()


def run():
    Versus(first_hand, second_hand).start_game(done)


root = tk.Tk()
root.withdraw()
first_hand, second_hand = DummyBrain(), RLBrain()
run()
# GenePetri())
root.mainloop()
