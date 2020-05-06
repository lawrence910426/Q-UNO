from Brains.Human_Brain import HumanBrain
from Brains.Dummy_Brain import DummyBrain
from Brains.Naive_Brain import *
from Brains.Gene_Brain import GeneBrain
from tkinter import *
from Versus import Versus
from Gene_Petri import GenePetri

root = Tk()
"""
first_hand, second_hand = GeneBrain(), GeneBrain()
Versus(first_hand, second_hand).start_game()
"""
GenePetri()
root.withdraw()
root.mainloop()
