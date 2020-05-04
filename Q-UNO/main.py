from Brains.Human_Brain import HumanBrain
from Brains.Dummy_Brain import DummyBrain
from tkinter import *
from Versus import Versus

root = Tk()
first_hand, second_hand = DummyBrain(), DummyBrain()
Versus(first_hand, second_hand).start_game()
root.withdraw()
root.mainloop()
