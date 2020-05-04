from Enviroment.UNO_Server import UNOServer
from Brains.Human_Brain import HumanBrain
from tkinter import *
import threading
from Versus import Versus

root = Tk()
first_hand, second_hand = HumanBrain(), HumanBrain()
Versus(first_hand, second_hand).start_game()
root.withdraw()
root.mainloop()
