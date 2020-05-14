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

root = tk.Tk()
petri = GenePetri(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
root.mainloop()
