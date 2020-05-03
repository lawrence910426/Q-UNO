from Enviroment.UNO_Server import UNOServer
from Brains.Human_Brain import HumanBrain
from tkinter import *
import threading

root = Tk()
first_hand, second_hand = HumanBrain(), HumanBrain()


def first_hand_play(state):
    return first_hand.get(state)


def first_hand_observe(s, a, r, s_):
    first_hand.add_observation(s, a, r, s_)


def second_hand_play(state):
    return second_hand.get(state)


def second_hand_observe(s, a, r, s_):
    second_hand.add_observation(s, a, r, s_)


Server = UNOServer({"init_hand_cards": 5, "max_plays": 100})
Clients = Server.init_game()
Clients[0].start_game(first_hand_observe, first_hand_play)
Clients[1].start_game(second_hand_observe, second_hand_play)
Server.start_game()

root.withdraw()
root.mainloop()
