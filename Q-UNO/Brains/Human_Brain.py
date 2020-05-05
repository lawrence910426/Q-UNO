from Brains.Brain import Brain
import tkinter as tk
from tkinter import *
from functools import partial
import time
import copy


class HumanBrain(Brain):
    """ -- ALWAYS CALL THIS FUNCTION WITH MAIN THREAD -- """
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.geometry("1800x900")
        self.window.title("Human Brain")
        self.pixel_virtual = tk.PhotoImage(width=1, height=1)
        self.chosen_card, self.chose = None, False

    def choose_card(self, card):
        self.chosen_card, self.chose = card, True

    def draw_card(self, x, y, card):
        if card is None:
            tk.Button(self.window, bg="black", fg="white",
                      width=40, height=90, image=self.pixel_virtual,
                      command=partial(self.choose_card, None)
                      ).place(x=x, y=y)
            tk.Label(self.window, text="?").place(x=x+20, y=y+45)
        else:
            if card.color == 0:
                for i in range(1, 5):
                    card.color = i
                    tk.Button(self.window, fg="white",
                              bg=card.get_color_string(),
                              width=40, height=90/4, image=self.pixel_virtual,
                              command=partial(self.choose_card, copy.deepcopy(card))
                              ).place(x=x, y=y+90/4*(i-1))
                card.color = 0
            else:
                tk.Button(self.window, fg="white",
                          bg=card.get_color_string(),
                          width=40, height=90, image=self.pixel_virtual,
                          command=partial(self.choose_card, card)
                          ).place(x=x, y=y)
            tk.Label(self.window, text=card.get_value_string()).place(x=x+10, y=y+45)

    def get(self, s):
        for item in self.window.winfo_children():
            item.destroy()
        self.chosen_card, self.chose = None, False
        self.window.title("Human Brain - Penalty: " + str(s[4]))
        self.window.lift()

        for i in range(len(s[2])):
            x, y = i % 36, int(i / 36)
            self.draw_card(x * 50, 100 * y, s[2][i])
        for i in range(s[3]):
            x, y = i % 36, int(i / 36)
            self.draw_card(x * 50, 600 + 100 * y, None)
        for i in range(min(len(s[0]), 30)):
            if s[0][i] is not None:
                self.draw_card(150 + i * 50, 300, s[0][i])

        while not self.chose:
            time.sleep(0.1)
        return self.chosen_card

    def add_observation(self, s, a, r, s_):
        return

    def learn(self):
        return
