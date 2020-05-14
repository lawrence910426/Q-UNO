from tkinter import *
from mttkinter import mtTkinter as tk
import time


class GUIOutput:
    """ -- ALWAYS CALL THIS FUNCTION WITH MAIN THREAD -- """
    def __init__(self, enable):
        self.enable = enable
        if self.enable:
            self.window = tk.Toplevel()
            self.window.geometry("1800x900")
            self.window.title("GUI Output")
            self.canvas = Canvas(self.window, width=1800, height=900)

    def draw_card(self, x, y, card):
        if self.enable:
            self.canvas.create_rectangle(x, y, 40 + x, 90 + y, fill=card.get_color_string())
            self.canvas.create_text(x + 20, y + 45, fill="white",
                                    font="Times 12 italic bold", text=card.get_value_string())

    def update(self, state):
        if self.enable:
            time.sleep(1)
            """ {0 -> used, 1 -> unused, 2 -> first hand, 3 -> second hand} """
            if isinstance(state, int):
                message = {0: "DRAW", 1: "UP WIN", 2: "DOWN WIN"}
                self.canvas.create_rectangle(350, 300, 1500, 550, fill="black")
                self.canvas.create_text(900, 425, fill="white",
                                        font="Times 100 bold", text=message[state])
            else:
                self.canvas.delete(ALL)
                self.canvas.pack()
                for i in range(len(state[2])):
                    x, y = i % 36, int(i / 36)
                    self.draw_card(x * 50, 100 * y, state[2][i])
                for i in range(len(state[3])):
                    x, y = i % 36, int(i / 36)
                    self.draw_card(x * 50, 600 + 100 * y, state[3][i])
                for i in range(0, min(30, len(state[0]))):
                    card = state[0][max(0, len(state[0]) - 30) + i]
                    if card is not None:
                        self.draw_card(150 + i * 50, 300, card)

    def destroy(self):
        if self.enable:
            self.window.destroy()
