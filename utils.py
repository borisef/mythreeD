import PIL
import json, os
import numpy as np
import tkinter as tk
from PIL import Image



def save_data_2_json(data, data3D, out_json):
    #TODO
    pass

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0],rgb[1],rgb[2])

def pil_image_translate(img,x,y):
    a = 1
    b = 0
    c = x  # left/right (i.e. 5/-5)
    d = 0
    e = 1
    f = y  # up/down (i.e. 5/-5)
    img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
    return img

class CustomTooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.tooltip_window = None
    def enter(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background='#ffffff', relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
    def leave(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()

