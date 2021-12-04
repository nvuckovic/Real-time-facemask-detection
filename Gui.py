import tkinter as tk
import os

def runDetect():
    os.system('python DetectVideo.py')

def runTraining():
    os.system('python trainModel.py')

gui = tk.Tk()
frame = tk.Frame(gui)
frame.pack()

button = tk.Button(frame, 
                   text="Run training script", 
                   fg="green",
                   command=runTraining)

button.pack(side=tk.LEFT)

button2 = tk.Button(frame,
                   text="Run Camera script",
                   fg="blue",
                   command=runDetect)

button2.pack(side=tk.LEFT)

button3 = tk.Button(frame, 
                   text="QUIT", 
                   fg="red",
                   command=quit)

button3.pack(side=tk.RIGHT)

gui.mainloop()