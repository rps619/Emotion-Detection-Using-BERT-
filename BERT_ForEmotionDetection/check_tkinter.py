import tkinter as tk
from time import sleep

def task():
    # The window will stay open until this function call ends.
    sleep(2) # Replace this with the code you want to run
    label= tk.Label(root, text="processing")
    label.pack()
    root.destroy()

root = tk.Tk()
root.title("Example")

label = tk.Label(root, text="Waiting for task to finish.")
label.pack()

root.after(200, task)
root.mainloop()

