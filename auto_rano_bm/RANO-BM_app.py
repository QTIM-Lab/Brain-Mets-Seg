import tkinter as tk
from tkinter.filedialog import askopenfilename
from RANO_BM_response_category import category_logic
import os
basedir = os.getcwd()

files_needed = ['a','b','c','d']

def AssignCategory(paths_list):
	path1, path2, path3, path4 = paths_list
	response = category_logic(path1, path2, path3, path4, basedir, output_images = False, immunotherapy = True)
	return response
def Beg(i):
	x = askopenfilename()
	files_needed[i-1] = str(x)
def popup_bonus(files):
	response = AssignCategory(files)
	win = tk.Toplevel()
	win.wm_title("Response")
	l = tk.Label(win, text=response,  width=25, height=5, bg="blue", fg="red")
	l.grid(row=0, column=0)
	b = tk.Button(win, text="Quit",  width=25, height=5, bg="blue", fg="white",command=win.destroy)
	b.grid(row=1, column=0)

window = tk.Tk()
w = tk.Label(window, text = 'Auto RANO-BM App')
w.pack() 
requester = tk.Button(window, text = 'Select Baseline Image', width=25, height=5, bg="black", fg="white", command = lambda: Beg(1))
requester.pack()
requester = tk.Button(window, text = 'Select Baseline ROI', width=25, height=5, bg="black", fg="white", command = lambda: Beg(2))
requester.pack()
requester = tk.Button(window, text = 'Select New Image', width=25, height=5, bg="black", fg="white", command = lambda: Beg(3))
requester.pack()
requester = tk.Button(window, text = 'Select New ROI', width=25, height=5, bg="black", fg="white", command = lambda: Beg(4))
requester.pack()
compiler = tk.Button(window, text = 'Get RANO-BM Response', width=25, height=5, bg="black", fg="white", command = lambda: popup_bonus(files_needed))
compiler.pack()
closer = tk.Button(window, text = 'Quit', width=25, height=5, bg="black", fg="white", command = lambda: window.destroy())
closer.pack()
	
window.mainloop()