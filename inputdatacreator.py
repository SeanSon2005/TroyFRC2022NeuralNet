import numpy as np
import cv2 as cv2
import os
import tkinter as tk
from PIL import Image, ImageTk
import csv
  
path = "C://Users//astro//Documents//Coding//AIDATA"
os.chdir(path+"//images")
  
def generate_output(original):
    def update_position1(event):
        x,y = event.x, event.y
        global xtop
        xtop = x
        global yleft
        yleft = y
        redraw()

    def update_position2(event):
        x,y = event.x, event.y
        global xbot
        xbot = x
        global yright
        yright = y
        redraw()
        
    def redraw():
        global xtop
        global yleft
        global xbot
        global yright
        canvas.delete("all")
        canvas.create_image(0,0,image=test, anchor="nw")
        if not xtop == 0:
            canvas.create_line(xtop-4, yleft, xtop+5, yleft)
            canvas.create_line(xtop, yleft-4, xtop, yleft+5)
            if not xbot == 0:
                canvas.create_line(xtop, yleft, xbot, yleft, width=2, fill='cyan')
                canvas.create_line(xtop, yright, xbot, yright, width=2, fill='cyan')
                canvas.create_line(xtop, yleft, xtop, yright, width=2, fill='cyan')
                canvas.create_line(xbot, yleft, xbot, yright, width=2, fill='cyan')

    def close(event):
        root.destroy()

    #frame_scaled = cv2.resize(frame, dsize=(x_res, y_res), interpolation=cv2.INTER_CUBIC)
    #blurred = cv2.blur(image,(4,4))

    root = tk.Tk()
    test = ImageTk.PhotoImage(Image.fromarray(original))

    canvas = tk.Canvas(root, width=len(original[0]), height=len(original))
    canvas.create_image(0,0,image=test, anchor="nw")
    canvas.pack()
    canvas.old_coords = None

    root.bind('<B1-Motion>', update_position1)
    root.bind('<ButtonPress-1>',update_position1)
    root.bind('<B3-Motion>', update_position2)
    root.bind('<ButtonPress-3>',update_position2)
    root.bind('<space>',close)

    root.mainloop()

    return xtop, yleft, xbot, yright

def generate_input(image):
    return image  

file_count = len(os.listdir())
image_array = np.empty(file_count,dtype=object)
coords_array = np.empty(file_count,dtype=object)
ind = 0
for file in os.listdir():
    xtop = 0
    yleft = 0
    xbot = 0
    yright = 0
    print("Opening",file)
    image = cv2.imread(file)
    coords_array[ind] = generate_output(image)
    image_array[ind] = generate_input(image)
    ind += 1
    print("Done")
    

os.chdir(path+"//labels")
with open('data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    header = ["class","x1","y1","x2","y2"]
    writer.writerow(header)         
    for i in range(file_count):
        writer.writerow((0,)+coords_array[i])
    csvfile.close()
    
        