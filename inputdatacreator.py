import numpy as np
import cv2 as cv2
import os
import tkinter as tk
from PIL import Image, ImageTk
  
path = "C://Users//astro//Documents//Coding//yolov7//train"
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
ind = 0
for file in os.listdir():
    xtop = 0
    yleft = 0
    xbot = 0
    yright = 0
    print("Opening",file)
    
    image = cv2.imread(file)
    image_width = len(image[0])
    image_height = len(image)

    coords = generate_output(image)
    os.chdir(path+"//labels")
    with open('image_'+str(ind)+".txt", 'w') as txtfile:
        txtfile.write('0 ' + str(round(coords[0]/image_width,6)) + " " + str(round(coords[1]/image_height,6)) + " "+ str(round(coords[2]/image_width,6)) + " "+ str(round(coords[3]/image_height,6)))
    ind += 1
    print("Done")
    
    
        