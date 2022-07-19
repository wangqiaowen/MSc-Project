from re import I
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
from synthesise import *

from tkinter import *


cf = open("/Users/qiaowenwang/Github/MSc-Project/data generation/synthesis_config.json")
data = json.load(cf)
tool_dir=data["Tool_DIR"]
save_dir=data["Save_DIR"]
lnd_dir=data["LND_DIR"]
image_count=data["IMAGE_COUNT"]
garbage_dir = data["Garbage_DIR"]
start_name=data["START_NAME"]
temp_dir = "/Users/qiaowenwang/Github/MSc-Project/data generation/temp"


def generate():
    global picture3

    global output_image
    global image_points
    global rx
    global ry
    global rz
    global image_points_temp
    global trn_temp
    global output_image_temp
    global trnMat
    global formula_acceptance
    global human_judge
    global reason
    global picture2
    global c
    global garbage_num
    global img_num
    global human_judge
    global dm_temp

    temp = img_num.get()
    output_image, image_points, rx, ry, rz, image_points_temp, trn_temp, output_image_temp, trnMat = synthesis_one(tool_dir, save_dir, lnd_dir, garbage_dir, temp)
    cv2.imwrite(os.path.join(temp_dir, 'temp.png'), output_image)
    formula_acceptance, reason, dm_temp = judge(output_image, image_points, rx, ry, rz, image_points_temp, trn_temp, output_image_temp, trnMat, output_image)

    picture3 = PhotoImage(file='./temp/'+'temp.png')
    c.itemconfigure(picture2, image = picture3)

def agree():
    global i
    global picture3 

    global output_image
    global image_points
    global rx
    global ry
    global rz
    global image_points_temp
    global trn_temp
    global output_image_temp
    global trnMat
    global formula_acceptance
    global human_judge
    global reason

    human_judge.set(True)
    print(human_judge.get())
    save_res()
    generate()


def garbage():
    global i
    global picture3 

    global output_image
    global image_points
    global rx
    global ry
    global rz
    global image_points_temp
    global trn_temp
    global output_image_temp
    global trnMat
    global formula_acceptance
    global human_judge
    global reason

    # formula_acceptance, reason = judge(output_image, image_points, rx, ry, rz, image_points_temp, trn_temp, output_image_temp, trnMat)
    human_judge.set(False)
    print(human_judge.get())
    save_res()
    generate()

def save_res():
    global formula_acceptance
    global human_judge
    global reason
    global garbage_num
    global img_num
    global dm_temp

    print(reason)

    decision = human_judge.get()
    print(decision)
        
    if(reason == "disobey" or reason == "outside"):
        if(decision is not True):
            temp = garbage_num.get()
            # judege = human_judge.get()
            save(decision, temp, garbage_dir, save_dir, output_image , image_points, temp, dm_temp)
            garbage_num.set(temp+1)
        else:
            temp = img_num.get()
            # judege = human_judge.get()
            save(decision, temp, garbage_dir, save_dir, output_image , image_points, temp, dm_temp)
            img_num.set(temp+1)

    if(reason == "comply"):
        if(decision is not True):
            temp = garbage_num.get()
            # judege = human_judge.get()
            save(decision, temp, garbage_dir, save_dir, output_image , image_points, temp, dm_temp)
            garbage_num.set(temp+1)
        else:
            temp = img_num.get()
            # judege = human_judge.get()
            save(decision, temp, garbage_dir, save_dir, output_image , image_points, temp, dm_temp)
            img_num.set(temp+1)
    
def show_window():

    global picture2
    global c
    global garbage_num
    global img_num
    global human_judge

    root = tk.Tk()

    garbage_num = IntVar()
    img_num = IntVar()
    garbage_num.set(start_name)
    img_num.set(start_name)

    human_judge = BooleanVar()

    c = Canvas(root, width=600, height=600)
    c.pack()

    picture = PhotoImage()
    picture2 = c.create_image(300,300,image=picture)

    label = tk.Label(root)
    label.pack()
    root.geometry('700x700')
    root.resizable(True, True)
    root.title('img selector')

    buttonframe = ttk.Frame(root)
    buttonframe.pack()

    # main = Tk()
    # c.bind("<Button-1>", stuff)

    button1=ttk.Button(buttonframe,text='generate', command=lambda: generate())
    button1.grid(row=0, column=2, padx=5, pady=5)
    button3=ttk.Button(buttonframe,text='garbage', command=lambda: garbage())
    button3.grid(row=1, column=1, padx=5, pady=5)
    button4=ttk.Button(buttonframe,text='accept', command=lambda: agree())
    button4.grid(row=1, column=2, padx=5, pady=5)
    button5=ttk.Button(buttonframe,text='save', command=lambda: save_res())
    button5.grid(row=1, column=3, padx=5, pady=5)
    button2=ttk.Button(buttonframe,text='Exit', command=lambda: root.quit())
    button2.grid(row=3, column=2, padx=5, pady=5)


    root.mainloop()

if __name__ == "__main__":
    show_window()