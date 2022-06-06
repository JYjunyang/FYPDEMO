from tkinter.font import BOLD
from tkinter import *
from PIL import Image, ImageTk
# from ttkthemes import ThemedTk
# import tkinter.ttk as ttk
import fypDEMOmainPage2


class fyp:
    # style = ttk.Style()
    # style.configure('vivaldi28bold.TLabel', font=("Vivaldi",28,BOLD))
    # style.configure('lucida11.TButton', font=("Lucida Handwriting",11))
    
    def __init__(self):
        self.window = Tk()
        
    def windowGeometry(self,window):
        
        window.title("Sentiment Analysis Tool")
        window.geometry("1000x600")

        w = 800
        h = 400

        ws = window.winfo_screenwidth()
        hs = window.winfo_screenheight()

        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

        window.geometry("%dx%d+%d+%d" % (w, h, x, y))
        
    def callMainApp(self, event, ind, frames, window, button):
        frame = frames[ind]
        ind += 1
        button.configure(image=frame,text="loading...")
        if(ind == 29):
            window.destroy()
            fyp2.window.deiconify()
            fyp2.windowGeometry(fyp2.window)
        window.after(30, lambda event=event, ind=ind, frames=frames, window=window, 
                     button=button: self.callMainApp(event, ind, frames, window, button))
        
    def materialWindow1(self,window):
        firstPageTitle = Label(window, text="Sentiment Analysis Tool", font=("Vivaldi",28,BOLD))
        firstPageTitle.grid(row = 1, column = 1,sticky=S)
        window.grid_rowconfigure(1,weight = 1)
        window.grid_columnconfigure(1,weight = 1)

        firstPageEmptyLabel = Label(window)
        firstPageEmptyLabel.grid(row = 2, column = 1, pady=10)

        img = ImageTk.PhotoImage(Image.open(r"C:\Users\USER\OneDrive\sem6\FYP2\fypVirtual\fypProject1\start.jpg").resize((60,60)),master=window)
        self.img = img
        firstPageButton = Button(window, text="Click here to begin!", compound=LEFT, image = img, font=("Lucida Handwriting",11))
        firstPageButton.grid(row = 3, column = 1,sticky=N)
        window.grid_rowconfigure(3, weight = 1)
        frames = [PhotoImage(file=r'C:\Users\USER\OneDrive\sem6\FYP2\fypVirtual\fypProject1\loadingGif.gif',
                     format = 'gif -index %i' %(i),master=window).subsample(10) for i in range(29)]
        firstPageButton.bind('<Button-1>',lambda event, ind=0, frames=frames, window=window, 
                             button=firstPageButton: self.callMainApp(event, ind, frames, window, button))
    
    
if __name__ == "__main__":
    fyp1 = fyp()
    fyp1.windowGeometry(fyp1.window)
    fyp2 = fypDEMOmainPage2.fypMenu()
    fyp1.materialWindow1(fyp1.window)
    fyp2.buttonCreation()
    fyp1.window.mainloop()