from tkinter import *
from tkinter import messagebox as tm
import pickle
from collections import Counter
#import cv2
import numpy as np
import photo_show
class Login(Frame):
	def __init__(self,abc):
		super().__init__(abc)
		self.log=0
		self.rf=0
		self.gb=0
		self.btnlogin=Button(self,padx=10,pady=10,bd=10,fg='black',font=('arial',17,'bold'),width=10,text="Load",bg='powder blue',command=self.load)
		self.btnpredict=Button(self,padx=10,pady=10,bd=10,fg='black',font=('arial',17,'bold'),width=10,text="predict",bg='powder blue',command=self.predict)
		self.btnpredict.grid(row=0,column=1)
		self.btnlogin.grid(row=0,column=0)
		self.canvas=Canvas(abc,width=400,height=200)
		self.canvas.pack()
		self.pack()
	def load(self):
		path='img.png'
		img=PhotoImage(file=path)
		self.canvas.create_image(200,100,anchor=W,image=img)
		tm.showinfo('completion box','loaded')
	def predict(self):
		l=[]
		path='img.png'
		x=cv2.imread(path)
		x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
		x=cv2.resize(x,(28,28))
		print(x.shape)
		f=open('training.txt','rb')
		k=pickle.load(f)
		x=k[0].transform(x.reshape(1,-1).astype(np.float64))
		x_pca=k[1].transform(x)
		for model in [k[2],k[3],k[4]]:
			l.append(int(model.predict(x_pca.reshape(1,-1))))
		print(l)
		p=Counter(l).most_common()[0][0]
		prediction='prediction='+str(p)
		tm.showinfo('completion box',prediction)

root=Tk()
obj=Login(root)
root.title('Digit Recognition')
root.geometry('800x400')
root.mainloop()

