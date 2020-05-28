import os
import tkinter
import pandas as pd

path = os.path.join(os.getcwd(),"Raw_Data.csv")

read = pd.read_csv(path)
print(read)
label = {"Bad":0,"Good":1}
top = tkinter.Tk()

def setValue1():
	print("Bad")

def setValue2():
	print("Bad")

def GUUI(rev):
	text1 = "Good"
	text2 = "Bad"
	review = tkinter.StringVar()
	message = tkinter.Message(top,textvariable = review,relief = tkinter.FLAT) 
	message.pack()
	review.set(rev)
	good = tkinter.Button(top,text = text1,command = setValue1 )
	bad = tkinter.Button(top,text = text2,command = setValue2 )

	good.pack()
	bad.pack()

	top.mainloop()



if __name__ == '__main__':
	for i in range(len(read)):
		review = read["Reviews"][i]
		GUUI(review)