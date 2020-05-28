import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


path12 = os.path.join(os.getcwd(),r"tmp\final.csv")
def assignTo():
	if os.path.exists(os.path.join(os.getcwd(),"tmp")):
		print()
	else:
		os.mkdir(os.path.join(os.getcwd(),"tmp"))
	read = pd.read_csv(os.path.join(os.getcwd(),r"tmp\Raw_Data.csv"))
	print(read)
	label = {"Bad":0,"Good":1}
	dicr = []
	for i in range(len(read["Reviews"])):
		print(read["Reviews"][i])
		print("Good: 1\nBad: 0")
		val = input(">>>")
		dicr.append((read["Reviews"][i].strip('"'),val))
		print(dicr)

	df = pd.DataFrame(dicr,columns = ["Reviews","sentiment"])
	print(df)

	df.to_csv(path12)

def alignDataForvariousModels():
	print("Aligning data for Training...")
	df = pd.read_csv(path12)
	dataset = list(df["Reviews"])
	data = list(df["Reviews"])
	sentiments = list(df["sentiment"])
	count = []
	dataset1 =[]
	tmp= []
	stop_words = set(stopwords.words('english'))
	print(dataset[1])
	for w in range(len(dataset)):
		dataset[w] = word_tokenize(dataset[w])
		for i in dataset[w]:
			if i not in stop_words:
				dataset1.append(i)
		dataset[w] = dataset1
		dataset1 = []
		print(dataset[w])
		count.append(len(dataset[w]))

	print(count)
	if os.path.exists(os.path.join(os.getcwd(),"Models\dataset")):
		print()
	else:
		os.mkdir(os.path.join(os.getcwd(),"Models\dataset"))
	dataset1 = []
	c= list(set(count))
	print(c)
	for w in c:
		for i in range(len(count)):
			if w == count[i]:
				tmp.append(data[i])
				tmp.append(sentiments[i])
				dataset1.append(tuple(tmp))
			tmp = []
		print(dataset1)
		df = pd.DataFrame(dataset1,columns = ["Reviews","sentiment"])
		df.to_csv(os.path.join(os.getcwd(),"Models\dataset\model"+str(w)+".csv"))
		dataset1 = []


	

if __name__ == '__main__':
	assignTo()
	alignDataForvariousModels()
	
#COMPLETED