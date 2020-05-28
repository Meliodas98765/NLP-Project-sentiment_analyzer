import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import datetime
import pickle
import sys
import emoji
import dataset_sentiment as ds
import classifier_sentiment as xs


conversation_stored_for_session = []
clean_inst_con_dict ={}
exit_sequence = "no data read..."
label = {0:"bad",1:"good"}
path1 = os.path.join(os.getcwd(),"Con_Sessions",str(datetime.date.today()),str(datetime.datetime.now().strftime("%H_%M_%S"))+".csv")
path2 = os.path.join(os.getcwd(),"Con_Sessions",str(datetime.date.today()))
path3 = os.path.join(os.getcwd(),"Con_Sessions")
path4 = os.path.join(os.getcwd(),r"tmp\Raw_Data.csv")

def chat(inst):
	instance_conversion = inst
	conversation_stored_for_session.append(instance_conversion)
	print(conversation_stored_for_session)
	clean_inst_con = clean_data_for_instance(instance_conversion)
	train_data =  create_instance_bag(clean_inst_con)
	decideTheinstance(train_data,clean_inst_con)


def clean_data_for_instance(instance_conversion):
	print("Cleaning")
	word_tokens = word_tokenize(instance_conversion)
	print(word_tokens)
	stop_words = set(stopwords.words('english'))
	clean_instance_conversation = []
	for w in word_tokens:
		if not w in stop_words:
			clean_instance_conversation.append(w)

	print(clean_instance_conversation)
	return clean_instance_conversation

def create_instance_bag(clean_con):
	
	print("Creating the bag")
	verctorizer = CountVectorizer(analyzer = "word",
								tokenizer = None,
								preprocessor = None,
								stop_words = None,
								max_features = 1000)

	train_data_feat = verctorizer.fit_transform(clean_con)
	train_data_feat = train_data_feat.toarray()
	return train_data_feat

def decideTheinstance(train_data,clean_data):
	#file =open(os.path.join(os.getcwd(),r"Models\model.pickle"),'rb')
	try:
		RFC_predict = pickle.load(open(os.path.join(os.getcwd(),r"Models\model"+str(len(clean_data))+".pickle"),"rb"))
		print("Predicting...")
		result = RFC_predict.predict(train_data)
		output = []
		for w in result:
			output.append(label[w])
		print(output)
	except FileNotFoundError:
		print("I dont know how to help with...")
	except ValueError:
		print("I dont know how to help with...but I'm learning..so I might know next time")
	else:
		print("\tCome again!!!\n\n")
	finally:
		pass
	

def settingUpRawData():
	print("Got till Here!!!")

	df = pd.DataFrame(conversation_stored_for_session)
	df.to_csv(path4,index = None,header = None,sep =" ",mode = 'a')
	ds.assignTo()
	ds.alignDataForvariousModels()
	data,model = xs.alignDataForvariousModels()
	for i in range(len(data)):
		xs.classify(model[i],data[i])

	print("All models trained!!!")


if __name__ == '__main__':
	if  os.path.exists(path3):
		print("Reading")
	else:
		os.mkdir(path3)
	if os.path.exists(path2):
			print("Following the line-up...")
	else:
		os.mkdir(path2)
	print("Hey how is your day...\n\n")

	inst =  input(">>>\t")

	try:
		exit_sequence = pd.read_csv(os.path.join(os.getcwd(),r"tmp\escape_seq.csv"))
		exit_sequence = list(exit_sequence["Sequence"])
	except FileNotFoundError:
		print("DataNotFound: escape_seq.csv")
	else:
		pass

	
	print(type(inst))
	try:
		while inst not in exit_sequence:
			chat(inst)
			inst =  input(">>>\t")
	except TypeError:
		print("I am not used to numbers yet, but I'm learning...")
	else:
		pass
	finally:
		pass

	print("Bye")
	settingUpRawData()