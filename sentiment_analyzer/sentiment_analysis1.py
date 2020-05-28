import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import datetime
import pickle

conversation_stored_for_session = []
clean_inst_con_dict ={}
label = {0:"bad",1:"good"}
path1 = os.path.join(os.getcwd(),r"Con_Sessions",str(datetime.date.today()),str(datetime.datetime.now().strftime("%H_%M_%S"))+".csv")
path2 = os.path.join(os.getcwd(),"Con_Sessions",str(datetime.date.today()))
path3 = os.path.join(os.getcwd(),"Con_Sessions")


def chat():
	print("Hey how is your day...\n\n")
	instance_conversion =  input(">>>\t")
	conversation_stored_for_session.append(instance_conversion)
	print(conversation_stored_for_session)
	clean_inst_con = clean_data_for_instance(instance_conversion)
	create_instance_bag(clean_inst_con)
	train_data =  create_instance_bag1(clean_inst_con)
	decideTheinstance(train_data)


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
	instance_calls = {}
	for w in clean_con:
		instance_calls[w] = 1
		if w in clean_inst_con_dict:
			clean_inst_con_dict[w] = int(clean_inst_con_dict[w])+1
		else:
			clean_inst_con_dict.update(instance_calls)
	val = [clean_inst_con_dict.keys(),clean_inst_con_dict.values()]
	df = pd.DataFrame(val)
	df.to_csv(path1)

def create_instance_bag1(clean_con):
	print("Creating the bag")
	verctorizer = CountVectorizer(analyzer = "word",
								tokenizer = None,
								preprocessor = None,
								stop_words = None,
								max_features = 1000)

	train_data_feat = verctorizer.fit_transform(clean_con)
	train_data_feat = train_data_feat.toarray()
	return train_data_feat

def decideTheinstance(train_data):
	#file =open(os.path.join(os.getcwd(),r"Models\model.pickle"),'rb')
	RFC_predict = pickle.load(open(os.path.join(os.getcwd(),r"Models\model.pickle"),"rb"))
	print("Predicting...")
	result = RFC_predict.predict(train_data)
	output = []
	for w in result:
		output.append(label[w])
	print(output)

if __name__ == '__main__':
	if not os.path.exists(path2):
		os.mkdir(path3)
		os.mkdir(path2)
	
	chat()