import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import date
import pickle

pathSent = os.path.join(os.getcwd(),r"Models\dataset")

def classify(pathMode,pathDataset):
	print("Training model...")
	sentiments = pd.read_csv(pathDataset)
	tain_data_features = sentiments["Reviews"]
	verctorizer = CountVectorizer(analyzer = "word",
								tokenizer = None,
								preprocessor = None,
								stop_words = None,
								max_features = 1000)
	tain_data_features = verctorizer.fit_transform(tain_data_features)
	tain_data_features = tain_data_features.toarray()

	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(tain_data_features,sentiments["sentiment"])

	pickle.dump(forest,open(pathMode,'wb'))


def alignDataForvariousModels():
	dict_Model_dataset = os.listdir(pathSent)
	data= []
	model = []
	for dataset in dict_Model_dataset:
		inst = os.path.join(os.getcwd()+r"\Models\dataset")+r"\\"+dataset
		data.append(inst)
		inst = os.path.join(os.getcwd()+r"\Models")+r"\\"+dataset
		model.append(inst.split(".")[0]+".pickle")

	return data,model

if __name__ == '__main__':
	data,model = alignDataForvariousModels()
	for i in range(len(data)):
		classify(model[i],data[i])

	print("All models trained!!!")


#COMPLETED