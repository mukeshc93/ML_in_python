import numpy as np
import pandas as pd
import random
import nltk
##The data is downladed  from https://archive.org/download/india-names-dataset/ap-names.txt.gz

#The extracted file path has to be pasted below
path="/home/mukesh/Downloads/ap-names-grouped.txt"

d=pd.read_table(path,sep="\t", header=None, usecols=[2,3,4],names=["Male", "Female","Name"])
#The data has number of males and females with names born during a year. I have grouped them by name, removed the count and added the gender
d=d.groupby("Name",as_index=False).sum()
d['gender']=np.where(d["Male"] > d["Female"],1,0)##1 denotes male and 0 is female
d=d[["Name","gender"]]#the final data has just 2 columns viz. name and gender

#Feature extraction
def namefeat(name):
	name=name.lower()
	features={}
	features["first_letter"] = name[0]
	features["last_2"] = name[-2:]
	features["last_3"] = name[-3:]
	features["last_letter"] = name[-1]
	features["last_vowel"] = (name[-1] in 'aeiou')
	features["first_vowel"] = (name[0] in 'aeiou')
	return features

feat = [(namefeat(n)) for n in d["Name"]]

gender=d["gender"]

list3 = [list(a) for a in zip(feat, gender)]##created a list with features and the gender
random.shuffle(list3)

#Training and testing part
#dtrain, dtest = list3[0:2000000], list3[2000001:len(list3)]
#classifier = nltk.NaiveBayesClassifier.train(dtrain)
#print(nltk.classify.accuracy(classifier, dtest))

#training on entire data
classifier = nltk.NaiveBayesClassifier.train(list3)
for i in range(1,10):
	name = input('Enter name to classify: ')
	x=classifier.classify(namefeat(name))
	if x==1:
		print(name, "is a MALE")
	else:
		print(name, "is a FEMALE")
	i = i+1
