from __future__ import division
from sklearn.naive_bayes import MultinomialNB
from dataPreprocess import DataPreprocess
from featureExtraction import FeatureExtract
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np
class Classifiers:

	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	
	def __init__(self):
		self.nbBOW = None
		self.fe = None

	def classifyNBBagOfWords(self):	
		self.fe = FeatureExtract(Classifiers.dp.trainTweets)
		self.fe.vectorizeFitTransform()
		self.nbBOW = MultinomialNB().fit(self.fe.counts, Classifiers.dp.trainLabels)

	def predict(self):
		predicted = self.nbBOW.predict(self.fe.vectorizeTransform(Classifiers.dp.testTweets))
		print predicted
		return predicted

	def calculateAccuracy(self,predicted):
		correctPredictions = 0
		for i,j in zip(predicted, Classifiers.dp.testLabels):
			if i == j:
				correctPredictions += 1
		return correctPredictions/len(predicted)

	def NBPipeline(self):
		text_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()), ])
		return text_clf

	def SVMPipeline(self):
		text_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)), ])
		return text_clf

	def usingPipeline(self):
		text_clf = self.SVMPipeline()
		text_clf = text_clf.fit(Classifiers.dp.trainTweets, Classifiers.dp.trainLabels)
		predicted = text_clf.predict(Classifiers.dp.testTweets)
		return np.mean(predicted == Classifiers.dp.testLabels)


if __name__ == '__main__':
	c = Classifiers()
	c.classifyNBBagOfWords()
	accuracy = c.calculateAccuracy(c.predict())
	print 'Accuracy:', accuracy 
	print 'using Pipeline', c.usingPipeline()

