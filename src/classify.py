from __future__ import division
from sklearn.naive_bayes import MultinomialNB
from dataPreprocess import DataPreprocess
from featureExtraction import FeatureExtract

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


if __name__ == '__main__':
	c = Classifiers()
	c.classifyNBBagOfWords()
	accuracy = c.calculateAccuracy(c.predict())
	print 'Accuracy:', accuracy 

