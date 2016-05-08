from dataManager import DataManager
from featureExtractor import FeatureExtractor
from nltk.classify import NaiveBayesClassifier as nbclf
from nltk.classify import accuracy
from nltk import classify
class StanceDetector:
	def __init__(self):
		self.data = DataManager('../data/train.csv','../data/test.csv')
		self.featext = FeatureExtractor(self.data)

	def buildBaseline(self):
		X = self.featext.getFeatures('train',True)
		print 'Training baseline...'
		clf = nbclf.train(X)
		print 'Training done'
		print 'Testing...'
		# print clf.classify(self.featext.getTweetFeatures(self.data.testTweets[0][0])), self.data.testTweets[0][1]
		print accuracy(clf, self.featext.getFeatures('test',True))
		print clf.show_most_informative_features(30)

			
if __name__=='__main__':
	sd = StanceDetector()
	sd.buildBaseline()



 
