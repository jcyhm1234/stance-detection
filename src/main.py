from dataManager import DataManager
from featureExtractor import FeatureExtractor
from nltk.classify import NaiveBayesClassifier as nbclf
from nltk.classify import accuracy
from nltk import classify
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC

class StanceDetector:
	def __init__(self):
		self.data = DataManager('../data/train.csv','../data/test.csv')
		self.featext = FeatureExtractor(self.data)

	def buildBaselineNB(self):
		X = self.featext.getFeatures('train',True)
		print 'Training baseline...'
		clf = nbclf.train(X)
		print 'Training done'
		print 'Testing...'
		# print clf.classify(self.featext.getTweetFeatures(self.data.testTweets[0][0])), self.data.testTweets[0][1]
		print accuracy(clf, self.featext.getFeatures('test',True))
		print clf.show_most_informative_features(30)
		self.clf_base_nb = clf

	def buildBaselineSVM(self):
		X = self.featext.getFeatures('train',True)
		print 'Training baseline...'
		clf = SklearnClassifier(LinearSVC())
		clf = clf.train(X)
		print 'Training done'
		print 'Testing...'
		print accuracy(clf, self.featext.getFeatures('test',True))
		# self.clf_base_svm = clf

	def buildSVM(self):
		X = self.featext.getFeatures('train',False)
		print 'Training baseline...'
		clf = SklearnClassifier(LinearSVC())
		clf = clf.train(X)
		print 'Training done'
		print 'Testing...'
		print accuracy(clf, self.featext.getFeatures('test',False))
		print clf.classify_many(self.featext.getFeatures('test',False,False))
		# self.clf_base_svm = clf
		# print self.featext.temp


	def buildNB(self):
		#with additioanl features
		#simple, no ensemble
		X = self.featext.getFeatures('train',False)
		print 'Training baseline...'
		clf = nbclf.train(X)
		print 'Training done'
		print 'Testing...'
		print accuracy(clf, self.featext.getFeatures('test',False))
		print clf.show_most_informative_features(30)

	def buildTwo(self):
		#builds two separate for topic and stance
		pass

if __name__=='__main__':
	# sd = StanceDetector()
	# sd.buildBaselineSVM()

	sd2 = StanceDetector()
	sd2.buildSVM()



 
