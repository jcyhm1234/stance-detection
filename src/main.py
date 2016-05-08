from dataManager import DataManager
from featureExtractor import FeatureExtractor
from nltk.classify import NaiveBayesClassifier as nbclf
from nltk.classify import accuracy
from nltk import classify
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
		clf = SklearnClassifier(SVC())
		clf = clf.train(X)
		print 'Training done'
		print 'Testing...'
		print accuracy(clf, self.featext.getFeatures('test',True))
		# self.clf_base_svm = clf

	def buildSVM(self):
		X = self.featext.getFeatures('train')
		clf = SklearnClassifier(SVC())
		clf = clf.train(X)
		print 'Training done'
		print 'Testing...'
		print accuracy(clf, self.featext.getFeatures('test'))
		print clf.classify_many(self.featext.getFeatures('test',labeled=False))
		# self.clf_base_svm = clf
		# print self.featext.temp


	def buildNB(self):
		#with additioanl features
		#simple, no ensemble
		X = self.featext.getFeatures('train')
		clf = nbclf.train(X)
		print 'Training done'
		print 'Testing...'
		print accuracy(clf, self.featext.getFeatures('test'))
		print clf.show_most_informative_features(30)

	def buildSeparate(self):
		#builds two separate for topic and stance
		#WIP
		test_features = self.featext.getFeatures('test',labeled=False)
		topic_clf = SklearnClassifier(SVC(probability=True))
		topic_clf = topic_clf.train(self.featext.getFeatures('train',withtopic=False))
		# Get probablility dist over all labels
		dist = topic_clf.prob_classify_many(test_features)
		boost_factors = []
		for d in dist:
			# Get the label which has max probablity
			prob = d.prob(d.max())
			boost = 1
			if prob < 0.4:
				boost = 2
			boost_factors.append(boost)
		print 'boost factor length '
		print len(boost_factors)
		stance_clf = SklearnClassifier(SVC())
		stance_clf = stance_clf.train(self.featext.getFeatures('train',withtopic=True))
		test_labels = stance_clf.classify_many(test_features)
		for i in range(0, len(test_labels)):
			if boost_factors[i] == 2:
				test_labels[i] = 'NONE'
		#calculate accuracy
		print self.data.testLabels
		print test_labels
		score = accuracy_score(self.data.testLabels, test_labels)
		print score


if __name__=='__main__':
	# sd = StanceDetector()
	# sd.buildBaselineSVM()

	# sd2 = StanceDetector()
	# sd2.buildSVM()

	sd3 = StanceDetector()
	sd3.buildSeparate()




 
