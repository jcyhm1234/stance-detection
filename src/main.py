from dataManager import DataManager
from featureExtractor import FeatureExtractor
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class StanceDetector:
	def __init__(self):
		self.data = DataManager('../data/train.csv','../data/test.csv')
		self.fe = FeatureExtractor(self.data)

	def buildBaseline(self, model):
		print 'Training baseline',model
		feats = ['words']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		for mode in ['simple','tfidf']:
			if model=='bayes':
				cl = MultinomialNB()
			elif model=='svm':
				cl = LinearSVC()

			if mode=='tfidf':
				X = TfidfTransformer().fit_transform(X).toarray()
				X_test = TfidfTransformer().fit_transform(X_test).toarray()
			
			clf = cl.fit(X, y)
			y_pred = clf.predict(X_test)
			print '\t',mode, accuracy_score(y_true, y_pred)

	def build(self, model):
		print 'Training NB '
		feats = ['words','lexiconsbyword','topic']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		for mode in ['simple','tfidf']:
			if model=='bayes':
				cl = MultinomialNB()
			elif model=='svm':
				cl = LinearSVC()

			if mode=='tfidf':
				X = TfidfTransformer().fit_transform(X).toarray()
				X_test = TfidfTransformer().fit_transform(X_test).toarray()
			
			clf = cl.fit(X, y)
			y_pred = clf.predict(X_test)
			print '\t',mode, accuracy_score(y_true, y_pred)

	def buildSeparate(self):
		#builds two separate for topic and stance
		#WIP
		
		feats = ['words','lexiconsbyword']
		y_attribute = 'topic'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)

		topic_clf = SVC(probability=True)
		topic_clf = topic_clf.fit(X,y)

		y_topic_proba_pred = topic_clf.predict_proba(X)

		# Get probablility dist over all labels
		# dist = topic_clf.prob_classify_many(test_features)
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
		
		stance_clf = SVC()
		feats = ['words','lexiconsbyword','topic']
		y_attribute = 'stance'
		X_stance,y_stance = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test_stance,y_true_stance = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		stance_clf = stance_clf.fit(X_stance,y_stance)
		stance_pred = stance_clf.predict(X_test_stance)
		
		for i in range(0, len(stance_pred)):
			if boost_factors[i] == 2:
				stance_pred[i] = self.fe.labelenc.transform(["NONE"])[0]
		
		#calculate accuracy
		pred_labels = self.fe.labelenc.inverse_transform(stance_pred)
		print [(self.data.testLabels[i], pred_labels[i] for i in range(len(stance_pred))]
		score = accuracy_score(y_true_stance, stance_pred)
		print score

if __name__=='__main__':
	sd = StanceDetector()
	# sd.buildBaseline('bayes')
	# sd.build('svm')
	sd.buildSeparate()



 
