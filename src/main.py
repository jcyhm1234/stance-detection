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

	def buildSimple(self, model):
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

	def buildSVC(self, feats, y_attribute, proba=False):
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		
		clf = SVC(probability=proba)
		clf = clf.fit(X,y)
		if prob:
			y_proba = clf.predict_proba(X_test)
			return clf, y_proba
		else:
			y = clf.predict(X_test)
			return clf, y
		

	def buildSeparate(self):
		#builds two separate for topic and stance
		topic_clf, y_topic_proba = self.buildSVC(feats = ['words','lexiconsbyword'],y_attribute = 'topic',proba=True)
		
		boost_factors = np.ones_like(y_true)
		#multiply by NONE (0) = 0
		#multiply by FAVOR (1) = 1
		#multiply by AGAINST (2) = 2

		#has index of class with max prob for each sample
		topic_preds = np.argmax(y_topic_proba,axis=1)
		for ind,s in enumerate(y_topic_proba):
			prob = y_topic_proba[ind][topic_preds[ind]]
			if prob < 0.4:
				boost_factors[ind] = 0 #corresponds to NONE
		
		stance_clf,stance_pred = self.buildSVC(feats = ['words','lexiconsbyword','topic'],y_attribute = 'stance')		
		
		# for i in range(0, len(stance_pred)):
		# 	if boost_factors[i] == 2:
		# 		stance_pred[i] = self.fe.labelenc.transform("NONE")
		
		#with numpy arrays now, above is equivalent to below , right?
		stance_pred = np.multiply(stance_pred, boost_factors)
		stance_pred_labels = self.fe.labelenc.inverse_transform(stance_pred)

		print [(self.data.testLabels[i], pred_labels[i]) for i in range(len(stance_pred))]
		score = accuracy_score(y_true_stance, stance_pred)
		print score

if __name__=='__main__':
	sd = StanceDetector()
	# sd.buildBaseline('bayes')
	# sd.build('svm')
	sd.buildSeparate()



 
