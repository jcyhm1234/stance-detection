from dataManager import DataManager
from featureExtractor import FeatureExtractor
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
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

	# def buildSeparate(self):
	# 	#builds two separate for topic and stance, and later for wikilink/word2vec
	# 	#WIP
	# 	topic_clf = SklearnClassifier(LinearSVC())
	# 	topic_clf = topic_clf.train(self.fe.getFeatures('train',withtopic=False))

	# 	stance_clf = SklearnClassifier(LinearSVC())
	# 	stance_clf = stance_clf.train(self.fe.getFeatures('train',withtopic=True))

	# def stacked(self):
	# 	count_vect = CountVectorizer(decode_error='ignore')
	# 	matrix_counts = self.fe.getFeatures('train', True)
	# 	print matrix_counts
	# 	tf_idf = TfidfTransformer()
	# 	tfidf_matrix = tf_idf.fit_transform(matrix_counts)
	# 	print tfidf_matrix




if __name__=='__main__':
	sd = StanceDetector()
	sd.buildBaseline('svm')
	sd.build('svm')



 
