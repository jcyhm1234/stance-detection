from dataManager import DataManager
from featureExtractor import FeatureExtractor
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import make_scorer
from evaluate import Evaluate
from pprint import pprint
import numpy as np
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import xgboost as xgb

class StanceDetector:
	def __init__(self):
		self.data = DataManager('../data/train.csv','../data/test.csv')
		self.fe = FeatureExtractor(self.data)
		self.eval = Evaluate()

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
				cl = Pipeline([('tfidf', TfidfTransformer()),
                      ('clf', cl), ])

			clf = cl.fit(X, y)
			y_pred = clf.predict(X_test)
			print mode, accuracy_score(y_true, y_pred)
			pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))


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
				cl = Pipeline([('tfidf', TfidfTransformer()),
                      ('clf', cl), ])
			
			clf = cl.fit(X, y)
			y_pred = clf.predict(X_test)
			print mode, accuracy_score(y_true, y_pred)
			pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))

	#train in name means helper function
	def trainSVC(self, feats, y_attribute, proba=False):
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		
		clf = SVC(probability=proba)
		clf = clf.fit(X,y)
		if proba:
			y_proba = clf.predict_proba(X_test)
			return clf, y_proba
		else:
			y_pr = clf.predict(X_test)
			return clf, y_pr
	
	def trainLinearSVC(self, feats, y_attribute, dec=False):
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		
		clf = LinearSVC()
		clf = clf.fit(X,y)
		if dec:
			y_pr = clf.decision_function(X_test)
			return clf, y_pr
		else:
			y_pr = clf.predict(X_test)
			return clf, y_pr

	#TODO: revisit
	#check lable transform encodings of NONE, FAVOR, AGAINST
	# def buildTopicStanceSeparate(self):
	# 	feats = ['words']
	# 	y_attribute = 'stance'
	# 	X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)

	# 	#builds two separate for topic and stance
	# 	topic_clf, y_topic_proba = self.trainLinearSVC(feats = ['words','lexiconsbyword'],y_attribute = 'topic',dec=True)
		
	# 	#WRONR
	# 	#WRONG
	# 	#WRONG
	# 	boost_factors = np.ones_like(y_true)
	# 	#multiply by NONE (0) = 0
	# 	#multiply by FAVOR (1) = 1
	# 	#multiply by AGAINST (2) = 2

	# 	#has index of class with max prob for each sample
	# 	topic_preds = np.argmax(y_topic_proba,axis=1)
	# 	for ind,s in enumerate(y_topic_proba):
	# 		prob = y_topic_proba[ind][topic_preds[ind]]
	# 		if prob < 0.4:
	# 			boost_factors[ind] = 0 #corresponds to NONE
		
	# 	stance_clf,stance_pred = self.trainLinearSVC(feats = ['words','lexiconsbyword','topic'],y_attribute = 'stance')		
		
	# 	# for i in range(0, len(stance_pred)):
	# 	# 	if boost_factors[i] == 2:
	# 	# 		stance_pred[i] = self.fe.labelenc.transform("NONE")
		
	# 	#with numpy arrays now, above is equivalent to below , right?
	# 	stance_pred = np.multiply(stance_pred, boost_factors)
	# 	stance_pred_labels = self.fe.labelenc.inverse_transform(stance_pred)

	# 	# print [(self.data.testLabels[i], stance_pred_labels[i]) for i in range(len(stance_pred))]
	# 	score = accuracy_score(y_true, stance_pred)
	# 	print score
	# 	pprint(self.eval.computeFscores(self.data.testTweets, stance_pred_labels))

	def buildTopicOnlyMultiple(self):
		#one svm for each topic
		feats = ['words']
		y_attribute = 'topic'
		clf_topic = {}
		for topic in list(self.fe.topicenc.classes_):
			X,y = self.fe.get('train',feats,y_attribute, topic)
			# Xt,yt = self.fe.getFeaturesTopicNontopic('test',feats,y_attribute, topic)
			clf = LinearSVC()
			clf = clf.fit(X,y)
			clf_topic[topic] = clf
			print topic, clf.score(Xt,yt)

		# not useful. still less than single SVM. but not as much as avg of above

		# X_whole,y_whole = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		# Xt,yt = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		# newX = []
		# newXt = []
		# for topic in clf_topic:
		# 	newX.append(clf_topic[topic].transform(X_whole))
		# 	newXt.append(clf_topic[topic].transform(Xt))
		# newX = np.concatenate(tuple(newX),axis=1)
		# newXt = np.concatenate(tuple(newXt),axis=1)
		# newclf = LinearSVC()
		# newclf = newclf.fit(newX, y_whole)
		# print newclf.score(newXt, yt)

	def trainTopicSVM(self, topic):
		feats = ['words2vec','clusteredLexicons','topic1hot']
		y_attribute = 'stance'
		
		X,y = self.fe.getFeaturesTopicNontopic('train',feats,y_attribute, topic=topic)
		X_test,y_true = self.fe.getFeaturesTopicNontopic('test',feats,y_attribute, topic=topic)
		clf = LinearSVC(C=0.01)
		clf = clf.fit(X,y)
		print topic #,clf.score(X_test, y_true)
		return clf
	
	#shit
	def buildTopicWise(self):
		#separate SVC for each topic, tests on that class only first, then on all
		topic_clf = {}
		feats = ['words2vec','clusteredLexicons','topic1hot']
		
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,y_attribute)

		#X matrix for new classifier which uses this as train matrix
		#has columns of each topic classifier's confidence function
		# X_fx = []
		# X_ftestx = []
		preds = []
		for topic in list(self.fe.topicenc.classes_):
			topic_clf[topic] = self.trainTopicSVM(topic)
			preds.append(topic_clf[topic].predict(X_test))
			# X_fx.append(topic_clf[topic].decision_function(X))
			# X_ftestx.append(topic_clf[topic].decision_function(X_test))

		allpreds = np.vstack(tuple(preds))
		topic1hot, temp = self.fe.getFeaturesMatrix('test',['topic1hot'],'stance')
		# print allpreds.shape, topic1hot.T.shape
		allpreds[allpreds==5] = 1
		final_pred = np.multiply(topic1hot.T,allpreds)

		prediction = np.sum(final_pred, axis=0).astype(int)
		# X_fx = np.concatenate(tuple(X_fx), axis=1)
		# X_ftestx = np.concatenate(tuple(X_ftestx), axis=1)
		# clf = LinearSVC().fit(X_fx, y)
		# y_pred = clf.predict(X_ftestx)
		print accuracy_score(y_true, prediction)
		pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(prediction)))


	#GOOD 66%acc
	#1.2 % increase with change topic to 1hot
	def buildSVMWord2Vec(self):
		feats = ['words2vec','topic1hot','pos']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		Xt,yt = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		clf = LinearSVC(C=0.01,penalty='l1',dual=False)
		clf = clf.fit(X,y)
		y_pred = clf.predict(Xt)
		print clf.score(Xt, yt)
		pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))

	def buildSVMTrial(self):
		feats = ['topic1hot','words2vec']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		Xt,yt = self.fe.getFeaturesMatrix('test',feats,y_attribute)		
		clf = LinearSVC(C=0.001)
		clf = clf.fit(X,y)
		y_pred = clf.predict(Xt)
		print clf.score(Xt, yt)
		pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))
	
	def buildTrial(self):
		# feats = ['pos','words2vec','clusteredLexicons','topic1hot']
		feats = ['givenSentiment','givenOpinion','words2vec','pos','clusteredLexicons','top1grams','top2grams']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		Xt,yt = self.fe.getFeaturesMatrix('test',feats,y_attribute)		
		# clf = DecisionTreeClassifier()
		# clf = LogisticRegression()
		clf = LinearSVC(C=1, class_weight='balanced', penalty='l1',dual=False)
		clf = clf.fit(X,y)
		y_pred = clf.predict(Xt)
		# print y_pred
		print len(np.where(y_pred==0)[0]),len(np.where(y_pred==1)[0]),len(np.where(y_pred>1)[0])
		print len(y_pred)
		print 'training accuracy',clf.score(X, y)
		print clf.score(Xt, yt)
		pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))

	def buildGithubSGDModel(self):
		# feats = ['words2vec','topic1hot','pos']
		y_attribute = 'stance'
		dataset = self.fe.getDataset('train')
		dataset2 = self.fe.getDataset('test')
		y_train = self.fe.getY('train',dataset, y_attribute)
		y_test = self.fe.getY('train',dataset2, y_attribute)

		tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=1.0, min_df=1, binary=True, norm='l2', use_idf=True, smooth_idf=False, sublinear_tf=True, encoding='latin1')
		
		X_train = tfidf.fit_transform(self.data.trainTweetsText)
		X_test = tfidf.transform(self.data.testTweetsText)
		tuned_parameters = {'alpha': [10 ** a for a in range(-12, 0)]}
		clf = GridSearchCV(SGDClassifier(loss='hinge', penalty='elasticnet',l1_ratio=0.75, n_iter=10, shuffle=True, verbose=False, n_jobs=4, average=False)
                      , tuned_parameters, cv=10, scoring='f1_weighted')

		clf.fit(X_train, y_train)
		print clf.best_params_
		print("Grid scores on development set:")
		for params, mean_score, scores in clf.grid_scores_:
			print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
		print classification_report(y_test, clf.predict(X_test))
		print clf.score(X_test, y_test)


	def getGridSearchParams(self):
		param_grid = [
				{'C': [0.001, 0.01, 0.1, 1], 'dual':[False, True]}
		 ]
		return param_grid

	def getGridSearchParamsForXGBoost(self):
		param_grid = [
			{'n_estimators':[10,20,30,40,50], 'max_depth': [1,2,3,4,5]}
		]

	def buildSVMWord2VecWithClusters(self):
		#feats = ['topic1hot']
		#feats = ['words2vec', 'top1grams', 'top2grams']
		#feats = ['words2vec', 'top1grams']
		#feats = ['words2vec', 'top2grams']
		feats = ['words2vec','topic1hot', 'pos','clusteredLexicons', 'top2grams']
		#feats = ['clusteredLexicons']
		#feats = ['pos']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		print (X.shape)
		Xt,yt = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		clf = LinearSVC(C=1,penalty='l1',dual=False)
		clf = clf.fit(X,y)
		y_pred = clf.predict(Xt)
		f = open('pred','w')
		for i in y_pred:
			#print type(i)
			f.write('{0}'.format(i))
		f.close()
		print clf.score(Xt, yt)
		pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))

	def buildSVMWord2VecWithClustersGridSearch(self):
		feats = ['words2vec','topic1hot','pos', 'clusteredLexicons']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		Xt,yt = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		
		svmclf = LinearSVC(C=0.01,penalty='l1',dual=False)
		clf = GridSearchCV(svmclf, self.getGridSearchParams())
		clf = clf.fit(X,y)
		print clf.best_params_

		y_pred = clf.predict(Xt)
		
		print clf.score(Xt, yt)
		pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))

	def trainStanceNone(self, feats):
		# feats = ['words2vec','topic1hot','pos']
		X,y = self.fe.getFeaturesStanceNone('train',feats)
		Xt,yt = self.fe.getFeaturesStanceNone('test',feats)
		svmclf = LinearSVC()
		stance_none_clf = GridSearchCV(svmclf, self.getGridSearchParams(), scoring='f1_macro').fit(X, y)
		# print stance_none_clf.score(Xt, yt)
		pred = stance_none_clf.predict(Xt)
		print classification_report(yt, pred)
		return stance_none_clf

	def trainFavorAgainst(self,feats):
		# feats = ['words2vec','topic1hot','pos']
		X,y = self.fe.getFeaturesFavorAgainst('train',feats)
		Xt,yt = self.fe.getFeaturesFavorAgainst('test',feats)
		svmclf = LinearSVC()
		fav_agnst_clf = GridSearchCV(svmclf, self.getGridSearchParams(), scoring='f1_macro').fit(X, y)
		pred = fav_agnst_clf.predict(Xt)
		print classification_report(yt, pred)

		# print fav_agnst_clf.score(Xt, yt)
		return fav_agnst_clf

	def buildModel2(self):
		#one SVM for Stance/None and other for Favor/Against
		feats = ['words2vec','topic1hot','pos','top2grams','top1grams']
		stance_none_clf = self.trainStanceNone(feats)
		fav_agnst_clf = self.trainFavorAgainst(feats)
		X_test,y_true = self.fe.getFeaturesMatrix('test',feats,'stance')
		st_pred = stance_none_clf.predict(X_test)
		favaga_pred = fav_agnst_clf.predict(X_test)
		for index,row in enumerate(st_pred):
			if row==3:
				st_pred[index] = favaga_pred[index]
		print classification_report(y_true, st_pred)
		print accuracy_score(y_true, st_pred)
		# assert(stance_none_clf.classes_[1]==3) #stance(3)
		# # >0 means this class - stance will be predicted
		# # <0 means none is predicted
		# confi = stance_none_clf.decision_function(X_test)
		# # treat as confident about none if confi<-0.25:
		# y_pred = fav_agnst_clf.predict(X_test)
		# print accuracy_score(y_true, y_pred)
		# pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))
		# threshold = -0.25
		# confi_high = np.where(confi<threshold)[0]
		# for loc in confi_high:
		# 	y_pred[loc] = self.fe.labelenc.transform('NONE')
		# print 'Boosted', accuracy_score(y_true, y_pred)
		# print len(np.where(y_pred==0)[0]),len(np.where(y_pred==1)[0]), len(np.where(y_pred==2)[0]),
		# pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))
		
	def get_proba_one(self, model, X):
	    predicted = model.predict_proba(X)
	    return predicted[:, 1]

	def runXGBoostModel(self,model, model_name, X, target, X_test, crossOn):
	    print "Trying to fit model"
	    print X.shape, target.shape
	    model.fit(X, target)
	    print "Successfully fit model"
	    predicted = self.get_proba_one(model, X)
	    predicted_test = self.get_proba_one(model, X_test)
	    predicted_test = model.predict(X_test)
	    print predicted_test
	    return predicted_test


	def word2VecXGBoost(self):
		feats = ['words2vec','pos','clusteredLexicons', 'top1grams','top2grams', 'topic1hot' ]
		#feats = ['words2vec']
		#feats = ['clusteredLexicons']
		#feats = ['pos']
		y_attribute = 'stance'
		X,y = self.fe.getFeaturesMatrix('train',feats,y_attribute)
		print (X.shape)
		Xt,yt = self.fe.getFeaturesMatrix('test',feats,y_attribute)
		#clf = LinearSVC(C=0.01,penalty='l1',dual=False)
		#clf = clf.fit(X,y)
		#y_pred = clf.predict(Xt)
		# f = open('pred','w')
		# for i in y_pred:
		# 	#print type(i)
		# 	f.write('{0}'.format(i))
		# f.close()
		#print clf.score(Xt, yt)
		#pprint(self.eval.computeFscores(self.data.testTweets, self.fe.labelenc.inverse_transform(y_pred)))
		m2_xgb = xgb.XGBClassifier(n_estimators=10, nthread=-1, max_depth = 2 	, seed=500)
		#m2_xgb = GridSearchCV(m2_xgb, self.getGridSearchParamsForXGBoost())
		print "Run Model"
		y_pred = self.runXGBoostModel(m2_xgb, "m2_xgb_OS_ENN", X, y, Xt, True)
		# print type(yt)
		# print type(y_pred)
		# print len(yt)
		# print len(y_pred)
		# print yt.shape
		# print y_pred.shape
		# print yt
		# print y_pred
		# print(m2_xgb)
		print accuracy_score(yt, y_pred)

if __name__=='__main__':
	sd = StanceDetector()
	# sd.buildBaseline('bayes')
	# sd.buildSimple('svm')
	# sd.buildTopicStanceSeparate()
	# sd.buildTopicWise()
	# sd.buildTopicOnlyIndiv()
	# sd.buildTopicOnlySingle()
	#sd.buildSVMWord2Vec()
	# sd.buildStanceNone()
	# sd.trainStanceNone()
	# sd.trainFavorAgainst()
	# sd.buildModel2()
	# sd.buildSVMWord2VecWithClusters()
	# sd.buildSVMWord2VecWithClustersGridSearch()
	# sd.buildTrial()	
	# sd.buildGithubSGDModel()
	#sd.buildModel2()
	#sd.buildSVMWord2VecWithClusters()
	# sd.buildSVMWord2VecWithClustersGridSearch()
	sd.buildTrial()
	# sd.word2VecXGBoost()




 
