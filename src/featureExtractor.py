import operator
import numpy as np
from CMUTweetTagger import runtagger_parse
from nltk.classify.util import apply_features
from word2vec import Word2Vec
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from lexicons import SubjLexicon, LiuLexicon
from dataManager import DataManager
from random import shuffle
class FeatureExtractor:
	def __init__(self, data):
		self.data = data
		self.corpus = None
		self.liu = LiuLexicon()
		self.subj = SubjLexicon()
		self.buildTweetCorpus()
		self.word_vec_model = Word2Vec(self.corpus)
		self.initEncoders()

	def buildTweetCorpus(self):
		self.corpus = []
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for sample in dataset:
				words = [w for w in sample[0]]
				self.corpus.extend(words)
		self.corpus = set(self.corpus)
		# print 'Built corpus'
		
	def initEncoders(self):
		self.topicenc = preprocessing.LabelEncoder()
		self.topicenc.fit(["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"])

		self.labelenc = preprocessing.LabelEncoder()
		self.labelenc.fit(["NONE","FAVOR","AGAINST"])		

		i = 0
		self.word2i = {}
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for row in dataset:
				#tuple of tweetwords, topic, label
				for w in row[0]:
					if w not in self.word2i:
						self.word2i[w] = i
						i+=1
		
		postags = ["N","O","S","^","Z","L","M","V","A","R","!","D","P","&","T","X","Y","#","@","~","U","E","$",",","G"]
		self.pos2i = {}
		i=0
		for tag in postags:
			self.pos2i[tag] = i
			i+=1
		
	def getSubjectivity(self, word, tweetwords):
		if word in tweetwords:
			return self.subj.getWordSubjectivity(word)
		else:
			return 0

	def getPolarity(self, word, tweetwords):
		if word in tweetwords:
			return self.subj.getWordPolarity(word)
		else:
			return 0

	def getLiuSentiment(self, word, tweetwords):
		if word in tweetwords:
			return self.liu.getWordFeatures(word)
		else:
			return 0

	def getWord2Vec(self, tweetwords):
		return self.word_vec_model.getFeatureVectorsFromBinary(tweetwords)

	def getPOSTags(self, tweets):
		tagtuples = runtagger_parse(tweets)
		#[ [(tuple-word,tag,confidence),('word','V',0.99)], [] ]
		pos_feats = np.zeros((len(tweets), len(self.pos2i)))
		for ind, tweettags in enumerate(tagtuples):
			for wordtags in tweettags:
				tag = wordtags[1]
				conf = wordtags[2]
				
				#adds confidence as weight for each tag found in tweet
				pos_feats[ind][self.pos2i[tag]] += conf
				
		return pos_feats

	# def getPosTagWord(self, word, doc, pos_tags):
	# 	if word not in doc:
	# 		return ''
	# 	else:
	# 		return pos_tags[word]

	# def getTweetFeatures(self, sample, listOfFeats):
	# 	#sample is tuple of tweet_words, topic, label
	# 	#list of feats can include 'words','topic','label','lexiconsbyword'
	# 	features = {}
	# 	tweet_words = set(sample[0])
		
	# 	if 'topic' in listOfFeats:
	# 		features['topic'] = sample[1]

	# 	#Gets the tweet feature for a single tweet
	# 	for word in self.corpus:
	# 		if 'words' in listOfFeats:
	# 			features['contains({})'.format(word)] = (word in tweet_words)
	# 		if 'lexiconsbyword' in listOfFeats:
	# 			features['subj({})'.format(word)] = (self.getSubjectivity(word, tweet_words))
	# 			features['pol({})'.format(word)] = (self.getPolarity(word, tweet_words))
	# 			features['senti({})'.format(word)] = (self.getLiuSentiment(word, tweet_words))

	# 	return features

	# def getFeatures(self, mode, listOfFeats, y_feat):
	# 	"""
	# 	labeled = True : returns[ (sample_features_dict, y_value),(.),..]
	# 			  False : returns [sample_features_dict,...]
	# 	"""
	# 	features = []
	# 	if mode=='train':
	# 		dataset = self.data.trainTweets
	# 	elif mode=='test':
	# 		dataset = self.data.testTweets
	# 	for t in dataset:
	# 		# t is of form ([preprocessedwords],topic, stance)
	# 		feats = self.getTweetFeatures(t, listOfFeats)
	# 		if y_feat:
	# 			if y_feat=='topic':
	# 				y = t[1]
	# 			elif y_feat=='stance':
	# 				y = t[2]
	# 			features.append((feats,y))
	# 		else:
	# 			features.append(feats)
	# 	return features

	def getWordIndex(self, word):
		return self.word2i[word]
		# return self.wordenc.transform([word])[0]
	
	def getDataset(self, mode, topic='All', includeGivenTopic=True):
		if mode=='train':
			dataset = self.data.trainTweets
		elif mode=='test':
			dataset = self.data.testTweets
		if topic=='All':
			return dataset
		elif includeGivenTopic:
			#includes only given topic
			d = [row for row in dataset if row[1]==topic]
			return d
		else:
			#includes all except given topic
			d = [row for row in dataset if row[1]!=topic]
			shuffle(d)
			return d

	def getX(self, mode, dataset, listOfFeats):
		features = []
		for feat in listOfFeats:
			if feat=='words':
				word_f = np.zeros((len(dataset),len(self.corpus)))
				for count, sample in enumerate(dataset):
					for word in sample[0]:
						word_f[count][self.getWordIndex(word)] += 1
				features.append(word_f)
			
			elif feat=='topic':
				topics = []
				for count, sample in enumerate(dataset):
					topics.append(sample[1])
				topic_f = self.topicenc.transform(topics)
				topic_f = topic_f.reshape(topic_f.shape[0],1)
				features.append(topic_f)

			elif feat=='topic1hot':
				topics = []
				topics_f = np.zeros((len(dataset),5))
				for sample in dataset:
					topics.append(sample[1])
				topic_ind = self.topicenc.transform(topics)
				for ind,t in enumerate(topic_ind):
					topics_f[ind][t] = 1
				features.append(topics_f)
			
			elif feat=='lexiconsbyword':
				lex_f = np.zeros((len(dataset),len(self.corpus)*3))
				for count, sample in enumerate(dataset):
					for word in sample[0]:
						lex_f[count][self.getWordIndex(word)] += self.getSubjectivity(word, sample[0])
						lex_f[count][self.getWordIndex(word)+len(self.corpus)] += self.getPolarity(word, sample[0])
						lex_f[count][self.getWordIndex(word)+2*len(self.corpus)] += self.getLiuSentiment(word, sample[0])
				features.append(lex_f)

			elif feat=='pos':
				#needs actual text, not tokenized list
				if mode=='train':
					features.append(self.getPOSTags(self.data.trainTweetsText))
				elif mode=='test':
					features.append(self.getPOSTags(self.data.testTweetsText))
			
			elif feat=='words2vec':
				vecs = []
				for tweet in dataset:
					vecs.append(self.getWord2Vec(tweet[0]))
				features.append(np.asarray(vecs))

			# elif feat=='topunigrams':


			else:
				print 'Feature not recognized'
		features = np.concatenate(tuple(features), axis=1)
		return features

	def getY(self, mode, dataset, y_feat):
		y = []
		if y_feat=='topic':
			for count, sample in enumerate(dataset):
				y.append(sample[1])
			y = self.topicenc.transform(y)
		elif y_feat=='stance':
			for count, sample in enumerate(dataset):
				y.append(sample[2])
			y = self.labelenc.transform(y)
		return y

	#this method is 20 times faster than above dictionary method
	def getFeaturesMatrix(self, mode, listOfFeats, y_feat, topic='All'):
		dataset = self.getDataset(mode, topic)
		X = self.getX(mode, dataset, listOfFeats)
		if y_feat:
			y = self.getY(mode, dataset, y_feat)
			rval = (X, y)
		else:
			rval = X
		return rval

	def getFeaturesTopicNontopic(self, mode, listOfFeats, y_feat, topic):
		rv = self.getFeaturesMatrix(mode, listOfFeats, y_feat, topic)
		if y_feat:
			X,y = rv
		else:
			X = rv
		#if a specific topic given, also adds negative samples
		nonTopicDataset = self.getDataset(mode, topic, includeGivenTopic=False)
		
		X_nontopic = self.getX(mode, nonTopicDataset, listOfFeats)[:len(X),:]
		X = np.concatenate((X,X_nontopic),axis=0)
		if y_feat:
			y_nontopic = np.empty(len(y))
			y_nontopic.fill(5)
			y = np.concatenate((y,y_nontopic),axis=0)
			rval = (X , y)
		else:
			rval = X
		return rval

if __name__ == '__main__':
	dp = DataManager('../data/train.csv','../data/test.csv')
	fe = FeatureExtractor(dp)
	# print fe.getFeaturesMatrix('train',['words'],'topic','Hillary Clinton')[0].shape
	# print fe.getFeaturesTopicNontopic('train',['words'],'topic', 'Hillary Clinton')[0].shape
	# print fe.getX('train',fe.data.trainTweets, ['words2vec']).shape