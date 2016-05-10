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
from glove import Glove
from clusterVectors import Cluster
from itertools import tee, izip

class FeatureExtractor:
	def __init__(self, data):
		self.data = data
		self.corpus = None
		self.liu = LiuLexicon()
		self.subj = SubjLexicon()
		self.buildTweetCorpus()
		self.word_vec_model = Word2Vec(self.corpus)
		self.glove_vec_model = Glove(100, self.corpus)
		self.clusters = Cluster(100)
		self.initEncoders()
		self.topicVecs = self.word_vec_model.getVectorsForTopics(self.topicenc.classes_)
		self.collectTopUnigrams()
		self.collectTopBigrams()

	def buildTweetCorpus(self):
		self.corpus = []
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for sample in dataset:
				words = [w for w in sample[0]]
				self.corpus.extend(words)
		self.corpus = set(self.corpus)
		# print 'Built corpus'
	
	def collectTopUnigrams(self):
		self.counts = {}
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for sample in dataset:
				for w in sample[0]:
					if w in self.counts:
						self.counts[w] +=1
					else:
						self.counts[w] = 1

		sorted_x = sorted(self.counts.items(), key=operator.itemgetter(1))
		self.topunigrams = {}
		i = 0
		for x in sorted_x[:-500]:
			self.topunigrams[x[0]] = i
			i+=1

	def pairwise(self, iterable):
	    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
	    a, b = tee(iterable)
	    next(b, None)
	    return izip(a, b)

	def collectTopBigrams(self):
		self.bigramCounts = {}
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for sample in dataset:
				for u, w in self.pairwise(sample[0]):
					#print u, w
					if (u,w) in self.bigramCounts:
						self.bigramCounts[(u,w)] +=1
					else:
						#print 'Duplicate of ',(u,w)
						self.bigramCounts[(u,w)] = 1					
		sorted_x = sorted(self.bigramCounts.items(), key=operator.itemgetter(1))
		self.topbigrams = {}
		i = 0
		for x in sorted_x[:-500]:
			self.topbigrams[x[0]] = i
			i+=1
				
	# print self.bigramCounts

	def initEncoders(self):
		self.topicenc = preprocessing.LabelEncoder()
		self.topicenc.fit(["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"])

		self.labelenc = preprocessing.LabelEncoder()
		self.labelenc.fit(["NONE","AGAINST","FAVOR"])		

		self.sentenc = preprocessing.LabelEncoder()
		self.sentenc.fit(["pos","neg","other"])		

		self.opintow = preprocessing.LabelEncoder()
		self.opintow.fit(["1.  The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target.","2. The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target.","3.  The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)"])		

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

	def getWords2Vectors(self, words):
		#returns vector for each word
		rval = []
		for word in words:
			rval.append(self.word_vec_model.getVectorForWord(word))
		return rval

	def getPOSTags(self, tweets):
		tagtuples = runtagger_parse(tweets)
		#[ [(tuple-word,tag,confidence),('word','V',0.99)], [] ]
		pos_feats = np.zeros((len(tweets), len(self.pos2i)))
		for ind, tweettags in enumerate(tagtuples):
			for wordtags in tweettags:
				tag = wordtags[1]
				conf = wordtags[2]
				
				#adds confidence as weight for each tag found in tweet
				pos_feats[ind][self.pos2i[tag]] = 1
				
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
			
			elif feat=='givenSentiment':
				sents = []
				if mode=='train':
					sent_f = np.zeros((len(self.data.trainData),3))
					for row in self.data.trainData:
						sents.append(row[4])
				if mode=='test':
					sent_f = np.zeros((len([row for row in self.data.testData if row[1]!='Donald Trump']),3))
					for row in self.data.testData:
						if row[1] !='Donald Trump':
							sents.append(row[4])

				sents = self.sentenc.transform(sents)
				for ind,t in enumerate(sents):
					sent_f[ind][t] = 1
				features.append(sent_f)

			elif feat=='givenOpinion':
				opin = []
				if mode=='train':
					opin_f = np.zeros((len(self.data.trainData),3))
					for row in self.data.trainData:
						opin.append(row[3])
				if mode=='test':
					opin_f = np.zeros((len([row for row in self.data.testData if row[1]!='Donald Trump']),3))
					for row in self.data.testData:
						if row[1] !='Donald Trump':
							opin.append(row[3])

				opin = self.opintow.transform(opin)
				for ind,t in enumerate(opin):
					opin_f[ind][t] = 1
				features.append(opin_f)

			elif feat=='topic':
				topics = []
				for count, sample in enumerate(dataset):
					topics.append(sample[1])
				topic_f = self.topicenc.transform(topics)
				topic_f = topic_f.reshape(topic_f.shape[0],1)
				features.append(np.asarray(topic_f))

			elif feat=='topic1hot':
				topics = []
				topics_f = np.zeros((len(dataset),5))
				for sample in dataset:
					topics.append(sample[1])
				topic_ind = self.topicenc.transform(topics)
				for ind,t in enumerate(topic_ind):
					topics_f[ind][t] = 1
				features.append(topics_f)
			
			elif feat=='topicVecs':
				topics = []
				for tweet in dataset:
					topics.append(self.topicVecs[tweet[1]])
				features.append(np.asarray(topics))
					
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

			elif feat=='glove':
				vecs = []
				for tweet in dataset:
					vecs.append(self.glove_vec_model.getVecOfTweet(tweet[0]))
				features.append(np.asarray(vecs))
			
			elif feat=='polarity':
				vecs = []
				for tweet in dataset:
					vecs.append(self.clusters.getPolarity(self.getWords2Vectors(tweet[0])))
				features.append(np.asarray(vecs))
			
			elif feat=='subjectivity':
				vecs = []
				for tweet in dataset:
					vecs.append(self.clusters.getSubjectivity(self.getWords2Vectors(tweet[0])))
				features.append(np.asarray(vecs))

			elif feat=='sentiment':
				vecs = []
				for tweet in dataset:
					vecs.append(self.clusters.getSentiment(self.getWords2Vectors(tweet[0])))
				features.append(np.asarray(vecs))

			elif feat=='clusteredLexicons':
				vecs = [[],[],[]]
				for tweet in dataset:
					wordsAsVecs = self.getWords2Vectors(tweet[0])
					vecs[0].append(self.clusters.getSentiment(wordsAsVecs))
					vecs[1].append(self.clusters.getSubjectivity(wordsAsVecs))
					vecs[2].append(self.clusters.getPolarity(wordsAsVecs))
				# allvecs = np.concatenate(tuple(vecs))
				for vecsi in vecs:
					#print 'Adding ', np.asarray(vecsi).shape
					features.append(np.asarray(vecsi))

			elif feat=='top1grams':
				vec = np.zeros((len(dataset),len(self.topunigrams)))
				for count, tweet in enumerate(dataset):
					for word in tweet[0]:
						if word in self.topunigrams:
							vec[count][self.topunigrams[word]] += 1
				features.append(vec)

			elif feat=='top2grams':
				vec = np.zeros((len(dataset),len(self.topbigrams)))
				for count, tweet in enumerate(dataset):
					for u, w in self.pairwise(tweet[0]):
						if (u, w) in self.topbigrams:
							#print 'Found ',(u,w)
							vec[count][self.topbigrams[(u, w)]] += 1
				features.append(vec)

			else:
				print 'Feature not recognized'
		print 'Final Feature set size:', len(features)
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

	def getFeaturesFavorAgainst(self, mode, listOfFeats):
		#only tweets with favor or against
		X, y = self.getFeaturesMatrix(mode, listOfFeats, 'stance')
		# print X,y
		nonerows = np.where(y==self.labelenc.transform('NONE'))[0]
		# print y
		# print nonerows
		X = np.delete(X, nonerows, axis=0)

		y = np.delete(y, nonerows)
		return X,y

	def getFeaturesStanceNone(self, mode, listOfFeats):
		X, y = self.getFeaturesMatrix(mode, listOfFeats, 'stance')
		y[y == self.labelenc.transform('FAVOR')] = 3
		y[y == self.labelenc.transform('AGAINST')] = 3
		return X, y

if __name__ == '__main__':
	dp = DataManager('../data/train.csv','../data/test.csv')
	fe = FeatureExtractor(dp)
	# fe.getYStanceNone('train')
	# fe.getFeaturesFavorAgainst('train',['words2vec'])
	# fe.getFeaturesStanceNone('train',['words2vec'])
	# X,y = fe.getFeaturesFavorAgainst('train',['words2vec'])

	# print fe.getFeaturesMatrix('train',['words'],'topic','Hillary Clinton')[0].shape
	# print fe.getFeaturesTopicNontopic('train',['words'],'topic', 'Hillary Clinton')[0].shape
	# print fe.getX('train',fe.data.trainTweets, ['words2vec']).shape