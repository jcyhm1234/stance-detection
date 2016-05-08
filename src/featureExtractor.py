import operator
import numpy as np

from CMUTweetTagger import runtagger_parse
from nltk.classify.util import apply_features
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

from lexicons import SubjLexicon, LiuLexicon
from dataManager import DataManager
class FeatureExtractor:
	def __init__(self, data):
		self.data = data
		self.corpus = None

		self.liu = LiuLexicon()
		self.subj = SubjLexicon()
		self.buildTweetCorpus()
		self.initEncoders()

	def buildTweetCorpus(self):
		self.corpus = []
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for sample in dataset:
				words = [w for w in sample[0]]
				self.corpus.extend(words)
		self.corpus = set(self.corpus)
		print 'Built corpus'

	def initEncoders(self):

		self.topicenc = preprocessing.LabelEncoder()
		self.topicenc.fit(["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Donald Trump", "Hillary Clinton", "Legalization of Abortion"])

		self.labelenc = preprocessing.LabelEncoder()
		self.labelenc.fit(["NONE","FAVOR","AGAINST"])
		
		i = 0
		# j = 0
		# k = 0
		self.word2i = {}
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for row in dataset:
				#tuple of tweetwords, topic, label
				for w in row[0]:
					if w not in self.word2i:
						self.word2i[w] = i
						i+=1
		# 		if row[1] not in self.topic2i:
		# 			self.topic2i[row[1]] = j
		# 			j+=1
		# 		if row[2] not in self.label2i:
		# 			self.label2i[row[2]] = k
		# 			k+=1

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

	def getTweetFeatures(self, sample, listOfFeats):
		#sample is tuple of tweet_words, topic, label
		#list of feats can include 'words','topic','label','lexiconsbyword'
		features = {}
		tweet_words = set(sample[0])
		
		if 'topic' in listOfFeats:
			features['topic'] = sample[1]

		#Gets the tweet feature for a single tweet
		for word in self.corpus:
			if 'words' in listOfFeats:
				features['contains({})'.format(word)] = (word in tweet_words)
			if 'lexiconsbyword' in listOfFeats:
				features['subj({})'.format(word)] = (self.getSubjectivity(word, tweet_words))
				features['pol({})'.format(word)] = (self.getPolarity(word, tweet_words))
				features['senti({})'.format(word)] = (self.getLiuSentiment(word, tweet_words))

		return features

	def getFeatures(self, mode, listOfFeats, y_feat):
		"""
		labeled = True : returns[ (sample_features_dict, y_value),(.),..]
				  False : returns [sample_features_dict,...]
		"""
		features = []
		if mode=='train':
			dataset = self.data.trainTweets
		elif mode=='test':
			dataset = self.data.testTweets
		
		for t in dataset:
			# t is of form ([preprocessedwords],topic, stance)
			feats = self.getTweetFeatures(t, listOfFeats)
			if y_feat:
				if y_feat=='topic':
					y = t[1]
				elif y_feat=='stance':
					y = t[2]
				features.append((feats,y))
			else:
				features.append(feats)
		return features

	def getWordIndex(self, word):
		return self.word2i[word]
		# return self.wordenc.transform([word])[0]
	
	#this method is 20 times faster than above dictionary method
	def getFeaturesMatrix(self, mode, listOfFeats, y_feat):
		if mode=='train':
			dataset = self.data.trainTweets
		elif mode=='test':
			dataset = self.data.testTweets
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
			elif feat=='lexiconsbyword':
				lex_f = np.zeros((len(dataset),len(self.corpus)*3))
				for count, sample in enumerate(dataset):
					for word in sample[0]:
						lex_f[count][self.getWordIndex(word)] += self.getSubjectivity(word, sample[0])
						lex_f[count][self.getWordIndex(word)+len(self.corpus)] += self.getPolarity(word, sample[0])
						lex_f[count][self.getWordIndex(word)+2*len(self.corpus)] += self.getLiuSentiment(word, sample[0])
				features.append(lex_f)
			else:
				print 'Feature not recognized'
		features = np.concatenate(tuple(features), axis=1)

		if y_feat:
			y = []
			if y_feat=='topic':
				for count, sample in enumerate(dataset):
					y.append(sample[1])
				y = self.topicenc.transform(y)
			elif y_feat=='stance':
				for count, sample in enumerate(dataset):
					y.append(sample[2])
				y = self.labelenc.transform(y)
			# y = y.reshape(y.shape[0],1)

			# print 'Shape ', features.shape, y.shape
			return features, y
		else:
			# print 'Shape of features', features.shape
			return features

if __name__ == '__main__':
	dp = DataManager('../data/train.csv','../data/test.csv')
	fe = FeatureExtractor(dp)
	fe.getFeaturesMatrix('train',['words'],'topic')