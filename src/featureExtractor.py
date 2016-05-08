import operator
import numpy as np

from CMUTweetTagger import runtagger_parse
from nltk.classify.util import apply_features
from sklearn.feature_extraction import DictVectorizer

from lexicons import SubjLexicon, LiuLexicon
from dataManager import DataManager

class FeatureExtractor:
	def __init__(self, data):
		self.data = data
		self.corpus = None

		self.liu = LiuLexicon()
		self.subj = SubjLexicon()
		self.buildTweetCorpus()
		self.mapStringToIndex()

	def buildTweetCorpus(self):
		self.corpus = []
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for sample in dataset:
				words = [w for w in sample[0]]
				self.corpus.extend(words)
		self.corpus = set(self.corpus)
		print 'Built corpus'

	def mapStringToIndex(self):
		i = 0
		j = 0
		k = 0
		self.word2i = {}
		self.topic2i = {}
		self.label2i = {}
		for dataset in [self.data.trainTweets, self.data.testTweets]:
			for row in dataset:
				#tuple of tweetwords, topic, label
				for w in row[0]:
					if w not in self.word2i:
						self.word2i[w] = i
						i+=1
				if row[1] not in self.topic2i:
					self.topic2i[row[1]] = j
					j+=1
				if row[2] not in self.label2i:
					self.label2i[row[2]] = k
					k+=1

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
						word_f[count][self.word2i[word]] += 1
				features.append(word_f)
			elif feat=='topic':
				topic_f = np.zeros((len(dataset),1))
				for count, sample in enumerate(dataset):
					topic_f[count] = self.topic2i[sample[1]]
				features.append(topic_f)
			elif feat=='lexiconsbyword':
				lex_f = np.zeros((len(dataset),len(self.corpus)*3))
				for count, sample in enumerate(dataset):
					for word in sample[0]:
						lex_f[count][self.word2i[word]] += self.getSubjectivity(word, sample[0])
						lex_f[count][self.word2i[word]+len(self.corpus)] += self.getPolarity(word, sample[0])
						lex_f[count][self.word2i[word]+2*len(self.corpus)] += self.getLiuSentiment(word, sample[0])
				features.append(lex_f)
			else:
				print 'Feature not recognized'
		features = np.concatenate(tuple(features), axis=1)

		if y_feat:
			y = np.zeros(len(dataset))			
			if y_feat=='topic':
				for count, sample in enumerate(dataset):
					y[count] = self.topic2i[sample[1]]
			elif y_feat=='stance':
				for count, sample in enumerate(dataset):
					y[count] = self.label2i[sample[2]]

			print 'Shape of features', features.shape
			return features, y
		else:
			print 'Shape of features', features.shape
			return features

if __name__ == '__main__':
	dp = DataManager('../data/train.csv','../data/test.csv')
	fe = FeatureExtractor(dp)
	fe.getFeaturesMatrix('train',['words'],'topic')