from dataManager import DataManager
from lexicons import SubjLexicon, LiuLexicon
from CMUTweetTagger import runtagger_parse
from nltk.classify.util import apply_features
from word2vec import Word2Vec
class FeatureExtractor:
	def __init__(self, data):
		self.data = data
		self.corpus = None

		self.liu = LiuLexicon()
		self.subj = SubjLexicon()
		self.word_vec_model = Word2Vec()
		self.buildTweetCorpus()

		self.temp = 0

	def buildTweetCorpus(self):
		self.corpus = []
		for sample in self.data.trainTweets:
			words = [w for w in sample[0] if not w.isdigit()]
			self.corpus.extend(words)
		for sample in self.data.testTweets:
			words = [w for w in sample[0] if not w.isdigit()]
			self.corpus.extend(words)
		self.corpus = set(self.corpus)

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
		return self.word_vec_model.getFeatureVectors(tweetwords)

	# def getPOSTags(self, tweets):
	# 	tagtuples = runtagger_parse(tweets)
		
	# 	tagdict = {}
	# 	for t in tagtuples:
	# 		if t[0] not in tagdict:
	# 			tagdict[t[0]] = (t[1],t[2])
	# 		else:
	# 			print tagdict[t[0]], (t[1], t[2]) 
	# 	return tagdict
	# 	#[[(tuple-word,tag,confidence),('word','V',0.99)][]]
		#change structure appropriatey

	# def getPosTagWord(self, word, doc, pos_tags):
	# 	if word not in doc:
	# 		return ''
	# 	else:
	# 		return pos_tags[word]


	def getTweetFeatures(self, sample, baseline, withtopic):
		# self.temp += 1
		features = {}
		tweet_words = set(sample[:-1])
		
		if withtopic:
			features['topic'] = sample[-1]
		#Gets the tweet feature for a single tweet
		for word in self.corpus:
			features['contains({})'.format(word)] = (word in tweet_words)
			if not baseline:
				features['subj({})'.format(word)] = (self.getSubjectivity(word, tweet_words))
				features['pol({})'.format(word)] = (self.getPolarity(word, tweet_words))
				features['senti({})'.format(word)] = (self.getLiuSentiment(word, tweet_words))
			# features['pos({})'.format(word)] = (self.getPosTagWord(word, tweet_words, pos_tags))
		# Add the word2vec representation for the entire tweet
		features['vector'] = (self.getWord2Vec(tweet_words))
		return features

	def getFeatures(self, mode, for_baseline=False, labeled=True, withtopic=True):
		"""
		labeled = True : returns[ (sample_features_dict, y_value),(.),..]
				  False : returns [sample_features_dict,...]
		"""
		#hack to pass this variable to getTweetFeatures when using lazy loading
		# self.for_baseline = for_baseline

		#lazy handling. doesnt use all samples for some reason
		# if mode=='train':
		# 	return apply_features(self.getTweetFeatures,self.data.trainTweets,labeled)
		# elif mode=='test':
		# 	return apply_features(self.getTweetFeatures,self.data.testTweets,labeled)

		features = []
		if mode=='train':
			for t in self.data.trainTweets:
				# type is (tweetPreprocess(row[0])+[topic], y)
				if withtopic:
					feats = self.getTweetFeatures(t[0], baseline=for_baseline, withtopic=withtopic)
					features.append((feats,t[1]))
					#print 'Features with topic and training ', features
				else:
					feats = self.getTweetFeatures(t[0], baseline=for_baseline, withtopic=withtopic)
					features.append((feats,t[0][-1]))
					# print 'Features without topic and training ', features
		elif mode=='test':
			for t in self.data.testTweets:
				# type is (tweetPreprocess(row[0])+[topic], y)
				feats = self.getTweetFeatures(t[0], baseline=for_baseline, withtopic=withtopic)
				if labeled:
					if withtopic:
						features.append((feats,t[1]))
						# print 'Features with topic and test ', features
					else:
						features.append((feats,t[0][-1]))
						# print 'Features without topic and test ', features
				else:
					features.append(feats)
					# print 'Features without label', features
		return features


if __name__ == '__main__':
	dp = DataManager('../data/train.csv','../data/test.csv')
	fe = FeatureExtractor(dp)
