from sklearn.feature_extraction.text import CountVectorizer
from dataPreprocess import DataPreprocess
from lexicon import SubjLexicon, LiuLexicon
from CMUTweetTagger import runtagger_parse
from nltk.tokenize import TweetTokenizer

class FeatureExtract:
	def __init__(self, data):
		self.data = data
		self.counts = []
		self.count_vect = CountVectorizer(decode_error='ignore')
		self.corpus = []
		self.tknzr = TweetTokenizer()
		self.document_corpus()
		self.liu = None
		self.subj = None


	#Unused 
	def vectorizeFitTransform(self):
		#TODO: Resolve decoding error ?
		self.counts = self.count_vect.fit_transform(self.data)
		# print self.counts
		# print 'Train count shape:'
		# print trainCounts.shape
		# print trainCounts[0]
	
	#Unused 
	def vectorizeTransform(self, testData):
		return self.count_vect.transform(testData)

	def getLexiconFeatures(self):
		l = LiuLexicon()
		s = SubjLexicon()
		#TODO: reshape to match the vector created above. right now its only for words in tweet
		self.liu = l.getFeatures(self.data)
		self.subj = s.getFeatures(self.data)

	def getPOSTags(self):
		pos = runtagger_parse(self.data)
		return pos
		#[[(tuple-word,tag,confidence),('word','V',0.99)][]]
		#change structure appropriateyl

	def document_corpus(self):
		for tweet in self.data:
			self.corpus.extend(self.tknzr.tokenize(tweet))
		self.corpus = set(self.corpus)


	def document_features(self, document): 
		#Gets the document feature for a single tweet
		word_features = self.corpus
		document_words = set(self.tknzr.tokenize(document))
		features = {}
		for word in word_features:
			features['contains({})'.format(word)] = (word in document_words)
		return features

	def get_BOW_feature(self):
		features = []
		for tweet in self.data:
			features.append(self.document_features(tweet))
		return features

	def get_all_features(self):
		pos = self.getPOSTags()
		self.getLexiconFeatures()
		liu = self.liu
		subj = self.subj
		bow = self.get_BOW_feature()
		print 'Length of pos features ', len(pos[0])
		print 'Length of liu features ', len(liu[0])
		print 'Length of subj features ', len(subj[0])
		print 'Length of bow features ', len(bow[0])



if __name__ == '__main__':
	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	fe = FeatureExtract(dp.trainTweets)
	print 'Extracting features for data'
	fe.get_all_features()
	# fe.vectorizeFitTransform()
	# fe.getPOSTags()
