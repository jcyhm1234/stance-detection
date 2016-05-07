from sklearn.feature_extraction.text import CountVectorizer
from dataPreprocess import DataPreprocess
from lexicon import SubjLexicon, LiuLexicon

class FeatureExtract:
	def __init__(self, data):
		self.data = data
		self.counts = []
		self.count_vect = CountVectorizer(decode_error='ignore')

	def vectorizeFitTransform(self):
		#TODO: Resolve decoding error ?
		self.counts = self.count_vect.fit_transform(self.data)
		# print self.counts
		# print 'Train count shape:'
		# print trainCounts.shape
		# print trainCounts[0]
	
	def vectorizeTransform(self, testData):
		return self.count_vect.transform(testData)

	def getLexiconFeatures(self):
		l = LiuLexicon()
		s = SubjLexicon()
		#TODO: reshape to match the vector created above. right now its only for words in tweet
		self.liu = l.getFeatures(data)
		self.subj = s.getFeatures(data)


if __name__ == '__main__':
	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	fe = FeatureExtract(dp.trainTweets)
	fe.vectorizeFitTransform()
