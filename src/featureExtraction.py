from sklearn.feature_extraction.text import CountVectorizer
from dataPreprocess import DataPreprocess


class FeatureExtract:
	def __init__(self, data):
		self.data = data
		self.counts = []
		self.count_vect = CountVectorizer(decode_error='ignore')

	def vectorizeFitTransform(self):
		#TODO: Resolve decoding error ?
		self.counts = self.count_vect.fit_transform(self.data)
		# print 'Train count shape:'
		# print trainCounts.shape
		# print trainCounts[0]

	def vectorizeTransform(self, testData):
		return self.count_vect.transform(testData)

if __name__ == '__main__':
	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	fe = FeatureExtract(dp.trainTweets)
	fe.vectorize()