from sklearn.feature_extraction.text import CountVectorizer
from dataPreprocess import DataPreprocess


class FeatureExtract:
	def __init__(self, trainData):
		self.trainData = trainData

	def vectorize(self):
		#TODO: Resolve decoding error ?
		count_vect = CountVectorizer(decode_error='ignore')
		trainCounts = count_vect.fit_transform(self.trainData)
		print 'Train count shape:'
		print trainCounts.shape
		print trainCounts[0]

if __name__ == '__main__':
	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	dp.loadData()
	dp.dataPreprocess()
	fe = FeatureExtract(dp.trainTweets)
	fe.vectorize()