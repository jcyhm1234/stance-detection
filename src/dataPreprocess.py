import json
import csv
import re
# check split hashtg
# TODO: handle punctuation
# TODO: remove numbers
# TODO: contractions

class DataPreprocess:
	
	def __init__(self, trainFile, testFile):
		self.trainFile = trainFile
		self.testFile = testFile
		self.trainData = []
		self.testData = []
		self.trainTweets = []
		self.testTweets = []
		self.trainLabels = []
		self.testLabels = []
		self.loadData()
		self.dataPreprocess()

	def loadData(self):
		with open(self.trainFile, 'r') as f:
			self.trainData = [row for row in csv.reader(f.read().splitlines())]
		with open(self.testFile, 'r') as f:
			self.testData = [row for row in csv.reader(f.read().splitlines())]
		# remove the header from the data
		self.trainData.pop(0)
		self.testData.pop(0)
		print('Loaded %s training samples', len(self.trainData))
		print('Loaded %s testing samples', len(self.testData))

	def dataPreprocess(self):
		self.trainTweets = [tweetPreprocess(row[0]) for row in self.trainData]
		self.testTweets = [tweetPreprocess(row[0]) for row in self.testData]
		self.trainLabels = [row[2] for row in self.trainData]
		self.testLabels = [row[2] for row in self.testData]
	
def tweetPreprocess(tw):
	#processHashtag
	tw = expandHashtag(tw)
	#lowercase
	tw = tw.lower()		
	#Convert @username to AT_USER
	tw = re.sub('@[^\s]+','AT_USER',tw)
	#Remove additional white spaces
	tw = re.sub('[\s]+', ' ', tw)
	#remove punctuations
	return tw
	
def expandHashtag(tw):
	rv = []
	for w in tw.split():
		if w[0]=='#':
			if w!='#SemST':
				#changes #MakeAmericaGreatAgain to #Make #America #Great #Again
				rv+=['#'+x for x in re.findall('[A-Z][^A-Z]*', w)]
				#changes #MakeAmericaGreatAgain to Make America Great Again
				# rv+=re.findall('[A-Z][^A-Z]*', w)
				#changes #MakeAmericaGreatAgain to Make America Great Again
				# rv+=re.findall('[A-Z][^A-Z]*', w)
		else:
			rv.append(w)

	return ' '.join(rv)


if __name__ == '__main__':
	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	dp.loadData()
	dp.dataPreprocess()