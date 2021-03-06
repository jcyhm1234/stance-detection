import json
import csv
import re
from nltk.tokenize import TweetTokenizer
from random import shuffle
import random

class DataManager:	
	def __init__(self, trainFile, testFile, n):
		self.trainFile = trainFile
		self.testFile = testFile
		
		self.trainData = []
		self.testData = []

		self.trainTweets = []
		self.testTweets = []
		
		self.trainLabels = []
		self.testLabels = []
		
		self.tknzr = TweetTokenizer()

		self.load(n)
		self.loadStopwords()
		self.preprocess()

	def load(self, n):
		with open(self.trainFile, 'r') as f:
			self.trainData = [row for row in csv.reader(f.read().splitlines())]
		with open(self.testFile, 'r') as f:
			self.testData = [row for row in csv.reader(f.read().splitlines())]
		# remove the header from the data
		self.trainData.pop(0)
		self.testData.pop(0)
		random.seed(1001)
		shuffle(self.trainData)
		if n!=-1:
			self.trainData = self.trainData[:n]

		# print 'Loaded %s training samples' % len(self.trainData),'and %s testing samples'%  len(self.testData)
	
	def loadStopwords(self):
		with open('../lexicons/stopwords.txt','r') as f:
			self.stopwords = set([row for row in f.read().splitlines()])

	def preprocess(self):
		#this does no preprocessing. actual tweets, used for POS tagging
		self.trainTweetsText = [row[0] for row in self.trainData]
		self.testTweetsText = [row[0] for row in self.testData if row[1]!='Donald Trump']
		
		self.trainTopics = [row[1] for row in self.trainData]
		self.testTopics = [row[1] for row in self.testData if row[1]!='Donald Trump']

		self.trainLabels = [row[2] for row in self.trainData]
		self.testLabels = [row[2] for row in self.testData if row[1]!='Donald Trump']

		# t is of form (preprocessedwords,topic, y)
		self.trainTweets = [(self.tweetPreprocess(row[0]),row[1],row[2]) for row in self.trainData]
		self.testTweets = [(self.tweetPreprocess(row[0]),row[1], row[2]) for row in self.testData if row[1]!='Donald Trump']

	def tweetPreprocess(self,tw):
		#remove nonascii
		tw = ''.join(i for i in tw if ord(i)<128)
		#processHashtag
		tw = expandHashtag(tw)
		tw = removeNumbers(tw)
		#lowercase
		tw = tw.lower()
		#Convert @username to AT_USER
		tw = re.sub('@[^\s]+','',tw)
		#Remove additional white spaces
		tw = re.sub('[\s]+', ' ', tw)
		#remove punctuations
		words = self.tknzr.tokenize(tw)
		words = self.removeStopwords(words)
		return words

	def removeStopwords(self, words):
		rval = []
		for w in words:
			if w not in self.stopwords:
				rval.append(w)
		return rval

def removeNumbers(tw):
	tw = re.sub("\d+[.,:]*\d+","",tw)
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
		else:
			rv.append(w)

	return ' '.join(rv)


if __name__ == '__main__':
	dp = DataManager('../data/train.csv','../data/test.csv')
	print dp.trainTweets[0]
	# print dp.trainTweets[0]
