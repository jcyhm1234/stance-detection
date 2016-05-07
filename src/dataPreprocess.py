import json
import csv

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
		self.trainTweets = [row[0] for row in self.trainData]
		self.testTweets = [row[0] for row in self.testData]
		self.trainLabels = [row[2] for row in self.trainData]
		self.testLabels = [row[2] for row in self.testData]

if __name__ == '__main__':
	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	dp.loadData()
	dp.dataPreprocess()