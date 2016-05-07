import json
import csv

class DataPreprocess:
	
	def __init__(self, trainFile, testFile):
		self.trainFile = trainFile
		self.testFile = testFile
		self.trainData = []
		self.testData = []


	def loadData(self):
		with open(self.trainFile, 'r') as f:
			self.trainData = [row for row in csv.reader(f.read().splitlines())]
		with open(self.testFile, 'r') as f:
			self.testData = [row for row in csv.reader(f.read().splitlines())]

	def dataPreprocess(self):
		pass

if __name__ == '__main__':
	dp = DataPreprocess('../data/train.csv','../data/test.csv')
	dp.loadData()