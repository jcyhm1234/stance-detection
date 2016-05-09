from gensim import models
from numpy import array
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
import json
import numpy as np

class Word2Vec:
	def __init__(self):
		self.w = models.Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True) 

	def getFeatureVectors(self, data):
		size = 300
		sum_vec = np.zeros(size).reshape((1, size))
		no_of_tokens = 0
		for token in data:
			try:
				sum_vec += w[token].reshape((1, size))
				no_of_tokens += 1
				type(sum_vec)
			except:
				sum_vec = sum_vec + 0
		if no_of_tokens != 0:
			sum_vec = sum_vec / no_of_tokens
			#print 'test'
		print 'Word2Vec feature vector', sum_vec 
		return sum_vec





