from gensim import models
from numpy import array
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
import json
import numpy as np
import pickle
import os

class Word2Vec:
	def __init__(self):
		self.corpus_vector_file = 'word2vec_corpus.p'
		if not os.path.isfile(self.corpus_vector_file):
			self.w = models.Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True) 
			self.buildVectorCorpus()
		#load the pickle file into a dictionary
		self.corpus_vectors = pickle.load( open( self.corpus_vector_file, "rb" ) )
		self.size = 300

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

	def getFeatureVectorsFromBinary(self, data):
		size = 300
		sum_vec = np.zeros(size).reshape((1, size))
		no_of_tokens = 0
		for token in data:
			try:
				sum_vec += corpus_vectors[token].reshape((1, self.size))
				no_of_tokens += 1
				type(sum_vec)
			except:
				sum_vec = sum_vec + 0
		if no_of_tokens != 0:
			sum_vec = sum_vec / no_of_tokens
			#print 'test'
		print 'Word2Vec feature vector', sum_vec 
		return sum_vec

	def buildVectorCorpus(self,corpus):
		corpus_vectors = dict()
		print 'Building corpus of word2vec vectors'
		for word in corpus:
			try:
				vec = self.w[word].reshape((1, self.size))
				corpus_vectors[word] = vec
			except:
				# Handle key error while using the vector set
				pass
		print 'Built corpus word2vec vectors'
		print len(corpus_vectors)
		# Save this as a pickle file
		pickle.dump( corpus_vectors, open( self.corpus_vector_file, "wb" ) )
		





