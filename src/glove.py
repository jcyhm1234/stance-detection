import numpy as np
import pickle
import os

class Glove:
	def __init__(self, num_dim, corpus_word_set):
		assert(num_dim in [25,50,100,200])
		self.num_dim = num_dim
		self.corpus_word_set = corpus_word_set
		self.path = '../data/glove.twitter.27B/glove.twitter.27B.'+str(num_dim)+'d.p'
		self.loadIntoHashmap()

	def loadIntoHashmap(self):
		self.vec = {}
		if not os.path.isfile(self.path):
			with open(self.path[:-1]+'txt') as f:
				for line in f:
					parts =  line.split()
					if len(parts)>0:
						if parts[0] in self.corpus_word_set:
							self.vec[parts[0]] = np.fromstring(line[line.find(' ')+1:], dtype=float, sep=' ')

			pickle.dump(self.vec, open(self.path, "wb" ))
		else:
			self.vec = pickle.load( open(self.path, "rb" ))
			# print len(self.vec)

	def getVecOfTweet(self, tweetwords):
		sum_vec = np.zeros(self.num_dim)
		no_of_tokens = 0
		for token in tweetwords:
			try:
				sum_vec += self.vec[token]
				no_of_tokens += 1
			except KeyError:
				try:
			 		#to handle hashtags, again search after removing hash
					if token[1:]:
						sum_vec += self.corpus_vectors[token[1:]]
						no_of_tokens += 1
				except:
					sum_vec = sum_vec + 0
		if no_of_tokens != 0:
			sum_vec = sum_vec / no_of_tokens
		return sum_vec