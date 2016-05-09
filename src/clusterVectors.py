from __future__ import division
import pickle
from numpy import array
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
import json
from numpy.linalg import norm

class Cluster:
	def __init__(self, num_clusters):
		self.num_clusters = num_clusters
		self.loadClusters()

	def generateClusters(self, filename, outputfile, n):
		d = pickle.load( open(filename, "rb"))
		keylist = []
		vallist = []
		for key, value in d.iteritems():
		    keylist.append(key)
		    vallist.append(value)

		keylist = np.array(keylist)
		vallist = np.array(vallist)

		kmeans_clustering = KMeans(n_clusters = n)
		labels = kmeans_clustering.fit_predict(vallist)
		print type(kmeans_clustering.cluster_centers_)
		print (kmeans_clustering.cluster_centers_.shape)
		# label_index = {}
		# label_counts = {}
		# for i,label in enumerate(labels):
		# 	if label not in label_index:
		# 		label_index[label] = 0
		# 		label_counts[label] = 0
		# 	label_index[label] = label_index[label] + (vallist[i])
		# 	label_counts[label] += 1 
		# #print((label_index[0])/label_counts[0])
		# #print((label_counts[0]))
		# for key in label_index.keys():
		# 	label_index[key] = label_index[key] / label_counts[key]
		# 	print label_index[key]
		# pickle.dump(label_index, open(outputfile, "wb"))

	def loadClustersDicts(self):
		self.senti_pos = pickle.load(open('../data/mean/dict/clusters_positive_'+str(self.num_clusters)+'.p','rb'))
		self.senti_neg = pickle.load(open('../data/mean/dict/clusters_negative_'+str(self.num_clusters)+'.p','rb'))
		
		self.polar_pos = pickle.load(open('../data/mean/dict/clusters_positive_pol_'+str(self.num_clusters)+'.p','rb'))
		self.polar_neg = pickle.load(open('../data/mean/dict/clusters_negative_pol_'+str(self.num_clusters)+'.p','rb'))

		self.sub_strong = pickle.load(open('../data/mean/dict/clusters_strong_sub_'+str(self.num_clusters)+'.p','rb'))
		self.sub_weak = pickle.load(open('../data/mean/dict/clusters_weak_sub_'+str(self.num_clusters)+'.p','rb'))

	def loadClusters(self):
		self.senti_pos = pickle.load(open('../data/mean/clusters_positive_'+str(self.num_clusters)+'.p','rb'))
		self.senti_neg = pickle.load(open('../data/mean/clusters_negative_'+str(self.num_clusters)+'.p','rb'))
		
		self.polar_pos = pickle.load(open('../data/mean/clusters_positive_pol_'+str(self.num_clusters)+'.p','rb'))
		self.polar_neg = pickle.load(open('../data/mean/clusters_negative_pol_'+str(self.num_clusters)+'.p','rb'))

		self.sub_strong = pickle.load(open('../data/mean/clusters_strong_sub_'+str(self.num_clusters)+'.p','rb'))
		self.sub_weak = pickle.load(open('../data/mean/clusters_weak_sub_'+str(self.num_clusters)+'.p','rb'))
	
	def convertClustersDictToArray(self):
		#convert to numpy arrow for faster distance calculation
		#each row is cluster representation
		#so num_clusters x num_dim
		allclusters = []
		for c in self.senti_pos:
			allclusters.append(self.senti_pos[c])
		senti_pos_arr = np.vstack(tuple(allclusters))
		pickle.dump(senti_pos_arr, open('../data/mean/clusters_positive_'+str(self.num_clusters)+'.p','wb'))

		allclusters = []
		for c in self.senti_neg:
			allclusters.append(self.senti_neg[c])
		senti_neg_arr = np.vstack(tuple(allclusters))
		pickle.dump(senti_neg_arr, open('../data/mean/clusters_negative_'+str(self.num_clusters)+'.p','wb'))

		allclusters = []
		for c in self.polar_pos:
			allclusters.append(self.polar_pos[c])
		polar_pos_arr = np.vstack(tuple(allclusters))
		pickle.dump(polar_pos_arr, open('../data/mean/clusters_positive_pol_'+str(self.num_clusters)+'.p','wb'))

		allclusters = []
		for c in self.polar_neg:
			allclusters.append(self.polar_neg[c])
		polar_neg_arr = np.vstack(tuple(allclusters))
		pickle.dump(polar_neg_arr, open('../data/mean/clusters_negative_pol_'+str(self.num_clusters)+'.p','wb'))

		allclusters = []
		for c in self.sub_strong:
			allclusters.append(self.sub_strong[c])
		sub_strong_arr = np.vstack(tuple(allclusters))
		pickle.dump(sub_strong_arr, open('../data/mean/clusters_strong_sub_'+str(self.num_clusters)+'.p','wb'))

		allclusters = []
		for c in self.sub_weak:
			allclusters.append(self.sub_weak[c])
		sub_weak_arr = np.vstack(tuple(allclusters))
		pickle.dump(sub_weak_arr, open('../data/mean/clusters_weak_sub_'+str(self.num_clusters)+'.p','wb'))

	def getPolarity(self, tweetword_vectors):
		sum_pos_dist = np.zeros(self.num_clusters)
		sum_neg_dist = np.zeros(self.num_clusters)
		c = 0
		for wv in tweetword_vectors:
			if wv is not None:
				sum_pos_dist += norm( self.polar_pos - wv,axis=1)
				sum_neg_dist += norm( self.polar_neg - wv,axis=1)
				c+=1
		if c!=0:
			sum_neg_dist /= c
			sum_pos_dist /= c
		return np.concatenate((sum_pos_dist,sum_neg_dist))

	def getSentiment(self, tweetword_vectors):
		sum_pos_dist = np.zeros(self.num_clusters)
		sum_neg_dist = np.zeros(self.num_clusters)
		c = 0
		for wv in tweetword_vectors:
			if wv is not None:
				sum_pos_dist += norm( self.senti_pos - wv,axis=1)
				sum_neg_dist += norm( self.senti_neg - wv,axis=1)
				c+=1
		if c!=0:
			sum_neg_dist /= c
			sum_pos_dist /= c
		return np.concatenate((sum_pos_dist,sum_neg_dist))

	def getSubjectivity(self, tweetword_vectors):
		sum_strong_dist = np.zeros(self.num_clusters)
		sum_weak_dist = np.zeros(self.num_clusters)
		c = 0
		for wv in tweetword_vectors:
			if wv is not None:
				sum_strong_dist += norm(self.sub_strong - wv,axis=1)
				sum_weak_dist += norm(self.sub_weak - wv,axis=1)
				c+=1
		if c!=0:
			sum_strong_dist /= c
			sum_weak_dist /= c
		return np.concatenate((sum_strong_dist,sum_weak_dist))

if __name__=='__main__':
	w = Cluster(100)

	# w.generateClusters('../data/pickle/positive_words_corpus.p', '../data/mean/clusters_positive_50.p', 50)
	# w.generateClusters('../data/pickle/positive_words_corpus.p', '../data/mean/clusters_positive_100.p', 100)
	# w.generateClusters('../data/pickle/negative_words_corpus.p', '../data/mean/clusters_negative_50.p', 50)
	# w.generateClusters('../data/pickle/negative_words_corpus.p', '../data/mean/clusters_negative_100.p', 100)
	# w.generateClusters('../data/pickle/positive_sub_corpus.p', '../data/mean/clusters_positive_pol_50.p', 50)
	# w.generateClusters('../data/pickle/positive_sub_corpus.p', '../data/mean/clusters_positive_pol_100.p', 100)
	# w.generateClusters('../data/pickle/negative_sub_corpus.p', '../data/mean/clusters_negative_pol_50.p', 50)
	# w.generateClusters('../data/pickle/negative_sub_corpus.p', '../data/mean/clusters_negative_pol_100.p', 100)
	# w.generateClusters('../data/pickle/neutral_sub_corpus.p', '../data/mean/clusters_neutral_pol_50.p', 50)
	# w.generateClusters('../data/pickle/neutral_sub_corpus.p', '../data/mean/clusters_neutral_pol_100.p', 100)

	w.generateClusters('../data/pickle/strong_sub_corpus.p', '../data/mean/clusters_strong_sub_50.p', 50)
	# w.generateClusters('../data/pickle/strong_sub_corpus.p', '../data/mean/clusters_strong_sub_100.p', 100)
	# w.generateClusters('../data/pickle/weak_sub_corpus.p', '../data/mean/clusters_weak_sub_50.p', 50)
	# w.generateClusters('../data/pickle/weak_sub_corpus.p', '../data/mean/clusters_weak_sub_100.p', 100)

	# w.generateClusters('../data/pickle/strong_sub_corpus.p', '../data/mean/clusters_strong_sub_50.p', 50)
	# w.generateClusters('../data/pickle/strong_sub_corpus.p', '../data/mean/clusters_strong_sub_100.p', 100)
	# w.generateClusters('../data/pickle/weak_sub_corpus.p', '../data/mean/clusters_weak_sub_50.p', 50)
	# w.generateClusters('../data/pickle/weak_sub_corpus.p', '../data/mean/clusters_weak_sub_100.p', 100)

	# w.convertClustersDictToArray()



	