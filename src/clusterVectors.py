from __future__ import division
import pickle
from numpy import array
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
import json

class Cluster:
	def __init__(self):
		pass

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
		label_index = {}
		label_counts = {}
		for i,label in enumerate(labels):
			if label not in label_index:
				label_index[label] = 0
				label_counts[label] = 0
			label_index[label] = label_index[label] + (vallist[i])
			label_counts[label] += 1 
		#print((label_index[0])/label_counts[0])
		#print((label_counts[0]))
		for key in label_index.keys():
			label_index[key] = label_index[key] / label_counts[key]
			print label_index[key]
		pickle.dump(label_index, open(outputfile, "wb"))


if __name__=='__main__':
	w = Cluster()
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
	w.generateClusters('../data/pickle/strong_sub_corpus.p', '../data/mean/clusters_strong_sub_100.p', 100)
	w.generateClusters('../data/pickle/weak_sub_corpus.p', '../data/mean/clusters_weak_sub_50.p', 50)
	w.generateClusters('../data/pickle/weak_sub_corpus.p', '../data/mean/clusters_weak_sub_100.p', 100)
	