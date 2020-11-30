#!/usr/bin/python2
# -*- coding: utf-8 -*-

import graphviz as gv
# from graphviz import Graph

import sklearn.cluster
from sklearn.metrics import adjusted_rand_score

import scipy.cluster
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
import copy
import sys
import re
import csv
import itertools

import networkx as nx
import nx2

class GraphReader(object):
    def __init__(self, filename):
        self._re_label = re.compile("^# *label\\(([0-9]+)\\) *= *\"(.*)\"$")
        self.filename = filename

    def _process(self, line):
        if not (line.startswith("#") or line.startswith(";")):
            return False

        m = self._re_label.match(line)
        if m is not None:
            i = int(m.group(1))
            self.labels[i] = m.group(2)
            return True

        sys.stderr.write("%s\n" % line)
        return True

    def iteritems(self):
        self.labels = {}
        with open(self.filename) as fp:
            for line in fp:
                line = line.rstrip()
                if line == "": continue
                if self._process(line): continue
                yield line

# used for slackner style input
def load_graph_and_labels(filename):
    """Load graph and labels from file."""
    A, labels = None, None
    reader = GraphReader(filename)
    for line in reader.iteritems():
        if A is None:
            num_nodes = int(line)
            A = np.zeros((num_nodes, num_nodes))
            continue

        values = line.split()
        i = int(values[0])
        j = int(values[1])
        weight = float(values[2]) if len(values) >= 3 else 1.0
        assert i >= 1 and i <= num_nodes
        assert j >= 1 and j <= num_nodes
        assert weight > 0.0
        A[i - 1, j - 1] = weight
        A[j - 1, i - 1] = weight

    if reader.labels:
        labels = [reader.labels[i + 1] for i in xrange(num_nodes)]

    return A, labels

def bfs(A, start):
    """Perform a breadth first search in a graph."""
    visited = [False for i in xrange(A.shape[0])]
    q = [(0, -1, start)]
    while len(q):
        c, f, t = heapq.heappop(q)
        if visited[t]: continue
        yield (c, f, t)
        visited[t] = True
        for i in xrange(A.shape[0]):
            if A[t, i] <= 0.0: continue
            if visited[i]: continue
            heapq.heappush(q, (c + 1, t, i))

def distance_trafo(A, start):
    v = [None for i in xrange(A.shape[0])]
    for c, f, t in bfs(A, start): v[t] = c
    return v

# slackner score
def compare_communities(com1, com2):
    mappings = collections.defaultdict(int)
    for i, j in zip(com1, com2):
        mappings[(i, j)] += 1

    used_i = set()
    used_j = set()
    error  = 0
    abort = False

    for (i, j), count in sorted(mappings.items(), key=lambda x: x[1], reverse=True):
        if i not in used_i:
            if j not in used_j:
                used_i.add(i)
                used_j.add(j)
            else:
                used_i.add(i)
                error += count
        else:
            if j not in used_j:
                used_j.add(j)
                error += count
            else:
                error += count

    return error

def rgb_color(r, g, b):
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return "#%02x%02x%02x" % (r, g, b)

# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def make_graph(A, labels=None, clusters=None, save_plot=False, filename='graph', nx_graph=None):
    assert A.shape[0] == A.shape[1]
    g = gv.Graph(comment='', engine='fdp')
    g.attr(outputorder="edgesfirst")

    if labels is None:
        labels = ["%d" % i for i in xrange(A.shape[0])]

    if nx_graph is not None:
        term_dict         = nx.get_node_attributes(nx_graph, 'term')          # ID-term dictionary [2S+1]LJ
        n_dict         = nx.get_node_attributes(nx_graph, 'n')                  # ID-n dictionary
        # labels = list(term_dict.values())
        # labels = ["%s %s" % ("".join(itertools.takewhile(str.isdigit, term_dict[nx_node])), label)  for (nx_node, label) in zip(nx_graph.nodes(), labels)]
        labels = ["%s %s" % (n_dict[nx_node], label)  for (nx_node, label) in zip(nx_graph.nodes(), labels)]
        print labels

    if clusters is not None:
        for c in np.unique(clusters):
            with g.subgraph(name='cluster_%d' % c) as sg:
                for i in np.where(clusters == c)[0]:
                    sg.node('node_%d' % i, labels[i], style='filled', fillcolor='white')
    else:
        for i in xrange(A.shape[0]):
            g.node('node_%d' % i, labels[i], style='filled', fillcolor='white')

    maxA = np.max(A)
    for i in xrange(A.shape[0]):
        for j in xrange(i + 1):
            if A[i, j] <= 0.0: continue
            c = 1.0 - A[i, j] / maxA
            g.edge('node_%d' % i, 'node_%d' % j, color=rgb_color(c, c, c))

    g.view()
    if save_plot:
        g.render(filename=filename)

def get_label(label):
    values = label.split(" ")
    return " ".join(values[1:])

# Calculate input feature vectors
# Each matrix row is considered as one node vector by the sklearn clustering algorithms
def get_feature_params(A, Graph):
    # Adjacency matrix
    # directly use connectivity matrix
    yield "A", np.array(A)


    # Distance transformed matrix
    B = np.zeros((A.shape[0], A.shape[0]))
    for start in xrange(A.shape[0]):
        B[start, :] = distance_trafo(A, start)
    yield "DT", B

    # n2v node vectors
    # ToDo: do not use via nx2
    n2v     = nx2.node2vec()
    n2v.G   = Graph
    for p in [x / 100.0 for x in range(400, 100, -100)]: # to use range with floats
        for q in [x / 100.0 for x in range(25, 100, 50)]:
            n2v.learn_features(p=2.0, q=0.5, name=Graph.graph['ion'], dimensions=40, save_file=False)
            X       = n2v.node_vec_array
            yield 'n2v' + ', p: ' + str(p) + ', q: ' + str(q), X

# Clustering algorithms
def get_hyper_params():
    # KMeans
    algo = sklearn.cluster.KMeans
    for clusters in xrange(2, 30):
        yield algo, {'n_clusters': clusters}

    # Spectral Clustering
    algo = sklearn.cluster.SpectralClustering
    for clusters in xrange(2, 30):
        for neighbors in xrange(1, 10):
            yield algo, {'n_clusters': clusters, 'n_neighbors': neighbors}

    # Agglomerative Clustering
    algo = sklearn.cluster.AgglomerativeClustering
    for clusters in xrange(2, 30):
        yield algo, {'n_clusters': clusters}

    # Affinitiy Propagation
    algo = sklearn.cluster.AffinityPropagation
    for damping in np.arange(0.5, 1.0, 10):
        yield algo, {'damping': damping}


def community_clustering(Graph,  Ground_Truth_LJ=True ,verbose=False, save_plot=False, graph_filename='graphviz_test'):
    """
    Function

    Parameters
    ----------
    Graph : NetworkX graph
        Graph
    verbose : boolean (optional)
        Changes output mode to verbose if true (default = False).

    To-Do
    -----
    Find permanent solution for the different methods: sklearn clustering, scipy hierarchical dendrograms.
    First one needs .fit() has labels, has a graphviz plot at the end etc.
    if else program flow is confusing and inelegant.
    """

    # adjacency matrix
    A = nx.to_numpy_matrix(Graph) # Adjacency matrix. Auszug aus Doku: nodelist (list, optional) - The rows and columns are ordered according to the nodes in nodelist. If nodelist is None, then the ordering is produced by G.nodes().
    # A = nx.to_numpy_matrix(Graph, weight='weight')
    # print A

    # create labels for ground truth
    l_dict = nx.get_node_attributes(Graph, 'l')          # ID-l dictionary
    j_dict = nx.get_node_attributes(Graph, 'J')          # ID-j dictionary
    if Ground_Truth_LJ:
        ground_truth_labels = ["%s%s" % (l_dict[n], j_dict[n]) for n in Graph.nodes()] #use L+J information
    else:
        ground_truth_labels = ["%s" % (l_dict[n]) for n in Graph.nodes()] #ground truth only l
    print 'GT labels', ground_truth_labels

    # slackner style input
    # A, ground_truth_labels = load_graph_and_labels('H-1.txt')
    # ground_truth_labels = [get_label(label) for label in ground_truth_labels]


    highscore = []
    for feature, X in get_feature_params(A, Graph):
        for algo, params in get_hyper_params():
            algo_name = algo.__name__
            if verbose:
                print ">>>>>", feature, algo_name, params

            try:
                # funktioniert nicht
                if algo == scipy.cluster.hierarchy:
                    Z = algo.linkage(X, **params)
                else:
                    # KMeans, Spectral Clustering, Agglomerative Clustering, Affinitiy Propagation
                    # X must be array-like or sparse matrix, shape=(n_samples, n_features) for the sklearn clustering methods to function.
                    algo = algo(**params) #wandelt dict eintraege in key-value paare um (die an algo übergeben werden)
                    algo.fit(X)
            except ValueError:
                continue
            except np.linalg.linalg.LinAlgError:
                continue

            # print algo.labels_
            if algo == scipy.cluster.hierarchy:
                score, coph_dists = cophenet(Z, pdist(X, metric = 'euclidean')) # TODO: how to only extract second argument (dictionary metric)?
            else:
                # score = compare_communities(ground_truth_labels, algo.labels_) # slackner score
                score = adjusted_rand_score(ground_truth_labels, algo.labels_)   # rand index

            # print score to terminal
            if verbose:
                print "SCORE: ", score

            # append current result to highscore list
            if algo == scipy.cluster.hierarchy:
                highscore.append((feature, algo_name, params, score, 'platzhalter'))
            else:
                highscore.append((feature, algo_name, params, score, algo.labels_))

    print "HIGHSCORE Top 50:"

    # Sort highscore list to actually make it a highscore; be aware what scoring method is used.
    # highscore = sorted(highscore, key=lambda x: x[3]) #für slackner score
    highscore = sorted(highscore, key=lambda x: x[3], reverse=True) #für rand index
    for feature, algo_name, params, score, _ in highscore[:50]:
        print ">>>", feature, algo_name, params, score

    # Show and save graph via graphviz
    if algo == scipy.cluster.hierarchy:
        # plot dendrogram
        plt.figure(figsize=(25, 10))
        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.xlabel('sample index')
        # plt.ylabel('distance')
        fancy_dendrogram( Z,
                    truncate_mode='lastp',  # show only the last p merged clusters
                    p=12,  # show only the last p merged clusters
                    # show_leaf_counts=False,  # otherwise numbers in brackets are counts
                    leaf_rotation=90.,  # rotates the x axis labels
                    leaf_font_size=8.,  # font size for the x axis labels
                    show_contracted=True,  # to get a distribution impression in truncated branches
                    max_d=1,  # plot a horizontal cut-off line
                    )
        plt.show()
    else:
        # only plot when not a dendrogram.
        make_graph(A, ground_truth_labels, highscore[0][4], save_plot=save_plot, filename=graph_filename)

    return highscore





def get_hierarchy_params():
    # Hierarchical Clustering Linkage
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    algo = scipy.cluster.hierarchy
    for method in ['ward', 'single', 'complete']:
        for metric in ['euclidean', 'cosine', 'correlation', 'hamming']:
        # ToDo: hamming gives sometimes nan scores. but other times best performer
            yield algo, {'method': method, 'metric': metric}



def hierachy_clustering(Graph,  Ground_Truth_LJ=True ,verbose=False, save_plot=False):
    """
    Spielwiese hierarchy clustering

    Parameters
    ----------

    To-Do
    -----
    """

    # adjacency matrix
    A = nx.to_numpy_matrix(Graph) # Adjacency matrix. Auszug aus Doku: nodelist (list, optional) - The rows and columns are ordered according to the nodes in nodelist. If nodelist is None, then the ordering is produced by G.nodes().
    # A = nx.to_numpy_matrix(Graph, weight='weight')
    # print A

    # create labels for ground truth
    l_dict = nx.get_node_attributes(Graph, 'l')          # ID-l dictionary
    j_dict = nx.get_node_attributes(Graph, 'J')          # ID-j dictionary
    if Ground_Truth_LJ:
        ground_truth_labels = ["%s%s" % (l_dict[n], j_dict[n]) for n in Graph.nodes()] #use L+J information
    else:
        ground_truth_labels = ["%s" % (l_dict[n]) for n in Graph.nodes()] #ground truth only l
    print 'GT labels', ground_truth_labels


    highscore = []
    for feature, X in get_feature_params(A, Graph):
        for algo, params in get_hierarchy_params():
            algo_name = algo.__name__
            if verbose:
                print ">>>>>", feature, algo_name, params

            try:
                Z = algo.linkage(X, **params)
            except ValueError:
                continue
            except np.linalg.linalg.LinAlgError:
                continue

            coph_score, coph_dists = cophenet(Z, pdist(X, metric = params['metric']))
            # score = adjusted_rand_score(ground_truth_labels, algo.labels_)   # rand index

            # print score to terminal
            if verbose:
                print "SCORE: ", coph_score

            # append current result to highscore list
            highscore.append((feature, algo_name, params, coph_score, Z))

    print "\n\n################ \nHIGHSCORE Top 50:"

    # Sort highscore list to actually make it a highscore; be aware what scoring method is used.
    # highscore = sorted(highscore, key=lambda x: x[3]) #für slackner score
    highscore = sorted(highscore, key=lambda x: x[3], reverse=True) #für coph score
    for feature, algo_name, params, score, Z in highscore[:50]:
        print ">>>", feature, algo_name, params, score


    # plot dendrogram with highest coph score
    best_Z = highscore[0][4]

    rand_scores = []
    # specify cutoff
    for cut_off in [x / 100.0 for x in range(800, 0, -1)]: # to use range with floats:
        max_d = cut_off
        # retrieve clusters
        clusters = fcluster(best_Z, max_d, criterion='distance')
        # calculate adj rand score
        score = adjusted_rand_score(ground_truth_labels, clusters)   # rand index
        print max_d
        print score
        print clusters
        rand_scores.append((max_d, score, clusters))
    rand_scores = sorted(rand_scores, key=lambda x: x[1], reverse=True) #für rand score


    plt.figure(figsize=(25, 10))
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('sample index')
    # plt.ylabel('distance')
    fancy_dendrogram( best_Z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=12,  # show only the last p merged clusters
                # show_leaf_counts=False,  # otherwise numbers in brackets are counts
                leaf_rotation=90.,  # rotates the x axis labels
                leaf_font_size=8.,  # font size for the x axis labels
                show_contracted=True,  # to get a distribution impression in truncated branches
                max_d=rand_scores[0][0],  # plot a horizontal cut-off line
                )
    plt.show()

    print 'best rand score', rand_scores[0], 'with coph score', highscore[0]
    make_graph(A, ground_truth_labels, rand_scores[0][2], save_plot=False, filename=None, nx_graph=Graph)



