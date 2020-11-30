# -*- coding: utf-8 -*-
"""
Description:

@author: Julian Heiss
@date: August 2017
"""

try:
	reload
except NameError:
	# Python 3
	from imp import reload
import nx2
reload(nx2)

import CommunityClustering as cc

from graphviz import Graph
import sklearn.cluster
import matplotlib.pyplot as plt
import collections
import numpy as np
import heapq
import copy
import re
import csv
import os
import networkx as nx


# %%
# GLOBAL VARIABLES
########################################

ion = 'He1.0'                          # ion to use as input, see nx2.py
experimental_global = True                    # use the experimental NIST data or the theoretical Jitrik-Bunke data
only_dipole_global = False              # global switch for taking the full network or only dipole lines
n_limit_global            = False               # limit is only to be used for one electron ions (hydrogenic ions)
if n_limit_global==False:
    max_n_global = None
else:
    max_n_global   = 8                 # maximal n considered in network

Ground_Truth_LJ = True

directory = './' + 'Community_detection_clustering' + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

save_plot_global = True             # whether to save the graphviz plot

keep_files = False                  # whether to keep the graphviz backend files

########################################

# Load network
print 'Network: ', ion
G_original = nx2.load_network(ion=ion, experimental=experimental_global, only_dipole=only_dipole_global, n_limit=n_limit_global, max_n=max_n_global)
print 'Only Dipole: ', str(only_dipole_global)
print 'Max n: ', str(max_n_global)
# set graph attribute
G_original.graph['ion']=ion

# show what you've got
print 'Number of nodes', nx.number_of_nodes(G_original)
print 'Number of edges', nx.number_of_edges(G_original)

# filtering of nodes without measured intensity
G_filtered = G_original
if experimental_global == True:
    for u,v,d in G_filtered.edges(data=True):
        if np.isnan(G_filtered[u][v]['intensity']):
            # print 'Intensity', G_filtered[u][v]['intensity']
            G_filtered.remove_edge(u,v)
    G_filtered = nx2.only_largest_component(G_filtered)

# show what you've got
print 'Number of nodes', nx.number_of_nodes(G_filtered)
print 'Number of edges', nx.number_of_edges(G_filtered)


print 'nodes', G_original.nodes()
# conf_dict         = nx.get_node_attributes(G_original, 'conf')          # ID-j dictionary
# print '0', conf_dict
l_dict         = nx.get_node_attributes(G_original, 'l')          # ID-l dictionary
print '1', l_dict
j_dict         = nx.get_node_attributes(G_original, 'J')          # ID-j dictionary
print '2', j_dict
term_dict         = nx.get_node_attributes(G_original, 'term')          # ID-term dictionary [2S+1]LJ
print '3', term_dict
print list(term_dict.values())
parity_dict   = nx.get_node_attributes(G_original, 'parity')
print '4', parity_dict
n_dict    =  nx.get_node_attributes(G_original, 'n')
print '5', n_dict
# liste = ["%s %s %s" % (conf_dict[n], term_dict[n], j_dict[n]) for n in G_original.nodes()]
# print liste



# Run community clustering
filename = ion + n_limit_global*'_max_n_' + n_limit_global*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental_global*'_NIST' + (not experimental_global)*'_Jitrik' + Ground_Truth_LJ*'_GT_LJ' + (not Ground_Truth_LJ)*'_GT_L' + '_Highscores'
highscore_list = cc.community_clustering(G_filtered, Ground_Truth_LJ=True ,verbose=True, save_plot=save_plot_global, graph_filename=directory+filename)

# Write results to file
with open(directory+filename + '.csv', 'w') as results_csv:
    # Todo: dont write with '"'
    spacewriter = csv.writer(results_csv, delimiter=' ')
    spacewriter.writerow(['# Network: ', ion])
    spacewriter.writerow(['# Type: ', experimental_global*'NIST' + (not experimental_global)*'Jitrik'])
    spacewriter.writerow(['# Only Dipole: ', str(only_dipole_global)])
    spacewriter.writerow(['# Max n: ', str(max_n_global)])
    spacewriter.writerow(['# Number of nodes', str(nx.number_of_nodes(G_original))])
    spacewriter.writerow(['# Number of edges', str(nx.number_of_edges(G_original))])

    csvwriter = csv.writer(results_csv, delimiter=',')
    csvwriter.writerow(['Feature Type', 'Algorithm', 'Parameters', 'Score'])
    for feature, algo_name, params, score, _ in highscore_list[:]:
        csvwriter.writerow([feature, algo_name, params, score])
        # TODO: properly write params dictionary (looks weird in .csv so far)
results_csv.close()


cc.hierachy_clustering(G_filtered, Ground_Truth_LJ=True ,verbose=True, save_plot=save_plot_global)


# if not keep_files:
#     os.remove(directory+filename)        # removes the one file without file ending
#     os.remove(directory+filename+'.pdf') # removes the graph pdf
#     os.remove(directory+filename+'.txt') # removes the highscore txt


