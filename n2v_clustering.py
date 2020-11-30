#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:07:35 2017

@author: arminkekic
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings
import itertools

from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.cluster.vq import kmeans2

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

# %%

n2v     = nx2.node2vec()
n2v.G   = nx2.model_network(Z=1, max_n=8, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True)

# %%

n2v.learn_features(p=1.0, q=1.0, name='test', dimensions=20, save_file=False)

# %%
X       = n2v.node_vec_array
Z       = linkage(X, method='average')
cut     = cut_tree(Z)
centroid, label = kmeans2(X, 8)

# %%
pos = nx.spring_layout(n2v.G)

# %%
for i in xrange(8, 7, -1):
    print '# groups:', i
    grouplist = cut[:,-i]
    for group in np.unique(grouplist):
        ind = np.argwhere(grouplist==group)
        group_nodes = np.array(n2v.node_list)[ind]
        print 'group', group
        print group_nodes

# %%
grouplist = label
for group in np.unique(grouplist):
    ind = np.argwhere(grouplist==group)
    group_nodes = np.array(n2v.node_list)[ind]
    print 'group', group
    print group_nodes

# %%
        
#for i in xrange(15, 2, -1):
#    grouplist = cut[:,-i]
#    g = dict((zip(n2v.node_list, grouplist)))
#    plt.figure()
#    plt.title('Vector clustering ' + str(i) + ' groups')
#    nx2.draw_groups(n2v.G, g, labels=g, pos=pos)



