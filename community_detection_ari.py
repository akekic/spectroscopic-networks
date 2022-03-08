"""
@author: David Wellnitz

Description: Community detection for the networks in graphlist and comparison to ground truth labels, saves results
into a file and graphs as a .gml

"""
import networkx as nx
import numpy as np
import graph_tool.all as gt
from sklearn.metrics import adjusted_rand_score

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

# ----------------------------- Parameters ------------------------------------

n_runs = 250  # for rand index statistics

# -----------------------------------------------------------------------------


nxG = nx2.spectroscopic_network('Th2.0', weighted=False)
nxG = nx2.only_largest_component(nxG)

G = nx2.nx2gt(nxG)

graphname = 'Th2.gml'

nested_state_lvls = []


def best_block(nested_state_list):
    sort = sorted(nested_state_list, key=lambda nested_state_list: nested_state_list.entropy(), reverse=False)
    return sort[0]


def find_label(G, BlockState):
    # g is the gtGraph, BlockState the group separation
    # returns a dictionary with nodes as keys and group assignment as values
    label = dict()
    # label keys should be node IDs and values numbers for each group
    PropertyMap = BlockState.get_blocks()
    PropertyMap_NodeNames = G.vp.nodeid
    # assign labels:
    for i, key in enumerate(PropertyMap_NodeNames):
        label[key] = PropertyMap[i]
    return label


def ground_truth(G, keys):
    # G should be a nx Graph, key needs to be a key to node attributes in nx
    # returns ground truth assignment of labels given by the key for a given graph
    labels = {n: ''.join([nx.get_node_attributes(G, k)[n] for k in keys]) for n in G.nodes_iter()}
    label_number = dict()
    group = dict()
    for i, name in enumerate(np.unique(labels.values())):
        label_number[name] = i
    for node in labels.keys():
        group[node] = label_number[labels[node]]
    return group


def rand_score(G, map_pred, map_true):
    """
    :param G: graph-tool Graph
    :param map_pred: PropertyMap of grouping
    :param map_true: PropertyMap of ground truth
    :param filename: file to save rand scores in
    :return: adjusted rand score
    """
    # dict to code property map values to ints
    dict_truth = {}
    count = 0
    for v in G.vertices():
        s = map_true[v]
        if s in dict_truth.keys():
            continue
        else:
            dict_truth[s] = count
            count += 1
    # need to turn property maps into integer arrays
    labels_pred = np.zeros(G.num_vertices(), dtype=int)
    labels_true = np.zeros(G.num_vertices(), dtype=int)
    for n in G.vertices():
        labels_pred[G.vertex_index[n]] = map_pred[n]
        labels_true[G.vertex_index[n]] = dict_truth[map_true[n]]
    score = adjusted_rand_score(labels_true, labels_pred)
    return score


# Combinations of properties
G.vp.ljp = G.new_vp('string')
for v in G.vertices():
    G.vp.ljp[v] = G.vp.l[v] + G.vp.J[v] + G.vp.parity[v]
G.vp.jparity = G.new_vp('string')
for v in G.vertices():
    G.vp.jparity[v] = G.vp.J[v] + G.vp.parity[v]
G.vp.lparity = G.new_vp('string')
for v in G.vertices():
    G.vp.lparity[v] = G.vp.l[v] + G.vp.parity[v]

num_levels = np.zeros(n_runs, dtype=int)
for j in range(n_runs):
    # find blockstates
    nested_state = gt.minimize_nested_blockmodel_dl(G, deg_corr=True, mcmc_args=dict(niter=10))
    nested_state_lvls.append(nested_state)
    num_levels[j] = len(nested_state.get_bs())
# calculate the statistics of the rand scores
# only look at the hierarchy level that fits the description best
scores_jp = np.zeros((n_runs, int(num_levels.max())))
scores_lp = np.zeros((n_runs, int(num_levels.max())))
scores_ljp = np.zeros((n_runs, int(num_levels.max())))
scores_term = np.zeros((n_runs, int(num_levels.max())))
scores_p = np.zeros(n_runs)
for l, state in enumerate(nested_state_lvls):
    maps = [state.project_level(j).get_blocks() for j in range(int(num_levels[l]))]
    for m, map in enumerate(maps):
        # sorry for the copy pasted code :(
        scores_jp[l, m] = rand_score(G, map, G.vp.jparity)
        scores_lp[l, m] = rand_score(G, map, G.vp.lparity)
        scores_ljp[l, m] = rand_score(G, map, G.vp.ljp)
        scores_term[l, m] = rand_score(G, map, G.vp.term)
    scores_p[l] = rand_score(G, maps[int(num_levels[l]) - 2], G.vp.parity)

G.save(graphname, fmt='gml')
