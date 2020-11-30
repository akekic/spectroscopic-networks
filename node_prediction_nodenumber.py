import graph_tool.all as gt
import numpy as np
import networkx as nx
import nx2
import matplotlib.pyplot as plt
import node_prediction_class as npc

reload(nx2)


def dropout(G, fraction):
    G_copy = G.copy()
    num_nodes = G_copy.num_vertices()
    # choose and delete a fraction p of nodes
    dropped_nodes = np.random.choice(num_nodes, int(fraction*num_nodes), replace=False)
    G_copy.remove_vertex(dropped_nodes)
    return G_copy

def add_random(G, fraction):
    G_copy = G.copy()
    num_nodes = G_copy.num_vertices()
    num_edges = G_copy.num_edges()
    num_new_nodes = int(fraction*num_nodes)
    deg = int(2*num_edges/num_nodes)

    for new_node in range(num_nodes, num_nodes+num_new_nodes):
        new_edges = np.random.choice(num_nodes, deg, replace=False)
        for node1 in new_edges:
            G_copy.add_edge(node1, new_node)
    return G_copy

def node_prediction(G, fraction):
    pred = npc.node_prediction()
    G1 = G.copy()
    G_copy = G.copy()
    pred.G_original = G1
    pred.group()
    pred.dropout(0.0)
    num_nodes = pred.G_original.num_vertices()
    new_nodes = pred.G_original.add_vertex(int(num_nodes*fraction))
    num_groups = pred.b.get_B()
    num_new_nodes = int(num_nodes*fraction)
    group_list = np.random.choice(num_groups, num_new_nodes, replace=True)
    pred.deleted_nodes = range(num_nodes, num_nodes + num_new_nodes)
    state = pred.b.copy()
    for node, group in zip(pred.deleted_nodes, group_list):
        state.move_vertex(node, group)
    pred.b = state
    pred.spectral_adjacency()
    edge_list = np.argwhere(pred.predicted_weights > 0.8)
    G_copy.add_edge_list(edge_list)
    return G_copy


dropout_list = [-1.0, -0.5, -0.1, 0.0, 0.05, 0.2, 0.4, 0.6]

Graph = gt.collection.data['celegans']
if Graph.is_directed():
    Graph.set_directed(False)
    gt.remove_parallel_edges(Graph)

# nxHe = nx2.spectroscopic_network('He1.0', line_data_path='../data/ASD54_lines.csv', level_data_path='../data/ASD54_levels.csv', weighted=True, check_accur=True, check_calc_wl=True)
# nxHe = nx2.only_largest_component(nxHe)
# Graph = nx2.nx2gt(nxHe)
# if Graph.is_directed():
#     Graph.set_directed(False)
#     gt.remove_parallel_edges(Graph)

# nxTheo = nx2.model_network(E1=True, max_n=8, datafolder='../data')
# Graph = nx2.nx2gt(nxTheo)
# if Graph.is_directed():
#     Graph.set_directed(False)
#     gt.remove_parallel_edges(Graph)


LP = nx2.LinkPrediction()

xdata = 1.0 - np.array(dropout_list)
ydata = np.zeros_like(xdata)
yerr = np.zeros_like(xdata)

for i, frac in enumerate(dropout_list):
    if frac < 0:
        addfrac = -frac
        G = node_prediction(Graph, addfrac)
    else:
        G = dropout(Graph, frac)

    sc = LP.gt_structural_consistency(G, n_repeat=10)
    ydata[i] = sc[0]
    yerr[i] = sc[1]/np.sqrt(10.0)
    print frac, ':     ', sc




plt.figure(1)
plt.errorbar(xdata, ydata, yerr)
plt.show()
