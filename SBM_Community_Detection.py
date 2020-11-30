import graph_tool.all as gt
try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)
import time
import numpy as np

# ------------------------------------------------------
minimize_runs = 10
nxG = nx2.spectroscopic_network('C1.0', weighted=False)

# ------------------------------------------------------

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

# -----------------------------------------------------

start = time.time()


G = nx2.nx2gt(nxG)
G = gt.Graph(G, directed=False, prune=True)
ground_state_estimation_list = [gt.minimize_nested_blockmodel_dl(G, deg_corr=True) for i in range(minimize_runs)]
ground_state_estimation_list = sorted(ground_state_estimation_list, key=lambda ground_state_estimation_list: ground_state_estimation_list.entropy(), reverse=False)
ground_state_estimation = ground_state_estimation_list[0]
entropy = ground_state_estimation.entropy()

print 'Duration: ', time.time() - start


# Alternative approach

# start = time.time()
#
#
# G = nx2.nx2gt(nxG)
# G = gt.Graph(G, directed=False, prune=True)
# ground_state_estimation = gt.minimize_nested_blockmodel_dl(G)
# entropy = ground_state_estimation.entropy()
# count = 1
# counter = 0
# start = time.time()
# while count < 5 and counter < 5:
#     b = gt.minimize_nested_blockmodel_dl(G)
#     S = b.entropy()
#     counter += 1
#     print S
#     if np.abs(S-entropy) < 0.0001:
#         count += 1
#         counter = 0
#         print count
#     elif S < entropy:
#         entropy = S
#         ground_state_estimation = b
#         count = 1
#         counter = 0
#         print count
#
# print entropy, counter
# print 'Time: ', time.time() - start


ground_state_estimation.draw(vertex_text=G.vp.term, vertex_size=7, vertex_font_size=1, output="../plots/carbon-community-detection-hierarchy.svg")
ground_state_estimation0 = ground_state_estimation.get_levels()[0]
ground_state_estimation0.draw(vertex_text=G.vp.term, vertex_size=7, vertex_font_size=1, output='../plots/carbon-community-detection.svg')
G.vp.group = ground_state_estimation0.get_blocks()
groups = G.new_edge_property('double')
G.ep.group_assignment = groups
for u in G.vertices():
    for v in G.vertices():
        if G.vp.group[u] == G.vp.group[v]:
            edge = G.edge(u, v, add_missing=True)
            G.ep.group_assignment[edge] = 1000.0
G.save('../plots/carbon' + str(entropy) + '-community-detection.gml')
