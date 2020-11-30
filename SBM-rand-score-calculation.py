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

# Compare nested an normal SBM for observed hydrogen, theoretical hydrogen and
# helium
nxH, conf, term, j = nx2.spectroscopic_network('H1.0', weighted=False, dictionaries=True)
# # Cut nodes without term for hydrogen
nodelist = list(nxH.nodes)
for n in nodelist:
    if term[n] == '':
        nxH.remove_node(n)
        term.pop(n)
        conf.pop(n)
        j.pop(n)
nxTheo = nx2.model_network(E1=True)
nxTheoFull = nx2.model_network(E1=True, E2=True, E3=True, M1=True, M2=True, M3=True)
nxTheo8 = nx2.model_network(E1=True, max_n=8)
nxTheoFull8 = nx2.model_network(E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=8)
nxHe = nx2.spectroscopic_network('He1.0', weighted=False)
nxHe = nx2.only_largest_component(nxHe)
# nodelist = nxHe.nodes()
# for n in nodelist:
#     if nx.get_node_attributes(nxHe, 'term')[n] == '3P0,1,2' or nx.get_node_attributes(nxHe, 'term')[n] == '3D1,2,3':
#         nxHe.remove_node(n)
nxFe = nx2.spectroscopic_network('Fe1.0', weighted=False)
nxFe = nx2.only_largest_component(nxFe)

nxC = nx2.spectroscopic_network('C1.0', weighted=False)
nxC = nx2.only_largest_component(nxC)


H = nx2.nx2gt(nxH)
Theo = nx2.nx2gt(nxTheo)
TheoFull = nx2.nx2gt(nxTheoFull)
Theo8 = nx2.nx2gt(nxTheo8)
TheoFull8 = nx2.nx2gt(nxTheoFull8)
He = nx2.nx2gt(nxHe)
Fe = nx2.nx2gt(nxFe)
C = nx2.nx2gt(nxC)

graphlist = [H, He, C, Fe, Theo, TheoFull, Theo8, TheoFull8]  # names of the graphs to use
graphnames = ['20180213H.gml', '20180213He.gml', '20180213C.gml', '20180213Fe.gml', '20180213Theo.gml', \
              '20180213TheoFull.gml', '20180213Theo8.gml', '20180213TheoFull8.gml']
# Theo.save("Hydrogen.gml", fmt="gml")

# Ground state nested SBM
nested_state_lvls = [[] for _ in graphlist]
# stateH_0 = []
# stateTheo_0 = []
# stateHe_0 = []


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


# property maps for blockstates

filename = './results/rand-scores'
f1 = open(filename, 'w+')
for i, G in enumerate(graphlist):
    print >> f1, '\n', '\n', G.graph_properties.name
    # new properties to compare to
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
        nested_state_lvls[i].append(nested_state)
        num_levels[j] = len(nested_state.get_bs())
    # calculate the statistics of the rand scores
    # only look at the hierarchy level that fits the description best
    scores_jp = np.zeros((n_runs, int(num_levels.max())))
    scores_lp = np.zeros((n_runs, int(num_levels.max())))
    scores_ljp = np.zeros((n_runs, int(num_levels.max())))
    scores_term = np.zeros((n_runs, int(num_levels.max())))
    scores_p = np.zeros(n_runs)
    for l, state in enumerate(nested_state_lvls[i]):
        maps = [state.project_level(j).get_blocks() for j in range(int(num_levels[l]))]
        for m, map in enumerate(maps):
            # sorry for the copy pasted code :(
            scores_jp[l, m] = rand_score(G, map, G.vp.jparity)
            scores_lp[l, m] = rand_score(G, map, G.vp.lparity)
            scores_ljp[l, m] = rand_score(G, map, G.vp.ljp)
            scores_term[l, m] = rand_score(G, map, G.vp.term)
        scores_p[l] = rand_score(G, maps[int(num_levels[l]) - 2], G.vp.parity)
    scores_jp_bestlvl = scores_jp.max(axis=1)
    bestlvl_j = np.argmax(scores_jp, axis=1)
    scores_lp_bestlvl = scores_lp.max(axis=1)
    bestlvl_l = np.argmax(scores_lp, axis=1)
    scores_ljp_bestlvl = scores_ljp.max(axis=1)
    bestlvl_lj = np.argmax(scores_ljp, axis=1)
    scores_term_bestlvl = scores_term.max(axis=1)
    bestlvl_term = np.argmax(scores_term, axis=1)
    print >> f1, 'Average and statistics for all ', n_runs, ' fits'
    print >> f1, 'j, parity', np.mean(scores_jp_bestlvl), ' +- ', np.std(scores_jp_bestlvl), ' at level number ', \
        np.mean(bestlvl_j), ' +- ', np.std(bestlvl_j)
    print >> f1, 'l, parity', np.mean(scores_lp_bestlvl), ' +- ', np.std(scores_lp_bestlvl), ' at level number ', \
        np.mean(bestlvl_l), ' +- ', np.std(bestlvl_l)
    print >> f1, 'l, j, parity', np.mean(scores_ljp_bestlvl), ' +- ', np.std(scores_ljp_bestlvl), ' at level number ', \
        np.mean(bestlvl_lj), ' +- ', np.std(bestlvl_lj)
    print >> f1, 'term', np.mean(scores_term_bestlvl), ' +- ', np.std(scores_term_bestlvl), ' at level number ', \
        np.mean(bestlvl_term), ' +- ', np.std(bestlvl_term)
    print G.graph_properties.name, 'j, parity', np.mean(scores_jp_bestlvl), ' +- ', np.std(scores_jp_bestlvl)
    print >> f1, 'parity', np.mean(scores_p), ' +- ', np.std(scores_p), ' at level number ', np.mean(num_levels)-2.0, \
                ' +- ', np.std(num_levels)
    print G.graph_properties.name, 'parity', np.mean(scores_p), ' +- ', np.std(scores_p)
    # evaluation for the blockstate with the lowest entropy: most likely state to describe data
    nested_state0 = best_block(nested_state_lvls[i])
    # nested_state0.draw(vertex_text=He.vp.term, vertex_size=7, vertex_font_size=4,
    #                     output="./results/plots/experimental-helium-nested-sbm-minimum.svg")
    nlevels = len(nested_state0.get_bs())
    maps = [nested_state0.project_level(j).get_blocks() for j in range(nlevels)]
    print >> f1, '\n', 'Lowest entropy  fit:'
    for count, map in enumerate(maps):
        score_p = rand_score(G, map, G.vp.parity)
        score_lp = rand_score(G, map, G.vp.lparity)
        score_j = rand_score(G, map, G.vp.J)
        score_term = rand_score(G, map, G.vp.term)
        score_ljp = rand_score(G, map, G.vp.ljp)
        score_jp = rand_score(G, map, G.vp.jparity)
        print >> f1, 'parity \t lp \t ljp \t term \t j \t jp'
        print >> f1, score_p, score_lp, score_ljp, score_term, score_j, score_jp
        print G.graph_properties.name, score_p, score_lp, score_ljp, score_term, score_j, score_jp
        # add communities as vp
        cname = 'communities' + str(count)
        G.vp[cname] = map
    G.save(graphnames[i], fmt='gml')
f1.close()


# for i in range(n_runs):
#     nested_stateH = gt.minimize_nested_blockmodel_dl(H, deg_corr=True, mcmc_args=dict(niter=10))
#     # nested_stateH.draw(vertex_text=H.vp.term, vertex_size=7, vertex_font_size=4, output="../plots/experimental-hydrogen-nested-sbm-minimum.svg")
#     nested_stateH_0.append(nested_stateH.get_levels()[0])
#     nested_stateH_lvls.append(nested_stateH)
#     # nested_stateH_0.draw(vertex_text=H.vp.term, vertex_size=7, vertex_font_size=4, output="../plots/experimental-hydrogen-nested-sbm-minimum_first_level.svg")
#     nested_stateTheo = gt.minimize_nested_blockmodel_dl(Theo, deg_corr=True, mcmc_args=dict(niter=10))
#     # nested_stateTheo.draw(vertex_text=Theo.vp.term, vertex_size=7, vertex_font_size=4, output="../plots/theoretical-hydrogen-nested-sbm-minimum.svg")
#     nested_stateTheo_0.append(nested_stateTheo.get_levels()[0])
#     nested_stateTheo_lvls.append(nested_stateTheo)
#     # filenameTheo = "../plots/communities/theoretical-hydrogen-nested-sbm-minimum-groundstate-run-" + str(i) + ".svg"
#     # nested_stateTheo_0[i].draw(vertex_text=Theo.vp.term, vertex_size=7, vertex_font_size=4, output=filenameTheo)
#     nested_stateHe = gt.minimize_nested_blockmodel_dl(He, deg_corr=True, mcmc_args=dict(niter=10))
#     # print i, nested_stateHe.entropy()
#     # nested_stateHe.draw(vertex_text=He.vp.term, vertex_size=7, vertex_font_size=4, output="../plots/experimental-helium-nested-sbm-minimum.svg")
#     nested_stateHe_0.append(nested_stateHe.get_levels()[0])
#     nested_stateHe_lvls.append(nested_stateHe)
#     # filename = "../plots/communities/experimental-helium-reduced-nested-sbm-minimum-groundstate-run-" + str(i) + ".svg"
#     # nested_stateHe_0[i].draw(vertex_text=He.vp.term, vertex_size=7, vertex_font_size=4, output=filename)
#     nested_stateFe = gt.minimize_nested_blockmodel_dl(Fe, deg_corr=True, mcmc_args=dict(niter=10))
#     # nested_stateFe.draw(vertex_text=Fe.vp.term, vertex_size=5, vertex_font_size=3, output="../plots/experimental-iron-nested-sbm-minimum.svg")
#     nested_stateFe_0 = nested_stateFe.get_levels()[0]
#     nested_stateFe_lvls.append(nested_stateFe)
#     nested_stateC = gt.minimize_nested_blockmodel_dl(C, deg_corr=True, mcmc_args=dict(niter=10))
#     # nested_stateC.draw(vertex_text=C.vp.term, vertex_size=5, vertex_font_size=3, output="../plots/experimental-carbon-nested-sbm-minimum.svg")
#     nested_stateC_0 = nested_stateC.get_levels()[0]
#     nested_stateC_lvls.append(nested_stateC)
#     # nested_stateFe_0.draw(vertex_text=Fe.vp.term, vertex_size=7, vertex_font_size=4, output="../plots/experimental-iron-nested-sbm-minimum_first_level.svg")
#     # stateH_0.append(gt.minimize_blockmodel_dl(H, deg_corr=True, mcmc_args=dict(niter=1000)))
#     # stateTheo_0.append(gt.minimize_blockmodel_dl(Theo, deg_corr=True, mcmc_args=dict(niter=1000)))
#     # stateHe_0.append(gt.minimize_blockmodel_dl(He, deg_corr=True, mcmc_args=dict(niter=1000)))
#     # stateFe_0 = gt.minimize_blockmodel_dl(Fe, deg_corr=True, mcmc_args=dict(niter=10000))






# Save Hydrogen Network
# gn0 = Theo.new_vertex_property("int")
# Theo.vp.gn0 = nested_stateTheo_0[0].get_blocks()
# Theo.vp.gn1 = nested_stateTheo_0[1].get_blocks()
# Theo.vp.gn2 = nested_stateTheo_0[2].get_blocks()
# Theo.vp.gn3 = nested_stateTheo_0[3].get_blocks()
# Theo.vp.gn4 = nested_stateTheo_0[4].get_blocks()
# Theo.vp.gn5 = nested_stateTheo_0[5].get_blocks()
# Theo.vp.gn6 = nested_stateTheo_0[6].get_blocks()
# Theo.vp.gn7 = nested_stateTheo_0[7].get_blocks()
# Theo.vp.gn8 = nested_stateTheo_0[8].get_blocks()
# Theo.vp.gn9 = nested_stateTheo_0[9].get_blocks()
# ground_truth_l = Theo.new_vertex_property("int")
# ground_truth_j = Theo.new_vertex_property("int")
# ground_truth_lj = Theo.new_vertex_property("int")
# gtl = ground_truth(nxTheo, ['l'])
# gtj = ground_truth(nxTheo, ['J'])
# gtlj = ground_truth(nxTheo, ['l', 'J'])
# gtpar = ground_truth(nxTheo, ['parity'])
# Theo.vp.ground_truth_l = ground_truth_l
# Theo.vp.ground_truth_j = ground_truth_j
# Theo.vp.ground_truth_lj = ground_truth_lj
# node_id = Theo.vp.nodeid


# for i in Theo.vertices():
#     Theo.vp.ground_truth_l[i] = gtl[node_id[i]]
#     Theo.vp.ground_truth_j[i] = gtj[node_id[i]]
#     Theo.vp.ground_truth_lj[i] = gtlj[node_id[i]]

# label = np.zeros(len(nxTheo))
# l = np.zeros(len(nxTheo))
# j = np.zeros(len(nxTheo))
# lj = np.zeros(len(nxTheo))
# par = np.zeros(len(nxTheo))








def rand(nxG, gtG, state_G_0, nested_state_G_0, filename):
    """
    Calculates different adjusted rand indices for different ground truth and prints them into a file.
    """
    truth_l = np.zeros(len(nxG.nodes()))
    truth_term = np.zeros(len(nxG.nodes()))
    truth_lj = np.zeros(len(nxG.nodes()))
    truth_j = np.zeros(len(nxG.nodes()))
    truth_parity = np.zeros(len(nxG.nodes()))
    groups_nSBM = np.zeros((len(nested_state_G_0), len(nxG.nodes())))
    groups_SBM = np.zeros((len(state_G_0), len(nxG.nodes())))
    groups_Newman = np.zeros(len(nxG.nodes()))
    gt_term = ground_truth(nxG, keys=['term'])
    gt_l = ground_truth(nxG, keys=['l'])
    gt_lj = ground_truth(nxG, keys=['l', 'J'])
    gt_j = ground_truth(nxG, keys=['J'])
    gt_parity = ground_truth(nxG, keys=['parity'])
    label_SBM = []
    label_nSBM = []
    for j in range(len(nested_state_G_0)):
        label_SBM.append(find_label(gtG, state_G_0[j]))
        label_nSBM.append(find_label(gtG, nested_state_G_0[j]))
    Newman = nx2.Newman_community_detection(nxG)
    for i, node in enumerate(nxG.nodes()):
        truth_term[i] = gt_term[node]
        truth_l[i] = gt_l[node]
        truth_lj[i] = gt_lj[node]
        truth_j[i] = gt_j[node]
        truth_parity[i] = gt_parity[node]
        for j in range(len(nested_state_G_0)):
            groups_SBM[j][i] = label_SBM[j][node]
            groups_nSBM[j][i] = label_nSBM[j][node]
        if node in Newman[0]:
            groups_Newman[i] = 1

    rand_score_sbm_term = np.zeros(len(nested_state_G_0))
    rand_score_nested_sbm_term = np.zeros(len(nested_state_G_0))
    rand_score_sbm_l = np.zeros(len(nested_state_G_0))
    rand_score_nested_sbm_l = np.zeros(len(nested_state_G_0))
    rand_score_sbm_lj = np.zeros(len(nested_state_G_0))
    rand_score_nested_sbm_lj = np.zeros(len(nested_state_G_0))
    rand_score_sbm_j = np.zeros(len(nested_state_G_0))
    rand_score_nested_sbm_j = np.zeros(len(nested_state_G_0))
    for i in range(len(nested_state_G_0)):
        rand_score_sbm_term[i] = adjusted_rand_score(truth_term, groups_SBM[i])
        rand_score_nested_sbm_term[i] = adjusted_rand_score(truth_term, groups_nSBM[i])
        rand_score_sbm_l[i] = adjusted_rand_score(truth_l, groups_SBM[i])
        rand_score_nested_sbm_l[i] = adjusted_rand_score(truth_l, groups_nSBM[i])
        rand_score_sbm_lj[i] = adjusted_rand_score(truth_lj, groups_SBM[i])
        rand_score_nested_sbm_lj[i] = adjusted_rand_score(truth_lj, groups_nSBM[i])
        rand_score_sbm_j[i] = adjusted_rand_score(truth_j, groups_SBM[i])
        rand_score_nested_sbm_j[i] = adjusted_rand_score(truth_j, groups_nSBM[i])
    rand_score_newman_l = adjusted_rand_score(truth_l, groups_Newman)
    rand_score_newman_term = adjusted_rand_score(truth_term, groups_Newman)
    rand_score_newman_lj = adjusted_rand_score(truth_lj, groups_Newman)
    rand_score_newman_j = adjusted_rand_score(truth_j, groups_Newman)
    rand_score_newman_parity = adjusted_rand_score(truth_parity, groups_Newman)

    rand_score_sbm_term_avg = rand_score_sbm_term.mean()
    rand_score_nested_sbm_term_avg = rand_score_nested_sbm_term.mean()
    rand_score_sbm_l_avg = rand_score_sbm_l.mean()
    rand_score_nested_sbm_l_avg = rand_score_nested_sbm_l.mean()
    rand_score_sbm_lj_avg = rand_score_sbm_lj.mean()
    rand_score_nested_sbm_lj_avg = rand_score_nested_sbm_lj.mean()
    rand_score_sbm_j_avg = rand_score_sbm_j.mean()
    rand_score_nested_sbm_j_avg = rand_score_nested_sbm_j.mean()

    rand_score_sbm_term_std = rand_score_sbm_term.std()
    rand_score_nested_sbm_term_std = rand_score_nested_sbm_term.std()
    rand_score_sbm_l_std = rand_score_sbm_l.std()
    rand_score_nested_sbm_l_std = rand_score_nested_sbm_l.std()
    rand_score_sbm_lj_std = rand_score_sbm_lj.std()
    rand_score_nested_sbm_lj_std = rand_score_nested_sbm_lj.std()
    rand_score_sbm_j_std = rand_score_sbm_j.std()
    rand_score_nested_sbm_j_std = rand_score_nested_sbm_j.std()




    filename = './../plots/rand-scores/' + filename + '-nruns' + str(n_runs)
    f1 = open(filename, 'w+')

    print >> f1, "Newman compared to parity: ", rand_score_newman_parity
    print >> f1, "nSBM compared to parity: "
    print >> f1, "SBM compared to term: ", rand_score_sbm_term_avg, " +/- ", rand_score_sbm_term_std
    print >> f1, "nSBM compared to term: ", rand_score_nested_sbm_term_avg, "+/-", rand_score_nested_sbm_term_std
    print >> f1, "Newman compared to term: ", rand_score_newman_term
    print >> f1, "SBM compared to l: ", rand_score_sbm_l_avg, "+/-", rand_score_sbm_l_std
    print >> f1, "nSBM compared to l: ", rand_score_nested_sbm_l_avg, "+/-", rand_score_nested_sbm_l_std
    print >> f1, "Newman compared to l: ", rand_score_newman_l
    print >> f1, "SBM compared to l, j: ", rand_score_sbm_lj_avg, "+/-", rand_score_sbm_lj_std
    print >> f1, "nSBM compared to l, j: ", rand_score_nested_sbm_lj_avg, "+/-", rand_score_nested_sbm_lj_std
    print >> f1, "Newman compared to l, j: ", rand_score_newman_lj
    print >> f1, "SBM compared to j: ", rand_score_sbm_j_avg, "+/-", rand_score_sbm_j_std
    print >> f1, "nSBM compared to j: ", rand_score_nested_sbm_j_avg, "+/-", rand_score_nested_sbm_j_std
    print >> f1, "Newman compared to j: ", rand_score_newman_j


# Theo.save("theoretical-hydrogen-graph.gml", fmt="gml")


# Save Helium network
# He.vp.gn1 = nested_stateHe_0[1].get_blocks()
# He.vp.gn0 = nested_stateHe_0[0].get_blocks()
# He.vp.gn2 = nested_stateHe_0[2].get_blocks()
# He.vp.gn3 = nested_stateHe_0[3].get_blocks()
# He.vp.gn4 = nested_stateHe_0[4].get_blocks()
# He.vp.gn5 = nested_stateHe_0[5].get_blocks()
# He.vp.gn6 = nested_stateHe_0[6].get_blocks()
# He.vp.gn7 = nested_stateHe_0[7].get_blocks()
# He.vp.gn8 = nested_stateHe_0[8].get_blocks()
# He.vp.gn9 = nested_stateHe_0[9].get_blocks()
# ground_truth_lHe = He.new_vertex_property("int")
# ground_truth_jHe = He.new_vertex_property("int")
# ground_truth_ljHe = He.new_vertex_property("int")
# gtlHe = ground_truth(nxHe, ['l'])
# gtjHe = ground_truth(nxHe, ['J'])
# gtljHe = ground_truth(nxHe, ['l', 'J'])
# He.vp.ground_truth_l = ground_truth_lHe
# He.vp.ground_truth_j = ground_truth_jHe
# He.vp.ground_truth_lj = ground_truth_ljHe
# node_idHe = He.vp.nodeid
#
# for i in He.vertices():
#     He.vp.ground_truth_l[i] = gtlHe[node_idHe[i]]
#     He.vp.ground_truth_j[i] = gtjHe[node_idHe[i]]
#     He.vp.ground_truth_lj[i] = gtljHe[node_idHe[i]]
#
#
#
#
# He.save("helium-graph.gml", fmt="gml")


# # Save Iron Network
# group_numberHe = Fe.new_vertex_property("int")
# ground_truth_lHe = Fe.new_vertex_property("int")
# ground_truth_jHe = Fe.new_vertex_property("int")
# ground_truth_ljHe = Fe.new_vertex_property("int")
# Fe.vp.group_number = nested_stateFe_0.get_blocks()
# gtlHe = ground_truth(nxFe, ['l'])
# gtjHe = ground_truth(nxFe, ['J'])
# gtljHe = ground_truth(nxFe, ['l', 'J'])
# Fe.vp.ground_truth_l = ground_truth_lHe
# Fe.vp.ground_truth_j = ground_truth_jHe
# Fe.vp.ground_truth_lj = ground_truth_ljHe
# node_idHe = Fe.vp.nodeid


# for i in Fe.vertices():
#     Fe.vp.ground_truth_l[i] = gtlHe[node_idHe[i]]
#     Fe.vp.ground_truth_j[i] = gtjHe[node_idHe[i]]
#     Fe.vp.ground_truth_lj[i] = gtljHe[node_idHe[i]]


# Fe.save("iron-graph.gml", fmt="gml")


# # Compute adjusted rand index
# # Theoretical Hydrogen
# Theo_truth_l = np.zeros(len(nxTheo.nodes()))
# Theo_truth_term = np.zeros(len(nxTheo.nodes()))
# Theo_groups_nested = np.zeros(len(nxTheo.nodes()))
# Theo_groups = np.zeros(len(nxTheo.nodes()))
# for i, node in enumerate(nxTheo.nodes()):
#     Theo_truth_term[i] = ground_truth(nxTheo, keys=['term'])[node]
#     Theo_truth_l[i] = ground_truth(nxTheo, keys=['l'])[node]
#     Theo_groups[i] = find_label(Theo, stateTheo_0)[node]
#     Theo_groups_nested[i] = find_label(Theo, nested_stateTheo_0)[node]

# rand_score_sbm_Theo_term = adjusted_rand_score(Theo_truth_term, Theo_groups)
# rand_score_nested_sbm_Theo_term = adjusted_rand_score(Theo_truth_term, Theo_groups_nested)
# rand_score_sbm_Theo_l = adjusted_rand_score(Theo_truth_l, Theo_groups)
# rand_score_nested_sbm_Theo_l = adjusted_rand_score(Theo_truth_l, Theo_groups_nested)

# f1 = open('./randscores-SBM', 'w+')

# print >> f1, "Theoretical hydrogen not nested compared to term: ", rand_score_sbm_Theo_term
# print >> f1, "Theoretical hydrogen nested compared to term: ", rand_score_nested_sbm_Theo_term
# print >> f1, "Theoretical hydrogen not nested compared to l: ", rand_score_sbm_Theo_l
# print >> f1, "Theoretical hydrogen nested compared to l: ", rand_score_nested_sbm_Theo_l

# # Experimental Hydrogen
# H_truth_l = np.zeros(len(nxH.nodes()))
# H_truth_term = np.zeros(len(nxH.nodes()))
# H_groups_nested = np.zeros(len(nxH.nodes()))
# H_groups = np.zeros(len(nxH.nodes()))
# for i, node in enumerate(nxH.nodes()):
#     H_truth_term[i] = ground_truth(nxH, keys=['term'])[node]
#     H_truth_l[i] = ground_truth(nxH, keys=['l'])[node]
#     H_groups[i] = find_label(H, stateH_0)[node]
#     H_groups_nested[i] = find_label(H, nested_stateH_0)[node]

# rand_score_sbm_H_term = adjusted_rand_score(H_truth_term, H_groups)
# rand_score_nested_sbm_H_term = adjusted_rand_score(H_truth_term, H_groups_nested)
# rand_score_sbm_H_l = adjusted_rand_score(H_truth_l, H_groups)
# rand_score_nested_sbm_H_l = adjusted_rand_score(H_truth_l, H_groups_nested)

# print >> f1, "Experimental hydrogen not nested compared to term: ", rand_score_sbm_H_term
# print >> f1, "Experimental hydrogen nested compared to term: ", rand_score_nested_sbm_H_term
# print >> f1, "Experimental hydrogen not nested compared to l: ", rand_score_sbm_H_l
# print >> f1, "Experimental hydrogen nested compared to l: ", rand_score_nested_sbm_H_l

# # Helium
# He_truth_l = np.zeros(len(nxHe.nodes()))
# He_truth_lj = np.zeros(len(nxHe.nodes()))
# He_truth_term = np.zeros(len(nxHe.nodes()))
# He_groups_nested = np.zeros(len(nxHe.nodes()))
# He_groups = np.zeros(len(nxHe.nodes()))
# for i, node in enumerate(nxHe.nodes()):
#     He_truth_term[i] = ground_truth(nxHe, keys=['term'])[node]
#     He_truth_l[i] = ground_truth(nxHe, keys=['l'])[node]
#     He_truth_lj[i] = ground_truth(nxHe, keys=['l', 'J'])[node]
#     He_groups[i] = find_label(He, stateHe_0)[node]
#     He_groups_nested[i] = find_label(He, nested_stateHe_0)[node]

# rand_score_sbm_He_term = adjusted_rand_score(He_truth_term, He_groups)
# rand_score_nested_sbm_He_term = adjusted_rand_score(He_truth_term, He_groups_nested)
# rand_score_sbm_He_l = adjusted_rand_score(He_truth_l, He_groups)
# rand_score_nested_sbm_He_l = adjusted_rand_score(He_truth_l, He_groups_nested)
# rand_score_sbm_He_lj = adjusted_rand_score(He_truth_lj, He_groups)
# rand_score_nested_sbm_He_lj = adjusted_rand_score(He_truth_lj, He_groups_nested)

# print >> f1, "Helium not nested compared to term: ", rand_score_sbm_He_term
# print >> f1, "Helium nested compared to term: ", rand_score_nested_sbm_He_term
# print >> f1, "Helium not nested compared to l: ", rand_score_sbm_He_l
# print >> f1, "Helium nested compared to l: ", rand_score_nested_sbm_He_l
# print >> f1, "Helium not nested compared to l and j: ", rand_score_sbm_He_lj
# print >> f1, "Helium nested compared to l and j: ", rand_score_nested_sbm_He_lj

# # Iron
# Fe_truth_l = np.zeros(len(nxFe.nodes()))
# Fe_truth_term = np.zeros(len(nxFe.nodes()))
# Fe_truth_lj = np.zeros(len(nxFe.nodes()))
# Fe_groups_nested = np.zeros(len(nxFe.nodes()))
# Fe_groups = np.zeros(len(nxFe.nodes()))
# for i, node in enumerate(nxFe.nodes()):
#     Fe_truth_term[i] = ground_truth(nxFe, keys=['term'])[node]
#     Fe_truth_l[i] = ground_truth(nxFe, keys=['l'])[node]
#     Fe_truth_lj[i] = ground_truth(nxFe, keys=['l', 'J'])[node]
#     Fe_groups[i] = find_label(Fe, stateFe_0)[node]
#     Fe_groups_nested[i] = find_label(Fe, nested_stateFe_0)[node]

# rand_score_sbm_Fe_term = adjusted_rand_score(Fe_truth_term, Fe_groups)
# rand_score_nested_sbm_Fe_term = adjusted_rand_score(Fe_truth_term, Fe_groups_nested)
# rand_score_sbm_Fe_l = adjusted_rand_score(Fe_truth_l, Fe_groups)
# rand_score_nested_sbm_Fe_l = adjusted_rand_score(Fe_truth_l, Fe_groups_nested)
# rand_score_sbm_Fe_lj = adjusted_rand_score(Fe_truth_lj, Fe_groups)
# rand_score_nested_sbm_Fe_lj = adjusted_rand_score(Fe_truth_lj, Fe_groups_nested)

# print >> f1, "ATTENTION: Only largest connected component used!"
# print >> f1, "Iron not nested compared to term: ", rand_score_sbm_Fe_term
# print >> f1, "Iron nested compared to term: ", rand_score_nested_sbm_Fe_term
# print >> f1, "Iron not nested compared to l: ", rand_score_sbm_Fe_l
# print >> f1, "Iron nested compared to l: ", rand_score_nested_sbm_Fe_l
# print >> f1, "Iron not nested compared to l and j: ", rand_score_sbm_Fe_lj
# print >> f1, "Iron nested compared to l and j: ", rand_score_nested_sbm_Fe_lj

# f1.close()


# Link Prediction
# potential_edges = [(v1, v2) for v1 in Theo.vertices() for v2 in Theo.vertices() if not ((v1, v2) in Theo.edges() or v1 == v2)]
# potential_edges = [(1, v1) for v1 in Theo.vertices() if not (v1 == 1 or (1, v1) in Theo.edges())]
# probs = [[] for i in range(len(potential_edges))]
# bs = nested_stateTheo.get_bs()
# bs += [np.zeros(1)] * (6 - len(bs))
# nested_stateTheo = nested_stateTheo.copy(bs=bs, sampling=True)


# def collect_edge_probs(s):
#     for i in range(len(probs)):
#         p = s.get_edges_prob([potential_edges[i]], entropy_args=dict(partition_dl=False))
#         probs[i].append(p)


# gt.mcmc_equilibrate(nested_stateTheo, force_niter=2, mcmc_args=dict(niter=1), callback=collect_edge_probs)


# def get_avg(p):
#     p = np.array(p)
#     p_avg = np.exp(p).mean()
#     return p_avg


# probabilities = [(Theo.vp.nodeid[potential_edges[i][0]], Theo.vp.nodeid[potential_edges[i][1]], get_avg(probs[i])) for i in range(len(probs))]
# f2 = open('edge-probabilities-theo', mode='w+')
# print >> f2, "edge probabilities: ", probs


# Function to analyze the nested SBM of a network: gives lowest hierarchy groups best n partitions and saves then in output

def nested_SBM_analysis(g, output, number_of_histograms):
    # Find groundstate
    state = gt.minimize_nested_blockmodel_dl(g)
    # Before doing model averaging, the need to create a NestedBlockState
    # by passing sampling = True.

    # We also want to increase the maximum hierarchy depth to L = 10

    # We can do both of the above by copying.

    bs = state.get_bs()                     # Get hierarchical partition.
    bs += [np.zeros(1)] * (10 - len(bs))    # Augment it to L = 10 with
                                            # single-group levels.

    state = state.copy(bs=bs, sampling=True)

    # Now we run 1000 sweeps of the MCMC

    dS, nmoves = state.mcmc_sweep(niter=1000)

    print("Change in description length:", dS)
    print("Number of accepted vertex moves:", nmoves)

    # We will first equilibrate the Markov chain
    gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))

    pv = [None] * len(state.get_levels())

    def collect_marginals(s):
        global pv
        pv = [sl.collect_vertex_marginals(pv[l]) for l, sl in enumerate(s.get_levels())]

    # Now we collect the marginals for exactly 100,000 sweeps
    gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10), callback=collect_marginals)

    # Now the node marginals for all levels are stored in property map
    # list pv. We can visualize the first level as pie charts on the nodes:
    state_0 = state.get_levels()[0]
    state_0.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv[0], edge_gradient=None, output=output+"-nested-sbm-marginals.svg")
    for i in range(number_of_histograms):
        state.mcmc_sweep(niter=1000)
        state.draw(output=output+"nested-sbm-partition-sample-%i.svg" % i, empty_branches=False)



# # Before doing model averaging, the need to create a NestedBlockState
# # by passing sampling = True.

# # We also want to increase the maximum hierarchy depth to L = 10

# # We can do both of the above by copying.

# bs = stateH.get_bs()                     # Get hierarchical partition.
# bs += [np.zeros(1)] * (10 - len(bs))    # Augment it to L = 10 with
#                                         # single-group levels.

# stateH = stateH.copy(bs=bs, sampling=True)

# # Now we run 1000 sweeps of the MCMC

# dS, nmoves = stateH.mcmc_sweep(niter=1000)

# print("Change in description length of H:", dS)
# print("Number of accepted vertex moves:", nmoves)

# # Before doing model averaging, the need to create a NestedBlockState
# # by passing sampling = True.

# # We also want to increase the maximum hierarchy depth to L = 10

# # We can do both of the above by copying.

# bs = stateHe.get_bs()                     # Get hierarchical partition.
# bs += [np.zeros(1)] * (10 - len(bs))    # Augment it to L = 10 with
#                                         # single-group levels.

# stateHe = stateHe.copy(bs=bs, sampling=True)

# # Now we run 1000 sweeps of the MCMC

# dS, nmoves = stateHe.mcmc_sweep(niter=1000)

# print("Change in description length of He:", dS)
# print("Number of accepted vertex moves:", nmoves)

# # Before doing model averaging, the need to create a NestedBlockState
# # by passing sampling = True.

# # We also want to increase the maximum hierarchy depth to L = 10

# # We can do both of the above by copying.

# bs = stateTheo.get_bs()                     # Get hierarchical partition.
# bs += [np.zeros(1)] * (10 - len(bs))    # Augment it to L = 10 with
#                                         # single-group levels.

# stateTheo = stateTheo.copy(bs=bs, sampling=True)

# # Now we run 1000 sweeps of the MCMC

# dS, nmoves = stateTheo.mcmc_sweep(niter=1000)

# print("Change in description length of H, theoretical:", dS)
# print("Number of accepted vertex moves:", nmoves)



# # We will first equilibrate the Markov chain
# gt.mcmc_equilibrate(stateH, wait=1000, mcmc_args=dict(niter=10))

# pv = [None] * len(stateH.get_levels())

# def collect_marginals(s):
#    global pv
#    pv = [sl.collect_vertex_marginals(pv[l]) for l, sl in enumerate(s.get_levels())]

# # Now we collect the marginals for exactly 100,000 sweeps
# gt.mcmc_equilibrate(stateH, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_marginals)

# # Now the node marginals for all levels are stored in property map
# # list pv. We can visualize the first level as pie charts on the nodes:
# state_0 = stateH.get_levels()[0]
# state_0.draw(vertex_shape="pie", vertex_pie_fractions=pv[0], vertex_text = H.vp.term, edge_gradient=None, output="../plots/experimental-hydrogen-nested-sbm-marginals.svg")


# for i in range(4):
#     stateH.mcmc_sweep(niter=1000)
#     stateH.draw(output="../plots/experimental-hydrogen-nested-sbm-partition-sample-%i.svg" % i,vertex_text = H.vp.term, vertex_size = 5, vertex_font_size = 3, output_size = (1000,1000),  empty_branches=False)

# #### Experimental Helium


# # We will first equilibrate the Markov chain
# gt.mcmc_equilibrate(stateHe, wait=1000, mcmc_args=dict(niter=10))

# pv = [None] * len(stateHe.get_levels())

# # Now we collect the marginals for exactly 100,000 sweeps
# gt.mcmc_equilibrate(stateHe, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_marginals)

# # Now the node marginals for all levels are stored in property map
# # list pv. We can visualize the first level as pie charts on the nodes:
# state_0 = stateHe.get_levels()[0]
# state_0.draw(vertex_shape="pie", vertex_pie_fractions=pv[0], vertex_text = He.vp.term, edge_gradient=None, output="../plots/experimental-helium-nested-sbm-marginals.svg")

# for i in range(4):
#     stateHe.mcmc_sweep(niter=1000)
#     stateHe.draw(output="../plots/experimental-helium-nested-sbm-partition-sample-%i.svg" % i, vertex_text = He.vp.term, vertex_size = 5, vertex_font_size = 3, output_size = (1000,1000), empty_branches=False)


# #### Theoretical Hydrogen

# # Number of groups for each level
# h = [np.zeros(Theo.num_vertices() + 1) for s in stateTheo.get_levels()]

# def collect_num_groups(s):
#     for l, sl in enumerate(s.get_levels()):
#        B = sl.get_nonempty_B()
#        h[l][B] += 1


# # Now we collect the marginal distribution for exactly 100,000 sweeps
# gt.mcmc_equilibrate(stateTheo, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_num_groups)


# # We will first equilibrate the Markov chain
# gt.mcmc_equilibrate(stateTheo, wait=1000, mcmc_args=dict(niter=10))

# pv = [None] * len(stateTheo.get_levels())

# # Now we collect the marginals for exactly 100,000 sweeps
# gt.mcmc_equilibrate(stateTheo, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_marginals)

# # Now the node marginals for all levels are stored in property map
# # list pv. We can visualize the first level as pie charts on the nodes:
# state_0 = stateTheo.get_levels()[0]
# state_0.draw(vertex_shape="pie", vertex_pie_fractions=pv[0], vertex_text = Theo.vp.term, edge_gradient=None, output="../plots/theoretical-hydrogen-nested-sbm-marginals.svg")

# for i in range(4):
#     stateTheo.mcmc_sweep(niter=1000)
#     stateTheo.draw(output="../plots/theoretical hydrogen-nested-sbm-partition-sample-%i.svg" % i, vertex_text = Theo.vp.term, vertex_size = 5, vertex_font_size = 3, output_size = (1000,1000), empty_branches=False)


# print "Theoretical Hydrogen Histogram:", h
# plt.figure()
# plt.hist(h)
# plt.xlabel('Number of Groups')
# plt.ylabel('Probability')
# plt.savefig('../plots/theoretical-hydrogen-histogram')


# ###################################### Now we play the same game for the not nested model ######################################################


# #Hydrogen

# stateH = gt.minimize_blockmodel_dl(H)
# stateH = stateH.copy(B=H.num_vertices())
# dS, nmoves = stateH.mcmc_sweep(niter=1000)

# print("Change in description length:", dS)
# print("Number of accepted vertex moves:", nmoves)

# # We will first equilibrate the Markov chain
# gt.mcmc_equilibrate(stateH, wait=1000, mcmc_args=dict(niter=10))

# pv_nonnested = None

# def collect_marginals_nonnested(s):
#    global pv_nonnested
#    pv_nonnested = s.collect_vertex_marginals(pv_nonnested)

# # Now we collect the marginals for exactly 100,000 sweeps, at
# # intervals of 10 sweeps:
# gt.mcmc_equilibrate(stateH, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_marginals_nonnested)

# # Now the node marginals are stored in property map pv_nonnested. We can
# # visualize them as pie charts on the nodes:
# stateH.draw(vertex_shape="pie", vertex_text = H.vp.term, vertex_pie_fractions=pv_nonnested,
#            edge_gradient=None, output="../plots/experimental-hydrogen-sbm-marginals.svg")


# h_nonnested = np.zeros(H.num_vertices() + 1)

# def collect_num_groups_nonnested(s):
#     B = s.get_nonempty_B()
#     h_nonnested[B] += 1

# # Now we collect the marginals for exactly 100,000 sweeps, at
# # intervals of 10 sweeps:
# gt.mcmc_equilibrate(stateH, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_num_groups_nonnested)

# #Helium

# stateHe = gt.minimize_blockmodel_dl(He)
# stateHe = stateHe.copy(B=He.num_vertices())
# dS, nmoves = stateHe.mcmc_sweep(niter=1000)

# print("Change in description length:", dS)
# print("Number of accepted vertex moves:", nmoves)

# # We will first equilibrate the Markov chain
# gt.mcmc_equilibrate(stateHe, wait=1000, mcmc_args=dict(niter=10))

# pv_nonnested = None

# # Now we collect the marginals for exactly 100,000 sweeps, at
# # intervals of 10 sweeps:
# gt.mcmc_equilibrate(stateHe, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_marginals_nonnested)

# # Now the node marginals are stored in property map pv_nonnested. We can
# # visualize them as pie charts on the nodes:
# stateHe.draw(vertex_shape="pie", vertex_text = He.vp.term, vertex_pie_fractions=pv_nonnested,
#            edge_gradient=None, output="../plots/experimental-helium-sbm-marginals.svg")


# h_nonnested = np.zeros(He.num_vertices() + 1)

# # Now we collect the marginals for exactly 100,000 sweeps, at
# # intervals of 10 sweeps:
# gt.mcmc_equilibrate(stateHe, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_num_groups_nonnested)

# #Hydrogen, theoretical

# stateTheo = gt.minimize_blockmodel_dl(Theo)
# stateTheo = stateTheo.copy(B=Theo.num_vertices())
# dS, nmoves = stateTheo.mcmc_sweep(niter=1000)

# print("Change in description length:", dS)
# print("Number of accepted vertex moves:", nmoves)

# # We will first equilibrate the Markov chain
# gt.mcmc_equilibrate(stateTheo, wait=1000, mcmc_args=dict(niter=10))

# pv_nonnested = None

# # Now we collect the marginals for exactly 100,000 sweeps, at
# # intervals of 10 sweeps:
# gt.mcmc_equilibrate(stateTheo, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_marginals_nonnested)

# # Now the node marginals are stored in property map pv_nonnested. We can
# # visualize them as pie charts on the nodes:
# stateTheo.draw(vertex_shape="pie", vertex_text = Theo.vp.term, vertex_pie_fractions=pv_nonnested,
#            edge_gradient=None, output="../plots/theoretical-hydrogen-sbm-marginals.svg")


# h_nonnested = np.zeros(Theo.num_vertices() + 1)

# # Now we collect the marginals for exactly 100,000 sweeps, at
# # intervals of 10 sweeps:
# gt.mcmc_equilibrate(stateTheo, force_niter=1000, mcmc_args=dict(niter=10),
#                     callback=collect_num_groups_nonnested)
