
import matplotlib.pyplot as plt
import pickle
import pandas
from sbm_thesis_plots import *
from roc_plots import *
from sklearn.metrics import adjusted_rand_score



# plot_ari(['H1.0_JB_dipole', 'He1.0'], save=True, fname='plots/master-ari-plot-HHe.eps')
# plot_ari(['Fe1.0', 'Th2.0'], save=True, fname='plots/master-ari-plot-FeTh.eps')
# plot_hyd_ari()
# plot_he_ari()
# plot_fe_ari()
# plot_th_ari()

#
# plot_pred(['Fe1.0'], attrs=['parity', 'J', 'term', 'configuration'],
#           save=True, fname='plots/pred-score-Fe.pdf')
# plot_pred(['H1.0_JB_dipole'], attrs=['parity', 'l', 'J', 'term'],
#           save=True, fname='plots/pred-score-H.pdf')
# plot_pred(['Th2.0'], attrs=['parity', 'J', 'configuration'],
#           save=True, fname='plots/pred-score-Th2.pdf')
# plot_pred(['He1.0'], attrs=['parity', 'l', 'sl', 'J', 'term'],
#           save=True, fname='plots/pred-score-He.pdf')
# plot_pred_no_energies('H1.0_JB_dipole', attrs=['parity', 'J', 'sl', 'term', 'configuration'],
#           save=True, fname='plots/master-pred-score-plot-theory.eps')
#
# weight_els = ['He1.0', 'Fe1.0']
# for el in weight_els:
#     with open(el + '-hierarchies.pickle') as file:
#         hs = pickle.load(file)
#     g = hs[0].g
#     edge_weight_histogram(g, g.ep.matrixElement, g.ep.type, save=True, fname='plots/weight-hist-'+el[:-2]+'.pdf')
# with open('H1.0_JB_all-hierarchies.pickle') as file:
#     hs = pickle.load(file)
# g = hs[0].g
# find_edge_type_theory(g)
# edge_weight_histogram(g, g.ep.matrixElement, g.ep.type, save=True, fname='plots/master-weight-hist-H1_JB_all.pdf')
#
# ----------------------------------------------------------------------------------
# with open('Th2.0-hierarchies.pickle') as file:
#     hs = pickle.load(file)
# g = hs[0].g
# fig, ax = plt.subplots()
# ax.hist(g.ep.wavelength.a ** (-1) * 10 ** 8, bins=50)
# ax.set_xlabel('Transition Energy in cm$^{-1}$')
# ax.set_ylabel('Number of Transitions')
# fig.savefig('thorium-wl-hist.pdf')

# -----------------------------------------------------------------------------------------
with open('He1.0-hierarchies_old.pickle') as file:
    hs = pickle.load(file)
g = hs[0].g
# find_configuration(g)
# clean_term(g)
# add_leading_percentages(g, 'Th', 2.)
# fig, ax = plot_th2_energies_by_j(hs[0])
# plt.savefig('th2-energies-vs-jp.svg')
# fig, ax = plot_energies(hs[0])
# plt.savefig('th2-energies.pdf')
# edge_weight_histogram(g, g.ep.wavelength, g.ep.type, save=True, fname='plots/thorium-wl-hist.pdf')
groups = [0, 1, 19, 20]

# Plot all groups to see the data
# for i in range(8):
#     groups = list(range(4*i, 4*i + 4))
#     group_vs_attr_plot(g, groups, hs[0].get_levels()[0].get_blocks(), g.vp.energy, save=True,
#                        fname='plots/energy-groups/groups-vs-energies-Th2- ' + str(4*i) + '.pdf')
#
# for group in groups:
#     if group < 20:
#         groups_vs_energies_with_attr_plot(g, group, hs[0].get_levels()[0].get_blocks(), g.vp.configuration,
#                                           fname='plots/energy-groups/configuration-vs-energy-group'
#                                           + str(group) + '.pdf')
#     if group == 20:
#         groups_vs_energies_with_attr_plot(g, group, hs[0].get_levels()[0].get_blocks(), g.vp.configuration,
#                                           fname='plots/energy-groups/configuration-vs-energy-group'
#                                                 + str(group) + '.pdf', legend=True)
    # groups_vs_energies_with_attr_plot(g, group, hs[0].get_levels()[0].get_blocks(), g.vp.term,
    #                                   fname='plots/energy-groups/term-vs-energy-group' + str(group) + '.pdf')


# group_vs_attr_plot(g, [0.], g.vp.dummyCommunity, g.vp.energy, save=True,
#                    fname='plots/energy-groups/all-nodes-Th2.pdf')
# groups_vs_energies_with_attr_plot(g, 'o1/2', concat_pmaps([g.vp.parity, g.vp.J]), g.vp.configuration,
#                                   fname='plots/energy-groups/o1-configuration.pdf')
# group_vs_attr_plot(g, ['o1/2', 'o3/2', 'o5/2', 'o7/2'], concat_pmaps([g.vp.parity, g.vp.J]), g.vp.energy, save=True,
#                    fname='plots/energy-groups/split-by-jpar-o1to7.pdf')
# group0 = g.new_vp('string')
# group1 = g.new_vp('string')
# group6 = g.new_vp('string')
# for v in g.vertices():
#     group0[v] = 'Other Nodes'
#     group1[v] = 'Other Nodes'
#     group6[v] = 'Other Nodes'
#     if hs[0].get_levels()[0].get_blocks()[v] == 0:
#         group0[v] = 'Group 0'
#     if hs[0].get_levels()[0].get_blocks()[v] == 1:
#         group1[v] = 'Group 1'
#     if hs[0].get_levels()[0].get_blocks()[v] == 6:
#         group6[v] = 'Group 6'
# groups_vs_energies_with_attr_plot(g, 0., g.vp.dummyCommunity, group1, fname='plots/energy-groups/all-with-group1.pdf')
# groups_vs_energies_with_attr_plot(g, 0., g.vp.dummyCommunity, group0, fname='plots/energy-groups/all-with-group0.pdf')
# groups_vs_energies_with_attr_plot(g, 0., g.vp.dummyCommunity, group6, fname='plots/energy-groups/all-with-group6.pdf')

# groups_vs_energies_with_attr_plot(g, 'o9/2', concat_pmaps([g.vp.parity, g.vp.J]), g.vp.configuration,
#                                   fname='plots/energy-groups/o9-configuration.pdf')
# groups_vs_energies_with_attr_plot(g, 'o9/2', concat_pmaps([g.vp.parity, g.vp.J]), hs[0].get_levels()[0].get_blocks(),
#                                   fname='plots/energy-groups/o9-groups.pdf')
# groups_vs_energies_with_attr_plot(g, 'o11/2', concat_pmaps([g.vp.parity, g.vp.J]), g.vp.configuration,
#                                   fname='plots/energy-groups/o11-configuration.pdf')
# groups_vs_energies_with_attr_plot(g, 'o11/2', concat_pmaps([g.vp.parity, g.vp.J]), hs[0].get_levels()[0].get_blocks(),
#                                   fname='plots/energy-groups/o11-groups.pdf')
# groups_vs_energies_with_attr_plot(g, 'e5/2', concat_pmaps([g.vp.parity, g.vp.J]), g.vp.configuration,
#                                   fname='plots/energy-groups/e5-configuration.pdf')
# groups_vs_energies_with_attr_plot(g, 'e5/2', concat_pmaps([g.vp.parity, g.vp.J]), hs[0].get_levels()[0].get_blocks(),
#                                   fname='plots/energy-groups/e5-groups.pdf')
# groups_vs_energies_with_attr_plot(g, 'o3/2', concat_pmaps([g.vp.parity, g.vp.J]), g.vp.configuration,
#                                   fname='plots/energy-groups/o3-configuration.pdf')
# groups_vs_energies_with_attr_plot(g, 'o3/2', concat_pmaps([g.vp.parity, g.vp.J]), hs[0].get_levels()[0].get_blocks(),
#                                   fname='plots/energy-groups/o3-groups.pdf')
# groups_vs_energies_with_attr_plot(g, '5f.6d.7s', g.vp.configuration, g.vp.J,
#                                   fname='plots/energy-groups/5f6d7s-J.pdf')
# groups_vs_energies_with_attr_plot(g, '5f.6d.7s', g.vp.configuration, hs[0].get_levels()[0].get_blocks(),
#                                   fname='plots/energy-groups/5f6d7s-groups.pdf')

# ---------------------------- ENTROPY EVALUATION -------------------------------

group_similarity = np.zeros((2, len(hs)))
next_neighbor_similarity = np.zeros((2, len(hs[1:])))
jp_group_similarity = np.zeros(len(hs))
# Only take nodes with known J
j_ok = np.array([not bool(re.findall(r'or', g.vp.J[v])) for v in g.vertices()], dtype=bool)
jp_array = np.array([int(re.findall(r'\d+', g.vp.J[v])[0]) - (g.vp.parity[v] == 'e') for v in g.vertices()], dtype=int)
jp_values = jp_array[j_ok]
for i, h in enumerate(hs[1:]):
    for level in range(2):
        group_similarity[level, i+1] = adjusted_rand_score(hs[0].project_level(level).b.a, h.project_level(level).b.a)
        next_neighbor_similarity[level, i] = adjusted_rand_score(hs[i-1].project_level(level).b.a,
                                                                 h.project_level(level).b.a)
for i, h in enumerate(hs):
    community_values = h.project_level(1).b.a[j_ok]
    jp_group_similarity[i] = adjusted_rand_score(jp_values, community_values)
fig, ax = plt.subplots(ncols=2, figsize=(14, 5))
for level in range(2):
    group_similarity[level, 0] = 1.
    ax[0].plot([h.entropy() for h in hs], group_similarity[level], 'o', label='Level ' + str(level))
    ax[1].plot([h.entropy() for h in hs[:-1]], next_neighbor_similarity[level], 'o', label='Level ' + str(level))
# ax[2]._get_lines.get_next_color()
# ax[2].plot([h.entropy() for h in hs], jp_group_similarity, 'o')
ax[0].set_ylabel('Adjusted Rand Index')
ax[0].set_xlabel('Entropy')
ax[1].set_xlabel('Entropy')
# ax[2].set_xlabel('Entropy')
ax[0].set_title('Similarity compared to best fit')
ax[1].set_title('Similarity compared to next worst fit')
# ax[2].set_title('Similarity compared to parity and J')
ax[0].legend(loc='best')
fig.savefig('entropy-eval.pdf')

# ------------------------------- J PREDICTION ----------------------------------

# # Find nodes with unknown J
# predict_nodes = np.argwhere(np.array([bool(re.search(r'or', g.vp.J[v])) for v in g.vertices()])).flatten()
# j_prediction = dict()
# js = [str(2*n+1) + '/2' for n in range(8)]
# data = {j: np.zeros(len(predict_nodes), dtype=float) for j in js}
# for i, node in enumerate(predict_nodes):
#     j_prediction[node] = attribute_probabilities_from_group(hs, node, g.vp.J)
#     for k in j_prediction[node].keys():
#         if re.search(r'or', k):
#             del j_prediction[node][k]
#     normalization = np.sum(j_prediction[node].values())
#     for k in j_prediction[node].keys():
#         j_prediction[node][k] /= normalization
#     for j in js:
#         try: data[j][i] = j_prediction[node][j]
#         except KeyError: pass
#     data.update({'Energy': g.vp.energy.a[predict_nodes], 'NIST': [g.vp.J[v] for v in predict_nodes]})
# prediction_result = pd.DataFrame(data=data, columns=['Energy', 'NIST']+js).set_index('Energy').sort_values('Energy')
# tables_from_data(prediction_result, 'J-pred-table.tex')





# ----------------------------------- ROCs --------------------------------------

atoms = ['He1.0', 'Th2.0']

# lp_methods = ['nSBM', 'SPM']
# lp_data = load_lp_data(atoms, lp_methods, [0.1])
# for atom in atoms:
#     plot_rocs(lp_data, [atom], lp_methods, [0.1], fname='plots/rocs/link-pred-rocs-' + atom[:-2] + '.pdf')

# np_methods = ['adjacency']  #, 'laplace', 'groups']
# np_data = load_np_data(atoms, np_methods)
# plot_rocs(np_data, atoms, np_methods, fname='qmat-poster-rocs.pdf')
# plot_rocs(np_data, atoms, np_methods, fname='qmat-poster-rocs.png')
# for atom in atoms:
#     plot_rocs(np_data, [atom], np_methods, fname='plots/rocs/node-pred-rocs-' + atom[:-2] + '.pdf')

# plot_some_energies()
# plot_feth_energies()



# --------------------------------- Edge deletion according to energies -----------------------------

# with open('H1.0-hierarchies.pickle') as file:
#     hs = pickle.load(file)
# g = hs[0].g
# find_configuration(g)
# clean_term(g)
# add_leading_percentages(g, 'Th', 2.)

# fig, ax = plot_th2_energies_by_j(hs[0])
# ax[0].set_facecolor('w')
# ax[1].set_facecolor('w')
# adjust_spines(ax[0], ['left', 'bottom'])
# adjust_spines(ax[1], ['bottom'])
# fig.savefig('th2-energies-paper.pdf')
# fig.show()
# fig.savefig('th2-energies-vs-jp.png')
# g.ep.delta_e = g.new_ep('double')
# g.ep.delta_e.a = g.ep.wavelength.a**(-1)*10**8
#
# g_cut = g.copy()
# no_high_energies = g_cut.new_ep('bool', g.ep.delta_e.a < 33000)
# no_low_energies = g_cut.new_ep('bool', g.ep.delta_e.a > 10000)
#
# g_cut.set_edge_filter(no_high_energies)
# g_cut.purge_edges()
# h_cut = gt.minimize_nested_blockmodel_dl(g_cut)
# fig_cut, ax_cut = plot_energies_by_j(h_cut)
# plt.savefig('th2-cut-high-energies-vs-jp.pdf')
# plot_energies(h_cut)
# plt.savefig('th2-cut-high-energies.pdf')

# g_values_uncleaned = get_vertex_information(g, 'H', 1., 'lande_g', 'string')
# g.vp.g_value = g.new_vp('float')
# g.vp.multiplicity = get_vertex_information(g, 'H', 1., 'g', 'float')
# for v in g.vertices():
#     g.vp.g_value[v] = float('.'.join(re.findall(r'\d+|nan', g_values_uncleaned[v])))
# fig, ax = plot_prop_by_j(hs[0], 'g_value')
# fig.savefig('th2-gs-test.pdf')
# fig.show()
