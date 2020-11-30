"""
@author: David Wellnitz

Description: finds the best grouping with several runs and saves result as a .gml, with the respective groups of the
nodes as node attributes and the entropy of the grouping as graph attribute. Only groups with entropy lower than lowest
entropy + 5 are saved.

"""
import sys
sys.path.append('../code')

import graph_tool.all as gt
import time
import numpy as np
import cPickle as pickle
import pylab
import re
# import pandas as pd
import scipy
import matplotlib.pyplot as plt
import networkx as nx
import nx2
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import mean_shift
from matplotlib.cm import get_cmap
from collections import OrderedDict
try:
    reload(nx2)
except NameError:
    # Python 3
    from imp import reload
    reload(nx2)


# ------------------------------------------------------


# plt.style.use('bmh')
tab20b = get_cmap('Dark2')


# -------------------------------- general functions ------------------------------

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def get_vertex_information(g, atom, ionization, property_name, property_type,
                           level_data_path='../data/ASD54_levels.csv'):
    """ Takes additional information from the csv of the levels and transforms it into a PropertyMap

    :param g: graph-tool graph: for which to find information
    :param atom: str: name of the atom
    :param ionization: float: ionization level
    :param property_name: str: name of column in the csv
    :param property_type: str: dtype of PropertyMap to be created
    :param level_data_path: str: path to the data file
    :return: PropertyMap: Data for the added property
    """
    level_data_all = pd.read_csv(level_data_path, low_memory=False)
    level_data_atom = level_data_all[np.logical_and(np.array(level_data_all['el'] == atom, dtype=bool),
                                                    np.array(level_data_all['sc'] == ionization, dtype=bool))]
    new_level_data = level_data_atom[['level_id', property_name]].set_index('level_id')
    new_pmap = g.new_vp(property_type)
    for v in g.vertices():
        new_pmap[v] = new_level_data[property_name][g.vp.nodeid[v]]
    return new_pmap


def add_leading_percentages(g, atom, ionization, level_data_path='../data/ASD54_levels.csv'):

    """Wrapper to add all relevant properties corresponding to the most important configurations.

    :param g: graph-tool graph: for which to find information
    :param atom: str: name of the atom
    :param ionization: float: ionization level
    :param level_data_path: str: path to the data file
    :return:
    """
    g.vp.perc1 = get_vertex_information(g, atom, ionization, property_name='perc', property_type='float',
                                   level_data_path=level_data_path)
    conf1 = get_vertex_information(g, atom, ionization, property_name='conf1', property_type='string',
                                   level_data_path=level_data_path)
    for v in g.vertices():
        if conf1[v] == 'nan':
            conf1[v] = g.vp.conf[v]
    g.vp.perc2 = get_vertex_information(g, atom, ionization, property_name='perc2', property_type='float',
                                   level_data_path=level_data_path)
    conf2 = get_vertex_information(g, atom, ionization, property_name='conf2', property_type='string',
                                   level_data_path=level_data_path)
    g.vp.conf1 = clean_conf(conf1)
    g.vp.conf2 = clean_conf(conf2)


def plot_leading_percentages(g, fname=None):
    configurations = np.unique(np.array([g.vp.configuration[v] for v in g.vertices()]))
    conf_fractions = g.new_vp('vector<double>')
    for v in g.vertices():
        conf_fractions[v] = np.zeros(len(configurations), dtype=int)
        for i, conf in enumerate(configurations):
            if g.vp.conf1[v] == conf:
                if not np.isnan(g.vp.perc1[v]):
                    conf_fractions[v][i] += g.vp.perc1[v]
            if g.vp.conf2[v] == conf:
                if not np.isnan(g.vp.perc2[v]):
                    conf_fractions[v][i] += g.vp.perc2[v]
        conf_fractions[v][0] = 100. - np.sum(conf_fractions[v][1:])
    if fname:
        spectral = get_cmap('nipy_spectral')
        colors = [1., 1., 1., 1.] + [list(spectral(0.05 * i)) for i in range(len(configurations)-1)]
        group_color = g.new_vp('vector<double>')
        for v in g.vertices():
            group_color[v] = tab20b(g.group0level0[v] / float(len(np.unique(g.group0level0.a))))
        gt.graph_draw(g, vertex_shape='pie', vertex_pie_fractions=conf_fractions, vertex_pie_colors=colors,
                      vertex_text=g.group0level0, vertex_text_color='k', vertex_font_size=5, vertex_halo=True,
                      vertex_halo_color=group_color, output='conf-pie.pdf', output_size=(1500, 1000))
    return conf_fractions


def group_assignment_dict(g, bs):
    """
    Creates a dictionary with node IDs as keys and group assignments as values
    :param g: gt Graph
    :param bs: BlockState of G with the desired groups
    :return: dict
    """
    label = dict()
    # label keys should be node IDs and values numbers for each group
    pmap = bs.get_blocks()
    node_ids = g.vp.nodeid
    # assign labels:
    for i, key in enumerate(node_ids):
        label[key] = pmap[i]
    return label


def concat_pmaps(pmaps):
    g = pmaps[0].get_graph()
    pmap = g.new_vp('string')
    for v in g.vertices():
        pmap[v] = ''
        for prop in pmaps:
            pmap[v] += str(prop[v])
    return pmap


def find_configuration(g):
    """
    Assign proper configurations for the vertices
    :param g: gt Graph
    :return:
    """
    term = re.compile(r'\.?\(.+?\)\.?')
    g.vp.configuration = g.new_vp('string')
    for n in g.vertices():
        state = g.vp.conf[n]
        shells = re.split(term, state)
        c = '.'.join(shells)
        c = re.split(r'\?', c)[0]
        c = re.split(r' ', c)[0]
        c = re.split(r'\.$', c)[0]
        g.vp.configuration[n] = c


def clean_conf(conf_pmap):
    """ cleans up the intermediate terms from configuration

    :param conf_pmap: PropertyMap with intermediate terms
    :return: PropertyMap without intermediate terms
    """
    term = re.compile(r'\.?\(.+?\)\.?')
    g = conf_pmap.get_graph()
    configuration = g.new_vp('string')
    for n in g.vertices():
        state = conf_pmap[n]
        shells = re.split(term, state)
        c = '.'.join(shells)
        c = re.split(r'\?', c)[0]
        c = re.split(r' ', c)[0]
        c = re.split(r'\.$', c)[0]
        configuration[n] = c
    return configuration


def find_conf_theory(g):
    """ Add vertex PropertyMap conf needed for interesting_ari_data to theory network with one electron.

    :param g: gt graph for the network
    :return:
    """
    g.vp.conf = g.new_vp('string')
    for v in g.vertices():
        g.vp.conf[v] = g.vp.n[v] + g.vp.l[v].lower()


def find_energy_theory(g):
    """ Add vertex PropertyMap energy needed for plot_pred to theory network with one electron.

    :param g: gt graph for the network
    :return:
    """
    g.vp.energy = g.new_vp('double')
    for v in g.vertices():
        g.vp.energy[v] = 109737*(float(g.vp.n[v])**(-2))


def create_dict(g, pmaps):
    """
    creates a dictionary with values of pmaps as keys and nodelist as values
    :param g: graph-tool graph
    :param pmaps: list of PropertyMaps to sort nodes by
    :return: property_dict string: list
    """
    property_dict = dict()
    for v in g.vertices():
        prop = ''
        for pmap in pmaps:
            prop += str(pmap[v])
        if prop in property_dict:
            property_dict[prop].append(g.vertex_index[v])
        else:
            property_dict[prop] = [g.vertex_index[v]]
    return property_dict


def load_hydrogen(intensity_cutoff, cut_attribute='EinsteinA'):
    nxg = nx2.model_network(E1=True, E2=True, E3=True, M1=True, M2=True, M3=True)
    einstein_a = nx.get_edge_attributes(nxg, cut_attribute)
    for e in einstein_a:
        if einstein_a[e] < intensity_cutoff:
            nxg.remove_edge(e[0], e[1])
    g = nx2.nx2gt(nxg)
    return g


def edge_weight_histogram(g, weight_pmap, type_pmap, save=False, fname='edge-weight-histogram.svg'):
    """
    creates a histogram of all edges of g, colorcoded by transition type and sorted by weight, ignoring nans
    :param g: gt graph
    :param weight_pmap: assigns weight to edges
    :PropertyMap type_pmap: assigns transition type to edges
    :bool save:
    :str fname: name file will be saved as
    :return: dictionary with weights (except nans), keyed by type
    """
    # weight_data = {'E1': [], 'E2': [], 'E3': [], 'M1': [], 'M2': [], 'M3': []}
    weight_data = {}
    for e in g.edges():
        weight = weight_pmap[e]
        if np.isnan(weight):
            continue
        trans_type = type_pmap[e]
        if trans_type == '':
            trans_type = 'E1'
        try:
            weight_data[trans_type].append(weight)
        except KeyError:
            weight_data[trans_type] = [weight]
    bottom = 0
    fig, ax = plt.subplots()
    bins = np.logspace(np.log10(np.nanmin(np.sum(weight_data.values()))),
                       np.log10(np.nanmax(np.sum(weight_data.values()))), num=50)
    # bins = np.logspace(-50, 1)
    order_number = {'E1': 0, 'M1': 1, 'E2': 2, 'M2': 3, 'E3': 4, 'M3': 5}
    for type in sorted(weight_data.keys(), key=lambda x: order_number[x]):
        height, bins, patches = plt.hist(weight_data[type], bins=bins, bottom=bottom, histtype='bar', label=type)
        bottom += height
    ax.set_xscale('log')
    ax.set_xlabel('$A_{ik} \lambda^3$ in $10^5$nm$^3$s$^{-1}$')
    ax.set_ylabel('Number of Links')
    plt.legend(loc='best')
    if save:
        plt.savefig(fname)
    else:
        plt.show()
    return weight_data


def find_edge_type_theory(g):
    """Find the edge type for the theory graph g which has only transitionType as a property and save it into g.ep.type

    :param g: gt graph
    """
    g.ep.type = g.new_ep('string')
    for e in g.edges():
        g.ep.type[e] = g.ep.transitionType[e][:2]


def find_l(g):
    g.vp.l = g.new_vp('string')
    for v in g.vertices():
        try:
            g.vp.l[v] = re.findall(r'[A-Z]', g.vp.term[v])[-1]
        except IndexError:
            g.vp.l[v] = ''


def find_spin(g):
    g.vp.spin = g.new_vp('string')
    for v in g.vertices():
        try:
            g.vp.spin[v] = re.findall(r'\d+', g.vp.term[v])[0]
        except IndexError:
            g.vp.spin[v] = ''


def find_sl(g):
    g.vp.sl = g.new_vp('string')
    spin = g.new_vp('string')
    for v in g.vertices():
        try:
            spin[v] = re.findall(r'\d+', g.vp.term[v])[0]
        except IndexError:
            spin[v] = ''
    ang_mom = g.new_vp('string')
    for v in g.vertices():
        try:
            ang_mom[v] = re.findall(r'[A-Z]', g.vp.term[v])[-1]
        except IndexError:
            ang_mom[v] = ''
    for v in g.vertices():
        if spin[v] and ang_mom[v]:
            g.vp.sl[v] = spin[v] + ang_mom[v]
        else:
            g.vp.sl[v] = ''


def find_slj(g):
    g.vp.slj = g.new_vp('string')
    try:
        g.vp.sl
    except KeyError:
        find_sl(g)
    for v in g.vertices():
        if g.vp.sl[v] and g.vp.J[v] and re.findall(r'or', g.vp.J[v]) == []:
            g.vp.slj[v] = g.vp.sl[v] + g.vp.J[v]
        else:
            g.vp.slj[v] = ''


def clean_term(g):
    """Looks at the term of the vertices of and brings it in a unified shape inplace:
    (2s+1)LJ for LS coupling
    j1[j2]J for jj coupling
    J if the rest is unknown
    If there are different possibilities, they are joined by or

    :param g: graph-tool graph
    :return: new term PropertyMap
    """
    term_template = re.compile(r'((?:\d+[A-Z]\d+(?:/2)?)|'
                               r'(?:\d+(?:/2)?\[\d+(?:/2)?\]\d+(?:/2)?)|'
                               r'(?:^\d{1,2}(?:/2)?(?: or \d{1,2}(?:/2)?)*$))')
    g.vp.old_term = g.new_vp('string')
    g.vp.old_term = g.vp.term.copy()
    for v in g.vertices():
        g.vp.term[v] = ' or '.join(re.findall(term_template, g.vp.term[v]))
    return g.vp.term


def only_dipole_transitions(g, return_edge_sets=False):
    """
    Returns graph only containing dipole transitions. This method iteratively cuts the edge with smallest weight until
    the graph is bipartitie, and then restores cut edges which do not violate bipartivity in reverse order.

    Parameters
    ----------
    g : graph-tool graph

    return_edge_sets : bool  (optional)

    Returns
    -------
    gc : graph-tool graph
        If return_edge_sets==False.
    (gc, actually_cut, cut_edge_set, dipole) : tuple
         If return_edge_sets==True.

    """
    cut_edge_set = []
    false_cuts = [] #Contains edges that will be readded later
    # n = min([e[2] for e in G.edges(data = 'logarithmicMatrixElement')])
    gc = g.copy()
    # order edges by logarithmicMatrixElement
    order_nans_end = np.argsort(g.ep.logarithmicMatrixElement.a).flatten()
    nans = np.argwhere(np.isnan(g.ep.logarithmicMatrixElement.a)).flatten()
    order_no_nans = order_nans_end[:g.num_edges()-len(nans)]
    order = np.concatenate((nans, order_no_nans))
    # find beginning and end of each edge to remove
    ordered_edges = gc.get_edges()[order, :2]
    # Cut all edges from the weakest to the strongest until Graph is bipartite
    for e in ordered_edges:
        if gt.is_bipartite(gc):
            print 'BIPARTITE!'
            break
        gc.remove_edge(gc.edge(e[0], e[1]))
        if not np.all(gt.label_largest_component(gc).a):
            print 'disconnect', e
            gc.add_edge(e[0], e[1])
        else:
            cut_edge_set.append(e)
        # n+=1
    is_bi, partition = gt.is_bipartite(gc, partition=True)
    actually_cut = [] # Edges that will be left cut
    # Read edges that are between even and odd states to the Graph
    for e in cut_edge_set:
        if partition[e[0]] != partition[e[1]]:
            false_cuts.append([e[0], e[1]])
        else:
            actually_cut.append([e[0], e[1]])
    gc.add_edge_list(false_cuts)
    if return_edge_sets:
        return gc, actually_cut, cut_edge_set, false_cuts
    else:
        return gc

# --------------------------------- Communtiy Detection ------------------------------


def find_partition_list(g, n_runs, cut):
    """
    returns a list of the most important found NestedBlockStates
    :param g: gt graph
    :param n_runs: int, number of tries to find a new partition
    :param cut: float, only states with entropy smaller minimal entropy plus cut are counted as relevant
    :return: list, int: list of relevant NestedBlockStates, integer gives how often the best fit was found (high number
     means fit usually converges
    """
    time0 = time.time()
    hierarchies = [gt.minimize_nested_blockmodel_dl(g, deg_corr=True) for _ in range(n_runs)]
    time1 = time.time()
    sorted_hierarchies = sorted(hierarchies, key=lambda
        hierarchies: hierarchies.entropy(), reverse=False)
    time2 = time.time()
    relevant_hierarchies = []
    last_entropy = np.inf
    num_best_fit = 0
    for h in sorted_hierarchies:
        log_likelyhood_diff = h.entropy() - sorted_hierarchies[0].entropy()
        if log_likelyhood_diff < 0.0001:
            num_best_fit += 1
        if log_likelyhood_diff < cut:
            if np.abs(h.entropy() - last_entropy) > 0.0001:
                relevant_hierarchies.append(h)
                last_entropy = h.entropy()
        else:
            break
    time3 = time.time()
    print 'minimization time:', time1-time0, '\nsorting time:', time2-time1, '\nfiltering time:', time3-time2
    return relevant_hierarchies, num_best_fit


def add_pmaps_for_hierarchies(g, hierarchies):
    """
    Add hierarchy information as property maps to the graph
    :param g: gt graph
    :param hierarchies: NestedBlockState to add hierarchies to
    :return:
    """
    for i, h in enumerate(hierarchies):
        g.gp['group' + str(i) + 'entropy'] = g.new_gp('double')
        g.gp['group' + str(i) + 'entropy'] = h.entropy()
        n_levels = len(h.get_bs())
        for n in range(n_levels):
            name = 'group' + str(i) + 'level' + str(n)
            g.vp[name] = g.new_vp('int')
            g.vp[name] = h.project_level(n).get_blocks()


def add_inner_group_edges(g, group):
    """
    Add for every node pair in the same group an edge with property group_assignment 1000.0
    :param g: gt graph
    :param group: PropertyMap: the nodes will be grouped by this
    :return:
    """
    g.ep.group_assignment = g.new_edge_property('double')
    for u in g.vertices():
        for v in g.vertices():
            if group[u] == group[v]:
                if not g.edge(u, v):
                    edge = g.edge(u, v, add_missing=True)
                    g.ep.group_assignment[edge] = 1000.0


def save_element_groups(el, n_runs, **kwargs):
    """
    Saves the Graph for el with most likely groups found with n_runs steps
    :param el: name of the ion to save
    :param n_runs: number of calls of gt.minimize_nested_blockmodel_dl()
    :return:
    """
    nxg = nx2.load_network(el, **kwargs)
    g = nx2.nx2gt(nxg)

    hierarchies, count_best_fit = find_partition_list(g, n_runs, 5.0)
    add_pmaps_for_hierarchies(g, hierarchies)
    g.gp.count_best_fit = g.new_gp('int')
    g.gp.count_best_fit = count_best_fit
    g.gp.num_fits = g.new_gp('int')
    g.gp.num_fits = len(hierarchies)
    with open(el + '-hierarchies.pickle', 'w') as f:
        pickle.dump(hierarchies, f)

    g.save('graphs/' + el + '-communities.gml')


# ------------------------------ Link Prediction -----------------------------------


def calc_edge_probs(h, pot_edges):
    probs = np.zeros(len(pot_edges))
    for i, e in enumerate(pot_edges):
        probs[i] = np.exp(h.get_edges_prob([(e[0], e[1])]))
    return probs


def weighted_avg(probs, hs):
    weights = np.array([np.exp(-(h.entropy()-hs[0].entropy())) for h in hs])
    prob_array = np.average(probs, axis=0, weights=weights)
    return prob_array


def link_prediction(hs, name):
    """
    Do SBM link prediction and save results to file with name
    :param hs: nestedBlockStates used for link prediction
    :param name: name of element to save file to
    :return:
    """
    g = hs[0].g

    # Link Prediction
    pot_edges = np.argwhere(np.logical_not(gt.adjacency(g).todense()))
    probs = np.zeros((len(hs), pot_edges.shape[0]))

    for i, h in enumerate(hs):
        probs[i] = calc_edge_probs(h, pot_edges)
    edge_probs = weighted_avg(probs, hs)

    # save edge predictions
    np.savetxt(name + '-edges.txt', pot_edges)
    np.savetxt(name + '-edge-probabilities.txt', edge_probs)


# ------------------------------- Attribute Prediction ------------------------------


def attribute_probabilities_from_group(hs, v, pmap):
    """Calculates probability of node to have a certain attribute value by other nodes in the group

    :param hs: list of NestedBlockStates as prediction basis
    :param v: Node to be predicted
    :param pmap: PropertyMap with attribute to be predicted
    :return: dict assigning probability to each property
    """
    attribute_prob = dict()
    for h in hs:
        group = h.get_bs()[0][v]
        members = np.argwhere(h.get_bs()[0] == group).flatten()
        members = np.delete(members, np.argwhere(members == v))
        weight = np.exp(-(h.entropy()-hs[0].entropy()))
        for n in members:
            try:
                attribute_prob[pmap[n]] += weight
            except KeyError:
                attribute_prob[pmap[n]] = weight
    normalization = np.sum(attribute_prob.values())
    attribute_prob = {a: attribute_prob[a] / normalization for a in attribute_prob}
    return attribute_prob


def attribute_guess_from_group(hs, v, pmap):
    """Calculates the most likely attribute for vertex v

    :param hs: list of NestedBlockStates as prediction basis
    :param v: Node to be predicted
    :param pmap: PropertyMap with attribute to be predicted
    :return: most likely attribute
    """
    pdict = attribute_probabilities_from_group(hs, v, pmap)
    pdict.pop('', None)
    for k in pdict.keys():
        if re.findall('or', k):
            pdict.pop(k)
    if len(pdict.keys()) > 1:  # to prevent systematic errors when chosing from random data
        pdict = OrderedDict(sorted(pdict.items(), key=lambda x: np.random.random()))
    guesses = np.array(pdict.keys())[np.argsort(pdict.values())][::-1]
    if not len(guesses):
        guess = None
    else:
        guess = guesses[0]
    return guess


def attr_prediction_from_groups(hs, pmap):
    """
    predict attributes of each node individually, disregarding predictions for nodes whos attribute contains 'or' and
    count correct predictions
    :param hs: NestedBlockStates used for prediction
    :param pmap: PropertyMap with attributes to predict
    :return: number of correct guesses, number of total guesses
    """
    g = hs[0].g
    # correct_guesses = np.sum([attribute_guess_from_group(hs, v, pmap) == pmap[v] for v in g.get_vertices()
    #                           if re.findall('or', pmap[v]) == [] if pmap[v] != ''])
    # total_guesses = len([v for v in g.get_vertices() if re.findall('or', pmap[v]) == [] if pmap[v] != ''])
    relevant_vertices = [v for v in g.get_vertices() if pmap[v] != '' if re.findall('or', pmap[v]) == []]
    guesses = [attribute_guess_from_group(hs, v, pmap) for v in g.get_vertices()]
    correct_guesses = np.sum([float(guesses[v] == pmap[v]) + 0.5*float(guesses[v] == 'unsure')
                              for v in relevant_vertices])
    total_guesses = len(relevant_vertices)
    return correct_guesses, total_guesses


def attribute_guesses_from_matrix(g, pmap, feature_matrix, normalize=False):
    """
    Predicts the tags of each node from the graph g assuming the tags are unknown in this node and using
    the feature_matrix to determine similarities between the nodes (each node corresponds to one row in feature_matrix)

    The features are turned into distances and weighted with gaussian cutoff with variance (closest neighbor/4)^2
    (empirical formula)
    :param g: graph-tool graph which has the tags as vertex properties
    :param pmap: attribute to be predicted
    :param feature_matrix: matrix with node features as row vectors (taken from network)
    :param normalize: bool, if true feature_matrix is normalized first
    :return: tag_pred, array sorted like the nodes in g giving the predicted feature of each node
    """
    if normalize:
        for i in range(g.num_vertices()):
            feature_matrix[i] /= np.linalg.norm(feature_matrix[i])
    # create matrix with distances between feature vectors as vertices
    dist = np.zeros((g.num_vertices(), g.num_vertices()))
    tag_pred = np.array(['' for _ in range(g.num_vertices())], dtype='|S8')
    tag_dict = create_dict(g, [pmap])

    for n in range(g.num_vertices()):
        for n2 in range(g.num_vertices()):
            dist[n, n2] = min(np.linalg.norm(feature_matrix[n] - feature_matrix[n2]),
                              np.linalg.norm(feature_matrix[n] + feature_matrix[n2]))
        dist[n, n] = np.inf  # to prevent use of information of this node
        neighbor_dist = np.amin(dist[n])
        tag_scores = np.zeros(len(tag_dict))

        for i, t in enumerate(tag_dict):
            for n2 in tag_dict[t]:
                tag_scores[i] += np.exp(-(dist[n, n2]/(neighbor_dist/2.82))**2)
        tag_pred[n] = tag_dict.keys()[np.argmax(tag_scores)]
    return tag_pred


def attr_prediction_from_matrix(g, pmap, feature_matrix, normalize=False):
    """
    Prediction with attribute_guesses_from_matrix and count correct predictions
    :param g: graph-tool graph which has the tags as vertex properties
    :param pmap: attribute to be predicted
    :param feature_matrix: matrix with node features as row vectors (taken from network)
    :param normalize: bool, if true feature_matrix is normalized first
    :return: number of correct guesses, number of total guesses
    """
    prediction = attribute_guesses_from_matrix(g, pmap, feature_matrix, normalize=normalize)
    truth = np.array([str(pmap[v]) for v in g.get_vertices()])
    correct_guesses = np.sum([prediction[v] == truth[v] for v in g.get_vertices() if re.findall('or', pmap[v]) == []
                              if pmap[v] != ''])
    total_guesses = len([v for v in g.get_vertices() if re.findall('or', truth[v]) == [] if pmap[v] != ''])
    return correct_guesses, total_guesses


def attribute_guess_from_pmap(g, v, pmap, attr, return_probability=False):
    """Predict the attribute of node v from pmap, which assigns communities

    :param g: graph which contains the data
    :param v: vertex to predict attribute for
    :param pmap: PropertyMap containing community
    :param attr: PropertyMap containing attribute to predict
    :return: best guess for attr
    """
    attribute_prob = dict()
    group = pmap[v]
    members = np.argwhere(pmap.a == group).flatten()
    members = np.delete(members, np.argwhere(members == v))
    for n in members:
        try:
            attribute_prob[attr[n]] += 1
        except KeyError:
            attribute_prob[attr[n]] = 1
    if return_probability:
        return attribute_prob
    else:
        attribute_prob.pop('', None)
        for k in attribute_prob.keys():
            if re.findall('or', k):
                attribute_prob.pop(k)
        guesses = np.array(attribute_prob.keys())[np.argsort(attribute_prob.values())][::-1]
        if not len(guesses):
            guess = None
        elif len(guesses) == 1:
            guess = guesses[0]
        elif attribute_prob[guesses[0]] > attribute_prob[guesses[1]]:
            guess = guesses[0]
        else:
            guess = 'unsure'
        return guess


def attr_prediction_from_pmap(g, pmap, attr):
    relevant_vertices = [v for v in g.get_vertices() if attr[v] != '' if re.findall('or', attr[v]) == []]
    guesses = [attribute_guess_from_pmap(g, v, pmap, attr) for v in g.get_vertices()]
    correct_guesses = float(np.sum([float(guesses[v] == attr[v]) + 0.5*float(guesses[v] == 'unsure')
                              for v in relevant_vertices]))
    total_guesses = len(relevant_vertices)
    return correct_guesses, total_guesses


# --------------------------------------- Eigenvector Analysis -------------------------------------------


def adjacency_spectrum_feature_matrix(g):
    """
    Calculates the feature matrix from the spectrum, given by eigenvector components weighted with eigenvalue^2
    :param g: gt graph
    :return: feature matrix
    """
    a = gt.adjacency(g).todense()
    ew, ev = scipy.linalg.eig(a)

    feature_matrix = ew ** 2 * ev
    return feature_matrix


# --------------------------------------- Evaluate Communities with ARI -----------------------------------


def calc_ari(b, pmap):
    """
    Calculate the ARI of the two groups
    :param b: BlockState
    :param pmap: PropertyMap
    :return: ari_score
    """
    if pmap.a is not None:
        pmap_indices = pmap.a
    else:
        attributes = np.unique([pmap[v] for v in range(b.get_N())])
        pmap_indices = np.array([np.argwhere(attributes == pmap[v])[0, 0] for v in range(b.get_N())])
    ari = adjusted_rand_score(b.get_blocks().a, pmap_indices)
    return ari


def calc_ari_data(hs, pmaps):
    """
    Calculate the ARI value comparing groups given by hs compared to groups given by pmaps
    :param hs: list of NestedBlockStates
    :param pmaps: list of pairs (name, PropertyMap)
    :return: DataFrame with ARIs
    """
    df = pd.DataFrame(columns=['hierarchy state', 'hierarchy level', 'hierarchy entropy', 'ground truth', 'ari'])
    for h_index, h in enumerate(hs):
        for level_index in range(len(h.get_levels())):
            b = h.project_level(level_index)
            for pmap_name, pmap in pmaps:
                ari = calc_ari(b, pmap)
                new_line = pd.DataFrame({'hierarchy state': [h_index], 'hierarchy level': [level_index],
                                         'hierarchy entropy': [h.entropy()], 'ground truth': [pmap_name], 'ari': [ari]})
                df = df.append(new_line)
    return df


def interesting_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_configuration(g)
    find_slj(g)
    find_l(g)
    find_spin(g)
    clean_term(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.configuration_parity = concat_pmaps([g.vp.configuration, g.vp.parity])
    g.vp.configuration_J_parity = concat_pmaps([g.vp.configuration, g.vp.J, g.vp.parity])
    g.vp.conf_parity = concat_pmaps([g.vp.conf, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.slj, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.term_configuration_parity = concat_pmaps([g.vp.configuration, g.vp.term, g.vp.parity])
    g.vp.spin_parity = concat_pmaps([g.vp.spin, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('par.', g.vp.parity),
                         ('S', g.vp.spin),
                         ('J, par.', g.vp.j_parity),
                         ('L, par.', g.vp.l_parity),
                         ('S, par.', g.vp.spin_parity),
                         ('L, J, par.', g.vp.lj_parity),
                         ('S, L, par.', g.vp.sl_parity),
                         ('term', g.vp.term_parity),
                         ('conf., par.', g.vp.configuration_parity),
                         ('conf., J, par.', g.vp.configuration_J_parity),
                         ('conf., term', g.vp.term_configuration_parity),
                         ('ex. conf.', g.vp.conf_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def presentation_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_configuration(g)
    find_slj(g)
    find_l(g)
    find_spin(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.configuration_parity = concat_pmaps([g.vp.configuration, g.vp.parity])
    g.vp.configuration_J_parity = concat_pmaps([g.vp.configuration, g.vp.J, g.vp.parity])
    g.vp.conf_parity = concat_pmaps([g.vp.conf, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.slj, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.term_configuration_parity = concat_pmaps([g.vp.configuration, g.vp.term, g.vp.parity])
    g.vp.spin_parity = concat_pmaps([g.vp.spin, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('par.', g.vp.parity),
                         ('S', g.vp.spin),
                         ('J, par.', g.vp.j_parity),
                         ('L, par.', g.vp.l_parity),
                         ('S, par.', g.vp.spin_parity),
                         ('L, J, par.', g.vp.lj_parity),
                         ('S, L, par.', g.vp.sl_parity),
                         ('term', g.vp.term_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def julian_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_slj(g)
    find_l(g)
    find_spin(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.slj, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('J, par.', g.vp.j_parity),
                         ('L, par.', g.vp.l_parity),
                         ('L, J, par.', g.vp.lj_parity),
                         ('term', g.vp.term_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def good_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_configuration(g)
    find_slj(g)
    find_l(g)
    find_spin(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.configuration_parity = concat_pmaps([g.vp.configuration, g.vp.parity])
    g.vp.configuration_J_parity = concat_pmaps([g.vp.configuration, g.vp.J, g.vp.parity])
    g.vp.conf_parity = concat_pmaps([g.vp.conf, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.slj, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.term_configuration_parity = concat_pmaps([g.vp.configuration, g.vp.term, g.vp.parity])
    g.vp.spin_parity = concat_pmaps([g.vp.spin, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('par.', g.vp.parity),
                         ('J, par.', g.vp.j_parity),
                         ('L, par.', g.vp.l_parity),
                         ('L, J, par.', g.vp.lj_parity),
                         ('term', g.vp.term_parity),
                         ('conf., par.', g.vp.configuration_parity),
                         ('conf., J, par.', g.vp.configuration_J_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def tables_from_data(data, fname):
    df = pd.DataFrame(data, copy=True)
    with open(fname, 'w') as f:
        f.write(df.to_latex(encoding='utf-8'))


def plot_ari(els, save=False, fname='ari.svg'):
    fig, ax = plt.subplots(len(els), sharex=True)
    if len(els) == 1:
        ax = [ax]
    for i, el in enumerate(els):
        ionization = re.findall(r'\d', el)[0]
        atom_name = re.findall(r'[a-zA-Z]+', el)[0]
        ion_name = atom_name + r' ' + int(ionization)*'I'
        if re.findall(r'JB', el):
            ion_name += ' (Theory)'
        with open(el + '-hierarchies.pickle') as f:
            hs = pickle.load(f)
        if re.findall(r'JB', el):
            find_conf_theory(hs[0].g)
        data = interesting_ari_data(hs[:1])
        df = data[data['hierarchy level'] < np.amax(data['hierarchy level'])]
        level_data = pd.DataFrame({'Ground Truth': df[df['hierarchy level'] == 0]['ground truth']})
        for level in np.unique(df['hierarchy level']):
            level_data['level ' + str(level)] = df[df['hierarchy level'] == level]['ari']
        level_data.plot(x='Ground Truth', kind='barh', ax=ax[i], title=ion_name)
        ax[i].set_xlim((0., 1.))
    ax[len(els)-1].set_xlabel('ARI')
    plt.legend(loc='upper right')
    fig.set_figheight(5*len(els))
    fig.subplots_adjust(left=0.2)
    if save:
        plt.savefig(fname)
    else:
        plt.show()
    return level_data


# -------------------------------------- Data selection for Thesis ------------------------------------

def hydrogen_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_slj(g)
    find_l(g)
    find_spin(g)
    clean_term(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.slj, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.spin_parity = concat_pmaps([g.vp.spin, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('parity', g.vp.parity),
                         ('J', g.vp.J),
                         ('L', g.vp.l_parity),
                         ('term', g.vp.term_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def helium_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_configuration(g)
    find_slj(g)
    find_l(g)
    find_spin(g)
    clean_term(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.configuration_parity = concat_pmaps([g.vp.configuration, g.vp.parity])
    g.vp.configuration_J_parity = concat_pmaps([g.vp.configuration, g.vp.J, g.vp.parity])
    g.vp.conf_parity = concat_pmaps([g.vp.conf, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.slj, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.term_configuration_parity = concat_pmaps([g.vp.configuration, g.vp.term, g.vp.parity])
    g.vp.spin_parity = concat_pmaps([g.vp.spin, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('parity', g.vp.parity),
                         ('S', g.vp.spin),
                         ('S, parity', g.vp.spin_parity),
                         ('L', g.vp.l_parity),
                         ('S, L', g.vp.sl_parity),
                         ('J, parity', g.vp.j_parity),
                         ('L, J', g.vp.lj_parity),
                         ('term', g.vp.term_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def iron_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_configuration(g)
    find_slj(g)
    find_l(g)
    find_spin(g)
    clean_term(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.configuration_parity = concat_pmaps([g.vp.configuration, g.vp.parity])
    g.vp.configuration_J_parity = concat_pmaps([g.vp.configuration, g.vp.J, g.vp.parity])
    g.vp.conf_parity = concat_pmaps([g.vp.conf, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.term, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.term_configuration_parity = concat_pmaps([g.vp.configuration, g.vp.term, g.vp.parity])
    g.vp.spin_parity = concat_pmaps([g.vp.spin, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.slj_parity = concat_pmaps([g.vp.sl, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('parity', g.vp.parity),
                         ('par., S', g.vp.spin_parity),
                         ('par., L', g.vp.l_parity),
                         ('par., S, L', g.vp.sl_parity),
                         ('par., J', g.vp.j_parity),
                         ('par., L, J', g.vp.lj_parity),
                         ('par., S, L, J', g.vp.slj_parity),
                         ('par., term', g.vp.term_parity),
                         ('conf.', g.vp.configuration),
                         ('J, conf.', g.vp.configuration_J_parity),
                         ('conf., term', g.vp.term_configuration_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def thorium_ari_data(hs):
    """
    Convenient wrapper to calculate ari for interesting properties compared to hs
    :param hs: list of NestedBlockStates
    :return: DataFrame
    """
    g = hs[0].g
    find_configuration(g)
    find_slj(g)
    find_l(g)
    find_spin(g)
    clean_term(g)
    g.vp.j_parity = concat_pmaps([g.vp.J, g.vp.parity])
    g.vp.configuration_parity = concat_pmaps([g.vp.configuration, g.vp.parity])
    g.vp.configuration_J_parity = concat_pmaps([g.vp.configuration, g.vp.J, g.vp.parity])
    g.vp.conf_parity = concat_pmaps([g.vp.conf, g.vp.parity])
    g.vp.term_parity = concat_pmaps([g.vp.slj, g.vp.parity])
    g.vp.sl_parity = concat_pmaps([g.vp.sl, g.vp.parity])
    g.vp.term_configuration_parity = concat_pmaps([g.vp.configuration, g.vp.term, g.vp.parity])
    g.vp.spin_parity = concat_pmaps([g.vp.spin, g.vp.parity])
    g.vp.lj_parity = concat_pmaps([g.vp.l, g.vp.J, g.vp.parity])
    g.vp.l_parity = concat_pmaps([g.vp.l, g.vp.parity])
    ground_truth_data = [('par.', g.vp.parity),
                         ('J, par.', g.vp.j_parity),
                         ('conf.', g.vp.configuration_parity),
                         ('conf., J', g.vp.configuration_J_parity)]
    data = calc_ari_data(hs, ground_truth_data)
    return data


def plot_hyd_ari():
    fig, ax = plt.subplots(figsize=(10, 4))
    ion_name = 'H I (Theory)'
    with open('H1.0_JB_dipole-hierarchies.pickle') as f:
        hs = pickle.load(f)
    data = hydrogen_ari_data(hs[:1])
    df = data[data['hierarchy level'] < np.amax(data['hierarchy level'])]
    level_data = pd.DataFrame({'Ground Truth': df[df['hierarchy level'] == 0]['ground truth']})
    for level in np.unique(df['hierarchy level']):
        level_data['level ' + str(level)] = df[df['hierarchy level'] == level]['ari']
    level_data.plot(x='Ground Truth', kind='barh', ax=ax)
    ax.set_xlim((0., 1.))
    ax.set_xlabel('Adjusted Rand Index')
    plt.legend(loc='center right')
    fig.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig('H1-ari.pdf')
    return level_data


def plot_he_ari():
    fig, ax = plt.subplots(figsize=(10, 8))
    ion_name = 'He I'
    with open('He1.0-hierarchies.pickle') as f:
        hs = pickle.load(f)
    data = helium_ari_data(hs[:1])
    df = data[data['hierarchy level'] < np.amax(data['hierarchy level'])]
    level_data = pd.DataFrame({'Ground Truth': df[df['hierarchy level'] == 0]['ground truth']})
    for level in np.unique(df['hierarchy level']):
        level_data['level ' + str(level)] = df[df['hierarchy level'] == level]['ari']
    level_data.plot(x='Ground Truth', kind='barh', ax=ax)
    ax.set_xlim((0., 1.))
    ax.set_xlabel('Adjusted Rand Index')
    plt.legend(loc='center right')
    fig.subplots_adjust(left=0.2)
    plt.savefig('He1-ari.pdf')
    return level_data


def plot_fe_ari():
    fig, ax = plt.subplots(figsize=(10, 10))
    ion_name = 'Fe I'
    with open('Fe1.0-hierarchies.pickle') as f:
        hs = pickle.load(f)
    data = iron_ari_data(hs[:1])
    df = data[data['hierarchy level'] < np.amax(data['hierarchy level'])]
    level_data = pd.DataFrame({'Ground Truth': df[df['hierarchy level'] == 0]['ground truth']})
    for level in np.unique(df['hierarchy level']):
        level_data['level ' + str(level)] = df[df['hierarchy level'] == level]['ari']
    level_data.plot(x='Ground Truth', kind='barh', ax=ax)
    ax.set_xlim((0., 1.))
    ax.set_xlabel('Adjusted Rand Index')
    plt.legend(loc='center right')
    fig.subplots_adjust(left=0.2)
    plt.savefig('Fe1-ari.pdf')
    return level_data


def plot_th_ari():
    fig, ax = plt.subplots(figsize=(10, 4))
    ion_name = 'Th II'
    with open('Th2.0-hierarchies.pickle') as f:
        hs = pickle.load(f)
    data = thorium_ari_data(hs[:1])
    df = data[data['hierarchy level'] < np.amax(data['hierarchy level'])]
    level_data = pd.DataFrame({'Ground Truth': df[df['hierarchy level'] == 0]['ground truth']})
    for level in np.unique(df['hierarchy level']):
        level_data['level ' + str(level)] = df[df['hierarchy level'] == level]['ari']
    level_data.plot(x='Ground Truth', kind='barh', ax=ax)
    ax.set_xlim((0., 1.))
    ax.set_xlabel('Adjusted Rand Index')
    plt.legend(loc='center right')
    fig.subplots_adjust(left=0.2, bottom=0.15)
    plt.savefig('Th2-ari.pdf')
    return level_data

# --------------------------------------- Evaluate Attribute Prediction -------------------------------------


def attr_pred_table(els, attr):
    """
    Looks through elements, calculates predictions and saves result percentage in latex table
    :param els: names of elements to compare
    :param attr: name of attribute to check prediction on
    :return: pandas.DataFrame with result
    """
    df = pd.DataFrame()
    for el in els:
        with open(el + '-hierarchies.pickle') as f:
            hs = pickle.load(f)
        g = hs[0].g
        if attr == 'configuration':
            find_configuration(g)
        group_pred_corr, group_pred_tot = attr_prediction_from_groups(hs, g.vp[attr])
        feat_mat = adjacency_spectrum_feature_matrix(g)
        mat_pred_corr, mat_pred_tot = attr_prediction_from_matrix(g, g.vp[attr], feat_mat)
        energy_grouping = find_new_groups(g, hs[0].get_levels()[0].get_blocks(), g.vp.energy)
        energy_pred_corr, energy_pred_tot = attr_prediction_from_pmap(g, energy_grouping, g.vp[attr])
        ionization = re.findall(r'\d', el)[0]
        ion_name = el[:-3] + ' ' + int(ionization)*'I'
        tmpdf = pd.DataFrame({'ion': ion_name, 'groups': [float(group_pred_corr)/float(group_pred_tot)],
                              'eigenvectors': [float(mat_pred_corr)/float(mat_pred_tot)],
                              'energies': [float(energy_pred_corr)/float(energy_pred_tot)]})
        df = df.append(tmpdf)
    df = df.set_index('ion')
    with open('table-pred-' + attr + '.tex', 'w') as f:
        f.write(df.to_latex(encoding='utf-8'))
    return df


def plot_pred(els, attrs, save=False, fname='attribute-prediction.svg'):
    """Predicts the attributes given for the elements and visualizes the result quality

    :param els: element names for the elements to be predicted (grouping must be done earlier
    :param attrs: attributes to be predicted (must exist as vertex properties in saved element data
    :param save: save or show the plot?
    :param fname: filename for saving
    :return: DataFrame of prediction quality data
    """
    fig, ax = plt.subplots(len(els) + 1)
    fig.set_figheight(1.25 * len(attrs) * len(els))
    for i, el in enumerate(els):
        ionization = re.findall(r'\d', el)[0]
        atom_name = re.findall(r'[a-zA-Z]+', el)[0]
        ion_name = atom_name + r' ' + int(ionization)*'I'
        if re.findall(r'JB', el):
            ion_name += ' (Theory)'
        with open(el + '-hierarchies.pickle') as f:
            hs = pickle.load(f)
        g = hs[0].g
        if re.findall(r'JB', el):
            find_conf_theory(g)
            print el , 'is theory'
            find_energy_theory(g)
        else:
            print el, 'is experiment'
            clean_term(g)
        if 'configuration' in attrs:
            find_configuration(g)
        if 'sl' in attrs:
            find_sl(g)
        if 'slj' in attrs:
            find_slj(g)
        pred_data = pd.DataFrame()
        pred_data['Attribute'] = attrs
        accuracies = np.zeros((3, len(attrs)))
        for j, attr in enumerate(pred_data['Attribute']):
            group_pred_corr, group_pred_tot = attr_prediction_from_groups(hs, g.vp[attr])
            feat_mat = adjacency_spectrum_feature_matrix(g)
            mat_pred_corr, mat_pred_tot = attr_prediction_from_matrix(g, g.vp[attr], feat_mat)
            energy_grouping = find_new_groups(g, hs[0].get_levels()[0].get_blocks(), g.vp.energy)
            energy_pred_corr, energy_pred_tot = attr_prediction_from_pmap(g, energy_grouping, g.vp[attr])
            accuracies[0, j] = float(group_pred_corr)/float(group_pred_tot)
            accuracies[1, j] = float(energy_pred_corr)/float(energy_pred_tot)
            accuracies[2, j] = float(mat_pred_corr)/float(mat_pred_tot)
        pred_data['Eigenvectors'] = accuracies[2]
        pred_data['Energies and Groups'] = accuracies[1]
        pred_data['Groups'] = accuracies[0]
        pred_data.plot(x='Attribute', kind='barh', ax=ax[i], legend=False)
        ax[i].set_xticks(np.linspace(0., 1., 6))
    ax[len(els) - 1].set_xlabel('Fraction of correct predictions')
    fig.subplots_adjust(left=0.15)
    plt.sca(ax[len(els)])
    ax[len(els)].axis('off')
    plt.legend(*ax[len(els)-1].get_legend_handles_labels(), loc='center')
    if save:
        fig.savefig(fname)
    else:
        plt.show()
    return pred_data


def plot_pred_no_energies(el, attrs, save=False, fname='attribute-prediction.svg'):
    """Predicts the attributes given for the elements and visualizes the result quality, here without energy group
    method and only one element

    :param els: element names for the elements to be predicted (grouping must be done earlier
    :param attrs: attributes to be predicted (must exist as vertex properties in saved element data
    :param save: save or show the plot?
    :param fname: filename for saving
    :return: DataFrame of prediction quality data
    """
    fig, ax = plt.subplots()
    ionization = re.findall(r'\d', el)[0]
    atom_name = re.findall(r'[a-zA-Z]+', el)[0]
    ion_name = atom_name + r' ' + int(ionization)*'I'
    if re.findall(r'JB', el):
        ion_name += ' (Theory)'
    with open(el + '-hierarchies.pickle') as f:
        hs = pickle.load(f)
    g = hs[0].g
    if re.findall(r'JB', el):
        find_conf_theory(g)
    clean_term(g)
    if 'configuration' in attrs:
        find_configuration(g)
    if 'sl' in attrs:
        find_sl(g)
    if 'slj' in attrs:
        find_slj(g)
    pred_data = pd.DataFrame()
    pred_data['Attribute'] = attrs
    accuracies = np.zeros((3, len(attrs)))
    for j, attr in enumerate(pred_data['Attribute']):
        group_pred_corr, group_pred_tot = attr_prediction_from_groups(hs, g.vp[attr])
        feat_mat = adjacency_spectrum_feature_matrix(g)
        mat_pred_corr, mat_pred_tot = attr_prediction_from_matrix(g, g.vp[attr], feat_mat)
        accuracies[0, j] = float(group_pred_corr)/float(group_pred_tot)
        accuracies[1, j] = float(mat_pred_corr)/float(mat_pred_tot)
    pred_data['Eigenvectors'] = accuracies[1]
    pred_data['Groups'] = accuracies[0]
    pred_data.plot(x='Attribute', kind='barh', ax=ax, title=ion_name, legend=True)
    ax.set_xlabel('Fraction of correct predictions')
    fig.set_figheight(0.9 * len(attrs))
    fig.subplots_adjust(left=0.23, bottom=0.13)
    if save:
        plt.savefig(fname)
    else:
        plt.show()
    return pred_data


# --------------------------------------- Groups and Energies Together --------------------------------------


def group_vs_attr_plot(g, groups, grouping, attribute, save=False, fname='groups-vs-energies.svg'):
    """Plots the nodes in the given groups on the x-axis against the energy on the y-axis, uses a new color for each group

    :param groups: list of groups to plot
    :param grouping: PropertyMap containing the group tag
    :param attribute: PropertyMap which contains the attribute for the y-axis
    :return:
    """
    fig, ax = plt.subplots()
    for group in groups:
        nodes = [n for n in g.vertices() if grouping[n] == group]
        if nodes is not None:
            y_data = np.array([attribute[n] for n in nodes])
            y_data = np.sort(y_data)
            x_data = np.arange(len(y_data))
            dots, = ax.plot(x_data, y_data, 'o', label=group)
            ax.plot(x_data, y_data, '-', color=dots.get_color(), alpha=0.3, lw=0.5)
    plt.legend(loc='best')
    if save:
        plt.savefig(fname)
    else:
        plt.show()
    return fig, ax


def split_by_attribute(g, grouping, attribute, groups='all'):
    """Sorts the nodes by groups and splits them if where the jump in attribute value is very high

    :param g: graph
    :param grouping: PropertyMap assigning groups to nodes
    :param attribute: PropertyMap assigning value to nodes
    :param groups: which groups to find
    :return: PropertyMap with new groups
    """
    if groups == 'all':
        groups = np.unique(grouping.a)
    group_map = g.new_vp('int')
    num_groups = 0
    for group in groups:
        nodes = np.array([n for n in g.vertices() if grouping[n] == group])
        attr_data = np.array([attribute[n] for n in nodes])
        attr_data_sorting = np.argsort(attr_data)
        attr_data_sorted = attr_data[attr_data_sorting]
        backsort = np.argsort(attr_data_sorting)

        distances = np.ediff1d(attr_data_sorted)
        max_fact = 0.
        cuts = []
        for i, dist in enumerate(np.sort(distances[1:-1])[:2:-1]):
            av_dist = np.mean(np.sort(distances[1:-1])[:-(i + 1)])
            std_dist = np.std(np.sort(distances[1:-1])[:-(i + 1)])
            fact = (dist - av_dist) / std_dist
            if fact < max_fact:
                print 'Break at', i
                break
            cut = np.argwhere(distances == dist)[0, 0]
            print 'Cut at', cut
            if not (cut == 0 or cut == len(distances) - 1 or cut + 1 in cuts or cut - 1 in cuts):
                cuts.append(cut)
                max_fact = fact
        if max_fact < 4.:
            cuts = []
        # av_dist = np.mean(np.sort(distances)[:-min(3, len(distances)/2)])
        # std_dist = np.std(np.sort(distances)[:-min(3, len(distances)/2)])
        # cuts = np.argwhere(distances > av_dist + 3.*std_dist)
        # if 0 in cuts:
        #     cuts = cuts[1:]
        # if len(distances) in cuts:
        #     cuts = cuts[:-1]
        new_groups = np.zeros(len(nodes), dtype='int') + num_groups
        num_groups += len(cuts) + 1
        for cut in cuts:
            new_groups[cut+1:] += 1
        for i, n in enumerate(nodes):
            group_map[n] += new_groups[backsort][i]
    return group_map


def find_new_groups(g, grouping, attribute, groups='all'):
    """Depreciated --- Sorts the nodes by groups and splits them if where the jump in attribute value is very high

    :param g: graph
    :param grouping: PropertyMap assigning groups to nodes
    :param attribute: PropertyMap assigning value to nodes
    :param groups: which groups to find
    :return: PropertyMap with new groups
    """
    if groups == 'all':
        groups = np.unique(grouping.a)
    group_map = g.new_vp('int')
    num_groups = 0
    for group in groups:
        nodes = np.array([n for n in g.vertices() if grouping[n] == group])
        attr_data = np.array([attribute[n] for n in nodes])
        attr_data_sorting = np.argsort(attr_data)
        attr_data_sorted = attr_data[attr_data_sorting]
        backsort = np.argsort(attr_data_sorting)

        distances = np.ediff1d(attr_data_sorted)
        av_dist = np.mean(distances)
        std_dist = np.std(distances)
        cuts = np.argwhere(distances > 2.0*av_dist + std_dist)
        if 0 in cuts:
            cuts = cuts[1:]
        new_groups = np.zeros(len(nodes), dtype='int') + num_groups
        num_groups += len(cuts) + 1
        for cut in cuts.flatten():
            new_groups[cut+1:] += 1
        for i, n in enumerate(nodes):
            group_map[n] += new_groups[backsort][i]
    return group_map


def scatter_attributes(g, attributes, group_list, sort_by, groups='all', style='both', save=True, fname=None):
    """ Plots a scatter plot colored by attributes from nodes in some groups

    :param g: graph-tool graph
    :param attributes: list of PropertyMaps
    :param group_list: array/list/dict/PropertyMap; assigns group numbers to nodes
    :param sort_by: array that assigns y-axis data to nodes
    :param groups: 'all' or list of groups; groups to draw in the diagram
    :param style: string 'scatter' or 'plot'
    :param save: bool; whether the plot is saved and closed
    :return:
    """
    fig, ax = plt.subplots()
    attr_dict = create_dict(g, attributes)
    for i, attr, nodes in zip(range(len(attr_dict)), attr_dict.keys(), attr_dict.values()):
        if not attr:
            attr = 'unknown'
        if groups != 'all':
            nodes = [n for n in nodes if group_list[n] in groups]
        if len(nodes):
            y_attr = sort_by[nodes]
            y_attr.sort()
            if style == 'scatter' or 'both':
                dots, = ax.plot(np.arange(len(nodes)), y_attr, 'o', label=attr)
            if style == 'plot':
                ax.plot(np.arange(len(nodes)), y_attr, label=attr)
            if style == 'both':
                ax.plot(np.arange(len(nodes)), y_attr, '-', color=dots.get_color(), alpha=0.3, lw=0.5)
    plt.ylabel('energy in 1/cm')
    plt.xlabel('node number')
    plt.title('Group ' + ', '.join([str(g) for g in groups]))
    plt.legend(loc='best')
    if save:
        if not fname:
            fname = 'attr-group-' + groups + '-plot.pdf'
        plt.savefig(fname)
        plt.close('attribute-scatter')
    return fig, ax


# next(ax._get_lines.prop_cycler) -> cycle through colors
def groups_vs_energies_with_attr_plot(g, group, group_assignment, attribute, fname=None, show=False, hard_order=True,
                                      legend=False):
    """ Plot nodes in group vs energies and color nodes according to attribute

    :param g: gt graph
    :param group: group to plot
    :param group_assignment: PropertyMap assigning groups to nodes
    :param attribute: PropertyMap to color nodes by
    :param fname: string, if given, plot is saved to fname
    :return:
    """
    members = np.array([n for n in g.get_vertices() if group_assignment[n] == group])
    fig, ax = plt.subplots()
    y_data = np.array([g.vp.energy[n] for n in members])
    order = np.argsort(y_data)
    y_data = y_data[order]
    x_data = np.arange(len(y_data))
    ax.plot(x_data, y_data, 'k-', lw=0.5)
    attr_dict = create_dict(g, [attribute])
    if hard_order:
        attr_list = ['5f.6d.7s', '5f.6d2', '5f.7s2', '5f2.7p', '6d2.7p', '6d.7s.7p', '5f.7p2', '']
        default_node = [n for n in members if n in attr_dict['']][0]
        for attr in attr_list:
            attr_nodes = attr_dict[attr]
            attr_members = [n for n in attr_nodes if group_assignment[n] == group]
            if not len(attr_members):
                attr_members = [default_node]
            if not attr:
                attr = 'unknown'
                y_attr = np.array([g.vp.energy[n] for n in attr_members])
                x_attr = np.array([np.argwhere(m == members[order]) for m in attr_members]).flatten()
                ax.plot(x_attr, y_attr, 'ko', label=attr, markersize=10)
            elif len(attr_members):
                y_attr = np.array([g.vp.energy[n] for n in attr_members])
                x_attr = np.array([np.argwhere(n == members[order]) for n in attr_members]).flatten()
                if attr == 'Other Nodes':
                    ax.plot(x_attr, y_attr, 'wo', alpha=0.2, label=attr, zorder=-1, markersize=10)
                else:
                    ax.plot(x_attr, y_attr, 'o', label=attr, markersize=10)

    else:
        for i, attr, attr_nodes in zip(range(len(attr_dict)), attr_dict.keys(), attr_dict.values()):
            attr_members = [n for n in attr_nodes if group_assignment[n] == group]
            if not attr:
                attr = 'unknown'
                y_attr = np.array([g.vp.energy[n] for n in attr_members])
                x_attr = np.array([np.argwhere(m == members[order]) for m in attr_members]).flatten()
                ax.plot(x_attr, y_attr, 'ko', label=attr, markersize=10)
            elif len(attr_members):
                y_attr = np.array([g.vp.energy[n] for n in attr_members])
                x_attr = np.array([np.argwhere(m == members[order]) for m in attr_members]).flatten()
                if attr == 'Other Nodes':
                    ax.plot(x_attr, y_attr, 'wo', alpha=0.2, label=attr, zorder=-1, markersize=10)
                else:
                    ax.plot(x_attr, y_attr, 'o', label=attr, markersize=10)

    ax.set_ylabel('State Energy in cm$^{-1}$')
    ax.set_xlabel('Nodes')
    ax.set_xticks([])
    ax.set_ylim(0, 70000)
    if legend:
        plt.legend(loc='lower right')
    if fname:
        fig.savefig(fname)
    if show:
        plt.show()
    return fig, ax


def plot_both_confs(g, group, group_assignment):
    configurations = np.unique(np.array([g.vp.configuration[v] for v in g.vertices()]))
    confcolor = {c: tab20b(0.05 * i) for i, c in enumerate(configurations)}
    confcolor['nan'] = (0., 0., 0., 1.)
    confcolor['unknown'] = (0., 0., 0., 1.)

    fig, ax = plt.subplots()
    members = np.array([n for n in g.get_vertices() if group_assignment[n] == group])
    y_data = np.array([g.vp.energy[n] for n in members])
    order = np.argsort(y_data)
    y_data = y_data[order]
    x_data = np.arange(len(y_data))
    ax.plot(x_data, y_data, 'k-', lw=0.5)
    attr_dict = create_dict(g, [g.vp.conf1])
    for i, attr, attr_nodes in zip(range(len(attr_dict)), attr_dict.keys(), attr_dict.values()):
        attr_members = [n for n in attr_nodes if group_assignment[n] == group]
        if not attr:
            attr = 'unknown'
        if len(attr_members):
            y_attr = np.array([g.vp.energy[n] for n in attr_members])
            x_attr = np.array([np.argwhere(m == members[order]) for m in attr_members]).flatten()
            if attr == 'Other Nodes':
                ax.plot(x_attr, y_attr, 'wo', alpha=0.2, label=attr, zorder=-1)
            else:
                ax.plot(x_attr, y_attr, 'o', label=attr, markersize=15., color=confcolor[attr])

    attr_dict = create_dict(g, [g.vp.conf2])
    for i, attr, attr_nodes in zip(range(len(attr_dict)), attr_dict.keys(), attr_dict.values()):
        attr_members = [n for n in attr_nodes if group_assignment[n] == group]
        if not attr:
            attr = 'unknown'
        if len(attr_members):
            y_attr = np.array([g.vp.energy[n] for n in attr_members])
            x_attr = np.array([np.argwhere(m == members[order]) for m in attr_members]).flatten()
            if attr == 'Other Nodes':
                ax.plot(x_attr, y_attr, 'wo', alpha=0.2, label=attr, zorder=-1)
            else:
                ax.plot(x_attr, y_attr, 'D', markersize=9.5, color=confcolor[attr])

    attr_dict = create_dict(g, [g.vp.group0level0])
    for i, attr, attr_nodes in zip(range(len(attr_dict)), attr_dict.keys(), attr_dict.values()):
        attr_members = [n for n in attr_nodes if group_assignment[n] == group]
        if not attr:
            attr = 'unknown'
        if len(attr_members):
            y_attr = np.array([g.vp.energy[n] for n in attr_members])
            x_attr = np.array([np.argwhere(m == members[order]) for m in attr_members]).flatten()
            if attr == 'Other Nodes':
                ax.plot(x_attr, y_attr, 'wo', alpha=0.2, label=attr, zorder=-1)
            else:
                ax.plot(x_attr, y_attr, 's', label=attr, markersize=5.5)
    plt.legend(loc='best')
    return fig

# ----------------------------------------------------- Main --------------------------------------------------


if __name__ == '__main__':

    # with open('H1.0-hierarchies.pickle') as f:
    #     hs = pickle.load(f)
    # g = hs[0].g
    # find_configuration(g)
    # groups_vs_energies_with_attr_plot(g, 0, hs[0].get_levels()[0].get_blocks(), g.vp.configuration)
    # groups = hs[0].get_levels()[0].get_blocks()
    # g.vp.old_en_groups = find_new_groups(g, hs[0].get_levels()[0].get_blocks(), g.vp.energy)
    # g.vp.new_en_groups = split_by_attribute(g, hs[0].get_levels()[0].get_blocks(), g.vp.energy)
    # find_configuration(g)
    # correct_old, total_old = attr_prediction_from_pmap(g, g.vp.old_en_groups, g.vp.configuration)
    # correct_new, total_new = attr_prediction_from_pmap(g, g.vp.new_en_groups, g.vp.configuration)
    # correct_0, total_0 = attr_prediction_from_pmap(g, hs[0].get_levels()[0].get_blocks(), g.vp.configuration)
    # for i in np.unique(groups.a):
    #     scatter_attributes(g, [g.vp.J, g.vp.parity], groups, g.vp.energy.a, groups=[i], save=False)
    #     plt.title('Group '+str(i))
    #     plt.show()
    #
    # # plot group data as expected if drawn randomly from energies
    # fig, ax = plt.subplots()
    # for _ in range(5):
    #     rand_group = np.random.choice(g.vp.energy.a, size=30, replace=False)
    #     rand_group = np.sort(rand_group)
    #     dots, = ax.plot(rand_group, 'o')
    #     ax.plot(rand_group, '-', color=dots.get_color(), alpha=0.3)
    # plt.show()


    # plot_ari(['He1.0'], save=True, fname='ari-helium.eps')
    # plot_ari(['Fe1.0', 'Fe2.0', 'Th2.0'], save=True, fname='ari-fe-th.eps')
    # elements = ['H1.0', 'Th2.0']
    # plot_pred(elements, ['parity', 'J', 'term', 'configuration'], save=True, fname='attr-pred-unsure-as-half.svg')
    # for el in elements:
    #     with open(el + '-hierarchies.pickle') as f:
    #         hs = pickle.load(f)
    #     g = hs[0].g
    #     group_vs_attr_plot(g, range(4), hs[0].get_levels()[0].get_blocks(), g.vp.configuration, save=True,
    #                        fname='groups-vs-energies-' + el + '.svg')

    # Julian ARI Data
    # with open('Th2.0-hierarchies.pickle') as f:
    #     hs = pickle.load(f)
    # data = julian_ari_data(hs)
    # tables_from_data(data, 'julian-hydro-jb-all.tex')


    # save_element_groups('H1.0_JB_dipole', n_runs=1000, experimental=False, only_dipole=True)
    # save_element_groups('H1.0_JB_all', n_runs=1000, experimental=False)
    # save_element_groups('H1.0', n_runs=100, experimental=True)
    # save_element_groups('He1.0', n_runs=1000, experimental=True)
    # save_element_groups('C1.0', n_runs=1000, experimental=True)
    # save_element_groups('Fe1.0', n_runs=1000, experimental=True)
    # save_element_groups('Fe2.0', n_runs=1000, experimental=True)
    # save_element_groups('Th1.0', n_runs=1000, experimental=True)
    # save_element_groups('Th2.0', n_runs=1000, experimental=True)

    # data = attr_pred_table(['He1.0', 'Fe1.0', 'Th2.0'], 'configuration')
    # print 'Done!'


    # 2018/12/17 Link prediction parameter tests
    runs = 3
    cutoff_list = [0.5, 1.0, 3.0]
    he = nx2.load_network('Th2.0')
    lp = nx2.LinkPrediction()
    lp.G_original = he
    data = {c: np.zeros(runs) for c in cutoff_list}
    durations = {c: np.zeros(runs) for c in cutoff_list}
    for i in range(runs):
        lp.dropout(0.1)
        for cutoff in cutoff_list:
            start = time.time()
            lp.predict_nested_SBM(cutoff=cutoff)
            lp.check_if_correct()
            auc = lp.calculate_AUC()
            durations[cutoff][i] = time.time() - start
            print i, cutoff, lp.calculate_AUC(), '\nDuration: ', time.time() - start
            data[cutoff][i] = auc