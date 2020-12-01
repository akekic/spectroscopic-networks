#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:14:19 2017

@author: arminkekic

Module containing all important functions related to networks that we
coded ourselves. Functions can be called using dot-notation
(e.g. nx2.func()). You can import the module like this:

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

"""
import numpy as np
import networkx as nx
import itertools
import csv
import time
import random
import warnings
import graph_tool.all as gt


def spectroscopic_network(ion_type, line_data_path='../data/ASD54_lines.csv', level_data_path='../data/ASD54_levels.csv', weighted=True, check_accur=False, check_wl=False, check_obs_wl=False, check_calc_wl=False, dictionaries=False, alt_read_in=False):
    """

    Function loading a graph from line and level data stored in .csv files.

    Returns
    -------

    G : NetworkX graph
        If dictionaries==False.
    (G, elevel, term, j) : tuple
        If dictionaries==True.


    Parameters
    ----------

    ion_type : str
        String specifying the ion for which thdatae data is to be loaded. The string needs to contain both the element name as well as the ionisation state sc in one combined string; e.g. 'H1.0' for hydrogen.
    line_data_path : str (optional)
        Path to the line data file.(default='../data/ASD54_lines.csv')
    level_data_path : str (optional)
        Path to the line data file.(default='../data/ASD54_levels.csv')
    weighted : bool (optional)
        Whether or not to include weights. Both intensity and wavelength are added as weights to the NetworkX graph. (default=True)
    check_accur : bool (optional)
        Check accuracies of intensities and only take lines with 'D' or better. (default=False)
    check_wl : bool (optional)
        Filter out transitions where no wavelength value (calculated or observed) is given. (default=False)
    check_obs_wl : bool (optional)
        Filter out transitions with empty observed wavelength value. (default=False)
    check_calc_wl : bool (optional)
        Filter out transitions with empty calculated wavelength value. (default=False)
    dictionaries : bool (optional)
        If true, dictionaries containing the energylevels, j, and terms are returned as well. (default=False)
    alt_read_in : bool (optional)
        If True, uses an alternative way to read in the data file (hardcoded column number). (default=False)

    Notes
    -----

    One should only use the checks that are actually necessary for the application at hand, as any additional check may decrease the number of available lines in the graph significantly.
    The n number attribute works only for ions with one electron.

    """
    ### Get the Data of Lines
    # if csv reader does not work, use alternative read in mode
    if alt_read_in==True:
        with open(line_data_path) as line_file:
            line_reader = csv.reader(line_file)
            line_data_temp = np.array(list(line_reader))
            row_count = len(line_data_temp)

        # create empty 2d array and fill it with the data. 33 is the number of columns *hardcoded!
        line_data = np.empty((row_count,33),dtype='|S64')
        for i in range(len(line_data_temp)):
           for k in range(33):
               line_data[i,k] = line_data_temp[i][k]
    else:
        # use csv reader
        with open(line_data_path) as line_file:
            line_reader = csv.reader(line_file)
            line_data = np.array(list(line_reader))
            row_count = len(line_data)

    # types of atoms atoms[i] gives atom+ionization
    atoms   = np.empty(row_count-1, dtype='|S8')
    for i,row in enumerate(line_data[1:]):
        atoms[i] = row[0]+row[1]

    # Index that is true for the correct atom and false otherwise
    atom_index = np.concatenate(([False],(atoms == ion_type)))

    # if there are no lines in the data return empty graph
    if np.all(np.logical_not(atom_index)):
        G = nx.empty_graph()
        if dictionaries:
            return (G, None, None, None)
        else:
            return G

    # lower and upper levels, intensities and frequencies of transitions
    low_levels = line_data[atom_index,25]
    upp_levels = line_data[atom_index,26]
    trans_type = line_data[atom_index,22] # transition type
    theo_ref   = line_data[atom_index,23] # theoretical reference (sometimes books etc.)
    exp_ref    = line_data[atom_index,24] # measurement reference (sometimes books etc.)
    if weighted:
        intensity  = line_data[atom_index,17] # The Einstein A coefficient
        obs_wl     = line_data[atom_index,2]  # observed wavelength
        calc_wl    = line_data[atom_index,5]  # calculated wavelength

        """
        Combine calculated and observed wavelength. If both are available for a given line, choose observed.
        """
        wavelength = np.asarray([obs_wl[i] if not obs_wl[i]=='' else calc_wl[i] for i in xrange(len(obs_wl))])

    # Delete edges with no intensity information if Graph should be weighted
    if weighted:
        if check_accur:
            accur_list = np.array(['AAA','AA','A+','A','B+','B','C+','C','D+','D']) #These accuracies for the intensity are ok
            accur      = np.array(line_data[atom_index,21])
            accur_ok   = np.in1d(accur, accur_list)
            low_levels = low_levels[accur_ok]
            upp_levels = upp_levels[accur_ok]
            intensity  = intensity[accur_ok]
            wavelength = wavelength[accur_ok]
            obs_wl     = obs_wl[accur_ok]
            calc_wl    = calc_wl[accur_ok]
        if check_obs_wl:
            obs_wl_ok  = (obs_wl != '')
            low_levels = low_levels[obs_wl_ok]
            upp_levels = upp_levels[obs_wl_ok]
            intensity  = intensity[obs_wl_ok]
            wavelength = wavelength[obs_wl_ok]
            obs_wl     = obs_wl[obs_wl_ok]
            calc_wl    = calc_wl[obs_wl_ok]
        if check_calc_wl:
            calc_wl_ok = (calc_wl != '')
            low_levels = low_levels[calc_wl_ok]
            upp_levels = upp_levels[calc_wl_ok]
            intensity  = intensity[calc_wl_ok]
            wavelength = wavelength[calc_wl_ok]
            obs_wl     = obs_wl[calc_wl_ok]
            calc_wl    = calc_wl[calc_wl_ok]
        if check_wl:
            wl_ok = (wavelength != '')
            low_levels = low_levels[wl_ok]
            upp_levels = upp_levels[wl_ok]
            intensity  = intensity[wl_ok]
            wavelength = wavelength[wl_ok]
            obs_wl     = obs_wl[wl_ok]
            calc_wl    = calc_wl[wl_ok]

        # convert empty entries to NaN's
        if weighted:
            intensity[intensity == '']      = 'NaN'
            wavelength[wavelength == '']    = 'NaN'
            obs_wl[obs_wl == '']            = 'NaN'
            calc_wl[calc_wl == '']          = 'NaN'
            intensity  = intensity.astype(float)
            wavelength = wavelength.astype(float)
            obs_wl     = obs_wl.astype(float)
            calc_wl    = calc_wl.astype(float)

    # Create the Graph
    G       = nx.Graph()
    G.name  = ion_type
    if weighted:
        G.add_weighted_edges_from(zip(low_levels, upp_levels,intensity), weight = 'intensity')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,(1e-8)*intensity*wavelength**3.0), weight = 'matrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.log((1e-8)*intensity*wavelength**3.0)), weight = 'logarithmicMatrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,(-1e-8)*intensity*wavelength**3.0), weight = 'negativeMatrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,(-1)*np.log((1e-8)*intensity*wavelength**3.0)), weight = 'negativeLogarithmicMatrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(intensity)), weight = 'inverseIntensity')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,wavelength), weight = 'wavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(wavelength)), weight = 'inverseWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,obs_wl), weight = 'observedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(obs_wl)), weight = 'inverseObservedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,calc_wl), weight = 'calculatedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(calc_wl)), weight = 'inverseCalculatedWavelength')
    else:
        G.add_edges_from(zip(low_levels,upp_levels))

    # set edge attributes
    nx.set_edge_attributes(G, name='type', values=dict(zip(zip(low_levels, upp_levels),trans_type)))
    nx.set_edge_attributes(G, name='expref', values=dict(zip(zip(low_levels, upp_levels),exp_ref)))
    nx.set_edge_attributes(G, name='theo_ref', values=dict(zip(zip(low_levels, upp_levels),theo_ref)))

    ### Characterization of nodes
    with open(level_data_path) as level_file:
        level_reader = csv.reader(level_file)
        level_data = np.array(list(level_reader))
        j = dict()
        conf = dict()
        term = dict()
        parity = dict()
        n_dict = dict()
        l = dict()
        modelName = dict()
        dummyCommunity = dict()
        energy = dict()
        # leading_configurations = dict()
        ### dictionary for l-states, used in calculation of modelName
        L_dictionary = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4, 'h':5, 'i':6, 'k':7, 'l':8, 'm':9, 'n':10, 'o':11, 'q':12, 'r':13, 't':14, 'u':15, 'v':16, 'w':17, 'x':18, 'y':19, 'z':20, 'a':21, 'b':22, 'c':23, 'e':24, 'j':25}

        for n in nx.nodes(G):

            # total J
            j_value = (level_data[level_data[:,24] == n,8])
            if j_value.size == 0:
                j[n] = ''
            else:
                j[n] = j_value[0]

            # energy
            energy_value = (level_data[level_data[:,24] == n,10])
            if energy_value.size == 0:
                energy[n] = ''
            elif energy_value[0] == '':
                energy[n] = ''
            else:
                energy[n] = float(energy_value[0])

            # configuration
            conf_value = (level_data[level_data[:,24] == n,5])
            if conf_value.size == 0:
                conf[n] = ''
            else:
                conf[n] = conf_value[0]

            # # leading configurations
            # j_value = (level_data[level_data[:,24] == n,8])
            # if j_value.size == 0:
            #     j[n] = ''
            # else:
            #     j[n] = j_value[0]

            # n
            # only makes sense for H1.0
            n_dict[n] = conf[n][:-1]

            # term [2S+1]LJ
            try:
                term[n] = (level_data[level_data[:,24] == n,6])[0]+j[n]
                term[n] = term[n].replace('*', '')
            except:
                term[n] = ''

            # l
            try:
                for i, char in enumerate(term[n]):
                    if not char.isdigit():
                        index = i
                        break
                l[n] = term[n][index]
            except:
                l[n] = ''

            # parity
            parity_value = (level_data[level_data[:,24] == n ,7])
            if parity_value.size == 0:
                parity[n] = ''
            else:
                parity[n] = parity_value[0]


            # only works for 1 electron ions
            try:
                if int(j[n][:-2]) == 2*L_dictionary[l[n].lower()] + 1:
                    upper_true = True
                    lower_true = False
                else:
                    upper_true = False
                    lower_true = True

                modelName[n] = n_dict[n] + upper_true*l[n] + lower_true*l[n].lower()
            except:
                modelName[n] = ''

            # dummy community for commonneighbours link prediction algorithm, which needs a community
            dummyCommunity[n] = 0


    # Give attributes to nodes
    nx.set_node_attributes(G,name='conf',values=conf)
    nx.set_node_attributes(G,name='term',values=term)
    nx.set_node_attributes(G,name='J',values=j)
    nx.set_node_attributes(G,name='n',values=n_dict)
    nx.set_node_attributes(G,name='l',values=l)
    nx.set_node_attributes(G,name='parity',values=parity)
    nx.set_node_attributes(G,name='modelName',values=modelName)
    nx.set_node_attributes(G,name='dummyCommunity',values=dummyCommunity)
    nx.set_node_attributes(G,name='energy',values=energy)

    if dictionaries:
        return (G, conf, term, j)
    else:
        return G


def only_largest_component(G):
    no_conn = nx.number_connected_components(G)
    if no_conn==1:
        return G
    else:
        Gc = G.copy()
        components = sorted(nx.connected_components(Gc), key = len, reverse=True)
        # take only largest component
        for comp in components[1:]:
            Gc.remove_nodes_from(comp)
        return Gc


def draw_groups(G, groups, labels=None, pos=None):
    """
    Function for drawing NetworkX graphs with different node colors indicating their group membership.

    Parameters
    ----------
    G : NetworkX graph
    groups : dictionary
        {nodes: group} dictionary with nodes as keys and an integer indicating the group it belongs to. The nodes should be of the same format as in G. group should be an integer {0,...,k}, where k is the number of groups.
    labels : dictionary, optional
        Node labels in a dictionary keyed by node of text labels. (default=None)
    pos : dictionary or None (optional)
        A dictionary with nodes as keys and positions as values. If not specified a spring layout positioning will be computed. (default=None)

    Notes
    -----
    For k larger than 19, the colors might not be unique, i.e. two different groups might be assigned the same color.

    """
    import matplotlib.pyplot as plt

    n = [] # nodes
    g = [] # assigned groups
    for node, group in groups.iteritems():
        n.append(node)
        g.append(group)
    n = np.asarray(n)
    g = np.asarray(g)

    # check set of group integer values
    gmin, gmax = g.min(), g.max()
    if gmin != 0:
        warnings.warn('Minimum group assignment number is not 0, this may result in multiple groups having the same color.',
                      RuntimeWarning)

    # choose colormap
    if gmax <= 9:
        colors = (plt.get_cmap('Vega10'))(np.linspace(0.0,1.0,num=gmax+1))
    elif gmax <= 19:
        colors = (plt.get_cmap('Vega20'))(np.linspace(0.0,1.0,num=gmax+1))
    else:
        colors = (plt.get_cmap('prism'))(np.linspace(0.0,1.0,num=gmax+1))
        warnings.warn('Maximum group assignment number is larger than 19, this may result in multiple groups having the same color.',
                      RuntimeWarning)

    # set position layout
    if not pos:
        pos=nx.spring_layout(G)

    # draw network
    for i in xrange(gmax+1):
        nodelist = list(n[g==i])
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, node_color=colors[i])
    nx.draw_networkx_edges(G, pos=pos)
    if labels != None:
        nx.draw_networkx_labels(G, pos=pos, labels=labels)


def remove_attributes(G):
    """
    This function returns a version of G where all the node and edge attributes have been removed.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    G2 : NetworkX graph

    """
    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes())
    G2.add_edges_from(G.edges())
    return G2


##
## @brief      Removes nodes from network that have an empty specified
##             attribute.
##
## @param      Graph      The networkx graph
## @param      attribute  The attribute (string)
##
## @return     Graph
##
def remove_empty_levels(Graph, attribute):
    # remove nodes with empty attribute
    temp_list=nx.get_node_attributes(Graph, attribute)
    for i, (node_ID, attribute) in enumerate(temp_list.items()):
        if not attribute: #if empty string, delete entries
            Graph.remove_node(node_ID)

    return Graph


class LinkPrediction:
    G_original         = None # original graph
    G_training         = None # training graph
    G_probe            = None # probe graph
    G_predict          = None # predicition graph

    P                  = 0 # number of positive samples
    N                  = 0 # number of negative samples

    prediction_list    = [] # list of tuples (edge, probability) (hast to be ordered by descending probability)
    gain_list          = [] # list of gain values
    probe_list         = [] # list of probe edges
    probe_rank         = [] # list of probe edge ranks

    is_correct         = None # array of bools

    # ROC
    ROC                = None # (2, # non-observed links) - array containing TPR and FPR
    FPR                = None
    TPR                = None

    # gain evaluation measures (arrays)
    CG                 = None
    DCG                = None
    ICG                = None
    nCG                = None
    nDCG               = None


    # AUC value
    AUC                = 0


    #########################FUNCTIONS#############################

    def dropout(self, p):
        """
        This function creates the probe and training graphs
        self.G_probe and self.G_training by randomly dropping every
        node in the original graph with probability p.

        Parameters
        ----------
        p : float
            Dropout probability.

        To-Do
        -----
        Transcribe weights of G_original to G_probe and G_training.
        """
        # randomly select which edges to drop
        edge_list = np.asarray(self.G_original.edges())
        r = np.random.sample(len(edge_list))
        dropout_mask = r<p

        edges_dropped   = edge_list[dropout_mask]
        edges_kept      = edge_list[np.logical_not(dropout_mask)]

        # write kept edges to training graph
        self.G_training = nx.Graph()
        self.G_training.add_edges_from(edges_kept)

        # write dropped edges to probe graph
        self.G_probe = nx.Graph()
        self.G_probe.add_edges_from(edges_dropped)

        # store number of positive samples
        self.P = len(edges_dropped)

        # calculate number of negative samples
        n_nodes_training = self.G_training.number_of_nodes()
        n_edges_training = self.G_training.number_of_edges()
        self.N = n_nodes_training*(n_nodes_training-1)/2
                 - n_edges_training - len(edges_dropped)

        # store list of probe edges
        self.probe_list = self.get_dropout_list()


    def cross_validation(self, edge_list, train_index, validation_index):
        """
        This function creates the probe and training graphs self.G_probe and self.G_training.

        Parameters
        ----------scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
        edge_list :
            Edgelist of G_original.
        train_index :
            train_index array (output of **.split()).
        validation_index :
            validation_index array (output of **.split()).

        TO-DO
        ----------
        fix the routine using this function.
        """
        edges_kept      = edge_list[train_index]
        edges_dropped   = edge_list[validation_index]

        # print edges_kept

        # write kept edges to training graph
        self.G_training = nx.Graph()
        self.G_training.add_edges_from(edges_kept)

        # write dropped edges to probe graph
        self.G_probe = nx.Graph()
        self.G_probe.add_edges_from(edges_dropped)

        # store number of positive samples
        self.P = len(edges_dropped)

        # calculate number of negative samples
        n_nodes_training = self.G_training.number_of_nodes()
        n_edges_training = self.G_training.number_of_edges()
        self.N = n_nodes_training*(n_nodes_training-1)/2 - n_edges_training - len(edges_dropped)

        # store list of probe edges
        self.probe_list = self.get_dropout_list()


    ################################ SPM ###############################
    @staticmethod
    def _SPM_delta_E_mat(V, E, M_cut):
        """
        Compute delta_E in terms of adjacency matrix indices.
        """
        # randomly select edges to cut
        choice = np.random.choice(len(E), size=M_cut, replace=False)
        delta_E = E[choice,:]

        # translate delta_E to indices in adjacency matrix
        delta_E_mat = np.asarray([(np.argwhere(V==edge1)[0,0],
                                   np.argwhere(V==edge2)[0,0])
                                  for edge1, edge2 in delta_E])
        return delta_E_mat


    @staticmethod
    def _gt_delta_E_mat(V, E, M_cut):
        """
        Compute delta_E in terms of adjacency matrix indices.
        """
        # randomly select edges to cut
        choice = np.random.choice(len(E), size=M_cut, replace=False)
        delta_E_mat = E[choice,:2]

        # translate delta_E to indices in adjacency matrix
        # delta_E_mat = np.asarray([(np.argwhere(V==edge1)[0,0], np.argwhere(V==edge2)[0,0]) for edge1, edge2 in delta_E])
        return delta_E_mat


    @staticmethod
    def _SPM_A_tilde_A_R(A, V, delta_E_mat):
        """
        This function computes A_tilde and A_R for SPM prediction or
        structural consistency.

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix.
        V : array-like
            Vertex list by which the adjacency matrix is ordered.
        delta_E_mat : np.ndarray
            Cut edges in terms of matrix indices.
        """
        # create delta_A and A_R
        delta_A = np.zeros_like(A, dtype=int)
        delta_A[delta_E_mat[:,0], delta_E_mat[:,1]] = 1
        delta_A[delta_E_mat[:,1], delta_E_mat[:,0]] = 1
        A_R = A - delta_A

        # compute eigenvalues and eigenvectors
        lam0, v = np.linalg.eigh(A_R)
        lam = lam0.round(decimals=4)

        # calculate first order eigenvalue corrections and A_tilde
        delta_lam = np.zeros_like(lam)
        A_tilde = np.zeros_like(A, dtype=float)

        # no degenerate eigenvalues vs degenerate eigenvalues
        if len(lam)==len(np.unique(lam)):
            for i in xrange(len(lam)):
                vec = np.asarray(v[:,i]).flatten()
                delta_lam[i] =
                    (np.dot(vec, np.asarray(np.dot(delta_A, vec)).flatten())
                     / np.dot(vec, vec))
                A_tilde += (lam[i] + delta_lam[i])*np.outer(vec, vec)
        else:
            for ew in np.unique(lam):
                counts = np.sum(lam==ew)
                if counts == 1:
                    i = np.argwhere(lam == ew)[0, 0]
                    vec = np.asarray(v[:, i]).flatten()
                    delta_lam[i] =
                        (np.dot(vec, np.asarray(np.dot(delta_A, vec)).flatten())
                         / np.dot(vec, vec))
                    A_tilde += (lam[i] + delta_lam[i]) * np.outer(vec, vec)
                else:
                    vecs = v[:, lam==ew]
                    W = np.dot(vecs.transpose(), np.dot(delta_A, vecs))
                    dlam, beta = np.linalg.eigh(W)
                    v_new = np.dot(vecs, beta)
                    for i, delta_ew in enumerate(dlam):
                        A_tilde +=
                            (ew + delta_ew) * np.outer(v_new[:, i], v_new[:, i])

        return A_tilde, A_R


    @staticmethod
    def _SPM_rank_mat(A_tilde, A):
        """
        Rank all adges not observed in A by the values in A_tilde.
        The edges are given in terms of matrix indices.
        """
        # rank non-observed links
        triu_ind = np.triu_indices(A_tilde.shape[0], k=1) # upper right triangular indices w/o diagonal
        triu_ind_flat = np.ravel_multi_index(triu_ind, dims=A_tilde.shape) # flattened

        ## sort upper right triangle
        sorted_ind = triu_ind_flat[np.asarray(np.argsort(
            A_tilde[triu_ind], axis=None)).flatten()][::-1]

        ## rank indices
        rank_ind = np.unravel_index(sorted_ind, dims=A_tilde.shape)
        rank_ind = np.stack((rank_ind[0], rank_ind[1]), axis=1)
        mask = np.asarray(A[rank_ind[:,0], rank_ind[:,1]] == 0).flatten() # remove edges that are already observed
        rank_mat = rank_ind[mask, :]

        return rank_mat


    @staticmethod
    def _SPM_prediction(G, p=0.1, n_selections=10):
        """
        This function predicts links in a network using the structural
        perturbation method (SPM).
        """

        # number of edges to cut rounded to next integer
        M = G.number_of_edges()
        M_cut = int(np.rint(p*M))

        # get adjacency matrix
        V = np.asarray(G.nodes()) # vertex list
        E = np.asarray(G.edges()) # edge list
        A = nx.to_numpy_matrix(G, nodelist=V).astype(int)

        delta_E_mat = LinkPrediction._SPM_delta_E_mat(V, E, M_cut)

        A_tilde_avg = np.zeros_like(A, dtype=float)
        for k in xrange(n_selections):
            A_tilde, A_R = LinkPrediction._SPM_A_tilde_A_R(A, V, delta_E_mat)
            A_tilde_avg += A_tilde

        A_tilde_avg /= n_selections # divide by number of selections

        rank_mat = LinkPrediction._SPM_rank_mat(A_tilde_avg, A)

        # translate rank_ind to edge labels
        # translate rank_ind to edge labels
        rank = [(V[ind1], V[ind2], A_tilde_avg[ind1, ind2]) for ind1, ind2 in rank_mat]
        return rank


    def predict_SPM(self):
        """
        Predict links which are not observed in the training graph using the structural perturbation method (SBM).

        References
        ----------
        .. [1] Lü, L., Pan, L., Zhou, T., Zhang, Y.-C., & Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
        """
        # calculate rank of non-observed links by SPM
        rank = self._SPM_prediction(self.G_training)

        # create prediction graph
        self.G_predict = nx.Graph()

        # store prediciton list and add edges to prediction graph
        prediction_list_dummy = []
        for node1, node2, score in rank:
            # set probabilities to score
            prediction_list_dummy.append(((node1, node2), score))

            # add edge to preediction graph
            self.G_predict.add_edge(node1, node2)

        self.prediction_list = prediction_list_dummy


    # ####################### nested SBM ############
    @staticmethod
    def _nested_SBM_prediction(nxG, force_niter=100, minimize_runs=10,
                               mcmc_args=10, pred_list=None, cutoff=1.0):
        """
        Link Prediction using nested SBM

        Parameters
        ----------
        nxG : NetworkX graph
            Graph to do link prediction on.
        wait : int
            Number of iterations to wait for a record-breaking event.
        mcmc_args : int
            niter passed to graph-tool.mcmc_sweeps

        Returns
        -------
        probabilities : list
            list of tuples (node1, node2, link probability) sorted by
            link probability
        """
        try:
            import graph_tool.all as gt
        except:
            warnings.warn("Need graph-tool for nested SBM!")
        import time
        G = nx2gt(nxG)
        G = gt.Graph(G, directed=False, prune=True)

        ground_state_estimation_list =
            [gt.minimize_nested_blockmodel_dl(G) for i in range(minimize_runs)]
        ground_state_estimation_list =
            sorted(ground_state_estimation_list,
                   key=lambda ground_state_estimation_list:
                       ground_state_estimation_list.entropy(),
                   reverse=False)
        ground_state_estimation = ground_state_estimation_list[0]
        if pred_list is not None:
            potential_edges =
                [(v1, v2) for v1 in G.vertices() for v2 in G.vertices()
                 if (G.vp.nodeid[v1], G.vp.nodeid[v2]) in pred_list]
        else:
            adj = gt.adjacency(G)
            ones = np.triu(np.ones((G.num_vertices(), G.num_vertices())))
            potential_edges = np.argwhere(np.logical_not(adj + ones))

        # probs = np.zeros((wait, len(potential_edges)))
        probs = [[] for _ in range(len(potential_edges))]
        bs = ground_state_estimation.get_bs()
        bs += [np.zeros(1)] * (6 - len(bs))
        ground_state = ground_state_estimation.copy(bs=bs, sampling=True)
        run_number = [0]
        blockstatesvisited = []

        def collect_edge_probs(s):
            S = s.entropy()
            # only look at new blockstates
            if np.all(np.abs(np.array(blockstatesvisited)-S) > 0.0001):
                # only look at blockstates with high likelyhood
                if len(blockstatesvisited) == 0 or
                  np.all((np.array(blockstatesvisited)-S) > -cutoff):
                    for i in range(len(potential_edges)):
                        p = s.get_edges_prob([potential_edges[i]],
                                entropy_args=dict(partition_dl=False))
                        probs[i].append((np.exp(p), S))
                    blockstatesvisited.append(S)
            run_number[0] = run_number[0] + 1

        collect_edge_probs(ground_state)

        gt.mcmc_equilibrate(ground_state,
                            force_niter=force_niter,
                            mcmc_args=dict(niter=mcmc_args),
                            callback=collect_edge_probs)

        def get_avg(p):
            p_avg = 0.0
            entropy_offset = p[0][1]
            norm = 0.0
            for prob in p:
                p_avg += prob[0]*np.exp(-prob[1] + entropy_offset)
                norm += np.exp(-prob[1] + entropy_offset)
            p_avg = p_avg/norm
            return p_avg

        probabilities = [(G.vp.nodeid[potential_edges[i][0]], G.vp.nodeid[potential_edges[i][1]], get_avg(probs[i])) for i in range(len(potential_edges))]
        probabilities = sorted(probabilities, key=lambda probabilities: probabilities[2], reverse=True)
        return probabilities


    def predict_nested_SBM(self, minimize_runs=10, force_niter=100,
                           mcmc_args=10, pred_list=None, cutoff=1.0):
        '''
        Link Prediction using nested SBM

        Parameters
        ----------
        wait : int
            Number of iterations to wait for a record-breaking event.
        mcmc_args : int
            niter passed to graph-tool.mcmc_sweeps
            should be around 10, but makes the algorithm last long

        Returns
        -------
        updates self with prediction
        '''
        probabilities = self._nested_SBM_prediction(self.G_training,
                                                    minimize_runs=10,
                                                    force_niter=force_niter,
                                                    mcmc_args=mcmc_args,
                                                    pred_list=pred_list,
                                                    cutoff=cutoff)
        self.G_predict = nx.Graph()
        self.G_predict.add_weighted_edges_from(probabilities,
                                               weight='likelyhood')
        self.prediction_list = [ ( (probability[0], probability[1]),
                                  probability[2])
                                for probability in probabilities]


    #####################EVALUATION#####################
    def check_if_correct(self):
        """
        Check if predicted edges are present in original graph.

        TO-DO
        -----
        The gain value edge attribute only exists for jitrik data at the moment.
        """
        is_correct_dummy = []
        gain_list_dummy = []
        for (node1, node2), prob in self.prediction_list:
            if self.G_probe.has_edge(node1, node2):
                is_correct_dummy.append(True)
                try: # to not raise exception for NIST data
                    gain_list_dummy.append(self.G_original[node1][node2]['gainValue'])
                except:
                    gain_list_dummy.append(True)
            else:
                is_correct_dummy.append(False)
                gain_list_dummy.append(0)
        self.is_correct = np.asarray(is_correct_dummy)
        self.gain_list = gain_list_dummy


    def get_prediction_list(self):
        import operator
        probability_dict = nx.get_edge_attributes(self.G_predict, 'probability')
        prediction_list_dummy = sorted(probability_dict.iteritems(), key=operator.itemgetter(1), reverse=True)

        # check whether predictions are correct
        prediction_list = []
        for prediction in prediction_list_dummy:
            prediction_list.append([prediction[0], prediction[1], self.G_original.has_edge(*(prediction[0]))])
        return prediction_list


    def get_dropout_list(self):
        return np.asarray(self.G_probe.edges())


    def get_probe_rank(self):
        self.probe_rank = np.zeros(self.probe_list.shape[0], dtype=int)

        for i in xrange(len(self.probe_list)):
            for j in xrange(len(self.prediction_list)):
                if set(self.probe_list[i]) == set(self.prediction_list[j][0]):
                    self.probe_rank[i] = j
        return self.probe_rank


    def calculate_P_and_N(self):
        self.P = len(self.G_probe.edges())
        self.N = len(self.G_training.nodes())*(len(self.G_training.nodes())-1)/2 - len(self.G_training.edges()) - len(self.G_probe.edges())


    def calculate_ROC(self):
        self.ROC = np.zeros((len(self.prediction_list), 2))
        TP = 0
        FP = 0
        for i, correct in enumerate(self.is_correct):
            if correct:
                TP += 1
            else:
                FP += 1

            self.ROC[i,0] = float(TP)/self.P   # TPR
            self.ROC[i,1] = float(FP)/self.N   # FPR

        return self.ROC


    def calculate_ROC_2(self):
        import numpy as np
        from sklearn.metrics import roc_curve

        # ranks  = np.arange(0,len(self.is_correct))
        # scores = 1 - np.divide(ranks.astype(float),len(lp_SPM.is_correct))
        scores = np.linspace(1,0,num=len(self.is_correct))

        fpr, tpr, thresholds = roc_curve(self.is_correct, scores)
        print tpr
        print fpr
        print thresholds

        self.ROC = np.zeros((len(self.prediction_list), 2))

        self.TPR = tpr   # TPR
        self.FPR = fpr   # FPR

        return (self.TPR, self.FPR)


    def plot_ROC(self, fig=None, save_switch=False, name='test',
                 plotdirectory='../plots/', plotlabel = ''):
        import matplotlib.pyplot as plt
        # if self.ROC==None:
            # self.calculate_ROC()
        TPR = self.ROC[:,0]
        FPR = self.ROC[:,1]
        if not fig:
            fig = plt.figure()
        ax = plt.axes()
        ax.scatter(FPR, TPR, marker='.', label = plotlabel)
        ax.plot((0, 1), (0, 1), 'r--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC')
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        if save_switch==True:
            fig.savefig(plotdirectory+name+'.png')
            plt.close(fig)
        else: fig.draw()


    def calculate_gain_measures(self, base):
        """
        ## @brief      Calculates the information retrieval measures
                       described by Järvelin, K., & Kekäläinen, J.
                       (2002). Cumulated gain-based evaluation of IR
                       techniques. ACM Transactions on Information
                       Systems, 20(4), 422–446.
                       https://doi.org/10.1145/582415.582418
        ##
        ## @param      masked_ranks  The masked array of predictions
        ## @param      base          The base of the logarithm in the
                                     discount
        ##                           factor (hyperparameter)
        ##
        ## @return     A tuple consisting of (CG, DCG, ICG, nCG, nDCG):
        ##             Cumulated gain, Discounted cumulated gain, ideal
        ##             cumulated gain, normalised cumulated gain,
        ##             normalised discounted cumulated gain.
        """

        # gain vector
        # G = self.is_correct.flatten() #old
        G = np.asarray(self.gain_list).flatten()

        # cumulated gain vector
        self.CG = np.cumsum(G)

        # discounted cumulative gain vector
        temp_log_arg  = np.log10(np.arange( base,G.shape[0] ))
        temp_discount = np.divide ( temp_log_arg , np.log10(base) )
        temp          = np.concatenate( (G[:base],
                                         np.divide(G[base:], temp_discount )))
        self.DCG = np.cumsum(temp)

        # Ideal gain vector
        IG   = np.sort(G, axis=0)[::-1]
        # Ideal cumulated gain vector
        self.ICG  = np.cumsum(IG)

        # normalised cumulated gain
        self.nCG  = np.divide(self.CG.astype(float), self.ICG.astype(float) )

        # normalised discounted cumulated gain
        self.nDCG = np.divide(self.DCG, self.ICG.astype(float) )


    def plot_gain_metrics(self):
        import matplotlib.pyplot as plt
        #TODO: not working
        # if self.CG == None: #TODO: check also self.DCG, self.ICG, self.nCG, self.nDCG
        #     self.calculate_gain_measures()
        # plot gain metrics
        plt.figure()
        plt.plot(self.CG[:], label='Cumulated Gain', marker='.')
        plt.plot(self.DCG[:], label='Discounted Cumulated Gain', marker='.')
        plt.plot(self.ICG[:], label='Ideal Cumulated Gain', marker='.')
        plt.xlabel('Rank')
        plt.ylabel('Gain')
        plt.title('Gain curves')
        # plt.legend(loc=8)
        plt.show()

        # plot normalised gain metrics
        plt.figure()
        plt.plot(self.nCG[:],
                 label='Normalised Cumulated Gain',
                 marker='.')
        plt.plot(self.nDCG[:],
                 label='Normalised Discounted Cumulated Gain',
                 marker='.')
        plt.title('Normalised Gain curves')
        plt.ylabel('Normalised Gain')
        plt.xlabel('Rank')
        plt.ylim([0,1])
        # plt.legend(loc=2)
        plt.show()


    # @brief      Calculates the AUC value.
    #
    # @param      predictions                  The predictions
    # @param      ranks                        The ranks
    # @param      ranks_considered_percentage  The ranks considered percentage
    #
    # @return     The AUC value.
    #
    # AUC: Lu, L., and Zhou, T. (2010). Link Prediction in Complex Networks: A
    # Survey. Physica A, 390(6), 1150-1170.
    # https://doi.org/10.1016/j.physa.2010.11.027
    #
    def calculate_AUC_MC(self, sample_size_percentage=0.3,
                         ranks_considered_percentage=0.99):

        b = 0 #number of times the predicted link has a higher rank than a randomly chosen non-existent link
        w = 0 #number of times the predicted link has a equal or worse rank than a randomly chosen non-existent link

        #sampling
        #TODO: what is a sensible sample size?
        #TODO: ranks_considered_percentage = 1.0 funktioniert nicht?
        sample_size = int(sample_size_percentage * self.is_correct.shape[0])
        for e in range(sample_size):
            temp_true =
                np.asarray(np.argwhere(self.is_correct==True).flatten())
            rank_probe =
                np.random.choice(temp_true[np.where(
                        temp_true < int(ranks_considered_percentage * self.is_correct.shape[0]) )], 1, replace=True) #choosing from E_p
            temp_false = np.asarray(np.argwhere(self.is_correct==False).flatten())
            rank_nonexist = np.random.choice( temp_false[np.where( temp_false < int(ranks_considered_percentage * self.is_correct.shape[0]) )], 1, replace=True) #choosing from non-existent links, = U-E

            if (rank_probe < rank_nonexist):
                b+=1
            elif (rank_probe == rank_nonexist):
                w+=1

        self.AUC = (b + 0.5 * w) / float(sample_size)
        # print 'b',b
        # print 'w',w
        # print 'sample size', sample_size

        return self.AUC

    def calculate_AUC(self):
        from sklearn.metrics import roc_auc_score

        y_true = self.is_correct
        scores = np.linspace(1,0,num=len(self.is_correct))
        # TODO fix exception problem: in case all entries of is_correct are either True or False, the next line rasises an exception
        self.AUC = roc_auc_score(y_true, scores)

        return self.AUC



# Copied from https://bbengfort.github.io/snippets/2016/06/23/graph-tool-from-networkx.html
def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    import graph_tool.all as gt

    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    import graph_tool.all as gt

    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'nodeid' -- modify as needed!
    gtG.vertex_properties['nodeid'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['nodeid'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG
