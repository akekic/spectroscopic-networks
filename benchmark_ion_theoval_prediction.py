"""
Description: Performs a prediction algorithm on a networkx graph derived from NIST data and checks the prediction afterwards on the model network based on Jitrik, Bunge.

Experimental data by NIST
Theoretical data by Jitrik, O., Bunge, C. F. (2004). Transition probabilities for hydrogen-like atoms. Journal of Physical and Chemical Reference Data, 33(4), 1059-1070. https://doi.org/10.1063/1.1796671
HRG C++ algorithm by Clauset, A., Moore, C., & Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
SPM algorithm by Lu, L., Pan, L., Zhou, T., Zhang, Y.-C., & Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
SBM algorithm by Newman, M. E. J., & Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7). https://doi.org/10.1103/PhysRevLett.117.078301

@author: Julian Heiss
@date: July 2017
"""

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

import networkx as nx
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# %%
# GLOBAL VARIABLES
########################################

ion = 'He2.0'                    # ion used as input
only_dipole_global = True      # global switch for taking the full network or only dipole lines
n_limit            = True      # limit is only to be used for one electron ions (hydrogenic ions)
if n_limit==False:
    max_n_global = None
else:
    max_n_global   = 8                 # maximal n considered in network
SPM                = False      # run SPM prediction algorithm
HRG                = False       # run HRG prediction algorithm
SBM                = True      # run SBM prediction algorithm
save_switch        = True        # True: save all figures. False: show
save_files         = False      #True: Do not delete edgelist and similar files used for the prediction algorithms (eg. HRG)

########################################

if SPM == True:
    plot_directory = '../plots/SPM/' + ion + '/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    # %%
    # predict using SPM
    print 'SPM'
    lp_SPM = nx2.LinkPrediction()

    lp_SPM.split_graph_into_sets(ion=ion, only_dipole=only_dipole_global, max_n=max_n_global)
    print 'Training set number of edges', nx.number_of_edges(lp_SPM.G_training)
    print 'Training set number of nodes', nx.number_of_nodes(lp_SPM.G_training)
    print 'Probe set number of edges',    nx.number_of_edges(lp_SPM.G_probe)
    print 'Probe set number of nodes',    nx.number_of_nodes(lp_SPM.G_probe)

    # %%
    lp_SPM.predict_SPM()

    # get is_correct array
    lp_SPM.check_if_correct()

    # ROC
    lp_SPM.calculate_P_and_N() # calculate number of positive and negative samples
    name = ion + '_theoval' + '_SPM_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'
    lp_SPM.plot_ROC(fig=None, save_switch=save_switch, name=name, plotdirectory=plot_directory)

    # # calculate gain measures
    # lp_SPM.calculate_gain_measures(base=2)
    # lp_SPM.plot_gain_metrics()

    #AUC
    lp_SPM.calculate_AUC(sample_size_percentage=0.5, ranks_considered_percentage=0.5)
    print lp_SPM.AUC


#----------------------------------------------------------------------
if HRG == True:
    HRG_PATH = './HRG/'
    directory = HRG_PATH + ion + '_' + 'theo_validation' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot_directory = '../plots/HRG/' + ion + '/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # %%
    # predict using HRG
    print 'HRG'
    lp_HRG = nx2.LinkPrediction()

    lp_HRG.split_graph_into_sets(ion=ion, only_dipole=only_dipole_global, max_n=max_n_global)
    print 'Training set number of edges', nx.number_of_edges(lp_HRG.G_training)
    print 'Training set number of nodes', nx.number_of_nodes(lp_HRG.G_training)
    print 'Probe set number of edges',    nx.number_of_edges(lp_HRG.G_probe)
    print 'Probe set number of nodes',    nx.number_of_nodes(lp_HRG.G_probe)

    # create dictionary
    lp_HRG.create_HRG_ID_dict(experimental=False) # experimental always false

    # %%
    lp_HRG.write_edgelist(Graph=lp_HRG.G_training, label='training', directory=directory) # write training set edgelist
    lp_HRG.write_edgelist(Graph=lp_HRG.G_probe, label='probe', directory=directory) # write probe set edgelist

    # %%
    # fit best dendrogram
    lp_HRG.fit_dendrogram(label='training', filedirectory=directory, HRG_PATH='./HRG/')
    # create second .lut file with nx labels
    # nx2.create_nx_string_lut_file('model', label, model_numID_2_strID_dict, label_switch, filedirectory=directory)

    # %%
    # run prediction
    lp_HRG.HRG_predict_links(label='training', filedirectory=directory, HRG_PATH='./HRG/')

    # %%
    # load prediction
    print 'Loading predicted links'
    lp_HRG.load_predicted_links_from_wpairs_file(label='training', filedirectory=directory)

    # get is_correct array
    lp_HRG.check_if_correct()

    # %%
    # ROC
    lp_HRG.calculate_P_and_N() # calculate number of positive and negative samples
    name = ion + '_theoval' + '_HRG_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'
    lp_HRG.plot_ROC(fig=None, save_switch=save_switch, name=name, plotdirectory=plot_directory)

    # # calculate gain measures
    # lp_HRG.calculate_gain_measures(base=2)
    # lp_HRG.plot_gain_metrics()

    #AUC
    lp_HRG.calculate_AUC(sample_size_percentage=0.5, ranks_considered_percentage=0.5)
    print lp_HRG.AUC

    if not save_file:
        os.remove(directory+'probe.pairs')
        os.remove(directory+'training.pairs')
        os.remove(directory+'training-names.lut')
        os.remove(directory+'training-L.xy')
        os.remove(directory+'training-ranked.wpairs')
        os.remove(directory+'training_best-dendro.hrg')
        os.remove(directory+'training_best-dendro.info')


#----------------------------------------------------------------------
if SBM == True:
    plot_directory = '../plots/SBM/' + ion + '/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    # %%
    # predict using SBM
    print 'SBM'
    lp_SBM = nx2.LinkPrediction()

    lp_SBM.split_graph_into_sets(ion=ion, only_dipole=only_dipole_global, max_n=max_n_global)
    print 'Training set number of edges', nx.number_of_edges(lp_SBM.G_training)
    print 'Training set number of nodes', nx.number_of_nodes(lp_SBM.G_training)
    print 'Probe set number of edges',    nx.number_of_edges(lp_SBM.G_probe)
    print 'Probe set number of nodes',    nx.number_of_nodes(lp_SBM.G_probe)

    lp_SBM.predict_SBM()

    # get is_correct array
    lp_SBM.check_if_correct()

    # %%
    # ROC
    lp_SBM.calculate_P_and_N() # calculate number of positive and negative samples
    name = ion + '_theoval' + '_SBM_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'
    lp_SBM.plot_ROC(fig=None, save_switch=save_switch, name=name, plotdirectory=plot_directory)

    # # calculate gain measures
    # lp_SBM.calculate_gain_measures(base=2)
    # lp_SBM.plot_gain_metrics()

    #AUC
    lp_SBM.calculate_AUC(sample_size_percentage=0.5, ranks_considered_percentage=0.5)
    print lp_SBM.AUC
