"""
Description: Performs a prediction algorithm on a the hydrogen networkx graph and validates the prediction via Dropout/bootstrapping.

Experimental data by NIST
Theoretical data by Jitrik, O., & Bunge, C. F. (2004). Transition probabilities for hydrogen-like atoms. Journal of Physical and Chemical Reference Data, 33(4), 1059-1070. https://doi.org/10.1063/1.1796671
HRG C++ algorithm by Clauset, A., Moore, C., Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
SPM algorithm by Lu, L., Pan, L., Zhou, T., Zhang, Y.-C., Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
SBM algorithm by Newman, M. E. J., Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7). https://doi.org/10.1103/PhysRevLett.117.078301

@author: Julian Heiss
@date: July 2017
"""

# TODO: debug common neighbours

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
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold

# %%
# GLOBAL VARIABLES
########################################

ion = 'Fe1.0'                          # ion to use as input, see nx2.py
experimental = True                    # use the experimental NIST data or the theoretical JitrikBunke data
if experimental == False:
    # get Z
    for i, char in enumerate(ion.split('.')[0]):
        if char.isdigit():
            index = i
            break
    Z = int(ion.split('.')[0][index:])
    print 'Model Data'
else:
    print 'NIST Data'

only_dipole_global = False              # global switch for taking the full network or only dipole lines
n_limit            = False               # limit is only to be used for one electron ions (hydrogenic ions)
if n_limit==False:
    max_n_global = None
else:
    max_n_global   = 8                 # maximal n considered in network

SPM                = False              # run SPM (Structural Perturbation Method) prediction algorithm
HRG                = True              # run HRG (Hierarchical Random Graph) prediction algorithm
SBM                = False              # run SBM (Stochastic Block Model) prediction algorithm
NX_METHODS         = False              # run all nx lp methods

dropout_list       = [0.1, 0.3, 0.5]    # the fractions of edges dropped
n_runs             = 5                 # the number of runs for each dropout fraction over which is averaged
save_switch        = True              # True: save all figures. False: show
save_files         = False              #True: Do not delete edgelist and similar files used for the prediction algorithms (eg. HRG)

########################################

print 'Network: ', ion

#----------------------------------------------------------------------

#----------------------------------------------------------------------
if HRG == True:
    # %%
    HRG_PATH = './HRG/'
    directory = HRG_PATH + ion + '_' + 'dropout' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot_directory = '../plots/HRG/' + ion + '/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # %%
    # predict using HRG
    print 'Method: ','HRG'
    lp_HRG = nx2.LinkPrediction()

    # load network
    if experimental == False:
        # load theoretical data
        if only_dipole_global == True:
            lp_HRG.G_original = nx2.model_network(Z=Z, E1=True, max_n=max_n_global) #only dipole lines up to max_n
        else:
            lp_HRG.G_original = nx2.model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=max_n_global) #all lines
    else:
        if only_dipole_global == True:
            lp_HRG.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
            lp_HRG.G_original = nx2.remove_empty_levels(lp_HRG.G_original, 'term') #remove nodes with empty term entry
            lp_HRG.G_original = nx2.remove_n_greater(lp_HRG.G_original, max_n=max_n_global) # maximal n considered
            lp_HRG.G_original = nx2.only_dipole_transitions_parity(lp_HRG.G_original) #only dipole lines
            lp_HRG.G_original = nx2.only_largest_component(lp_HRG.G_original)
        else:
            lp_HRG.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
            lp_HRG.G_original = nx2.remove_empty_levels(lp_HRG.G_original, 'term') #remove nodes with empty term entry
            lp_HRG.G_original = nx2.remove_n_greater(lp_HRG.G_original, max_n=max_n_global) # maximal n considered
            lp_HRG.G_original = nx2.only_largest_component(lp_HRG.G_original)

    # create dict
    lp_HRG.create_HRG_ID_dict(experimental=experimental)


    # %%
    ROC_curves     = [] # to store the result for each dropout fraction
    ROC_curves_std = []
    nCG_curves_avg = [] # to store the result for each dropout fraction
    nCG_curves_std = []
    AUC_avg        = [] # to store the result for each dropout fraction
    AUC_std        = []
    for dropout in dropout_list:

        x_values = np.linspace(0,1,num=nx.number_of_nodes(lp_HRG.G_original)**2)
        ROC_curves_runs = [] # to store the result of each individual run
        nCG_curves_runs = [] # to store the result of each individual run
        AUC_runs = [] # to store the result of each individual run
        for k in xrange(n_runs):
            # dropout
            lp_HRG.dropout(dropout)

            # print 'Number of edges', nx.number_of_edges(lp_HRG.G_training)
            # print 'Number of nodes', nx.number_of_nodes(lp_HRG.G_training)

            # %%
            lp_HRG.write_edgelist(Graph=lp_HRG.G_training,
                            label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                            directory=directory) # write training set edgelist
            lp_HRG.write_edgelist(Graph=lp_HRG.G_probe,
                            label='probe_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                            directory=directory) # write probe set edgelist

            # %%
            # fit best dendrogram
            lp_HRG.fit_dendrogram(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                                    filedirectory=directory, HRG_PATH='./HRG/')
            # create second .lut file with nx labels
            # nx2.create_nx_string_lut_file('model', label, model_numID_2_strID_dict, label_switch, filedirectory=directory)

            # %%
            # run prediction
            lp_HRG.HRG_predict_links(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                                    filedirectory=directory, HRG_PATH='./HRG/')

            # %%
            # load prediction
            print 'Loading predicted links'
            lp_HRG.load_predicted_links_from_wpairs_file(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                                                    filedirectory=directory)

            # get is_correct array
            lp_HRG.check_if_correct()

            # ROC
            lp_HRG.calculate_P_and_N() # calculate number of positive and negative samples
            ROC_curves_runs.append(lp_HRG.calculate_ROC())

            # nCG
            lp_HRG.calculate_gain_measures(2)
            nCG_curves_runs.append(lp_HRG.nCG)

            # AUC
            AUC_runs.append(lp_HRG.calculate_AUC())
            #(end for-loop)

        # ROC
        TPR_interp = np.zeros((n_runs, len(x_values)))
        for i,ROC in enumerate(ROC_curves_runs):
            TPR = ROC[:,0]
            FPR = ROC[:,1]
            TPR_interp[i,:] = np.interp(x_values, FPR, TPR)
        TPR_avg = np.mean(TPR_interp,axis=0)
        TPR_std = np.std(TPR_interp, axis=0)
        ROC_curves.append(TPR_avg)
        ROC_curves_std.append(TPR_std)

        # nCG
        nCG_interp = np.zeros((n_runs, len(x_values)))
        for i,nCG in enumerate(nCG_curves_runs):
            #TODO: hier gibt es manchmal rundungsfehler bei der stepsize, deswegen ist der rundungsfehler doppelt eingebaut, damit er sich aufhebt.
            nCG_interp[i,:] = np.interp(x_values, np.arange(0,1.0/float(len(nCG))*float(len(nCG)),1.0/float(len(nCG))), nCG)
        nCG_avg = np.mean(nCG_interp,axis=0)
        nCG_std = np.std(nCG_interp, axis=0)
        nCG_curves_avg.append(nCG_avg)
        nCG_curves_std.append(nCG_std)

        # AUC
        AUC_runs_avg = np.mean(AUC_runs)
        AUC_runs_std = np.std(AUC_runs)
        AUC_avg.append(AUC_runs_avg)
        AUC_std.append(AUC_runs_std)
        with open(directory + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik' + '_AUC_Results', 'a') as results_file:
            results_file.write('Average (over '+ str(n_runs) +' runs) AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)+ '\n')
            results_file.close()
        print 'Average AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)
        #(end for-loop)




    if not save_files:
        os.remove(directory +'probe_'    + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
        os.remove(directory +'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
        os.remove(directory +'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-names.lut')
        os.remove(directory +'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-L.xy')
        os.remove(directory +'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-ranked.wpairs')
        os.remove(directory +'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.hrg')
        os.remove(directory +'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.info')


