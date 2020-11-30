"""
Description: Performs a prediction algorithm on a the hydrogen networkx graph and validates the prediction via cross-validated dropout/bootstrapping.

Experimental data by NIST
Theoretical data by Jitrik, O., & Bunge, C. F. (2004). Transition probabilities for hydrogen-like atoms. Journal of Physical and Chemical Reference Data, 33(4), 1059-1070. https://doi.org/10.1063/1.1796671
HRG C++ algorithm by Clauset, A., Moore, C., Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
SPM algorithm by Lu, L., Pan, L., Zhou, T., Zhang, Y.-C., Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
SBM algorithm by Newman, M. E. J., Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7). https://doi.org/10.1103/PhysRevLett.117.078301

@author: Julian Heiss
@date: August 2017
"""

# TODO: debug CG averaging
#       debug proximity measures
#
# introduce ShuffleSplit if wanted
# write n_splits instead of n_repeats into labels!

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
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold

# %%
# GLOBAL VARIABLES
########################################

ion = 'H1.0'                          # ion to use as input, see nx2.py
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

SPM                = True              # run SPM (Structural Perturbation Method) prediction algorithm
HRG                = False              # run HRG (Hierarchical Random Graph) prediction algorithm
SBM                = False              # run SBM (Stochastic Block Model) prediction algorithm
NX_METHODS         = False              # run all nx lp methods
method_names = ['SPM'] # methods to un in NX_METHODS


n_splits_list = [10, 5, 2] # Gibt das k von k-fold validation
n_repeats_list = [1, 2, 5] # Wie oft die k-fold validation wiederholt werden soll
# Obige parameter sind so gewaehlt, dass fuer jede rate das experiment 10 mal wiederholt wird.
dropout_list = [1/float(x) for x in n_splits_list] # calculate dropout fraction from k-folding

# for ShuffleSplit:
# n_splits_list = [10, 10, 10] # Gibt das k von k-fold validation
# n_repeats_list = [1, 1, 1] # Wie oft die k-fold validation wiederholt werden soll
# dropout_list = [0.1, 0.2, 0.5] # calculate dropout fraction from k-folding

save_switch        = True              # True: save all figures. False: show
save_files         = False              #True: Do not delete edgelist and similar files used for the prediction algorithms (eg. HRG)

########################################

print 'Network: ', ion

#----------------------------------------------------------------------
if SPM == True:
    plot_directory = '../plots/SPM/' + ion + '/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    # %%
    # predict using SPM
    print 'Method: ','SPM'
    lp_SPM = nx2.LinkPrediction()

    # load network
    if experimental == False:
        # load theoretical data
        if only_dipole_global == True:
            lp_SPM.G_original = nx2.model_network(Z=Z, E1=True, max_n=max_n_global) #only dipole lines up to max_n
        else:
            lp_SPM.G_original = nx2.model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=max_n_global) #all lines
    else:
        # load NIST data
        if only_dipole_global == True:
            lp_SPM.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
            lp_SPM.G_original = nx2.remove_empty_levels(lp_SPM.G_original, 'term') #remove nodes with empty term entry
            lp_SPM.G_original = nx2.remove_n_greater(lp_SPM.G_original, max_n=max_n_global) # maximal n considered
            lp_SPM.G_original = nx2.only_dipole_transitions_parity(lp_SPM.G_original) #only dipole lines
            lp_SPM.G_original = nx2.only_largest_component(lp_SPM.G_original)
        else:
            lp_SPM.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
            lp_SPM.G_original = nx2.remove_empty_levels(lp_SPM.G_original, 'term') #remove nodes with empty term entry
            lp_SPM.G_original = nx2.remove_n_greater(lp_SPM.G_original, max_n=max_n_global) # maximal n considered
            lp_SPM.G_original = nx2.only_largest_component(lp_SPM.G_original)

    print 'Number of edges', nx.number_of_edges(lp_SPM.G_original)
    print 'Number of nodes', nx.number_of_nodes(lp_SPM.G_original)

    # %%
    ROC_curves     = [] # to store the result for each dropout fraction
    ROC_curves_std = []
    nCG_curves_avg = [] # to store the result for each dropout fraction
    nCG_curves_std = []
    AUC_avg        = [] # to store the result for each dropout fraction
    AUC_std        = []

    edge_list = np.asarray(lp_SPM.G_original.edges())

    for n_splits, n_repeats, dropout in zip(n_splits_list, n_repeats_list, dropout_list):

        x_values = np.linspace(0,1,num=nx.number_of_nodes(lp_SPM.G_original)**2)
        ROC_curves_runs = [] # to store the result of each individual run
        nCG_curves_runs = [] # to store the result of each individual run
        AUC_runs = [] # to store the result of each individual run

        # initialise k-folding
        # old version: kf = KFold(n_splits=n_splits, shuffle=True)
        rs = ShuffleSplit(n_splits=n_splits, test_size=dropout) #für ShuffleSplit ist n_splits so etwas wie unser n_runs
        # rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
        for train_index, validation_index in rs.split(edge_list):
            # get sets
            lp_SPM.cross_validation(edge_list, train_index, validation_index)

            # %%
            lp_SPM.predict_SPM()

            # get is_correct array
            lp_SPM.check_if_correct()

            # Evaluations
            # ROC
            lp_SPM.calculate_P_and_N() # calculate number of positive and negative samples
            ROC_curves_runs.append(lp_SPM.calculate_ROC())

            # nCG
            lp_SPM.calculate_gain_measures(2)
            nCG_curves_runs.append(lp_SPM.nCG)

            # AUC
            AUC_runs.append(lp_SPM.calculate_AUC())
            #(end for-loop)

        # ROC
        TPR_interp = np.zeros((n_splits*n_repeats, len(x_values)))
        for i,ROC in enumerate(ROC_curves_runs):
            TPR = ROC[:,0]
            FPR = ROC[:,1]
            TPR_interp[i,:] = np.interp(x_values, FPR, TPR)
        TPR_avg = np.mean(TPR_interp,axis=0)
        TPR_std = np.std(TPR_interp, axis=0)
        ROC_curves.append(TPR_avg)
        ROC_curves_std.append(TPR_std)

        # nCG
        nCG_interp = np.zeros((n_splits*n_repeats, len(x_values)))
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
        print 'Average AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)
        #(end for-loop)

    # ROC
    rocfig = plt.figure()
    ax     = rocfig.gca()
    cmap = plt.get_cmap('Set1')
    for i, (dropout, n_repeats) in enumerate(zip(dropout_list, n_repeats_list)):
        ROC     = ROC_curves[i]
        TPR     = ROC
        TPR_err = ROC_curves_std[i]
        FPR     = x_values

        color=cmap(i / float(len(dropout_list)))

        ax.plot((0, 1), (0, 1), 'r--') # red diagonal line
        ax.plot(FPR, TPR, label='dropout='+str(dropout)+', n_repeats='+str(n_repeats), color=color)
        ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        ax.legend(loc=4) #lower right corner
    if save_switch==True:
        name = ion + '_crossval' + '_SPM_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        rocfig.savefig(plot_directory+name+'.png')
    else: rocfig.show()

    # nCG
    ncgfig = plt.figure()
    ax     = ncgfig.gca()
    cmap = plt.get_cmap('Set1')
    for i, (dropout, n_repeats) in enumerate(zip(dropout_list, n_repeats_list)):
        nCG     = nCG_curves_avg[i]
        nCG_err = nCG_curves_std[i]

        color=cmap(i / float(len(dropout_list)))

        ax.plot(x_values, nCG, label='dropout='+str(dropout)+', n_repeats='+str(n_repeats), color=color)
        ax.fill_between(x_values, nCG+0.5*nCG_err, nCG-0.5*nCG_err, alpha=0.3, color=color)
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        ax.legend(loc=4) #lower right corner
    if save_switch==True:
        name = ion + '_crossval' + '_SPM_nCG'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        ncgfig.savefig(plot_directory+name+'.png')
    else: ncgfig.show()

    # AUC
    aucfig = plt.figure()
    ax     = aucfig.gca()
    cmap = plt.get_cmap('Set1')
    color=cmap(i / float(len(dropout_list)))

    fraction_kept_list = [1-x for x in dropout_list]
    ax.errorbar(fraction_kept_list, AUC_avg, yerr=AUC_std, fmt='o')
    ax.set_xlim(xmin=0.0, xmax=1.0)
    ax.set_ylim(ymin=0.0, ymax=1.0)
    if save_switch==True:
        name = ion + '_crossval' + '_SPM_AUC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        aucfig.savefig(plot_directory+name+'.png')
    else: aucfig.show()

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
                            label='training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                            directory=directory) # write training set edgelist
            lp_HRG.write_edgelist(Graph=lp_HRG.G_probe,
                            label='probe_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                            directory=directory) # write probe set edgelist

            # %%
            # fit best dendrogram
            lp_HRG.fit_dendrogram(label='training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                                    filedirectory=directory, HRG_PATH='./HRG/')
            # create second .lut file with nx labels
            # nx2.create_nx_string_lut_file('model', label, model_numID_2_strID_dict, label_switch, filedirectory=directory)

            # %%
            # run prediction
            lp_HRG.HRG_predict_links(label='training' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                                    filedirectory=directory, HRG_PATH='./HRG/')

            # %%
            # load prediction
            print 'Loading predicted links'
            lp_HRG.load_predicted_links_from_wpairs_file(label='training' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
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
            AUC_runs.append(lp_HRG.calculate_AUC(sample_size_percentage=0.5, ranks_considered_percentage=0.5))
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
        print 'Average AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)
        #(end for-loop)


    # ROC
    rocfig = plt.figure()
    ax     = rocfig.gca()
    cmap = plt.get_cmap('Set1')
    for i,dropout in enumerate(dropout_list):
        ROC     = ROC_curves[i]
        TPR     = ROC
        TPR_err = ROC_curves_std[i]
        FPR     = x_values

        color=cmap(i / float(len(dropout_list)))

        ax.plot((0, 1), (0, 1), 'r--') # red diagonal line
        ax.plot(FPR, TPR, label='dropout='+str(dropout)+', n_runs='+str(n_runs), color=color)
        ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc=4) #lower right corner
    if save_switch==True:
        name = ion + '_crossval' + '_HRG_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        rocfig.savefig(plot_directory+name+'.png')
    else: rocfig.show()

    # nCG
    ncgfig = plt.figure()
    ax     = ncgfig.gca()
    cmap = plt.get_cmap('Set1')
    for i,dropout in enumerate(dropout_list):
        nCG     = nCG_curves_avg[i]
        nCG_err = nCG_curves_std[i]

        color=cmap(i / float(len(dropout_list)))

        ax.plot(x_values, nCG, label='dropout='+str(dropout)+', n_runs='+str(n_runs), color=color)
        ax.fill_between(x_values, nCG+0.5*nCG_err, nCG-0.5*nCG_err, alpha=0.3, color=color)
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        ax.legend(loc=4) #lower right corner
    if save_switch==True:
        name = ion + '_crossval' + '_HRG_nCG'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        ncgfig.savefig(plot_directory+name+'.png')
    else: ncgfig.show()

    # AUC
    aucfig = plt.figure()
    ax     = aucfig.gca()
    cmap = plt.get_cmap('Set1')
    color=cmap(i / float(len(dropout_list)))

    fraction_kept_list = [1-x for x in dropout_list]
    ax.errorbar(fraction_kept_list, AUC_avg, yerr=AUC_std, fmt='o')
    ax.set_xlim(xmin=0.0, xmax=1.0)
    ax.set_ylim(ymin=0.0, ymax=1.0)
    if save_switch==True:
        name = ion + '_crossval' + '_HRG_AUC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        aucfig.savefig(plot_directory+name+'.png')
    else: aucfig.show()


    if not save_files:
        os.remove(directory+'probe' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
        os.remove(directory+'training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
        os.remove(directory+'training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-names.lut')
        os.remove(directory+'training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-L.xy')
        os.remove(directory+'training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-ranked.wpairs')
        os.remove(directory+'training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.hrg')
        os.remove(directory+'training_' + ion + '_crossval' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.info')


#----------------------------------------------------------------------
if SBM == True:
    plot_directory = '../plots/SBM/' + ion + '/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    # %%
    # predict using SBM
    print 'Method: ','SBM'
    lp_SBM = nx2.LinkPrediction()

    # load network
    if experimental == False:
        # load theoretical data
        if only_dipole_global == True:
            lp_SBM.G_original = nx2.model_network(Z=Z, E1=True, max_n=max_n_global) #only dipole lines up to max_n
        else:
            lp_SBM.G_original = nx2.model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=max_n_global) #all lines
    else:
        if only_dipole_global == True:
            lp_SBM.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
            lp_SBM.G_original = nx2.remove_empty_levels(lp_SBM.G_original, 'term') #remove nodes with empty term entry
            lp_SBM.G_original = nx2.remove_n_greater(lp_SBM.G_original, max_n=max_n_global) # maximal n considered
            lp_SBM.G_original = nx2.only_dipole_transitions_parity(lp_SBM.G_original) #only dipole lines
            lp_SBM.G_original = nx2.only_largest_component(lp_SBM.G_original)
        else:
            lp_SBM.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
            lp_SBM.G_original = nx2.remove_empty_levels(lp_SBM.G_original, 'term') #remove nodes with empty term entry
            lp_SBM.G_original = nx2.remove_n_greater(lp_SBM.G_original, max_n=max_n_global) # maximal n considered
            lp_SBM.G_original = nx2.only_largest_component(lp_SBM.G_original)

    # %%
    ROC_curves     = [] # to store the result for each dropout fraction
    ROC_curves_std = []
    nCG_curves_avg = [] # to store the result for each dropout fraction
    nCG_curves_std = []
    AUC_avg        = [] # to store the result for each dropout fraction
    AUC_std        = []
    for dropout in dropout_list:

        x_values = np.linspace(0,1,num=nx.number_of_nodes(lp_SBM.G_original)**2)
        ROC_curves_runs = [] # to store the result of each individual run
        nCG_curves_runs = [] # to store the result of each individual run
        AUC_runs = [] # to store the result of each individual run
        for k in xrange(n_runs):
            # dropout
            lp_SBM.dropout(dropout)

            # print 'Number of edges', nx.number_of_edges(lp_SBM.G_training)
            # print 'Number of nodes', nx.number_of_nodes(lp_SBM.G_training)

            # %%
            lp_SBM.predict_SBM()

            # get is_correct array
            lp_SBM.check_if_correct()

            # ROC
            lp_SBM.calculate_P_and_N() # calculate number of positive and negative samples
            ROC_curves_runs.append(lp_SBM.calculate_ROC())

            # nCG
            lp_SBM.calculate_gain_measures(2)
            nCG_curves_runs.append(lp_SBM.nCG)

            # AUC
            AUC_runs.append(lp_SBM.calculate_AUC(sample_size_percentage=0.5, ranks_considered_percentage=0.5))
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
        print 'Average AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)
        #(end for-loop)


    # ROC
    rocfig = plt.figure()
    ax     = rocfig.gca()
    cmap = plt.get_cmap('Set1')
    for i,dropout in enumerate(dropout_list):
        ROC     = ROC_curves[i]
        TPR     = ROC
        TPR_err = ROC_curves_std[i]
        FPR     = x_values

        color=cmap(i / float(len(dropout_list)))

        ax.plot((0, 1), (0, 1), 'r--') # red diagonal line
        ax.plot(FPR, TPR, label='dropout='+str(dropout)+', n_runs='+str(n_runs), color=color)
        ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc=4) #lower right corner
    if save_switch==True:
        name = ion + '_crossval' + '_SBM_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        rocfig.savefig(plot_directory+name+'.png')
    else: rocfig.show()

    # nCG
    ncgfig = plt.figure()
    ax     = ncgfig.gca()
    cmap = plt.get_cmap('Set1')
    for i,dropout in enumerate(dropout_list):
        nCG     = nCG_curves_avg[i]
        nCG_err = nCG_curves_std[i]

        color=cmap(i / float(len(dropout_list)))

        ax.plot(x_values, nCG, label='dropout='+str(dropout)+', n_runs='+str(n_runs), color=color)
        ax.fill_between(x_values, nCG+0.5*nCG_err, nCG-0.5*nCG_err, alpha=0.3, color=color)
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        ax.legend(loc=4) #lower right corner
    if save_switch==True:
        name = ion + '_crossval' + '_SBM_nCG'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        ncgfig.savefig(plot_directory+name+'.png')
    else: ncgfig.show()

    # AUC
    aucfig = plt.figure()
    ax     = aucfig.gca()
    cmap = plt.get_cmap('Set1')
    color=cmap(i / float(len(dropout_list)))

    fraction_kept_list = [1-x for x in dropout_list]
    ax.errorbar(fraction_kept_list, AUC_avg, yerr=AUC_std, fmt='o')
    ax.set_xlim(xmin=0.0, xmax=1.0)
    ax.set_ylim(ymin=0.0, ymax=1.0)
    if save_switch==True:
        name = ion + '_crossval' + '_SBM_AUC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        aucfig.savefig(plot_directory+name+'.png')
    else: aucfig.show()




#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------



if NX_METHODS == True:
    # method_names = ['RA', 'AA', 'PA', 'JC']
    lp = {} #emtpy dict
    for method in method_names:

        plot_directory = '../plots/' + method + '/' + ion + '/'
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        print 'Method: ',method
        lp[method] = nx2.LinkPrediction()

        # load network
        if experimental == False:
            # load theoretical data
            if only_dipole_global == True:
                lp[method].G_original = nx2.model_network(Z=Z, E1=True, max_n=max_n_global) #only dipole lines up to max_n
            else:
                lp[method].G_original = nx2.model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=max_n_global) #all lines
        else:
            # load NIST data
            if only_dipole_global == True:
                lp[method].G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
                lp[method].G_original = nx2.remove_empty_levels(lp[method].G_original, 'term') #remove nodes with empty term entry
                lp[method].G_original = nx2.remove_n_greater(lp[method].G_original, max_n=max_n_global) # maximal n considered
                lp[method].G_original = nx2.only_dipole_transitions_parity(lp[method].G_original) #only dipole lines
                lp[method].G_original = nx2.only_largest_component(lp[method].G_original)
            else:
                lp[method].G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
                lp[method].G_original = nx2.remove_empty_levels(lp[method].G_original, 'term') #remove nodes with empty term entry
                lp[method].G_original = nx2.remove_n_greater(lp[method].G_original, max_n=max_n_global) # maximal n considered
                lp[method].G_original = nx2.only_largest_component(lp[method].G_original)

        print 'Number of edges', nx.number_of_edges(lp[method].G_original)
        print 'Number of nodes', nx.number_of_nodes(lp[method].G_original)

        # %%
        ROC_curves     = [] # to store the result for each dropout fraction
        ROC_curves_std = []
        nCG_curves_avg = [] # to store the result for each dropout fraction
        nCG_curves_std = []
        AUC_avg        = [] # to store the result for each dropout fraction
        AUC_std        = []

        edge_list = np.asarray(lp[method].G_original.edges())

        for n_splits, n_repeats, dropout in zip(n_splits_list, n_repeats_list, dropout_list):
            x_values = np.linspace(0,1,num=nx.number_of_nodes(lp[method].G_original)**2)
            ROC_curves_runs = [] # to store the result of each individual run
            nCG_curves_runs = [] # to store the result of each individual run
            AUC_runs = [] # to store the result of each individual run

            # initialise k-folding
            # kf = KFold(n_splits=n_splits, shuffle=True)
            rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
            for train_index, validation_index in rkf.split(edge_list):
                lp[method].cross_validation(edge_list, train_index, validation_index)

                ######################
                #carry out prediction for specified method
                if method == 'SPM':
                    lp[method].predict_SPM()
                if method == 'SBM':
                    lp[method].predict_SBM()
                if method  == 'RA':
                    preds = nx.resource_allocation_index(lp[method].G_training)
                    prediction_list_dummy = []
                    for u, v, p in preds:
                        prediction_list_dummy.append( ((u, v), p) )
                    #sort prediction list
                    lp[method].prediction_list = sorted(prediction_list_dummy, key=lambda x: x[1], reverse=True)
                elif method == 'AA':
                    preds = nx.adamic_adar_index(lp[method].G_training)
                    prediction_list_dummy = []
                    for u, v, p in preds:
                        prediction_list_dummy.append( ((u, v), p) )
                    #sort prediction list
                    lp[method].prediction_list = sorted(prediction_list_dummy, key=lambda x: x[1], reverse=True)
                elif method == 'PA':
                    preds = nx.preferential_attachment(lp[method].G_training)
                    prediction_list_dummy = []
                    for u, v, p in preds:
                        prediction_list_dummy.append( ((u, v), p) )
                    #sort prediction list
                    lp[method].prediction_list = sorted(prediction_list_dummy, key=lambda x: x[1], reverse=True)
                elif method == 'JC':
                    preds = nx.jaccard_coefficient(lp[method].G_training)
                    prediction_list_dummy = []
                    for u, v, p in preds:
                        prediction_list_dummy.append( ((u, v), p) )
                    #sort prediction list
                    lp[method].prediction_list = sorted(prediction_list_dummy, key=lambda x: x[1], reverse=True)
                ######################


                # get is_correct array
                lp[method].check_if_correct()

                # ROC
                lp[method].calculate_P_and_N() # calculate number of positive and negative samples
                ROC_curves_runs.append(lp[method].calculate_ROC())

                # nCG
                lp[method].calculate_gain_measures(2)
                nCG_curves_runs.append(lp[method].nCG)

                # AUC
                AUC_runs.append(lp[method].calculate_AUC())
                #(end for-loop)

            # ROC
            TPR_interp = np.zeros((n_splits*n_repeats, len(x_values)))
            for i,ROC in enumerate(ROC_curves_runs):
                TPR = ROC[:,0]
                FPR = ROC[:,1]
                TPR_interp[i,:] = np.interp(x_values, FPR, TPR)
            TPR_avg = np.mean(TPR_interp,axis=0)
            TPR_std = np.std(TPR_interp, axis=0)
            ROC_curves.append(TPR_avg)
            ROC_curves_std.append(TPR_std)

            # nCG
            nCG_interp = np.zeros((n_splits*n_repeats, len(x_values)))
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
            print 'Average AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)
            #(end for-loop)

        # ROC
        rocfig = plt.figure()
        ax     = rocfig.gca()
        cmap = plt.get_cmap('Set1')
        for i, (dropout, n_repeats) in enumerate(zip(dropout_list, n_repeats_list)):
            ROC     = ROC_curves[i]
            TPR     = ROC
            TPR_err = ROC_curves_std[i]
            FPR     = x_values

            color=cmap(i / float(len(dropout_list)))

            ax.plot((0, 1), (0, 1), 'r--') # red diagonal line
            ax.plot(FPR, TPR, label='dropout='+str(dropout)+', n_repeats='+str(n_repeats), color=color)
            ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC')
            ax.set_xlim(xmin=0.0, xmax=1.0)
            ax.set_ylim(ymin=0.0, ymax=1.0)
            ax.legend(loc=4) #lower right corner
        if save_switch==True:
            name = ion + '_crossval' + '_'+method+'_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
            rocfig.savefig(plot_directory+name+'.png')
        else: rocfig.show()

        # nCG
        ncgfig = plt.figure()
        ax     = ncgfig.gca()
        cmap = plt.get_cmap('Set1')
        for i, (dropout, n_repeats) in enumerate(zip(dropout_list, n_repeats_list)):
            nCG     = nCG_curves_avg[i]
            nCG_err = nCG_curves_std[i]

            color=cmap(i / float(len(dropout_list)))
            ax.plot(x_values, nCG, label='dropout='+str(dropout)+', n_repeats='+str(n_repeats), color=color)
            ax.fill_between(x_values, nCG+0.5*nCG_err, nCG-0.5*nCG_err, alpha=0.3, color=color)
            # ax.set_xlim(xmin=0.0, xmax=1.0)
            # ax.set_ylim(ymin=0.0, ymax=1.0)
            ax.legend(loc=4) #lower right corner
        if save_switch==True:
            name = ion + '_crossval' + '_'+method+'_nCG'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
            ncgfig.savefig(plot_directory+name+'.png')
        else: ncgfig.show()

        # AUC
        aucfig = plt.figure()
        ax     = aucfig.gca()
        cmap = plt.get_cmap('Set1')
        color=cmap(i / float(len(dropout_list)))

        fraction_kept_list = [1-x for x in dropout_list]
        ax.errorbar(fraction_kept_list, AUC_avg, yerr=AUC_std, fmt='o')
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        if save_switch==True:
            name = ion + '_crossval' + '_'+method+'_AUC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
            aucfig.savefig(plot_directory+name+'.png')
        else: aucfig.show()