"""
Description: Performs a prediction algorithm on a the ion networkx graphs and validates the prediction via Dropout/bootstrapping.

Experimental data by NIST https://www.nist.gov/
Theoretical data by Jitrik, O., & Bunge, C. F. (2004). Transition probabilities for hydrogen-like atoms. Journal of Physical and Chemical Reference Data, 33(4), 1059-1070. https://doi.org/10.1063/1.1796671

SPM algorithm by Lu, L., Pan, L., Zhou, T., Zhang, Y.-C., Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
HRG C++ algorithm by Clauset, A., Moore, C., Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
SBM algorithm by Newman, M. E. J., Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7). https://doi.org/10.1103/PhysRevLett.117.078301
nSBM algorithm by Peixoto, T

@author: Julian Heiss
@date: October 2017
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
import matplotlib           # needed if running on server
matplotlib.use('Agg')       # needed if running on server
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import ShuffleSplit


# %%time
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
nSBM               = False              # run nested SBM (Stochastic Block Model) prediction algorithm
RA                 = False              # run Resource Allocation Index prediction algorithm
AA                 = False              # run Adamic Adar prediction algorithm
PA                 = False              # run Preferential attachment prediction algorithm
JC                 = False              # run Jaccard Coefficient prediction algorithm
CN                 = False              # run Common neighbours prediction algorithm

dropout_list       = [0.1, 0.3, 0.5]    # the fractions of edges dropped
n_runs             = 100                  # the number of runs for each dropout fraction over which is averaged
save_switch        = True               # True: save all figures. False: show
save_files         = False              # True: Do not delete edgelist and similar files used for the prediction algorithms (eg. HRG)

########################################


#----------------------------------------------------------------------
#----------------------------------------------------------------------
print 'Network: ', ion


method_names = SPM*['SPM'] + HRG*['HRG'] + SBM*['SBM'] + nSBM*['nSBM'] + RA*['RA'] + AA*['AA'] + PA*['PA'] + JC*['JC'] + CN*['CN']
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

    # print  lp[method].G_original.node['001001.001.000013']['dummyCommunity']

    edge_list = np.asarray(lp[method].G_original.edges())
    # print edge_list

    if method == 'HRG':
        # create dict
        lp[method].create_HRG_ID_dict(experimental=experimental)

    # %%
    ROC_curves     = [] # to store the result for each dropout fraction
    ROC_curves_std = []
    nCG_curves_avg = [] # to store the result for each dropout fraction
    nCG_curves_std = []
    AUC_results    = [] # to store the result for each dropout fraction
    AUC_avg        = []
    AUC_std        = []


    for dropout in dropout_list:
        x_values        = np.linspace(0,1,num=nx.number_of_nodes(lp[method].G_original)**2)
        ROC_curves_runs = [] # to store the result of each individual run
        nCG_curves_runs = [] # to store the result of each individual run
        AUC_runs        = [] # to store the result of each individual run

        rs = ShuffleSplit(n_splits=n_runs, test_size=dropout, random_state=0) #fuer ShuffleSplit ist n_splits so etwas wie unser n_runs
        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
        for train_index, validation_index in rs.split(edge_list):

            # get sets / do dropout
            lp[method].cross_validation(edge_list, train_index, validation_index)
            print lp[method].probe_list

            # print 'Number of edges probe', nx.number_of_edges(lp[method].G_probe)
            # print 'Number of nodes probe', nx.number_of_nodes(lp[method].G_probe)
            # print 'Number of edges training', nx.number_of_edges(lp[method].G_training)
            # print 'Number of edges training', nx.number_of_nodes(lp[method].G_training)


            #carry out prediction for specified methods
            if method == 'SPM':
                lp[method].predict_SPM()
            elif method == 'HRG':
                # %%
                HRG_PATH = './HRG/'
                directory = HRG_PATH + ion + '_' + 'dropout' + '/'
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # %%
                lp[method].write_edgelist(Graph=lp[method].G_training,
                                label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                                directory=directory) # write training set edgelist
                lp[method].write_edgelist(Graph=lp[method].G_probe,
                                label='probe_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                                directory=directory) # write probe set edgelist

                # %%
                # fit best dendrogram
                lp[method].fit_dendrogram(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout), filedirectory=directory, HRG_PATH='./HRG/')
                # create second .lut file with nx labels
                # nx2.create_nx_string_lut_file('model', label, model_numID_2_strID_dict, label_switch, filedirectory=directory)

                # %%
                # run prediction
                lp[method].HRG_predict_links(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout), filedirectory=directory, HRG_PATH='./HRG/')

                # %%
                # load prediction
                print 'Loading predicted links'
                lp[method].load_predicted_links_from_wpairs_file(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout), filedirectory=directory)
            elif method == 'SBM':
                lp[method].predict_SBM()
            elif method == 'nSBM':
                lp[method].predict_nested_SBM()
            elif method  == 'RA':
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
            # elif method == 'CN':
            #     preds = nx.cn_soundarajan_hopcroft(lp[method].G_training, community='dummyCommunity')
            #     prediction_list_dummy = []
            #     for u, v, p in preds:
            #         prediction_list_dummy.append( ((u, v), p) )
            #     #sort prediction list
            #     lp[method].prediction_list = sorted(prediction_list_dummy, key=lambda x: x[1], reverse=True)


            # get is_correct array
            lp[method].check_if_correct()

            # ROC
            # calculate ROC values
            lp[method].calculate_P_and_N() # calculate number of positive and negative samples
            ROC_curves_runs.append(lp[method].calculate_ROC())
            # ROC_curves_runs.append(lp[method].calculate_ROC_2())

            ####### # save averaged ROC data to txt file after each run
            ####### TPR_array = np.zeros((len(ROC_curves_runs), len(ROC_curves_runs[0])))
            ####### for i,ROC in enumerate(ROC_curves_runs):
            #######     print ROC_curves_runs[i].shape
            #######     # TPR = ROC[:,0]
            #######     TPR_array[i,:] = ROC[:,0]
            #######     FPR = ROC[:,1]
            #######     # TPR_interp[i,:] = np.interp(x_values, FPR, TPR)
            ####### TPR_avg = np.mean(TPR_array,axis=0)
            ####### TPR_std = np.std(TPR_array,axis=0)
            ####### # FPR = x_values

            # save averaged ROC data to txt file after each run
            TPR_interp = np.zeros((len(ROC_curves_runs), len(x_values)))
            for i,ROC in enumerate(ROC_curves_runs):
                TPR = ROC[:,0]
                FPR = ROC[:,1]
                TPR_interp[i,:] = np.interp(x_values, FPR, TPR)
            TPR_avg = np.mean(TPR_interp,axis=0)
            TPR_std = np.std(TPR_interp, axis=0)
            FPR = x_values
            name = ion + '_dropout' + '_'+method+'_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
            ROC_results = np.stack((FPR, TPR_avg, TPR_std), axis=-1) # hstack arrays
            header_string = ' Ion: ' + ion + '\n ' + 'Number of nodes: ' + str(nx.number_of_nodes(lp[method].G_original)) + '\n ' + 'Number of edges: ' + str(nx.number_of_edges(lp[method].G_original)) + '\n ' + 'Dropout Method: Random Dropout' + '\n ' + 'Dropout fraction value: ' + str(dropout) + '\n ' + 'Link Prediction Method: '+ method + '\n ' + 'Transition Type: ' + only_dipole_global*'Only E1'+ (not only_dipole_global)*'Full'+ '\n ' + 'n Limit: ' + (not n_limit)*'No Limit' + n_limit*(str(max_n_global)+' (inclusive)') + '\n ' + 'Data Type: ' + experimental*'NIST' + (not experimental)*'Model Data (Jitrik)' + '\n ' + 'Columns: FPR, TPR, TPR_err'
            np.savetxt(plot_directory+name+'_dropout_value_'+str(dropout)+'.csv', ROC_results, fmt='%.18e', delimiter=',', header=header_string, comments='#')
            # TODO: bisher wird nach vollstaendiger Ausfuehrung des Programms weiter unten nochmal die txt ueberspeichert. Evtl unten loeschen

            # nCG
            # lp[method].calculate_gain_measures(2)
            # nCG_curves_runs.append(lp[method].nCG)

            # AUC
            AUC_runs.append(lp[method].calculate_AUC())
            print len(AUC_runs)
            # save progress to file
            AUC_runs_array = np.asarray(AUC_runs)
            name = ion + '_dropout' + '_'+method+'_AUC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
            header_string = ' Ion: ' + ion + '\n ' + 'Number of nodes: ' + str(nx.number_of_nodes(lp[method].G_original)) + '\n ' + 'Number of edges: ' + str(nx.number_of_edges(lp[method].G_original)) + '\n ' + 'Dropout Method: Random Dropout' + '\n ' + 'Dropout fraction value: ' + str(dropout) + '\n ' + 'Link Prediction Method: '+ method + '\n ' + 'Transition Type: ' + only_dipole_global*'Only E1'+ (not only_dipole_global)*'Full'+ '\n ' + 'n Limit: ' + (not n_limit)*'No Limit' + n_limit*(str(max_n_global)+' (inclusive)') + '\n ' + 'Data Type: ' + experimental*'NIST' + (not experimental)*'Model Data (Jitrik)' + '\n ' + 'Columns: AUC_value'
            np.savetxt(plot_directory+name+'_dropout_value_'+str(dropout)+'.csv', AUC_runs_array, fmt='%.18e', delimiter=',', header=header_string, comments='#')

            #(end for-loop) ##################################

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
        # nCG_interp = np.zeros((n_runs, len(x_values)))
        # for i,nCG in enumerate(nCG_curves_runs):
            ##TODO: hier gibt es manchmal rundungsfehler bei der stepsize, deswegen ist der rundungsfehler doppelt eingebaut, damit er sich aufhebt.
            # nCG_interp[i,:] = np.interp(x_values, np.arange(0,1.0/float(len(nCG))*float(len(nCG)),1.0/float(len(nCG))), nCG)
        # nCG_avg = np.mean(nCG_interp,axis=0)
        # nCG_std = np.std(nCG_interp, axis=0)
        # nCG_curves_avg.append(nCG_avg)
        # nCG_curves_std.append(nCG_std)

        # AUC
        AUC_results.append(AUC_runs)
        AUC_runs_avg = np.mean(AUC_runs)
        AUC_runs_std = np.std(AUC_runs)
        AUC_avg.append(AUC_runs_avg)
        AUC_std.append(AUC_runs_std)
        print 'Average AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)
        #(end for-loop) #####################################

    # # ROC #############################################
    # rocfig = plt.figure()
    # ax     = rocfig.gca()
    # cmap = plt.get_cmap('Set1')
    # for i,dropout in enumerate(dropout_list):
    #     ROC     = ROC_curves[i]
    #     TPR     = ROC
    #     TPR_err = ROC_curves_std[i]
    #     FPR     = x_values

    #     name = ion + '_dropout' + '_'+method+'_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'

    #     ## write ROC data to txt file
    #     # ROC_results = np.stack((FPR, TPR, TPR_err), axis=-1) #hstack arrays
    #     # header_string = ' Ion: ' + ion + '\n ' + 'Dropout Method: Random Dropout' + '\n ' + 'Dropout fraction value: ' + str(dropout) + '\n ' + 'Link Prediction Method: '+ method + '\n ' + 'Transition Type: ' + only_dipole_global*'Only E1'+ (not only_dipole_global)*'Full'+ '\n ' + 'n Limit: ' + (not n_limit)*'No Limit' + n_limit*(str(max_n_global)+' (inclusive)') + '\n ' + 'Data Type: ' + experimental*'NIST' + (not experimental)*'Model Data (Jitrik)' + '\n ' + 'Columns: FPR, TPR, TPR_err'
    #     # np.savetxt(plot_directory+name+'_dropout_value_'+str(dropout)+'.csv', ROC_results, fmt='%.18e', delimiter=',', header=header_string, comments='#')
    #     ## try: np.savetxt('../plots/ROC_data/'+name+'_dropout_value_'+str(dropout)+'.csv', ROC_results, fmt='%.18e', delimiter=',', header=header_string, comments='#')

    #     color=cmap(i / float(len(dropout_list)))

    #     ax.plot((0, 1), (0, 1), 'r--') # red diagonal line
    #     ax.plot(FPR, TPR, label='dropout='+str(dropout)+', n_runs='+str(n_runs), color=color)
    #     ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color)
    #     ax.set_xlabel('False Positive Rate')
    #     ax.set_ylabel('True Positive Rate')
    #     ax.set_title('ROC')
    #     ax.set_xlim(xmin=0.0, xmax=1.0)
    #     ax.set_ylim(ymin=0.0, ymax=1.0)
    #     ax.legend(loc=4) #lower right corner
    # if save_switch==True:
    #     rocfig.savefig(plot_directory+name+'.png')
    # else: rocfig.show()
    # ####################################################

    # # nCG ##############################################
    # ncgfig = plt.figure()
    # ax     = ncgfig.gca()
    # cmap = plt.get_cmap('Set1')
    # for i,dropout in enumerate(dropout_list):
    #     nCG     = nCG_curves_avg[i]
    #     nCG_err = nCG_curves_std[i]

    #     color=cmap(i / float(len(dropout_list)))

    #     ax.plot(x_values, nCG, label='dropout='+str(dropout)+', n_runs='+str(n_runs), color=color)
    #     ax.fill_between(x_values, nCG+0.5*nCG_err, nCG-0.5*nCG_err, alpha=0.3, color=color)
    #     # ax.set_xlim(xmin=0.0, xmax=1.0)
    #     # ax.set_ylim(ymin=0.0, ymax=1.0)
    #     ax.legend(loc=4) #lower right corner
    # if save_switch==True:
    #     name = ion + '_dropout' + '_'+method+'_nCG'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
    #     ncgfig.savefig(plot_directory+name+'.png')
    # else: ncgfig.show()
    # ####################################################

    # # AUC ##############################################
    # aucfig = plt.figure()
    # ax     = aucfig.gca()
    # cmap = plt.get_cmap('Set1')
    # color=cmap(i / float(len(dropout_list)))

    # fraction_kept_list = [1-x for x in dropout_list]
    # ax.errorbar(fraction_kept_list, AUC_avg, yerr=AUC_std, fmt='o')
    # ax.set_xlim(xmin=0.0, xmax=1.0)
    # ax.set_ylim(ymin=0.0, ymax=1.0)
    # if save_switch==True:
    #     name = ion + '_dropout' + '_'+method+'_AUC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
    #     aucfig.savefig(plot_directory+name+'.png')
    # else: aucfig.show()
    # ####################################################


    # delete some files created in the prediction process but which are not needed anymore
    if method == 'HRG':
        if not save_files:
            os.remove(directory+'probe_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
            os.remove(directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
            os.remove(directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-names.lut')
            os.remove(directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-L.xy')
            os.remove(directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-ranked.wpairs')
            os.remove(directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.hrg')
            os.remove(directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.info')
