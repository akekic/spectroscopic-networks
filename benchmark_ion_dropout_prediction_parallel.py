"""
Description: Performs a prediction algorithm on a the ion networkx graphs and validates the prediction via Dropout/bootstrapping.

Experimental data by NIST https://www.nist.gov/
Theoretical data by Jitrik, O., & Bunge, C. F. (2004). Transition probabilities for hydrogen-like atoms. Journal of Physical and Chemical Reference Data, 33(4), 1059-1070. https://doi.org/10.1063/1.1796671

SPM algorithm by Lu, L., Pan, L., Zhou, T., Zhang, Y.-C., Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
HRG C++ algorithm by Clauset, A., Moore, C., Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
SBM algorithm by Newman, M. E. J., Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7). https://doi.org/10.1103/PhysRevLett.117.078301

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
# import matplotlib           # needed if running on server
# matplotlib.use('Agg')       # needed if running on server
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import ShuffleSplit
import random

from joblib import Parallel, delayed
import multiprocessing as mp


# %%
# GLOBAL VARIABLES
########################################

ion = 'C1.0'                          # ion to use as input, see nx2.py
experimental = True                    # use the experimental NIST data or the theoretical JitrikBunge data
if experimental == False:
    # get Z
    for i, char in enumerate(ion.split('.')[0]):
        if char.isdigit():
            index = i
            break
    Z = int(ion.split('.')[0][index:])
    # print 'Model Data'
# else:
    # print 'NIST Data'

only_dipole_global = False              # global switch for taking the full network or only dipole lines
n_limit            = False              # limit is only to be used for one electron ions (hydrogenic ions)
check_obs_wl_global  = False
check_calc_wl_global = False
weighted_global      = True             # if True, load the weighted graphs

if n_limit==False:
    max_n_global = None
else:
    max_n_global   = 8                 # maximal n considered in network

SPM                = True              # run SPM (Structural Perturbation Method) prediction algorithm
HRG                = False              # run HRG (Hierarchical Random Graph) prediction algorithm
nSBM               = True              # run nested SBM (Stochastic Block Model) prediction algorithm
RA                 = True              # run Resource Allocation Index prediction algorithm
AA                 = True              # run Adamic Adar prediction algorithm
PA                 = True              # run Preferential attachment prediction algorithm
JC                 = True              # run Jaccard Coefficient prediction algorithm
SBM                = False              # run SBM (Stochastic Block Model) prediction algorithm
# CN                 = False              # run Common neighbours prediction algorithm

dropout_list       = [0.1, 0.3, 0.5]    # the fractions of edges dropped
n_runs             = 100                # the number of runs for each dropout fraction over which is averaged
# dropout_list       = [0.1]    # the fractions of edges dropped
# n_runs             = 5                # the number of runs for each dropout fraction over which is averaged
# save_switch        = True               # True: save all figures. False: show
save_files         = False              # True: Do not delete edgelist and similar files used for the prediction algorithms (eg. HRG)
HRG_PATH = './HRG/'


num_cores = mp.cpu_count()  # count the number of availabe kernels
n_cores   = 3
RANDX     = 2**32-1             # used for seeding the random processes during multiprocessing

########################################




#----------------------------------------------------------------------
def single_run(G_global, method, dropout):

    # create instance
    lp[method] = nx2.LinkPrediction()

    # get graph
    lp[method].G_original = G_global

    edge_list = np.asarray(lp[method].G_original.edges())
    # print edge_list

    # create dict if necessary
    if method == 'HRG':
        lp[method].create_HRG_ID_dict(experimental=experimental)

    #TODO random seed
    rs = ShuffleSplit(n_splits=1, test_size=dropout, random_state=random.randint(0, RANDX)) #fuer ShuffleSplit ist n_splits so etwas wie unser n_runs
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    for train_index, validation_index in rs.split(edge_list):

        # get sets / do dropout
        lp[method].cross_validation(edge_list, train_index, validation_index)



        # # choose calculated graph as G_probe
        # lp[method].G_probe = nx2.spectroscopic_network(ion, weighted=weighted_global, alt_read_in=False, check_obs_wl=None, check_calc_wl=True)
        # lp[method].G_probe = lp[method].G_probe.subgraph(lp[method].G_training.nodes())

        print 'Number of edges original', nx.number_of_edges(lp[method].G_original)
        print 'Number of nodes original', nx.number_of_nodes(lp[method].G_original)
        print 'Number of edges probe', nx.number_of_edges(lp[method].G_probe)
        print 'Number of nodes probe', nx.number_of_nodes(lp[method].G_probe)
        print 'Number of edges training', nx.number_of_edges(lp[method].G_training)
        print 'Number of nodes training', nx.number_of_nodes(lp[method].G_training)
        #
        # n_train = lp[method].G_training.nodes()
        # n_probe = lp[method].G_probe.nodes()
        # print n_train
        # print n_probe
        # print np.in1d(n_probe, n_train)




        #carry out prediction for specified methods
        if method == 'SPM':
            lp[method].predict_SPM()
        elif method == 'HRG':
            # %%
            HRG_directory = HRG_PATH + ion + '_' + 'dropout' + '/'
            if not os.path.exists(HRG_directory):
                os.makedirs(HRG_directory)

            # %%
            lp[method].write_edgelist(Graph=lp[method].G_training,
                            label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                            directory=HRG_directory) # write training set edgelist
            lp[method].write_edgelist(Graph=lp[method].G_probe,
                            label='probe_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                            directory=HRG_directory) # write probe set edgelist

            # %%
            # fit best dendrogram
            # lp[method].fit_dendrogram(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout), filedirectory=HRG_directory, HRG_PATH='./HRG/')
            # create second .lut file with nx labels
            # nx2.create_nx_string_lut_file('model', label, model_numID_2_strID_dict, label_switch, filedirectory=HRG_directory)

            # %%
            # run prediction
            lp[method].HRG_predict_links(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout), filedirectory=HRG_directory, HRG_PATH='./HRG/')

            # %%
            # load prediction
            print 'Loading predicted links'
            lp[method].load_predicted_links_from_wpairs_file(label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout), filedirectory=HRG_directory)
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

        # get is_correct array
        lp[method].check_if_correct()
        print len(lp[method].is_correct)

        # ROC
        # calculate ROC values
        lp[method].calculate_P_and_N() # calculate number of positive and negative samples
        lp[method].calculate_ROC()

        # AUC
        lp[method].calculate_AUC()

    return (lp[method].ROC, lp[method].AUC)
    # return lp[method].ROC


#----------------------------------------------------------------------
print 'Network: ', ion

# load network
if experimental == False:
    # load theoretical data
    if only_dipole_global == True:
        G_global = nx2.model_network(Z=Z, E1=True, max_n=max_n_global) #only dipole lines up to max_n
    else:
        G_global = nx2.model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=max_n_global) #all lines
else:
    # load NIST data
    if only_dipole_global == True:
        G_global = nx2.spectroscopic_network(ion, weighted=weighted_global, alt_read_in=False, check_obs_wl=check_obs_wl_global, check_calc_wl=check_calc_wl_global)
        G_global = nx2.remove_empty_levels(G_global, 'term') #remove nodes with empty term entry
        G_global = nx2.remove_n_greater(G_global, max_n=max_n_global) # maximal n considered
        G_global = nx2.only_dipole_transitions_parity(G_global) #only dipole lines
        G_global = nx2.only_largest_component(G_global)
    else:
        G_global = nx2.spectroscopic_network(ion, weighted=weighted_global, alt_read_in=False, check_obs_wl=check_obs_wl_global, check_calc_wl=check_calc_wl_global)
        G_global = nx2.remove_empty_levels(G_global, 'term') #remove nodes with empty term entry
        G_global = nx2.remove_n_greater(G_global, max_n=max_n_global) # maximal n considered
        G_global = nx2.only_largest_component(G_global)

# print info to terminal
print experimental*'NIST' + (not experimental)*'Model Data (Jitrik)'
print 'max n: ', max_n_global
print 'only dipole: ', only_dipole_global
print 'n_runs: ', n_runs
print 'check_obs_wl', check_obs_wl_global
print 'check_calc_wl', check_calc_wl_global
print 'weighted', weighted_global

print 'Number of edges: ', nx.number_of_edges(G_global)
print 'Number of nodes: ', nx.number_of_nodes(G_global)
print 'Max missing edges: ', nx.number_of_nodes(G_global) * (nx.number_of_nodes(G_global) - 1) / 2- nx.number_of_edges(G_global)


method_names = SPM*['SPM'] + HRG*['HRG'] + SBM*['SBM'] + nSBM*['nSBM'] + RA*['RA'] + AA*['AA'] + PA*['PA'] + JC*['JC']
lp = {} #emtpy dict
for method in method_names:

    print 'Method: ', method

    plot_directory = '../plots/' + method + '/' + ion + '/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)


    # %%
    # ROC_curves     = [] # to store the result for each dropout fraction
    # ROC_curves_std = []
    # nCG_curves_avg = [] # to store the result for each dropout fraction
    # nCG_curves_std = []
    # AUC_results    = [] # to store the result for each dropout fraction
    # AUC_avg        = []
    # AUC_std        = []


    for dropout in dropout_list:
        x_values        = np.linspace(0,1,num=nx.number_of_nodes(G_global)**2)
        ROC_curves_runs = [] # to store the result of each individual run
        nCG_curves_runs = [] # to store the result of each individual ru
        AUC_runs        = [] # to store the result of each individual run


        # joblib multiprocessing
        if method == 'HRG':
            n_jobs = 1 # gibt noch probleme wegen zwischengespeicherten files.
        else:
            n_jobs = min(mp.cpu_count(),n_cores)
        joblib_results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(single_run)(G_global, method, dropout) for x in xrange(n_runs)) #traceback error

        # extract ROC and AUC from joblib results
        for single_job_result in joblib_results:
            ROC_curves_runs.append(single_job_result[0])
            AUC_runs.append(single_job_result[1])


        # ROC
        # save averaged ROC data to txt file for each dropout
        TPR_interp = np.zeros((n_runs, len(x_values)))
        for i,ROC in enumerate(ROC_curves_runs):
            TPR = ROC[:,0]
            FPR = ROC[:,1]
            TPR_interp[i,:] = np.interp(x_values, FPR, TPR)
        TPR_avg = np.mean(TPR_interp,axis=0)
        TPR_std = np.std(TPR_interp, axis=0)
        FPR = x_values
        ROC_results = np.stack((FPR, TPR_avg, TPR_std), axis=-1) # hstack arrays

        name = ion + '_dropout' + '_'+method+'_ROC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        header_string = ' Ion: ' + ion + '\n ' + 'Data Type: ' + experimental*'NIST' + (not experimental)*'Model Data (Jitrik)' + '\n ' + 'Transition Type: ' + only_dipole_global*'Only E1'+ (not only_dipole_global)*'Full'+ '\n ' + 'n Limit: ' + (not n_limit)*'No Limit' + n_limit*(str(max_n_global)+' (inclusive)') + '\n ' + 'Number of nodes: ' + str(nx.number_of_nodes(G_global)) + '\n ' + 'Number of edges: ' + str(nx.number_of_edges(G_global)) + '\n ' + 'Dropout Method: Random Dropout' + '\n ' + 'Dropout fraction value: ' + str(dropout) + '\n ' + 'Number of runs: ' + str(n_runs) + '\n ' + 'Link Prediction Method: '+ method + '\n ' + 'Columns: FPR, TPR, TPR_err'
        np.savetxt(plot_directory+name+'_dropout_value_'+str(dropout)+'.csv', ROC_results, fmt='%.18e', delimiter=',', header=header_string, comments='#')


        # AUC
        # AUC_results.append(AUC_runs)
        AUC_runs_avg = np.mean(AUC_runs)
        AUC_runs_std = np.std(AUC_runs)
        # AUC_avg.append(AUC_runs_avg)
        # AUC_std.append(AUC_runs_std)
        print 'Average AUC for dropout of ' + str(dropout) + ' : ' + str(AUC_runs_avg) + '+-' + str(AUC_runs_std)

        # save averaged ROC data to txt file for each dropout
        AUC_runs_array = np.asarray(AUC_runs)
        name = ion + '_dropout' + '_'+method+'_AUC'+ n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'
        header_string = ' Ion: ' + ion + '\n ' + 'Data Type: ' + experimental*'NIST' + (not experimental)*'Model Data (Jitrik)' + '\n ' + 'Transition Type: ' + only_dipole_global*'Only E1'+ (not only_dipole_global)*'Full'+ '\n ' + 'n Limit: ' + (not n_limit)*'No Limit' + n_limit*(str(max_n_global)+' (inclusive)') + '\n ' + 'Number of nodes: ' + str(nx.number_of_nodes(G_global)) + '\n ' + 'Number of edges: ' + str(nx.number_of_edges(G_global)) + '\n ' + 'Dropout Method: Random Dropout' + '\n ' + 'Dropout fraction value: ' + str(dropout) + '\n ' + 'Number of runs: ' + str(n_runs) + '\n ' + 'Link Prediction Method: '+ method + '\n ' + 'Columns: AUC_value'
        np.savetxt(plot_directory+name+'_dropout_value_'+str(dropout)+'.csv', AUC_runs_array, fmt='%.18e', delimiter=',', header=header_string, comments='#')
        #(end for-loop) #####################################


    # delete some files created in the prediction process but which are not needed anymore
    if method == 'HRG':
        HRG_directory = HRG_PATH + ion + '_' + 'dropout' + '/'
        if not save_files:
            for dropout in dropout_list:
                os.remove(HRG_directory+'probe_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
                os.remove(HRG_directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
                os.remove(HRG_directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-ranked.wpairs')
                # os.remove(HRG_directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-names.lut')
                # os.remove(HRG_directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '-L.xy')
                # os.remove(HRG_directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.hrg')
                # os.remove(HRG_directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '_best-dendro.info')
