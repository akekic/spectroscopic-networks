"""
Description: Performs a prediction algorithm on a the ion networkx graphs.

Experimental data by NIST https://www.nist.gov/

SPM algorithm by Lu, L., Pan, L., Zhou, T., Zhang, Y.-C., Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
HRG C++ algorithm by Clauset, A., Moore, C., Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
SBM algorithm by Newman, M. E. J., Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7). https://doi.org/10.1103/PhysRevLett.117.078301

@author: Julian Heiss, David Wellnitz
@date: October 2018
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
import pandas as pd
import csv
from sklearn.model_selection import ShuffleSplit


ion = 'Th2.0'

# load network
G_global = nx2.load_network(ion=ion, print_info=True)


## Link Prediction
method = 'nSBM' # For different method, also adjust code below
data = {}
lp = {} #emtpy dict


# create instance
lp = nx2.LinkPrediction()

# get graph
lp.G_original = G_global

edge_list = np.asarray(lp.G_original.edges())


dropout = 0.1    # the fractions of edges dropped
n_runs = 100     # the number of runs for each dropout fraction over which is averaged


x_values        = np.linspace(0,1,num=nx.number_of_nodes(lp.G_original)**2)
ROC_curves_runs = [] # to store the result of each individual run
AUC_runs        = [] # to store the result of each individual run

rs = ShuffleSplit(n_splits=n_runs, test_size=dropout, random_state=0)
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
for train_index, validation_index in rs.split(edge_list):

    # get sets / do dropout
    lp.cross_validation(edge_list, train_index, validation_index)

    lp.predict_nested_SBM() # Change according to method used

    # if method == 'AA':
    #     preds = nx.adamic_adar_index(lp.G_training)
    #     prediction_list_dummy = []
    #     for u, v, p in preds:
    #         prediction_list_dummy.append( ((u, v), p) )
    #     #sort prediction list
    #     lp.prediction_list = sorted(prediction_list_dummy, key=lambda x: x[1], reverse=True)
    # if method == 'PA':
    #     preds = nx.preferential_attachment(lp.G_training)
    #     prediction_list_dummy = []
    #     for u, v, p in preds:
    #         prediction_list_dummy.append( ((u, v), p) )
    #     #sort prediction list
    #     lp.prediction_list = sorted(prediction_list_dummy, key=lambda x: x[1], reverse=True)


    # get is_correct array
    lp.check_if_correct()

    # ROC
    # calculate ROC values
    lp.calculate_P_and_N() # calculate number of positive and negative samples
    ROC_curves_runs.append(lp.calculate_ROC())
    AUC_runs.append(lp.calculate_AUC())

# save averaged ROC data to txt file after each run
TPR_interp = np.zeros((len(ROC_curves_runs), len(x_values)))
for i, ROC in enumerate(ROC_curves_runs):
    TPR = ROC[:,0]
    FPR = ROC[:,1]
    TPR_interp[i,:] = np.interp(x_values, FPR, TPR)
TPR_avg = np.mean(TPR_interp,axis=0)
TPR_std = np.std(TPR_interp, axis=0)

FPR = x_values # Is it correct to include self-links and true links?
