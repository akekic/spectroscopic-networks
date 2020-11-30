"""
Description: Performs a prediction algorithm on a the ion networkx graphs.

Experimental data by NIST https://www.nist.gov/

SPM algorithm by Lu, L., Pan, L., Zhou, T., Zhang, Y.-C., Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
HRG C++ algorithm by Clauset, A., Moore, C., Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
SBM algorithm by Newman, M. E. J., Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7). https://doi.org/10.1103/PhysRevLett.117.078301

@author: Julian Heiss
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


ion = 'He1.0'
print 'Network: ', ion

# load network
G_global = nx2.load_network(ion=ion, print_info=True)


## Link Prediction
method_names = ['SPM', 'nSBM']
data = {}
lp = {} #emtpy dict

for method in method_names:
    print 'Method: ', method

    # create instance
    lp[method] = nx2.LinkPrediction()

    # get graph
    lp[method].G_original = G_global
    lp[method].G_training = G_global #needed for executing nsbm


    edge_list = np.asarray(lp[method].G_original.edges())
    # print edge_list

    if method == 'SPM':
        ## calculate rank of non-observed links by SPM
        lp[method].predict_SPM()   ## edgelist by nodeID
    elif method == 'nSBM':
        lp[method].predict_nested_SBM()


    if ion == 'He1.0':
        cols = ['Score', 'Energy 1', 'Energy 2', 'Term 1', 'Term 2', 'Conf 1', 'Conf 2']
        data[method] = {'Energy 1': [G_global.node(data=True)[n1]['energy'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Energy 2': [G_global.node(data=True)[n2]['energy'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Term 1': [G_global.node(data=True)[n1]['term'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Term 2': [G_global.node(data=True)[n2]['term'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Conf 1': [G_global.node(data=True)[n1]['conf'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Conf 2': [G_global.node(data=True)[n2]['conf'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Score': [score for (n1, n2), score in lp[method].prediction_list[:100]]}
    else:
        cols = ['Score', 'Energy 1', 'Energy 2', 'J 1', 'J 2']
        data[method] = {'Energy 1': [G_global.node(data=True)[n1]['energy'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Energy 2': [G_global.node(data=True)[n2]['energy'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'J 1': [G_global.node(data=True)[n1]['J'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'J 2': [G_global.node(data=True)[n2]['J'] for (n1, n2), score in lp[method].prediction_list[:100]],
                        'Score': [score for (n1, n2), score in lp[method].prediction_list[:100]]}

pred_spm = pd.DataFrame(data['SPM'], columns=cols).set_index('Score')
pred_nsbm = pd.DataFrame(data['nSBM'], columns=cols).set_index('Score')
with open(ion[:-2] + '-spm-link-prediction.tex', 'w+') as f:
    pred_spm.to_latex(f)
with open(ion[:-2] + '-nsbm-link-prediction.tex', 'w+') as f:
    pred_nsbm.to_latex(f)


print pred_spm.head()
print pred_nsbm.head()
