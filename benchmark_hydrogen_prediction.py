"""
Description: Performs the prediction algorithm on a the hydrogen networkx graph and checks the prediction afterwards on the model hydrogen network (Jitrik, bunge). (C++ algorithm by Clauset, A., Moore, C., & Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.)
@author: Julian Heiss

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

exp_ion = 'H1.0'
label = ''
label_switch = False
HRG_PATH = './HRG/'
directory = HRG_PATH + 'H1.0' + '_' + 'theo_validation' + '/'
if not os.path.exists(directory):
	os.makedirs(directory)
plot_directory = '../plots/HRG/' + 'H1.0' + '_' + 'theo_validation' + '/'
if not os.path.exists(plot_directory):
	os.makedirs(plot_directory)

########################################


# auxiliary functions
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# %%
#load networks

# todo: only up to n=8
print 'Loading spectroscopic network:', exp_ion
exp_Graph = nx2.spectroscopic_network(exp_ion, weighted=True, alt_read_in=False)
exp_Graph = nx2.remove_empty_levels(exp_Graph, 'term')
exp_Graph = nx2.remove_n_greater(exp_Graph, max_n=8) #no
exp_Graph = nx2.only_dipole_transitions_parity(exp_Graph) #only dipole lines

exp_term      = nx.get_node_attributes(exp_Graph, 'term')		# ID-term dictionary
exp_n         = nx.get_node_attributes(exp_Graph, 'n')			# ID-n dictionary
exp_l         = nx.get_node_attributes(exp_Graph, 'l')			# ID-l dictionary
exp_modelName = nx.get_node_attributes(exp_Graph, 'modelName')	# ID-modelID dictionary

# relabel node IDs
nx.relabel_nodes(exp_Graph,exp_modelName,copy=False)
print exp_Graph.nodes()


# %%
print 'Loading model network'
model_Graph = nx2.model_network(E1=True, E2=False, M1=False, max_n=8) #only dipole lines

model_term = nx.get_node_attributes(model_Graph, 'term')	# ID-term dictionary
model_n    = nx.get_node_attributes(model_Graph, 'n')		# ID-n dictionary


# %%
# create two sets: training set used for calculating predictions and probe set for evaluating these predictions
probe_set_graph = model_Graph
training_set_graph = nx.Graph()

# delete nodes from model graph that do not appear in exp graph (we wont be able to predict these anyways)
probe_set_graph = probe_set_graph.subgraph(exp_Graph.nodes())


# delete links from probe set graph which occur in the experimental network (training set)
# hits=0
# for e in probe_set_graph.edges():
# 	term1 	= model_term[e[0]]
# 	term2 	= model_term[e[1]]
# 	n1 		= model_n[e[0]]
# 	n2 		= model_n[e[1]]
# 	node1 = [node for node in exp_Graph.nodes() if exp_term[node]==term1 and exp_n[node]==n1]
# 	node2 = [node for node in exp_Graph.nodes() if exp_term[node]==term2 and exp_n[node]==n2]
# 	if node1 and node2:
# 		if exp_Graph.has_edge(node1[0], node2[0]):
# 			# hits+=1
# 			# print hits
# 			## exp_Graph.remove_edge(node1[0], node2[0])
# 			probe_set_graph.remove_edge(*e)
# 			training_set_graph.add_edge(*e)


# for e in probe_set_graph.edges():
# 	if exp_Graph.has_edge(*e):
# 			## exp_Graph.remove_edge(*e) #see following TODO
# 			probe_set_graph.remove_edge(*e)
# 			training_set_graph.add_edge(*e) #create training graph. note that there are only dipole lines in the probe set, so also in the training set.


for e in exp_Graph.edges(data=True):
		## exp_Graph.remove_edge(*e) #see following TODO
		if probe_set_graph.has_edge(*e[:2]): #need if, since some lines are missing in the model graph
			probe_set_graph.remove_edge(*e[:2])
		training_set_graph.add_edge(*e) #create training graph. note that there are only dipole lines in the probe set, so also in the training set.


#TODO: Problem: some links are in the exp network, but not in the model network (delta l = 1, delta n = 0, delta j = 0)
## for e in exp_Graph.edges():
## 	print exp_n[e[0]]+exp_term[e[0]], exp_n[e[1]]+exp_term[e[1]]
## 	print exp_Graph[e[0]][e[1]]['matrixElement']

print 'Model number of edges',        nx.number_of_edges(model_Graph)
print 'Model number of nodes',        nx.number_of_nodes(model_Graph)
print 'Exp number of edges',          nx.number_of_edges(exp_Graph)
print 'Exp number of nodes',          nx.number_of_nodes(exp_Graph)
print 'Training set number of edges', nx.number_of_edges(training_set_graph)
print 'Training set number of nodes', nx.number_of_nodes(training_set_graph)
print 'Probe set number of edges',    nx.number_of_edges(probe_set_graph)
print 'Probe set number of nodes',    nx.number_of_nodes(probe_set_graph)

print exp_Graph.nodes(data=True)
print probe_set_graph.nodes(data=True)
print training_set_graph.nodes(data=True)


#-------------------------------------------------------------------

# %%
# store IDs and terms of nodes in array
# exp_matching_array = nx2.create_ID_term_array(exp_Graph, exp_ion)
model_matching_array = nx2.create_ID_term_array(model_Graph, 'model')

# %%
#create dictionaries from matching array, transalting between IDs and terms
model_strID_2_numID_dict = dict(zip(model_matching_array[:,0], model_matching_array[:,1])) #keys: string ID ('7S'), values: numerical generated ID (saved as string: '33')
#create second dict with reversed order ("decryption")
model_numID_2_strID_dict = {y:x for x,y in model_strID_2_numID_dict.items()} #keys: numerical generated ID (saved as string: '33'), values: string ID ('7S')

# exp_strID_2_numID_dict = dict(zip(exp_matching_array[:,0], exp_matching_array[:,1])) #keys: string ID ('00001.000.0033'), values: numerical generated ID (saved as string: '33')

# term dict NICHT EINEINDEUTIG!
# exp_numID_2_term_dict = dict(zip(exp_matching_array[:,1], exp_matching_array[:,2])) #keys: numerical node ID (last digits, saved as string..), values: string of term (2S1/2)

#TODO dictionary to match one network to the other
# model_strID_2_exp_numID_dict = dict(zip(model_matching_array[:,0], exp_matching_array[:,1])) #keys: string ID ('7S'), values: numerical generated ID (saved as string: '33')


# %%
# #write the edgelist of the experimental graph to file
# nx2.write_edgelist(exp_Graph, exp_ion, label, exp_strID_2_numID_dict)


# %%
#we removed everything from the model graph that weas in the training graph, so "probe_set_graph" is now the probe set.
#write the edgelist of the model graph probe set to file
nx2.write_edgelist(probe_set_graph, 'model_probe', label, model_strID_2_numID_dict, directory, label_switch=label_switch)

#write the edgelist of the model graph training set to file
# nx2.write_edgelist(training_set_graph, 'model_training', label, model_strID_2_numID_dict, HRG_PATH)
nx2.write_edgelist(training_set_graph, 'model', label, model_strID_2_numID_dict, directory, label_switch=label_switch)


# %%
fit best dendrogram
nx2.fit_dendrogram('model', label, label_switch, filedirectory=directory, HRG_PATH=HRG_PATH)
# create second .lut file with nx labels
nx2.create_nx_string_lut_file('model', label, model_numID_2_strID_dict, label_switch, filedirectory=directory)

# %%
#run prediction
nx2.predict_links(ion='model', label=label, label_switch=label_switch, filedirectory=directory, HRG_PATH=HRG_PATH)


# %%
#load prediction
print 'Loading predicted links'
predicted_links, predicted_links_prob = nx2.load_predicted_links_from_wpairs_file('model', label, label_switch=label_switch, filedirectory=directory)
print len(predicted_links)

#load removed links
print 'Loading probe links'
probe_links = nx2.load_links_from_pairs_file('model_probe' + label + '.pairs', directory)
print len(probe_links)


#----------------------------------------------------------------------------------

# %%
#evaluate prediction
print 'Evaluating prediction...'
predictions, correct_mask, info_tuple, roc_array, ranks = nx2.evaluate_prediction(predicted_links, predicted_links_prob, probe_links)


info_tuple.sort(key=lambda k: k[1], reverse=False) #sort in ascending order of rank
print '( #correct predictions, rank of prediction, link probability, level 1, level 2 )'
# for tup in info_tuple: print(tup) #print result of prediction check

# print 'ranks of correct predictions', ranks[correct_mask]
# print np.sum(correct_mask)
# print 'ranks of wrong predictions', ranks[~correct_mask]
# print np.sum(~correct_mask)


# %%
# plot ROC curve
nx2.plot_ROC_curve(roc_array, 'model', label, label_switch=label_switch, plot_directory=plot_directory)

# %%
# calculate AUC value
AUC = nx2.calculate_AUC_value(predictions, ranks, correct_mask, ranks_considered_percentage=0.05)
print 'AUC value: ', AUC
plt.plot((0, 1), (0, 1), 'r--')
# %%
# calculate nDCG values
CG, DCG, ICG, nCG, nDCG = nx2.calculate_gain_measures(correct_mask, base=2)

# plot gain metrics
nx2.plot_gain_metrics(CG, DCG, ICG, nCG, nDCG)
