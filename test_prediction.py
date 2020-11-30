"""
@author: Julian Heiss
Description: Loads the predicted links by Newman HRG

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
import sys
import csv


ion = sys.argv[1]
label = sys.argv[2]
# File with prediction has to be in the same folder as python script
#####################################################

#remove nodes that dont have a term entry from graph
def remove_empty_levels(Graph):
	# remove nodes with empty term entry
	temp_term=nx.get_node_attributes(Graph,'term')
	for i, (node_ID, term) in enumerate(temp_term.items()):
		if not term: #if empty string, delete entries
			Graph.remove_node(node_ID)

	#ID-term dictionary without empty term entries
	term = nx.get_node_attributes(G,'term')
	return Graph, term

def load_predicted_links():
	# load predicted links (has to be in the same folder)
	with open(ion + '_' + label + '-ranked.wpairs', 'r') as predict_file:
		line_reader = csv.reader(predict_file, delimiter='\t')
		line_data_temp = np.array(list(line_reader))
	# print line_data_temp

	pred_links = np.empty((len(line_data_temp), 2), dtype=int)
	pred_links_prob = np.empty((len(line_data_temp)), dtype=float)
	for i in range(len(line_data_temp)):	
		for k in range(2): pred_links[i,k] = int( line_data_temp[i][k] )
		pred_links_prob[i] = float( line_data_temp[i][2] )
	return pred_links, pred_links_prob

def load_removed_links():
	# load removed links (has to be in the same folder)
	with open(ion + '_removed_edges.pairs', 'r') as removed_edges_file:
		line_reader = csv.reader(removed_edges_file, delimiter='\t')
		line_data_temp = np.array(list(line_reader))
	# print line_data_temp

	removed_links = np.empty((len(line_data_temp), 2), dtype=int)
	for i in range(len(line_data_temp)):	
		for k in range(2): removed_links[i,k] = int( line_data_temp[i][k] )
	return removed_links

def check_prediction(pred_links, removed_links):
	links_found = []
	right_predictions = 0
	for i in range(removed_links.shape[0]):
		for j in range(pred_links.shape[0]):
			if ( set(removed_links[i,:]) == set(pred_links[j,:]) ):
				links_found.append( (j, pred_links_prob[j], ID_dict2[str(removed_links[i,0])], ID_dict2[str(removed_links[i,1])]) )
				# links_found.append( (i, j, pred_links_prob[j], removed_links[i,0], removed_links[i,1]) )
	# print links_found
	return links_found

##################################################
# # load network
# if ion == 'model':
# 	print 'Loading model network'
# 	G = nx2.model_network()
# else:
# 	print 'Loading spectroscopic network:', ion
# 	G = nx2.spectroscopic_network(ion, weighted=False, alt_read_in=True)

# #get rid of levels without term entry
# G, term = remove_empty_levels(G)

#load array with IDs and terms of nodes
print 'Loading matching.npy'
matching_array = np.load('matching.npy')
# print 'matching.npy loaded'

#load dict with IDs and terms of nodes
print 'Loading ID_dict.npy'
ID_dict = np.load('ID_dict.npy').item()
# print 'matching.npy loaded'
print ID_dict
ID_dict2 = {y:x for x,y in ID_dict.items()}
print ID_dict2

# cast trimmed string IDs as integersAFTER
# int_IDs = [int(n) for n in matching_array[:,1]]

#load prediction
print 'Loading predicted links'
pred_links, pred_links_prob = load_predicted_links()
# print pred_links
# print pred_links_prob

#load removed links
print 'Loading removed links'
removed_links = load_removed_links()
# print removed_links

print 'Checking prediction...'
links_found = check_prediction(pred_links, removed_links)
links_found.sort(key=lambda k: k[0], reverse=False) #sort in ascending order of rank
for tup in links_found: print(tup) #print result of prediction check






