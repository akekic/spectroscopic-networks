"""
@author: Julian Heiss
Description: Loads a networkx network and writes its edgelist to file for further use as input for C++ algorithm by Clauset, A., Moore, C., & Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.
Also performs a dropout of edges and saves the removed edge tuples to file (in format of Clauset et al.) to be used as check.

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

##########################################################################

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

def dropout_nodes(Graph, dropout_fraction):
	k = int(np.rint(dropout_fraction * nx.number_of_nodes(G))) #number of nodes to remove
	rand = np.random.choice(int_IDs, k, replace=False) #get k randomly chosen IDs from the int ID set
	#store the IDs of the nodes which have been removed
	nodes = []
	for i in range(matching_array.shape[0]):
		if np.any(int(matching_array[i,1]) == rand): nodes.append(matching_array[i,0])
	#remove randomly chosen nodes
	for n in nodes:
		Graph.remove_node(n)
	#return smaller network and list of nodes which have been removed
	return Graph, nodes

def dropout_edges(Graph, dropout_fraction):
	#create edgelist, split it and join it without weight column
	temp = np.asarray(nx.to_edgelist(G))
	splits = np.split(temp, 3, axis=1)
	edgelist = np.hstack((splits[0], splits[1]))

	k = int(np.rint(dropout_fraction * len(edgelist[:,0]) )) #number of edges to remove
	rand = np.random.choice(list(range(len(edgelist[:,0]))), k, replace=False) #get k randomly chosen integers (within a range of between zero and the number of existing edges)

	#container of edges to be removed
	edges = []
	for i, e in enumerate(rand):
			edges.append( (edgelist[e,0], edgelist[e,1]) ) 
	#remove edges
	for e in edges:
		Graph.remove_edges_from(edges)

	# save removed edges to file
	print 'Writing', ion + '_removed_edges.pairs'
	with open(ion + '_removed_edges.pairs', 'w') as edge_file:
		writer = csv.writer(edge_file, delimiter='\t')
		for i in range(len(edges)):
			if ion == 'model':
				writer.writerow( [ ID_dict[edges[i][0]] , ID_dict[edges[i][1]] ] ) #save value of dict (numerical ID)
			else:
				writer.writerow( [int(edges[i][0][-4:]) , int(edges[i][1][-4:])] ) #trimming string, only take last four characters and convert to int type
		edge_file.close()
	print '-> ' + ion + '_removed_edges.pairs', 'saved'
	#return Graph and list of edges which have been removed
	return Graph, edges


def write_ID_file(Graph):
	# ID-term dictionary. remove empty nodes beforehand!
	term = nx.get_node_attributes(Graph,'term')
	#create array to store IDs and terms of nodes
	print 'Writing matching.npy'
	matching_array = np.ones((nx.number_of_nodes(Graph), 3), dtype='|S20')
	#fill array with nodeID, shortened nodeID and the term
	#####TODO other ID names with model network
	if ion == 'model':
		for i, (node_ID, term) in enumerate(term.items()):
			matching_array[i,0] = i
			matching_array[i,1] = node_ID
			matching_array[i,2] = term
		### string_IDs = [n for n in Graph.nodes() if deg[n] == 0]
		ID_dict = dict(zip(matching_array[:,1], matching_array[:,0]))
		# print ID_dict
		#save dict to file
		np.save('ID_dict.npy', ID_dict)
		print '-> ' + 'ID_dict.npy saved'
	else:
		for i, (node_ID, term) in enumerate(term.items()):
			matching_array[i,0] = node_ID
			matching_array[i,1] = str(int(node_ID[-4:]))
			matching_array[i,2] = term
	#save array to file
	np.save('matching.npy', matching_array)
	print '-> ' + 'matching.npy saved'

##TODO change to same routine for writing edgelist as in dropout_edges? (using to_edgelist instead of changing file)
#create an edgelist file exactly in the format that the Newman c++ algorithm needs
def write_edgelist(Graph, ion):
	print 'Writing', ion + '.pairs'
	nx.write_edgelist(G, ion + '.pairs', data=False) #see nx documentation

	#####TODO reading and writing could be done in one session with 'w+' option?
	# trimming node IDs and converting from string to integers: 
	# open and read file which has just been created by nx.write_edgelist()
	with open(ion + '.pairs', 'r') as temp_file:
		line_reader = csv.reader(temp_file, delimiter=' ')
		next(temp_file) #skip first line
		line_data_temp = np.array(list(line_reader))
		temp_file.close()
	# write file again
	with open(ion + '.pairs', 'w') as edge_file:
		writer = csv.writer(edge_file, delimiter='\t')
		for i in range(len(line_data_temp)):
			if ion == 'model':
				writer.writerow( [ ID_dict[line_data_temp[i][0]] , ID_dict[line_data_temp[i][1]] ] ) #saving numerical ID for model network
			else:
				writer.writerow( [int(line_data_temp[i][0][-4:]) , int(line_data_temp[i][1][-4:])] ) #trimming string, only take last four characters and convert to int type
		edge_file.close()
	print '-> ' + ion + '.pairs', 'saved'


	# Run Cpp routine by Clauset et al.
def run_cpp_prediction(ion, label)
	############################################

	# # create gml-file of G
	# gml_path = os.getcwd() + '/' + G.name + '.gml' 
	# nx.write_gml(G, gml_path)
	# gml_file = open(gml_path)

	# call ./estimate
	# args = ["./estimate", str(MCSWEEPS)]
	# a = Popen(args, stdin=gml_file, stdout=PIPE)
	######################################
	from subprocess import Popen, PIPE
	cmd = './HRG/hrg_20120527_predictHRG_v1.0.4/predictHRG_GPL/predictHRG -f ' + ion + '.pairs -s ' + ion + '_' + label + '_best-dendro.hrg -t ' + label
	args = shlex.split(cmd)
	subprocess.call(args)

	
##########################################################################
#load network
if ion == 'model':
	print 'Loading model network'
	# G = nx2.model_network()
	#only dipole up to n=8
	G = nx2.model_network(max_n=8)
	edges = G.edges()
	transitionType = nx.get_edge_attributes(G, 'transitionType')
	for e in edges:
		if transitionType[e] != 'E1':
			G.remove_edge(*e)
else:
	print 'Loading spectroscopic network:', ion
	G = nx2.spectroscopic_network(ion, weighted=False, alt_read_in=False)

#get rid of levels without term entry
G, term = remove_empty_levels(G)

# store IDs and terms of nodes in array and save it to .npy file
write_ID_file(G)

if ion == 'model':
	#load dict with IDs and terms of nodes
	print 'Loading ID_dict.npy'
	ID_dict = np.load('ID_dict.npy').item()
	print ID_dict

#perform dropout on network nodes(removing a specified fraction)
# G, removed_nodes = dropout_nodes(G, 0.25)

#perform dropout on the network's edges (removing a specified fraction)
print len(np.asarray(nx.to_edgelist(G)))
G, removed_edges = dropout_edges(G, 0.1)
print len(np.asarray(nx.to_edgelist(G)))
print len(removed_edges)


#save edgelist of G to file (AFTER dropout) in format needed by Clauset et al. C++ algorithm
write_edgelist(G, ion)

