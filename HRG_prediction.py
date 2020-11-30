"""
Description: Performs Dropout and the prediction algorithm on a networkx graph and checks the prediction afterwards. (C++ algorithm by Clauset, A., Moore, C., & Newman, M. (2008). Hierarchical Structure and the Prediction of Missing Links in Networks.)
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
import sys
import csv
import subprocess
import shlex
import matplotlib.pyplot as plt
import simplejson


ion = sys.argv[1]
label = sys.argv[2]
# dropout_fraction = float(sys.argv[3])
# File with prediction has to be in the same folder as python script

######################################
# Mengen-Legende:
# U: Menge aller mglichen Links
# E: Set of links in graph
# E_T: trainign set: known info
# E_p: probe set, not used for prediction, only validation of prediction. -> removed_links
# U-E_t: predicted links pred_links
# U-E: non-existent links = U - E_t - E_p = pred_links - removed_links
######################################


#-------------------------------------------------------------------------------
def load_network(ion, only_dipole=True, alt_read_in=False):
	## @brief      Loads a network.
	##
	## @param      ion          The ion
	## @param      only_dipole  Use only dipole lines
	## @param      alt_read_in  Use the alternate read in option (see nx2.py)
	##
	## @return     Returns the networkx graph of the chosen network.
	##
	#load network
	if ion == 'model':
		print 'Loading model network'
		if only_dipole==True:
			#only dipole up to n=8
			Graph = nx2.model_network(max_n=8)
			edges = Graph.edges()
			transitionType = nx.get_edge_attributes(Graph, 'transitionType')
			for e in edges:
				if transitionType[e] != 'E1':
					Graph.remove_edge(*e)
		else: 
			Graph = nx2.model_network()
	else:
		print 'Loading spectroscopic network:', ion
		Graph = nx2.spectroscopic_network(ion, weighted=False, alt_read_in=alt_read_in)
	return Graph

#remove nodes that dont have a term entry from graph
def remove_empty_levels(Graph):
	# remove nodes with empty term entry
	temp_term=nx.get_node_attributes(Graph,'term')
	for i, (node_ID, term) in enumerate(temp_term.items()):
		if not term: #if empty string, delete entries
			Graph.remove_node(node_ID)

	#ID-term dictionary without empty term entries
	term = nx.get_node_attributes(Graph,'term')
	return Graph, term

def dropout_nodes(Graph, dropout_fraction):
	##
	## @brief      Removes a specified fraction of nodes from the networkx graph
	##
	## @param      Graph             The graph
	## @param      dropout_fraction  The dropout fraction
	##
	## @return     Returns the networkx graph with a random fraction of nodes removed and a list of the removed nodes (their IDs)
	##
	k = int(np.rint(dropout_fraction * nx.number_of_nodes(Graph))) #number of nodes to remove
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

#TODO: need to recover nodes with degree zero. those are not parsed by the cpp routine
def dropout_edges(Graph, dropout_fraction, strID_2_numID_dict):
	##
	## @brief      Randomly removes a specified fraction of edges. Also saves
	##             the removed edges to file.
	##
	## @param      Graph             The graph
	## @param      dropout_fraction  The dropout fraction
	##
	## @return     Returns the nx graph with a fraction of edges missing and the
	##             list of the removed edges.
	##
	##             #create edgelist, split it and join it without weight column
	##
	original_nodes = nx.nodes(Graph) #list of nodes before removal

	temp = np.asarray(nx.to_edgelist(Graph))
	splits = np.split(temp, 3, axis=1)
	edgelist = np.hstack((splits[0], splits[1]))

	k = int(np.rint(dropout_fraction * len(edgelist[:,0]) )) #number of edges to remove
	rand = np.random.choice(list(range(len(edgelist[:,0]))), k, replace=False) #get k randomly chosen integers (within a range of between zero and the number of existing edges)

	#container of edges to be removed
	edges = []
	for e in rand:
			edges.append( (edgelist[e,0], edgelist[e,1]) ) 
	#remove edges
	for e in edges:
		Graph.remove_edges_from(edges)
	# save removed edges to file
	print 'Writing', ion + '_' + label + '_removed_edges.pairs'
	with open(ion + '_' + label + '_removed_edges.pairs', 'w') as edge_file:
		writer = csv.writer(edge_file, delimiter='\t')
		for i in range(len(edges)):
			writer.writerow( [ strID_2_numID_dict[edges[i][0]] , strID_2_numID_dict[edges[i][1]] ] ) #save value of dict (numerical ID)
		edge_file.close()
	print '-> ' + ion + '_' + label + '_removed_edges.pairs', 'saved'

	#add nodes from original nodes list, in case nodes have been removed while removing edges
	Graph.add_nodes_from(original_nodes)
	print 'number of nodes', nx.number_of_nodes(Graph)

	expected_no_of_pred = ( nx.number_of_nodes(Graph) * (nx.number_of_nodes(Graph) - 1) / float(2)) - nx.number_of_edges(Graph)
	print 'expected number of predictions: ', expected_no_of_pred

	#return Graph and list of edges which have been removed
	return Graph, edges, original_nodes

def write_edgelist(Graph, ion, strID_2_numID_dict, original_nodes):
	##
	## @brief      Create an edgelist file exactly in the format that the Clauset/Newman c++ algorithm needs.
	##
	## @param      Graph  The graph
	## @param      ion    The ion
	##
	## @return     { description_of_the_return_value }
	##
	temp = np.asarray(nx.to_edgelist(Graph))
	splits = np.split(temp, 3, axis=1)
	edgelist = np.hstack((splits[0], splits[1]))

	print 'Writing', ion + '_' + label + '.pairs'
	with open(ion + '_' + label +  '.pairs', 'w') as edge_file:
		writer = csv.writer(edge_file, delimiter='\t')
		for i in range(edgelist.shape[0]):
			writer.writerow( [ strID_2_numID_dict[edgelist[i][0]] , strID_2_numID_dict[edgelist[i][1]] ] ) #save value of dict (numerical ID)
		for n in original_nodes:
			writer.writerow( [ strID_2_numID_dict[n] , strID_2_numID_dict[n] ] ) #save value of dict (numerical ID)
		edge_file.close()
	print '-> ' + ion + '_' + label +  '.pairs', 'saved'

#used to write an edgelist without dropout
def write_full_edgelist():
	#load network
	G = load_network(ion, only_dipole=True, alt_read_in=False)

	#get rid of levels without term entry
	G, term = remove_empty_levels(G)

	# store IDs and terms of nodes in array and save it to .npy file
	matching_array = create_ID_term_array(G)
	# print matching_array

	#create dictionaries from matching array, transalting between IDs and terms
	if ion == 'model':
		strID_2_numID_dict = dict(zip(matching_array[:,0], matching_array[:,1])) #keys: string ID ('7S'), values: numerical generated ID (saved as string: '33')

		#create second dict with reversed order ("decryption")
		numID_2_strID_dict = {y:x for x,y in strID_2_numID_dict.items()} #keys: numerical generated ID (saved as string: '33'), values: string ID ('7S')
	else:
		strID_2_numID_dict = dict(zip(matching_array[:,0], matching_array[:,1])) #keys: string ID ('00001.000.0033'), values: numerical generated ID (saved as string: '33')

		# term dict NICHT EINEINDEUTIG!
		numID_2_term_dict = dict(zip(matching_array[:,1], matching_array[:,2])) #keys: numerical node ID (last digits, saved as string..), values: string of term (2S*1/2)

	#save edgelist of G to file (WITHOUT dropout) in format needed by Clauset et al. C++ algorithm
	write_edgelist(G, ion, strID_2_numID_dict, original_nodes)

##TODO change to same routine for writing edgelist as in dropout_edges? (using to_edgelist instead of changing file)
def write_edgelist_deprecated(Graph, ion, strID_2_numID_dict):
	##
	## @brief      Create an edgelist file exactly in the format that the Clauset/Newman c++ algorithm needs.
	##
	## @param      Graph  The graph
	## @param      ion    The ion
	##
	## @return     { description_of_the_return_value }
	##
	print 'Writing', ion + '.pairs'
	nx.write_edgelist(Graph, ion + '.pairs', data=False) #see nx documentation

				#####TODO reading and writing could be done in one session with 'w+' option?
	# trimming node IDs and converting from string to integers: 
	# open and read file which has just been created by nx.write_edgelist()
	with open(ion + '.pairs', 'r') as temp_file:
		line_reader = csv.reader(temp_file, delimiter=' ')
		next(temp_file) #skip first line #wrong
		line_data_temp = np.array(list(line_reader))
		temp_file.close()
	# write file again
	with open(ion + '.pairs', 'w') as edge_file:
		writer = csv.writer(edge_file, delimiter='\t')
		for i in range(len(line_data_temp)):
			writer.writerow( [ strID_2_numID_dict[line_data_temp[i][0]] , strID_2_numID_dict[line_data_temp[i][1]] ] ) #saving numerical ID
		edge_file.close()
	print '-> ' + ion + '.pairs', 'saved'

def create_ID_term_array(Graph):
	##
	##
	## @brief      Creates an npy array with node IDs and their terms to file. In case of
	##             the model network, it generates numerical IDs from the order of the
	##             nodes as an alternative to the string IDs (like '7S'). For the
	##             experimental network it uses the last non-zero digits of the string
	##             ID ('000.001.000034' -> 34). It also creates a dictionary containing
	##             this info in the model case.
	##
	## @param      Graph  The graph
	##
	## @return     Numpy array with 3 columns and as many row as there are edges in the graph.
	##

	# strID_2_term dictionary. remove empty nodes beforehand!
	term = nx.get_node_attributes(Graph,'term')
	# print term
	
	#create array to store IDs and terms of nodes
	print 'Writing matching.npy'
	matching_array = np.ones((nx.number_of_nodes(Graph), 3), dtype='|S20')

	#fill array with nodeID, shortened nodeID and the term
	if ion == 'model':
		for i, (node_ID, term) in enumerate(term.items()):
			matching_array[i,0] = node_ID   #strID ('4D')
			matching_array[i,1] = i         #numID ('26')
			matching_array[i,2] = term		#term ('2G7/2')
		### string_IDs = [n for n in Graph.nodes() if deg[n] == 0]
		# #save dict to file
		# np.save('ID_dict.npy', ID_dict)
		# print '-> ' + 'ID_dict.npy saved'
	else:
		for i, (node_ID, term) in enumerate(term.items()):
			matching_array[i,0] = node_ID				 	#strID ('0000.001.0033')
			matching_array[i,1] = str(int(node_ID[-4:])) 	#numID ('33')
			matching_array[i,2] = term 						#term ('2S*3/2')

	# #save array to file
	# np.save('matching.npy', matching_array)
	# print '-> ' + 'matching.npy saved'
	# cast trimmed string IDs as integers
	# int_IDs = [int(n) for n in matching_array[:,1]]
	return matching_array

#TODO if clause for usage of best dendrogram
def run_cpp_prediction(ion, label):
	##
	## @brief       Run Cpp routine by Clauset et al. which has to be located in the folder specified in the command.
	##
	## @param      ion    The ion
	## @param      label  The run label
	##
	## @return     { description_of_the_return_value }
	##
	# cmd = './HRG/hrg_20120527_predictHRG_v1.0.4/predictHRG_GPL/predictHRG -f ' + ion + '.pairs -s ' + ion + '_' + label + '_best-dendro.hrg -t ' + label
	cmd = './HRG/hrg_20120527_predictHRG_v1.0.4/predictHRG_GPL/predictHRG -f ' + ion + '_' + label + '.pairs -t ' + label
	args = shlex.split(cmd)
	subprocess.call(args)
	######################################
	# from subprocess import Popen, PIPE
	# # create gml-file of G
	# gml_path = os.getcwd() + '/' + G.name + '.gml' 
	# nx.write_gml(G, gml_path)
	# gml_file = open(gml_path)

	# call ./estimate
	# args = ["./estimate", str(MCSWEEPS)]
	# a = Popen(args, stdin=gml_file, stdout=PIPE)

def load_predicted_links():
	##
	## @brief      Loads predicted links from ranked.wpairs file created by the Clauset C++ script.
	##
	## @return     Returns array of predicted links ([u,v]) and a second array with the corresponding probabilities.
	##
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

#TODO reorganize to avoid saving stuff to file
def load_removed_links():
	##
	## @brief      Loads removed links from earlier written file. (this python
	##             script consisted originally of two parts.)
	##
	## @return     Returns a npy array of the removed links
	##

	# load removed links (has to be in the same folder)
	with open(ion + '_' + label + '_removed_edges.pairs', 'r') as removed_edges_file:
		line_reader = csv.reader(removed_edges_file, delimiter='\t')
		line_data_temp = np.array(list(line_reader))
	# print line_data_temp

	removed_links = np.empty((len(line_data_temp), 2), dtype=int)
	for i in range(len(line_data_temp)):	
		for k in range(2): removed_links[i,k] = int( line_data_temp[i][k] )

	return removed_links

def check_prediction(pred_links, pred_links_prob, removed_links, numID_2_strID_dict):
	##
	## @brief      Create an edgelist file exactly in the format that the
	##             Clauset/Newman c++ algorithm needs.
	##
	## @param      Graph          The graph
	## @param      ion            The ion
	##
	## @return     { description_of_the_return_value } d_links):
	##
	## @brief      Checks the predicted links against the removed ones
	##
	## @param      pred_links     The predicted links
	## @param      removed_links  The removed links
	##
	## @return     Returns list of tuple with rank of predicted link,
	##             probability, and both atomic link levels. Returns an array
	##             with true positive and false positive rates used later for
	##             plotting the ROC curve.
	##

	links_found_info_tup = [] #list with result tuples to print
	found_mask = np.zeros((pred_links.shape[0], 1), dtype=bool) #masks of the elements which have been found correctly. Equivalent to the removed_links set.
	ranks =  np.zeros((pred_links.shape[0], 1), dtype=int) #array of the ranks
	right_predictions = 0 #number of right predictions
	roc_array = np.empty((pred_links.shape[0], 2), dtype=float) # array used for calculating roc curve


	
	for j in range(pred_links.shape[0]):
		ranks[j] = j #needed to have the index permanently
		for i in range(removed_links.shape[0]):
			if ( set(list(removed_links[i,:])) == set(list(pred_links[j,:])) ):
				right_predictions+=1
				found_mask[j] = True

				#tupel with results
				if ion == 'iuasd':
				# if ion == 'model':
					links_found_info_tup.append((right_predictions,
										j, #rank
										pred_links_prob[j], #probability
										numID_2_strID_dict[str(removed_links[i,0])], numID_2_strID_dict[str(removed_links[i,1])] #level strID
										))
				else:
					links_found_info_tup.append((right_predictions,
										j, #rank
										pred_links_prob[j], #probability
										removed_links[i,0], removed_links[i,1] #term
										))
										# numID_2_term_dict[str(removed_links[i,0])], numID_2_term_dict[str(removed_links[i,1])] #term
				# break #back to j loop

		#still in j-loop
		no_of_eval_links = j+1 #start at 1 #number of links in the pred_links list that have been checked up to this point
		#true_pos = r_p / ( r_p + f_n )
		roc_array[j,0] = float(right_predictions ) / ( float(removed_links.shape[0]) )
		#false_pos  = f_p / ( r_n + f_p )
		roc_array[j,1] = (float(no_of_eval_links) - float(right_predictions)) / ( float(pred_links.shape[0]) - float(removed_links.shape[0]) )
		#true_pos / false_pos+1
		# roc_array[j,2] = float(right_predictions) / ( float(no_of_eval_links) - float(right_predictions) + float(1) )

	#make array consisting of the predicted edges, their rank and the probability
	predictions = np.hstack((pred_links, ranks)) 
	# print predictions

	return links_found_info_tup, roc_array, predictions, found_mask, ranks

#AUC: Lu, L., and Zhou, T. (2010). Link Prediction in Complex Networks: A Survey. Physica A, 390(6), 1150-1170. https://doi.org/10.1016/j.physa.2010.11.027
def calc_auc(roc_array, removed_links, pred_links, predictions, ranks, found_mask):
	print 'Calculating AUC...'
	b = 0 #number of times the predicted link has a higher rank than a randomly chosen non-existent link
	w = 0 #number of times the predicted link has a equal or worse rank than a randomly chosen non-existent link

	#sampling
	#TODO: what is a sensible sample size?
	#TODO: use predictions[2] instead of ranks
	if label == '5pauc':
		sample_size = int(0.3 * predictions.shape[0])
	else:
		sample_size = int(0.05 * predictions.shape[0])
	for e in  range(sample_size):
		if label == '5pauc':
			rank_removed = np.random.choice(ranks[found_mask]<int(0.05 * predictions.shape[0]), 1, replace=False) #choosing from E_p
			rank_nonexist = np.random.choice(ranks[~found_mask]<int(0.05 * predictions.shape[0]), 1, replace=False) #choosing from non-existent links, = U-E
		else:
			rank_removed = np.random.choice(ranks[found_mask], 1, replace=False) #choosing from E_p
			rank_nonexist = np.random.choice(ranks[~found_mask], 1, replace=False) #choosing from non-existent links, = U-E

		if (rank_removed > rank_nonexist):
			b+=1
		else:
			w+=1

	auc = (b + 0.5 * w) / float(sample_size)
	print 'b',b
	print 'w',w
	print 'sample size', sample_size

	return auc

def plot_roc_curve(roc_array, dropout_fraction):
	##
	## @brief      Plots a ROC curve
	##
	## @param      roc_array         The roc array
	## @param      dropout_fraction  The dropout fraction
	##
	## @return     Saves a figure to file.
	##
	fig = plt.figure(0)
	plt.plot(roc_array[:,1], roc_array[:,0], label='dropout p =' + str(dropout_fraction), linewidth=1.5)
	plt.title('ROC curve for '+ion+' Network')
	plt.ylabel('True Positive rate')
	plt.xlabel('False Positive rate')
	plt.legend(loc=8)
	plt.savefig('../plots/HRG/ROC_'+ion+'_'+label+'_drop_'+str(dropout_fraction)+'.png')
	# fig = plt.figure(1)
	# plt.plot(roc_array[:,2], label='dropout p =' + str(dropout_fraction))
	# plt.show()


def plot_auc(auc_list, fraction_kept_list):
	##
	## @brief      Plots the AUC value over the fraction of observed links. Hard-coded x axis
	##
	## @param      auc_list               	The auc list
	## @param      fraction_kept_list  	The list of the fraction of kept edges
	##
	## @return     Save a figure to file.
	##

	fig = plt.figure()
	plt.plot(fraction_kept_list, auc_list)
	plt.title('AUC curve for '+ion+' Network')
	plt.ylabel('AUC')
	plt.xlabel('Fraction of kept links ( 1 - dropout )')
	# plt.xlabel('Fraction of edges observed')
	plt.legend(loc=8)
	plt.ylim((0.4,1.0))
	plt.plot((0.0, 1.0), (0.5, 0.5), 'b--')
	plt.savefig('../plots/HRG/AUC_'+ion+'_'+label+'.png')
	plt.show()


##########################################################################################################
# performs a single run of dropout, prediction an evaluation for one specified network and fraction of dropout
def single_run(dropout_fraction):
	#load network
	G = load_network(ion, only_dipole=True, alt_read_in=False)

	#get rid of levels without term entry
	G, term = remove_empty_levels(G)
	

	# store IDs and terms of nodes in array and save it to .npy file
	matching_array = create_ID_term_array(G)
	# print matching_array

	"""
		# ####TODO sauberer organisieren. Dict global erstellen, nicht als datei speichern.
		# #load array with IDs and terms of nodes
		# print 'Loading matching.npy'
		# matching_array = np.load('matching.npy')
		# # print 'matching.npy loaded'
		# 
		# if ion == 'model':
		# 	#load dict with IDs and terms of nodes
		# 	print 'Loading ID_dict.npy'
		# 	ID_dict = np.load('ID_dict.npy').item()
		# 	# print ID_dict

		# 	#create second dict with reversed order (decryption...)
		# 	print 'Creating ID_dict2'
		# 	ID_dict2 = {y:x for x,y in ID_dict.items()}
		# 	# print ID_dict2
		# else:
		# 	# term dict
		# 	term_dict = dict(zip(matching_array[:,1], matching_array[:,2]))
		# 	print term_dict
	"""

	#create dictionaries from matching array, transalting between IDs and terms
	if ion == 'model':
		strID_2_numID_dict = dict(zip(matching_array[:,0], matching_array[:,1])) #keys: string ID ('7S'), values: numerical generated ID (saved as string: '33')

		#create second dict with reversed order ("decryption")
		numID_2_strID_dict = {y:x for x,y in strID_2_numID_dict.items()} #keys: numerical generated ID (saved as string: '33'), values: string ID ('7S')
	else:
		strID_2_numID_dict = dict(zip(matching_array[:,0], matching_array[:,1])) #keys: string ID ('00001.000.0033'), values: numerical generated ID (saved as string: '33')

		# term dict NICHT EINEINDEUTIG!
		numID_2_term_dict = dict(zip(matching_array[:,1], matching_array[:,2])) #keys: numerical node ID (last digits, saved as string..), values: string of term (2S*1/2)


	#perform dropout on network nodes(removing a specified fraction)
	# G, removed_nodes = dropout_nodes(G, 0.25)

	#perform dropout on the network's edges (removing a specified fraction)
	print 'Links before dropout: ', len(np.asarray(nx.to_edgelist(G)))
	print nx.number_of_edges(G)
	G, removed_edges, original_nodes = dropout_edges(G, dropout_fraction, strID_2_numID_dict)
	print nx.number_of_edges(G)
	print 'Links after dropout: ', len(np.asarray(nx.to_edgelist(G)))
	print 'Removed links: ', len(removed_edges)


	#save edgelist of G to file (AFTER dropout) in format needed by Clauset et al. C++ algorithm
	write_edgelist(G, ion, strID_2_numID_dict, original_nodes)

	#run cpp prediction script
	run_cpp_prediction(ion, label)

	#load prediction
	print 'Loading predicted links'
	pred_links, pred_links_prob = load_predicted_links()
	print len(pred_links)
	# print pred_links_prob

	#load removed links
	print 'Loading removed links'
	removed_links = load_removed_links()
	print len(removed_links)

	#evaluate prediction
	#TODO ROC curve
	print 'Checking prediction...'
	if ion == 'model':
		links_found_info_tup, roc_array, predictions, found_mask, ranks = check_prediction(pred_links, pred_links_prob, removed_links, numID_2_strID_dict)
	else: 
		links_found_info_tup, roc_array, predictions, found_mask, ranks = check_prediction(pred_links, pred_links_prob, removed_links, numID_2_term_dict)

	links_found_info_tup.sort(key=lambda k: k[1], reverse=False) #sort in ascending order of rank
	# for tup in links_found_info_tup: print(tup) #print result of prediction check

	# print predictions[:15]
	# print roc_array[:15]
	print ranks[found_mask]
	with open(ion + '_' + label + '_found_ranks.txt', 'a') as file:
		file.write('drop_' + str(dropout_fraction) + ': ')
		simplejson.dump(list(ranks[found_mask]), file)
		file.write('\n')
		file.close()
	# np.savetxt(ion + '_' + label + 'drop_' + str(dropout_fraction) + '_found_ranks.txt', ranks[found_mask])
	# print np.sum(found_mask)
	print ranks[~found_mask]
	# print np.sum(~found_mask)

	# print roc_array[:]
	# plot_roc_curve(roc_array, dropout_fraction)

	auc = calc_auc(roc_array, removed_links, pred_links, predictions, ranks, found_mask)
	print 'AUC: ', auc

	return auc, roc_array

def run_whole():
	dropout_fraction_list = [0.5,0.4,0.3,0.2,0.1,0.075,0.05,0.03,0.02,0.01]
	print 'Dropout fractions: ', dropout_fraction_list
	fraction_kept_list = [1-x for x in dropout_fraction_list]
	auc_list = []

	fig = plt.figure(0)
	plt.title('ROC curve for '+ion+' Network')
	plt.ylabel('True Positive rate')
	plt.xlabel('False Positive rate')

	# #calculate the mean auc out of 10 runs for each dropout fraction
	# m = 1 #sampling size
	# for dropout_fraction in dropout_fraction_list:
	# 	sum_auc = 0
	# 	for i in range(m):
	# 		print '#########################################################', dropout_fraction
	# 		auc, roc_array = single_run(dropout_fraction)
	# 		sum_auc += auc
	# 	mean = sum_auc / float(m)
	# 	auc_list.append(mean)
	# 	fig = plt.figure(0)
	# 	plt.plot(roc_array[:,1], roc_array[:,0], label='dropout p =' + str(dropout_fraction), linewidth=1.5)
	# 	plt.legend(loc=8)
	# 	plt.savefig('../plots/HRG/ROC_'+ion+'_'+label+'.png')

	# for dropout_fraction in dropout_fraction_list:
	for fraction_kept in fraction_kept_list:
		dropout_fraction = 1 - fraction_kept
		print '######################################################### Dropout: ', dropout_fraction
		auc, roc_array = single_run(dropout_fraction)
		auc_list.append(auc)
		fig = plt.figure(0)
		plt.plot(roc_array[:,1], roc_array[:,0], label='dropout p =' + str(dropout_fraction), linewidth=1.5)
		plt.legend(loc=8)
		plt.savefig('../plots/HRG/ROC_'+ion+'_'+label+'.png')


	#plot the auc graph
	plot_auc(auc_list, fraction_kept_list)

###################################################################################################################

# single_run(0.1)
run_whole()

#TODO while running exports ranked.wparis twice
# import file: scan and read?
