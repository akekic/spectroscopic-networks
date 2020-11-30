"""
@author: Julian Heiss
@date: May 2017

Description: Compares the experimental hydrogen network (NIST data) with the model network (after Jitrik-Bunge) by calculating several 'bias measures' (eg. (no of links measured on a exp-node)/(no of links on corresponding model node)=B0 ). Will in the end plot some boxgraphs visualising these measures. One can compare the whole datasets or make a wavelength cut that just includes edges in the specified wavelength window.
Uses the local nx2 library. Be aware of version incompabilities (eg. naming changes).


"""
try:
	reload
except NameError:
	# Python 3
	from imp import reload
import nx2
reload(nx2)

import csv
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import re




class Parameter:
	def __init__(self, value):
			self.value = value

	def set(self, value):
			self.value = value

	def __call__(self):
			return self.value

### dictionary for l-states
L_dictionary = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4, 'h':5, 'i':6, 'k':7, 'l':8, 'm':9, 'n':10, 'o':11, 'q':12, 'r':13, 't':14, 'u':15, 'v':16, 'w':17, 'x':18, 'y':19, 'z':20, 'a':21, 'b':22, 'c':23, 'e':24, 'j':25}

#set global path names
set1_e1_path = '../data/jitrik-bunge-e1-set1.csv'
set1_e2_path = '../data/jitrik-bunge-e2-set1.csv'
set1_m1_path = '../data/jitrik-bunge-m1-set1.csv'
set2_m1_path = '../data/jitrik-bunge-m1-set2.csv'
set2_e1_path = '../data/jitrik-bunge-e1-set2.csv'
set2_e2_path = '../data/jitrik-bunge-e2-set2.csv'


###########################################################################
###########################################################################
###########################################################################


# return all the necessary quantities to carry on with a fit
#
# @param      G     Networkx graph.
#
# @return     Returns several versions of the degree distribution needed for
#             further analysis
#
def create_deg_dist(G):
	# read all degrees into an numpy array
	deg_dist            = np.array(G.degree().values())

	num_data_points     = len(deg_dist)

	log_deg_dist        = np.log(deg_dist)

	# formula to normalise to range [a,b]: (b-a)*( (x - min(x)/(max(x) - min(x)) +a )
	normalised_deg_dist = ( deg_dist.astype(np.float) - deg_dist.astype(np.float).min() ) /( deg_dist.astype(np.float).max() - deg_dist.astype(np.float).min() )

	# calculate quantile function (inverse cumulative distribution function)
	quantile            = np.percentile(normalised_deg_dist, np.linspace(0,100,num_data_points))
	# quantile          = np.percentile(deg_dist, np.linspace(0,100,len(deg_dist)))

	return deg_dist, log_deg_dist, normalised_deg_dist, quantile, num_data_points


"""
# customized boxplot
# https://stackoverflow.com/questions/27214537/is-it-possible-to-draw-a-matplotlib-boxplot-given-the-percentile-values-instead
# #how to incorporate outliers?
# 
def customized_box_plot(percentiles, axes, redraw = True, *args, **kwargs):
	# Generates a customized boxplot based on the given percentile values
	n_box = len(percentiles)
	box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, *args, **kwargs) 
	# Creates len(percentiles) no of box plots

	min_y, max_y = float('inf'), -float('inf')

	for box_no, pdata in enumerate(percentiles):
		if len(pdata) == 6:
			(q1_start, q2_start, q3_start, q4_start, q4_end, fliers_xy) = pdata
		elif len(pdata) == 5:
			(q1_start, q2_start, q3_start, q4_start, q4_end) = pdata
			fliers_xy = None
		else:
			raise ValueError("Percentile arrays for customized_box_plot must have either 5 or 6 values")

		# Lower cap
		box_plot['caps'][2*box_no].set_ydata([q1_start, q1_start])
		# xdata is determined by the width of the box plot

		# Lower whiskers
		box_plot['whiskers'][2*box_no].set_ydata([q1_start, q2_start])

		# Higher cap
		box_plot['caps'][2*box_no + 1].set_ydata([q4_end, q4_end])

		# Higher whiskers
		box_plot['whiskers'][2*box_no + 1].set_ydata([q4_start, q4_end])

		# Box
		path = box_plot['boxes'][box_no].get_path()
		path.vertices[0][1] = q2_start
		path.vertices[1][1] = q2_start
		path.vertices[2][1] = q4_start
		path.vertices[3][1] = q4_start
		path.vertices[4][1] = q2_start

		# Median
		box_plot['medians'][box_no].set_ydata([q3_start, q3_start])

		# Outliers
		if fliers_xy is not None and len(fliers_xy[0]) != 0:
			# If outliers exist
			box_plot['fliers'][box_no].set(xdata = fliers_xy[0],
										   ydata = fliers_xy[1])

			min_y = min(q1_start, min_y, fliers_xy[1].min())
			max_y = max(q4_end, max_y, fliers_xy[1].max())

		else:
			min_y = min(q1_start, min_y)
			max_y = max(q4_end, max_y)

		# The y axis is rescaled to fit the new box plot completely with 10% 
		# of the maximum value at both ends
		axes.set_ylim([min_y*1.1, max_y*1.1])

	# If redraw is set to true, the canvas is updated.
	if redraw:
		axes.figure.canvas.draw()

	return box_plot
"""

##
## @brief      Creates a plot for the degree - l-state - dependence for the
##             experimental network
##
## @param      L          L state dictionary
## @param      draw_plot  bool: If plot should be shown on run (True) or only
##                        saved to file (False)
##
## @return     Saves a plot to hard-coded-path. Could return array of l states
##             and the mean degree of the nodes in this state (averaged over all
##             hydrogenic ions, but is not doing so at the moment)
##
def real_network_analysis(L, draw_plot):
	ions_list       = nx2.one_electron[:] #what ions to include
	empty_ions      = []
	deg_l 			= np.zeros(shape=(len(L),2)) #containing the temporary information about one ion during the loop
	deg_l_total		= np.zeros(shape=(len(L),len(ions_list))) #to store information about all networks 

	# loop through all ions in ions_list
	for ion_number, ion in enumerate(ions_list):
	# create network with the NIST lines
		G, elevel, term, j = nx2.spectroscopic_network(ion, weighted=True, check_wl=True, dictionaries=True)
		#check if empty or small
		if G.size() == 0:
			print('Empty graph.')
			empty_ions.append(ion)
			continue
		elif G.size() < 5:
			print('Less than 5 nodes in Graph.')
			empty_ions.append(ion)
			continue
		else:
			print '#######Ion:', ion

			# Cut nodes
			init_nodes = G.nodes()
			for n in init_nodes:
				if not term[n]: #if empty string, delete entries
					G.remove_node(n)
					term.pop(n)
					elevel.pop(n)
					j.pop(n)

			#list without weird nodes
			nodes = G.nodes()

			#get the degrees of the nodes
			nodes_degree = np.array(G.degree().values())
			# print nodes_degree

			# l_nodes = nx.get_node_attributes(G,'term')
			l_nodes = []
			for k, v in elevel.iteritems():
				l_value = "".join(re.findall("[a-zA-Z]+", v))
				l_nodes.append(l_value)

			temp = np.zeros_like(deg_l)
			for i, value in enumerate(L.itervalues()):
				temp[i][0] = value
				deg_l = np.sort(temp, axis=0)
			for i in range(len(l_nodes)):
				deg_l[L[l_nodes[i]]][1] = deg_l[L[l_nodes[i]]][1] + nodes_degree[i]  #verwirrende zeile

			#copy information to a np array for all ions
			# avg_deg = np.sum(deg_l[:,1],0) / nx.number_of_nodes(G)
			avg_deg = np.mean(np.array(nx.degree(G).values()))
			deg_l_total[:,ion_number] = deg_l[:,1] / avg_deg

			if draw_plot == True:
				fig = plt.figure()
				plt.plot(deg_l[:,0], deg_l[:,1], label='data') #plot the nth ion
				plt.title("Degree of l-state")
				plt.legend(loc=8)
				plt.show()


	#printing and plotting the results
	print deg_l_total
	sum = np.sum(deg_l_total, axis=1)
	mean = np.mean(deg_l_total, axis=1)
	print sum
	print mean
	plt.plot(deg_l[:,0], mean, label='data')
	plt.title("Degree of l-states")
	plt.ylabel('mean degree of all hydrogenic ions')
	plt.xlabel('l state')
	plt.show()
	plt.savefig('../plots/degree_l_bias_mean.png')


##
## @brief      Creates a plot for the degree - l-state - dependence for the
##             model network (Hydrogen)
##
## @param      L          L state dictionary
## @param      draw_plot  bool: If plot should be shown on run (True) or only
##                        saved to file (False)
##
## @return     Saves a plot to hard-coded-path. Could return array of l states
##             (up to the 5th lvl (hardcoded)) and the degree of the nodes in
##             this state (but is not doing so at the moment)
##
def model_network_analysis(L, draw_plot):

	# Take Model Network set 1 as input
	G_model = nx2.Model_Network_dipole(set1_e1_path, set1_e2_path, set1_m1_path, set2_e1_path, set2_e2_path, set2_m1_path)
	# G_model = nx2.Model_Network(set1_e1_path, set1_e2_path, set1_m1_path, set2_e1_path, set2_e2_path, set2_m1_path)
	# E1 = [(u,v) for (u,v) in G0.edges() if G0[u][v]['label'] == 'E1']
	# model_dipole_G = nx.Graph()
	# model_dipole_G.add_edges_from(E1)

	nodes        = G_model.nodes() 						#list of nodes
	nodes_degree = np.array(G_model.degree().values()) 	#get the degrees of the nodes
	l_nodes      = [] 									#list of l states of nodes
	deg_l 	     = np.zeros(shape=(len(L),2))			#array to save the histogram of the occurrences of l states
	temp         = np.zeros_like(deg_l)					#temporary array

	#read in the l state of each node and save to list
	for i, n in enumerate(nodes):
		l_value = n[-1].lower() #convert to lower case
		l_nodes.append(l_value)

	#create first column of arrays (the l states)
	for i, value in enumerate(L.itervalues()):
		temp[i][0] = value
		deg_l = np.sort(temp, axis=0)
	#fill in second colum (the occurrences of the l states)
	for i in range(len(l_nodes)):
				deg_l[L[l_nodes[i]]][1] = deg_l[L[l_nodes[i]]][1] + nodes_degree[i]


	# avg_deg = np.sum(deg_l[:,1],0) / nx.number_of_nodes(G)
	avg_deg = np.mean(np.array(nx.degree(G_model).values()))


	#printing and plotting the results
	if draw_plot == True:
		fig = plt.figure()
		plt.plot(deg_l[:6,0], deg_l[:6,1], label='data') #until 5th l state
		plt.title("Degree of l-state in the Model Network")
		plt.ylabel('degree of l state')
		plt.xlabel('l state')
		plt.legend(loc=8)
		plt.show()
		plt.savefig('../plots/degree_l_bias_model_network.png')


	
##
## @brief      Calculate bias for each node in a given experimental network on
##             the basis of a model network. The bias of node i is defined as
##             b_i = s_i / p_i where s_i is the sum of all weights of the edges
##             ending in node i and p_i the same quantity in the model network
##
## @param      G_exp           Experimental Networkx Graph
## @param      exp_elevel      The elevel dictionary of the exp Graph
## @param      exp_term        The term dictionary of the exp Graph
## @param      exp_j           The j dictionary of the exp Graph
## @param      G_model         The model spectroscopic network
## @param      L_dictionary    The l-states dictionary
## @param      wavelength_cut  bool: if true, make a cut to specified range
## @param      ion_name  The ion name for the experimental network that is to be
##                       checked
##
## @return     A Panda dataframe containing the node-levels and the respective
##             sum of weights in the experimental and model network and several
##             bias measures calculated from these quantities.
##
def calculate_bias(G_exp, exp_elevel, exp_term, exp_j, G_model, L_dictionary, wavelength_cut):
	#remove empty nodes
	G_exp, exp_elevel, exp_term, exp_j = remove_empty_nodes(G_exp, exp_elevel, exp_term, exp_j)

	# remove all edges from experimental graph  and model graph that are not in specified wavelength
	if wavelength_cut==True: G_exp, exp_elevel, exp_term, exp_j, G_model = make_wavelength_cut(900, 9000, G_exp, exp_elevel, exp_term, exp_j, G_model)

	# remove specific node
	## G_exp.remove_node('001001.001.000025')
	## exp_term.pop('001001.001.000025')
	## exp_elevel.pop('001001.001.000025')
	## exp_j.pop('001001.001.000025')

	# compare exp and model graph and remove all nodes that are not in model graph
	## assign labels in exp graph similar to the ones in the model graph
	for n in G_exp.nodes():
		if exp_elevel[n][-1] == 's':
			exp_elevel[n] = exp_elevel[n].upper()
		if exp_elevel[n][-1] == 'p' and exp_j[n] == '3/2':
			exp_elevel[n] = exp_elevel[n].upper()
		if exp_elevel[n][-1] == 'd' and exp_j[n] == '5/2':
			exp_elevel[n] = exp_elevel[n].upper()
		if exp_elevel[n][-1] == 'f' and exp_j[n] == '7/2':
			exp_elevel[n] = exp_elevel[n].upper()
		if exp_elevel[n][-1] == 'g' and exp_j[n] == '9/2':
			exp_elevel[n] = exp_elevel[n].upper()
		if exp_elevel[n][-1] == 'h' and exp_j[n] == '11/2':
			exp_elevel[n] = exp_elevel[n].upper()

	# Alternative way
	# TODO
	# for n in G_exp.nodes():
	# 	if j[n].astype(float) == L_dictionary[exp_elevel[n][-1]] + 1/2

	

	# take subgraph of model so that model graph is matching the experimental graph	
	G_model = G_model.subgraph(exp_elevel.values())

	# check if there are any nodes in exp graph that are not in the model graph
	if G_exp.number_of_nodes() != G_model.number_of_nodes():
		print('number of nodes exp:', G_exp.number_of_nodes())
		print('number of nodes model:', G_model.number_of_nodes())
		warnings.warn(
		"Networks don't match!",
		RuntimeWarning
		 )


	# Panda data frame with node IDs and Node Levels (which are the Node IDs in the Model graph)
	df = pd.DataFrame({ 'Node ID' 	 	 : exp_elevel.keys(),
						'Exp Node ID' 	 : exp_elevel.keys(),
						'Node Level'     : exp_elevel.values() #node level is read in from exp
						})


	"""
	# calculate sum of weights of node i (s_n) for experimental graph
	exp_sum_of_weights = np.zeros(shape=(G_exp.number_of_nodes()))
	exp_no_of_links = np.zeros(shape=(G_exp.number_of_nodes()))
	# for n in G_exp.nodes(): #old 
	exp_index = []
	for i, n in enumerate(df['Exp Node ID']):
		sum = 0
		# get links of this node
		links = G_exp.edges(n, data = 'dipole element')
		# calculate sum of all weigths of this node
		for k in range(len(links)):
			sum = sum + links[k][2]
		# exp_sum_of_weights[pd.Index(df).get_loc(n)] = sum		#geht nicht
		exp_no_of_links[i] = len(links)
		exp_sum_of_weights[i] = sum
		exp_index.append(n)
		if n=='001001.001.000001': print(exp_sum_of_weights[i])
	"""

	# calculate sum of weights, number of links and maximal weight of node i
	# in the experimental graph
	exp_index = [] #generate node id index on the go
	no_of_links_exp = np.zeros(shape=(G_exp.number_of_nodes()))
	exp_max_weights = []
	for i, n in enumerate(G_exp.nodes()):
		exp_index.append(n)

		no_of_links_exp[i] = len(G_exp.edges(n))

		links = G_exp.edges(n, data = 'matrix element')
		max_weight = 0
		for k in range(len(links)):
			if links[k][2] > max_weight: max_weight = links[k][2]
		exp_max_weights.append(max_weight)

		# if n=='001001.001.000025': print links



	# calculate sum of weights	
	sum_of_weights_exp = nx.degree(G_exp, weight = 'matrix element')

	# change index to the Node ID system (temporarily)
	df.set_index('Node ID', inplace=True)
	# fill in experimental weigths
	df['Exp Sum of weights S_n'] = pd.Series(sum_of_weights_exp, index=exp_index, dtype='float32')
	df['Exp Number of links'] = pd.Series(no_of_links_exp, index=exp_index, dtype='float32')
	df['Exp Max weight'] = pd.Series(exp_max_weights, index=exp_index, dtype='float32')

	
	"""
	# calculate sum of weights (sow) for model graph (p_n)
	model_index = [] #generate level name index on the go
	model_sum_of_weights = np.zeros(shape=(G_model.number_of_nodes()))
	model_no_of_links = np.zeros(shape=(G_model.number_of_nodes()))
	for i, n in enumerate(G_model.nodes()):
		model_sum = 0
		# get links of this node
		links = G_model.edges(n, data = 'dipole element')
		# calculate sum of all weigths of this node
		for k in range(len(links)):
			model_sum = model_sum + links[k][2]
		model_no_of_links[i] = len(links)
		model_sum_of_weights[i] = model_sum
		model_index.append(n)
		if n=='1S': print(model_sum_of_weights[i])
	"""

	# calculate sum of weights (using implented function), number of links
	# and the maximal weight for one node in the model graph 
	model_index = [] #generate level name index on the go
	no_of_links_model = np.zeros(shape=(G_model.number_of_nodes()))
	model_max_weights = []
	for i, n in enumerate(G_model.nodes()):
		model_index.append(n)

		no_of_links_model[i] = len(G_model.edges(n))

		links = G_model.edges(n, data = 'dipole element')
		# print links
		max_weight = 0
		for k in range(len(links)):
			if links[k][2] > max_weight:
				max_weight = links[k][2]
				max_id = k
		model_max_weights.append(max_weight)

		#
		max_llvl = links[max_id][0]
		max_ulvl = links[max_id][1]
		# print(max_llvl, max_ulvl)
		
		# if n=='6d': print links

	sum_of_weights_model = nx.degree(G_model, weight = 'dipole element')


	#change index to the level system
	df.set_index('Node Level', inplace=True)
	#fill in model weights using the level names as index
	df['Model Sum of weights P_n'] = pd.Series(sum_of_weights_model, index=model_index,dtype='float32')
	df['Model Number of links'] = pd.Series(no_of_links_model, index=model_index,dtype='float32')
	df['Model Max weight'] = pd.Series(model_max_weights, index=model_index,dtype='float32')


	# calculate ratio of number of edges per node in both graphs
	df['Bias_Measure_B0'] = df['Exp Number of links'] / df['Model Number of links']

	# calculate the bias_measure_B1 (ratio of sum of weights of one node (exp/nodel))             
	df['Bias_Measure_B1'] = df['Exp Sum of weights S_n'] / df['Model Sum of weights P_n']

	# calculate the bias_measure_B1 (ratio of max weights
	df['Bias_Measure_B2'] = df['Exp Max weight'] / df['Model Max weight']
	
	# sort df by descending bias measure
	df.sort_values(by='Bias_Measure_B1', ascending=False, inplace=True)


	# checking single transitions
	# print G_model['1S']['2p']['dipole element']
	# print G_exp['001001.001.000001']['001001.001.000002']['dipole element']

	return df
	

																		 

##
## @brief      Makes boxplots of data.
##
## @param      dataframe  The panda dataframe to be plotted
##
## @return     Saves plots to plots folder.
##
def make_boxplot(dataframe, wavelength_cut):
	#empirical quantiles

	#custom boxplot or standard matplotlib boxplot
	custom = False
 

	bias_measures = [dataframe['Bias_Measure_B0'], dataframe['Bias_Measure_B1'], dataframe['Bias_Measure_B2']]
	bias_names = ['B0', 'B1', 'B2']

	if custom == True:
		# calculate percentiles
		quantiles = np.array([0,25,50,75,100])
		percentiles = []
		for b in bias_measures:
			percentiles.append(np.percentile(bias_measures, quantiles)) #loop klappt nicht
		print percentiles
		fig, ax = plt.subplots()
		plt.title("Custom Boxplots of bias measures")
		customized_box_plot(percentiles, ax, redraw=True, notch=0, sym='+', vert=1, whis=[5, 95])
		plt.savefig('../plots/bias_custom_boxplot.png')
	else: #standard boxplot
		for i, bias_measure in enumerate(bias_measures):
			fig = plt.figure()
			plt.grid(True, axis='y') # let's add a grid on y-axis
			plt.boxplot(bias_measure, notch=True, sym='+', vert=1, whis=[5, 95], showmeans=True)
			if wavelength_cut == True:
				plt.title("Boxplot of bias measure after wavelength cut: "+bias_names[i])
				plot_path = '../plots/'+bias_names[i]+'_boxplot_wl_cut.png'
			else:
				plt.title("Boxplot of bias measure "+bias_names[i])
				plot_path = '../plots/'+bias_names[i]+'_boxplot.png'
			plt.savefig(plot_path)


#TODO: without dictionary?
def remove_empty_nodes(G_exp, exp_elevel, exp_term, exp_j):
	# delete empty nodes
	for n in G_exp.nodes():
		if not exp_term[n]: #if empty string, delete entries
			G_exp.remove_node(n)
			exp_term.pop(n)
			exp_elevel.pop(n)
			exp_j.pop(n)
	return G_exp, exp_elevel, exp_term, exp_j

#
# @brief      Makes a wavelength cut.
#
# @param      lower_wl    The lower wl bound in Angstroem
# @param      upper_wl    The upper wl bound in Angstroem
# @param      G_exp       The experimental graph
# @param      exp_elevel  Dictionary
# @param      exp_term    Dictionary
# @param      exp_j       Dictionary
# @param      G_model     The model Graph
#
# @return     Experimental Graph and its dictionaries with edges outside the
#             specified range removed and nodes without edges removed
#
def make_wavelength_cut(lower_wl, upper_wl, G_exp, exp_elevel, exp_term, exp_j, G_model):
	# print len(G_exp.edges())
	# Wavelength cut: remove edges that are not in the specified range
	for u,v,w in G_exp.edges(data='wavelength'):
		# print u,v,w
		if w < lower_wl or w > upper_wl:
			G_exp.remove_edge(u,v)
	# print len(G_exp.edges())


	# remove loners (nodes without connection)
	deg = G_exp.degree()
	loners = [n for n in deg if deg[n] == 0]
	# print loners
	G_exp.remove_nodes_from(loners)
	for n in loners:
		exp_term.pop(n)
		exp_elevel.pop(n)
		exp_j.pop(n)


	# Wavelength cut: remove edges that are not in the specified range
	for u,v,w in G_model.edges(data='wavelength'):
		# print u,v,w
		if w < lower_wl or w > upper_wl:
			G_model.remove_edge(u,v)


	return G_exp, exp_elevel, exp_term, exp_j, G_model




################## RUN ########################################

print('Bias Check')

# real_network_analysis(L_dictionary, draw_plot=True)
# model_network_analysis(L_dictionary, draw_plot=True)


G_model = nx2.Model_Network(set1_e1_path, set2_e1_path, set1_e2_path, set2_e2_path, set1_m1_path, set2_m1_path)

#get exp graph
ion = 'H1.0' #does not work for other ions, since model graph is only known for hydrogen
G_exp, exp_elevel, exp_term, exp_j = nx2.spectroscopic_network(ion, weighted=True, check_wl=False, check_accur=True, dictionaries=True, alt_read_in=True)

df = calculate_bias(G_exp, exp_elevel, exp_term, exp_j, G_model, L_dictionary, wavelength_cut=False)
print df
make_boxplot(df, wavelength_cut=False)

df_wl_cut = calculate_bias(G_exp, exp_elevel, exp_term, exp_j, G_model, L_dictionary, wavelength_cut=True)
print df_wl_cut
make_boxplot(df_wl_cut, wavelength_cut=True)
