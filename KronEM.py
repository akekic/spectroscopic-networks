"""
Description
-----------
Using Snap by Stanford

References
----------
Experimental data by NIST
Theoretical data by Jitrik, O., Bunge, C. F. (2004). Transition probabilities for hydrogen-like atoms. Journal of Physical and Chemical Reference Data, 33(4), 1059-1070. https://doi.org/10.1063/1.1796671

-----------
@author: Julian Heiss
@date: August 2017
"""

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

import snap

import networkx as nx
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# %%
# GLOBAL VARIABLES
########################################

ion = 'H1.0'                          # ion to use as input, see nx2.py
experimental = False                    # use the experimental NIST data or the theoretical JitrikBunke data
if experimental == False:
    # get Z
    for i, char in enumerate(ion.split('.')[0]):
        if char.isdigit():
            index = i
            break
    Z = int(ion.split('.')[0][index:])

only_dipole_global = True      # global switch for taking the full network or only dipole lines

n_limit            = True      # limit is only to be used for one electron ions (hydrogenic ions)
if n_limit==False:
    max_n_global = None
else:
    max_n_global   = 8                 # maximal n considered in network

save_switch        = True        # True: save all figures. False: show
save_files         = True      #True: Do not delete edgelist and similar files used for the prediction algorithms (eg. HRG)

########################################
# %%
# %%
Kron_PATH = './Kron/'
directory = Kron_PATH + ion + '_' + 'dropout' + '/'
if not os.path.exists(directory):
    os.makedirs(directory)
plot_directory = '../plots/Kron/' + ion + '/'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# %%
# predict using HRG
print 'Method: ','KronEM'
lp_Kron = nx2.LinkPrediction()

# load network
if experimental == False:
    # load theoretical data
    if only_dipole_global == True:
        lp_Kron.G_original = nx2.model_network(Z=Z, E1=True, max_n=max_n_global) #only dipole lines up to max_n
    else:
        lp_Kron.G_original = nx2.model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=max_n_global) #all lines
else:
    if only_dipole_global == True:
        lp_Kron.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
        lp_Kron.G_original = nx2.remove_empty_levels(lp_Kron.G_original, 'term') #remove nodes with empty term entry
        lp_Kron.G_original = nx2.remove_n_greater(lp_Kron.G_original, max_n=max_n_global) # maximal n considered
        lp_Kron.G_original = nx2.only_dipole_transitions_parity(lp_Kron.G_original) #only dipole lines
        lp_Kron.G_original = nx2.only_largest_component(lp_Kron.G_original)
    else:
        lp_Kron.G_original = nx2.spectroscopic_network(ion, weighted=True, alt_read_in=False)
        lp_Kron.G_original = nx2.remove_empty_levels(lp_Kron.G_original, 'term') #remove nodes with empty term entry
        lp_Kron.G_original = nx2.remove_n_greater(lp_Kron.G_original, max_n=max_n_global) # maximal n considered
        lp_Kron.G_original = nx2.only_largest_component(lp_Kron.G_original)

# create dict
lp_Kron.create_HRG_ID_dict(experimental=experimental)



# dropout
dropout = 0.1
lp_Kron.dropout(dropout)
# lp_Kron.G_training = lp_Kron.G_original

print 'Number of edges G_original', nx.number_of_edges(lp_Kron.G_original)
print 'Number of nodes G_original', nx.number_of_nodes(lp_Kron.G_original)
print 'Number of edges G_training', nx.number_of_edges(lp_Kron.G_training)
print 'Number of nodes G_training', nx.number_of_nodes(lp_Kron.G_training)

# %%
# save nx graph to edgelist
lp_Kron.write_edgelist(Graph=lp_Kron.G_training,
                label='training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                directory=directory) # write training set edgelist
lp_Kron.write_edgelist(Graph=lp_Kron.G_probe,
                label='probe_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout),
                directory=directory) # write probe set edgelist

# %%
# create snap graph from edgelist
G_snap_training = snap.LoadEdgeList(snap.PNGraph,directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs',0,1)

print snap.IsConnected(G_snap_training)


# %%
import shlex
import subprocess

path_main         = 'Snap-4.0/examples/kronem/kronem'
input_folder      = directory
output_folder     = directory
edgelist_filename = 'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs'
output_prefix     = 'test' #doesnt do anything

cmd = path_main + ' -i:' + input_folder + edgelist_filename + ' -n0:2 -ei:5 ' + '-o:' + output_prefix # Command in command-line style which will passed to the program.
args = shlex.split(cmd)

# run
subprocess.call(args)

# %%
if not save_files:
    os.remove(directory+'probe_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
    os.remove(directory+'training_' + ion + '_dropout' + n_limit*'_max_n_' + n_limit*str(max_n_global) + only_dipole_global*'_only_dipole'+ (not only_dipole_global)*'_full'+ experimental*'_NIST' + (not experimental)*'_Jitrik'+'_drop_'+str(dropout) + '.pairs')
    os.remove('KronEM-'+edgelist_filename+'.tab')


# %%
# Read in results
with open('results.txt') as results_file:
    line_reader = csv.reader(results_file, delimiter='\n')
    lines = np.array(list(line_reader))
    row_count = len(lines)

    initiator = np.asarray(lines[0,0][1:-1].replace(';', ',').split(',')).astype(float)
    # print initiator
    permutation = np.asarray(lines[1,0][:-1].split(',')).astype(int)
    # print permutation
#%%
os.remove('results.txt')

#%%
theta = np.array([[initiator[0],initiator[1]],[initiator[2],initiator[3]]])
# initialise loop
G_Kron = theta

# calculate prediction by kronecker powers
for i in xrange(int(np.log2(nx.number_of_nodes(lp_Kron.G_training)))):
    G_Kron = np.kron(G_Kron, theta)


# %%
# rearrange matrix
i = np.argsort(permutation)
G_Kron = G_Kron[:,i]
G_Kron = G_Kron[i,:]


# %%
# get adjacency matrix
G_original_adj = nx.to_numpy_matrix(lp_Kron.G_original)

# zeropad the adjacency matrix
G_original_adj_padded = np.zeros((permutation.shape[0],permutation.shape[0]))
G_original_adj_padded[:G_original_adj.shape[0],:G_original_adj.shape[1]] = G_original_adj


# %%
print np.corrcoef(G_original_adj.astype(float).flatten(), G_original_adj.astype(float).flatten())
print np.corrcoef(G_Kron.astype(float).flatten(), G_original_adj_padded.astype(float).flatten())
print np.cov(G_Kron.astype(float).flatten(), G_original_adj_padded.astype(float).flatten())


