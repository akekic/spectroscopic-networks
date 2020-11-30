try:
	reload
except NameError:
	# Python 3
	from imp import reload
import nx2
reload(nx2)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import subprocess
import shlex
# import node2vec.src.main as n2v




n2v = nx2.node2vec()
n2v.G = nx2.model_network(Z=1, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True)

n2v.learn_features(name='test2', path_main = './node2vec/src/main.py', input_folder =  './node2vec/graph/', output_folder = './node2vec/emb/', dimensions = 20, save_file = True)

print n2v.node_vec_array[0,:]



