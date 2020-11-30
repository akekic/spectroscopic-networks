#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:37:26 2017

@author: arminkekic
"""
import networkx as nx
import sys

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)


def only_dipole_transitions_parity(G):
    Gc = G.copy()
    for edge in G.edges_iter():
        p0 = nx.get_node_attributes(Gc, 'parity')[edge[0]]
        p1 = nx.get_node_attributes(Gc, 'parity')[edge[1]]
        if p0 == p1:
            Gc.remove_edge(*edge)
    return Gc


ion = sys.argv[1]

if ion == 'model':
    print 'Loading model network'
    G = nx2.model_network(max_n=8)
    edges = G.edges()
    transitionType = nx.get_edge_attributes(G, 'transitionType')
    for e in edges:
        if transitionType[e] != 'E1':
            G.remove_edge(*e)
    print nx.get_edge_attributes(G, 'transitionType')
else:
    print 'Loading spectroscopic network:', ion
    G = nx2.spectroscopic_network(ion, weighted=False)
    G = only_dipole_transitions_parity(G)

print 'Writing', ion + '.gml'
nx.write_gml(G, ion + '.gml')
print ion + '.gml', 'saved'
