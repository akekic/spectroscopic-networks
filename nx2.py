#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:14:19 2017

@author: arminkekic

Module containing all important functions related to networks that we coded ourselves. Functions can be called using dot-notation (e.g. nx2.func()). You can import the module like this:

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

"""
import numpy as np
import networkx as nx
import itertools
import csv
import time
import random
import warnings

# ions with two electrons and more than 50 lines
good_list = ['He1.0', 'Li2.0', 'Be3.0', 'B4.0', 'C5.0', 'N6.0', 'O7.0', 'F8.0', 'Ne9.0', 'Na10.0', 'Mg11.0', 'Al12.0',
             'Si13.0', 'S15.0', 'Ti21.0', 'V22.0', 'Cr23.0', 'Mn24.0', 'Fe25.0', 'Co26.0', 'Ni27.0', 'Cu28.0',
             'Ga30.0', 'Kr35.0','Sr37.0', 'Mo41.0']

ions = np.array(['10B1.0','10B2.0','11B1.0','11B2.0','198Hg1.0','3He1.0','Ac1.0','Ac2.0','Ac3.0','Ac4.0','Ag1.0','Ag2.0','Ag3.0','Al1.0','Al10.0','Al11.0','Al12.0','Al13.0','Al2.0','Al3.0','Al4.0','Al5.0','Al6.0','Al7.0','Al8.0','Al9.0','Am1.0','Am2.0','Ar1.0','Ar10.0','Ar11.0','Ar12.0','Ar13.0','Ar14.0','Ar15.0','Ar16.0','Ar17.0','Ar18.0','Ar2.0','Ar3.0','Ar4.0','Ar5.0','Ar6.0','Ar7.0','Ar8.0','Ar9.0','As1.0','As2.0','As3.0','As4.0','As5.0','At1.0','Au1.0','Au2.0','Au3.0','B1.0','B2.0','B3.0','B4.0','B5.0','Ba1.0','Ba10.0','Ba11.0','Ba12.0','Ba2.0','Ba20.0','Ba21.0','Ba22.0','Ba24.0','Ba25.0','Ba26.0','Ba27.0','Ba28.0','Ba29.0','Ba3.0','Ba30.0','Ba31.0','Ba32.0','Ba33.0','Ba34.0','Ba35.0','Ba36.0','Ba37.0','Ba38.0','Ba4.0','Ba40.0','Ba41.0','Ba42.0','Ba43.0','Ba44.0','Ba45.0','Ba46.0','Ba47.0','Ba48.0','Ba49.0','Ba5.0','Ba50.0','Ba52.0','Ba53.0','Ba54.0','Ba55.0','Ba56.0','Ba6.0','Ba7.0','Ba8.0','Ba9.0','Be1.0','Be2.0','Be3.0','Be4.0','Bi1.0','Bi2.0','Bi3.0','Bi4.0','Bi5.0','Bk1.0','Bk2.0','Br1.0','Br2.0','Br3.0','Br4.0','Br5.0','C1.0','C2.0','C3.0','C4.0','C5.0','C6.0','Ca1.0','Ca10.0','Ca11.0','Ca12.0','Ca13.0','Ca14.0','Ca15.0','Ca16.0','Ca2.0','Ca3.0','Ca4.0','Ca5.0','Ca6.0','Ca7.0','Ca8.0','Ca9.0','Cd1.0','Cd2.0','Cd3.0','Cd4.0','Ce1.0','Ce2.0','Ce3.0','Ce4.0','Ce5.0','Cf1.0','Cf2.0','Cl1.0','Cl10.0','Cl2.0','Cl3.0','Cl4.0','Cl5.0','Cl6.0','Cl7.0','Cl8.0','Cl9.0','Cm1.0','Cm2.0','Co1.0','Co10.0','Co11.0','Co12.0','Co13.0','Co14.0','Co15.0','Co16.0','Co17.0','Co18.0','Co19.0','Co2.0','Co20.0','Co21.0','Co22.0','Co23.0','Co24.0','Co25.0','Co26.0','Co27.0','Co3.0','Co4.0','Co5.0','Co8.0','Co9.0','Cr1.0','Cr10.0','Cr11.0','Cr12.0','Cr13.0','Cr14.0','Cr15.0','Cr16.0','Cr17.0','Cr18.0','Cr19.0','Cr2.0','Cr20.0','Cr21.0','Cr22.0','Cr23.0','Cr24.0','Cr3.0','Cr4.0','Cr5.0','Cr6.0','Cr7.0','Cr8.0','Cr9.0','Cs1.0','Cs10.0','Cs11.0','Cs19.0','Cs2.0','Cs20.0','Cs23.0','Cs25.0','Cs26.0','Cs27.0','Cs28.0','Cs29.0','Cs3.0','Cs34.0','Cs37.0','Cs39.0','Cs4.0','Cs40.0','Cs41.0','Cs42.0','Cs43.0','Cs44.0','Cs45.0','Cs46.0','Cs47.0','Cs48.0','Cs49.0','Cs5.0','Cs51.0','Cs52.0','Cs53.0','Cs54.0','Cs55.0','Cs6.0','Cs7.0','Cs8.0','Cs9.0','Cu1.0','Cu10.0','Cu11.0','Cu12.0','Cu13.0','Cu14.0','Cu15.0','Cu16.0','Cu17.0','Cu18.0','Cu19.0','Cu2.0','Cu20.0','Cu21.0','Cu22.0','Cu23.0','Cu24.0','Cu25.0','Cu26.0','Cu27.0','Cu28.0','Cu29.0','Cu3.0','Cu4.0','Cu5.0','D1.0','Dy1.0','Dy2.0','Er1.0','Er2.0','Er3.0','Es1.0','Es2.0','Eu1.0','Eu2.0','Eu3.0','F1.0','F2.0','F3.0','F4.0','F5.0','F6.0','F7.0','F8.0','Fe1.0','Fe10.0','Fe11.0','Fe12.0','Fe13.0','Fe14.0','Fe15.0','Fe16.0','Fe17.0','Fe18.0','Fe19.0','Fe2.0','Fe20.0','Fe21.0','Fe22.0','Fe23.0','Fe24.0','Fe25.0','Fe26.0','Fe3.0','Fe4.0','Fe5.0','Fe6.0','Fe7.0','Fe8.0','Fe9.0','Fr1.0','Ga1.0','Ga13.0','Ga14.0','Ga15.0','Ga16.0','Ga17.0','Ga18.0','Ga19.0','Ga2.0','Ga20.0','Ga21.0','Ga22.0','Ga23.0','Ga24.0','Ga25.0','Ga26.0','Ga29.0','Ga3.0','Ga30.0','Ga31.0','Ga4.0','Ga5.0','Ga6.0','Ga7.0','Gd1.0','Gd2.0','Gd3.0','Gd4.0','Ge1.0','Ge2.0','Ge3.0','Ge4.0','Ge5.0','H1.0','He1.0','He2.0','Hf1.0','Hf2.0','Hf3.0','Hf4.0','Hf5.0','Hg1.0','Hg2.0','Hg3.0','Ho1.0','Ho2.0','I1.0','I2.0','I3.0','I4.0','I5.0','In1.0','In2.0','In3.0','In4.0','In5.0','Ir1.0','Ir2.0','Ir4.0','K1.0','K10.0','K11.0','K12.0','K13.0','K14.0','K15.0','K16.0','K17.0','K18.0','K19.0','K2.0','K3.0','K4.0','K5.0','K6.0','K7.0','K8.0','K9.0','Kr1.0','Kr10.0','Kr18.0','Kr19.0','Kr2.0','Kr20.0','Kr21.0','Kr22.0','Kr23.0','Kr24.0','Kr25.0','Kr26.0','Kr27.0','Kr28.0','Kr29.0','Kr3.0','Kr30.0','Kr31.0','Kr32.0','Kr33.0','Kr34.0','Kr35.0','Kr36.0','Kr4.0','Kr5.0','Kr6.0','Kr7.0','Kr8.0','Kr9.0','La1.0','La2.0','La3.0','La4.0','La5.0','Li1.0','Li2.0','Li3.0','Lu1.0','Lu2.0','Lu3.0','Lu4.0','Lu5.0','Mg1.0','Mg10.0','Mg11.0','Mg12.0','Mg2.0','Mg3.0','Mg4.0','Mg5.0','Mg6.0','Mg7.0','Mg8.0','Mg9.0','Mn1.0','Mn10.0','Mn11.0','Mn12.0','Mn13.0','Mn14.0','Mn15.0','Mn16.0','Mn17.0','Mn18.0','Mn19.0','Mn2.0','Mn20.0','Mn21.0','Mn22.0','Mn23.0','Mn24.0','Mn25.0','Mn3.0','Mn4.0','Mn5.0','Mn6.0','Mn7.0','Mn8.0','Mn9.0','Mo1.0','Mo10.0','Mo11.0','Mo12.0','Mo13.0','Mo14.0','Mo15.0','Mo16.0','Mo17.0','Mo18.0','Mo2.0','Mo23.0','Mo24.0','Mo25.0','Mo26.0','Mo27.0','Mo28.0','Mo29.0','Mo3.0','Mo30.0','Mo31.0','Mo32.0','Mo33.0','Mo34.0','Mo35.0','Mo38.0','Mo39.0','Mo4.0','Mo40.0','Mo41.0','Mo42.0','Mo5.0','Mo6.0','Mo7.0','Mo8.0','Mo9.0','N1.0','N2.0','N3.0','N4.0','N5.0','N6.0','N7.0','Na1.0','Na10.0','Na11.0','Na2.0','Na3.0','Na4.0','Na5.0','Na6.0','Na7.0','Na8.0','Na9.0','Nb1.0','Nb2.0','Nb3.0','Nb4.0','Nb5.0','Nd1.0','Nd2.0','Ne1.0','Ne2.0','Ne3.0','Ne4.0','Ne5.0','Ne6.0','Ne7.0','Ne8.0','Ne9.0','Ni1.0','Ni10.0','Ni11.0','Ni12.0','Ni13.0','Ni14.0','Ni15.0','Ni16.0','Ni17.0','Ni18.0','Ni19.0','Ni2.0','Ni20.0','Ni21.0','Ni22.0','Ni23.0','Ni24.0','Ni25.0','Ni26.0','Ni27.0','Ni28.0','Ni3.0','Ni4.0','Ni5.0','Ni7.0','Ni9.0','Np1.0','O1.0','O2.0','O3.0','O4.0','O5.0','O6.0','O7.0','O8.0','Os1.0','Os2.0','P1.0','P10.0','P11.0','P12.0','P13.0','P2.0','P3.0','P4.0','P5.0','P6.0','P7.0','P8.0','P9.0','Pa1.0','Pa2.0','Pb1.0','Pb2.0','Pb3.0','Pb4.0','Pb5.0','Pd1.0','Pd2.0','Pd3.0','Pm1.0','Pm2.0','Po1.0','Pr1.0','Pr2.0','Pr3.0','Pr4.0','Pr5.0','Pt1.0','Pt2.0','Pt4.0','Pt5.0','Pu1.0','Pu2.0','Ra1.0','Ra2.0','Rb1.0','Rb10.0','Rb11.0','Rb12.0','Rb13.0','Rb19.0','Rb2.0','Rb20.0','Rb21.0','Rb22.0','Rb23.0','Rb24.0','Rb25.0','Rb26.0','Rb27.0','Rb28.0','Rb29.0','Rb3.0','Rb30.0','Rb31.0','Rb33.0','Rb34.0','Rb35.0','Rb36.0','Rb37.0','Rb4.0','Rb5.0','Rb6.0','Rb7.0','Rb8.0','Rb9.0','Re1.0','Re2.0','Rh1.0','Rh2.0','Rh3.0','Rn1.0','Ru1.0','Ru2.0','Ru3.0','S1.0','S10.0','S11.0','S12.0','S13.0','S14.0','S15.0','S16.0','S2.0','S3.0','S4.0','S5.0','S6.0','S7.0','S8.0','S9.0','Sb1.0','Sb2.0','Sb3.0','Sb4.0','Sb5.0','Sc1.0','Sc10.0','Sc11.0','Sc12.0','Sc13.0','Sc14.0','Sc15.0','Sc16.0','Sc17.0','Sc18.0','Sc19.0','Sc2.0','Sc20.0','Sc21.0','Sc3.0','Sc4.0','Sc5.0','Sc6.0','Sc7.0','Sc8.0','Sc9.0','Se1.0','Se2.0','Se3.0','Se4.0','Se5.0','Si1.0','Si10.0','Si11.0','Si12.0','Si13.0','Si2.0','Si3.0','Si4.0','Si5.0','Si6.0','Si7.0','Si8.0','Si9.0','Sm1.0','Sm2.0','Sn1.0','Sn2.0','Sn3.0','Sn4.0','Sn5.0','Sr1.0','Sr10.0','Sr11.0','Sr12.0','Sr13.0','Sr14.0','Sr2.0','Sr20.0','Sr21.0','Sr22.0','Sr23.0','Sr24.0','Sr25.0','Sr26.0','Sr27.0','Sr28.0','Sr29.0','Sr3.0','Sr30.0','Sr31.0','Sr32.0','Sr33.0','Sr34.0','Sr35.0','Sr36.0','Sr37.0','Sr38.0','Sr4.0','Sr5.0','Sr6.0','Sr7.0','Sr8.0','Sr9.0','T1.0','Ta1.0','Ta2.0','Ta4.0','Ta5.0','Tb1.0','Tb2.0','Tb4.0','Tc1.0','Tc2.0','Te1.0','Te2.0','Th1.0','Th2.0','Th3.0','Th4.0','Ti1.0','Ti10.0','Ti11.0','Ti12.0','Ti13.0','Ti14.0','Ti15.0','Ti16.0','Ti17.0','Ti18.0','Ti19.0','Ti2.0','Ti20.0','Ti21.0','Ti22.0','Ti3.0','Ti4.0','Ti5.0','Ti6.0','Ti7.0','Ti8.0','Ti9.0','Tl1.0','Tl2.0','Tl3.0','Tl4.0','Tm1.0','Tm2.0','Tm3.0','U1.0','U2.0','V1.0','V10.0','V11.0','V12.0','V13.0','V14.0','V15.0','V16.0','V17.0','V18.0','V19.0','V2.0','V20.0','V21.0','V22.0','V23.0','V3.0','V4.0','V5.0','V6.0','V7.0','V8.0','V9.0','W1.0','W14.0','W2.0','W28.0','W29.0','W3.0','W30.0','W31.0','W32.0','W33.0','W34.0','W35.0','W36.0','W37.0','W38.0','W39.0','W4.0','W40.0','W41.0','W42.0','W43.0','W44.0','W45.0','W46.0','W47.0','W48.0','W49.0','W5.0','W50.0','W51.0','W52.0','W53.0','W54.0','W55.0','W56.0','W57.0','W58.0','W59.0','W6.0','W60.0','W61.0','W62.0','W63.0','W64.0','W65.0','W66.0','W67.0','W68.0','W69.0','W7.0','W70.0','W71.0','W72.0','W73.0','W74.0','W8.0','Xe1.0','Xe10.0','Xe11.0','Xe19.0','Xe2.0','Xe25.0','Xe26.0','Xe27.0','Xe28.0','Xe29.0','Xe3.0','Xe4.0','Xe43.0','Xe44.0','Xe45.0','Xe5.0','Xe51.0','Xe52.0','Xe53.0','Xe54.0','Xe6.0','Xe7.0','Xe8.0','Xe9.0','Y1.0','Y2.0','Y3.0','Y4.0','Y5.0','Yb1.0','Yb2.0','Yb3.0','Yb4.0','Zn1.0','Zn2.0','Zn3.0','Zn4.0','Zr1.0','Zr2.0','Zr3.0','Zr4.0','Zr5.0','Zr6.0'])

one_electron = np.array(['D1.0','H1.0','T1.0','He2.0','Li3.0','Be4.0','C6.0','N7.0','O8.0','Na11.0','Mg12.0','Al13.0','S16.0','Ar18.0','K19.0','Sc21.0','Ti22.0','V23.0','Cr24.0','Mn25.0','Fe26.0','Co27.0','Ni28.0','Cu29.0','Ga31.0','Kr36.0','Rb37.0','Sr38.0','Mo42.0','Cs55.0','Ba56.0','W74.0'])

two_electron = np.array(['He1.0','Li2.0','Be3.0','B4.0','C5.0','N6.0','O7.0','F8.0','Ne9.0','Na10.0','Mg11.0','Al12.0','Si13.0','S15.0','Ar17.0','K18.0','Sc20.0','Ti21.0','V22.0','Cr23.0','Mn24.0','Fe25.0','Co26.0','Ni27.0','Cu28.0','Ga30.0','Kr35.0','Rb36.0','Sr37.0','Mo41.0','Xe53.0', 'Cs54.0','Ba55.0', 'W73.0'])

five_electron = np.array(['B1.0', 'C2.0', 'N3.0', 'O4.0', 'F5.0', 'Ne6.0', 'Na7.0', 'Mg8.0', 'Al9.0', 'Si10.0', 'P11.0',
                          'S12.0', 'Ar14.0', 'K15.0', 'Ca16.0', 'Sc17.0', 'Ti18.0', 'V19.0', 'Cr20.0', 'Mn21.0',
                          'Fe22.0', 'Co23.0', 'Ni24.0', 'Cu25.0', 'Kr32.0', 'Rb33.0', 'Sr34.0', 'Mo38.0', 'Cs51.0', 'Ba52.0', 'W70.0'])

six_electron = np.array(['C1.0', 'N2.0', 'O3.0', 'F4.0', 'Ne5.0', 'Na6.0', 'Mg7.0', 'Al8.0', 'Si9.0', 'P10.0',
                          'S11.0', 'Cl12.0', 'Ar13.0', 'K14.0', 'Ca15.0', 'Sc16.0', 'Ti17.0', 'V18.0', 'Cr19.0',
                         'Mn20.0', 'Fe21.0', 'Co22.0', 'Rb32.0', 'Sr33.0', 'Cs50.0', 'Ba51.0', 'W69.0'])

L_dictionary = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4, 'h':5, 'i':6, 'k':7, 'l':8, 'm':9, 'n':10, 'o':11, 'q':12, 'r':13, 't':14, 'u':15, 'v':16, 'w':17, 'x':18, 'y':19, 'z':20, 'a':21, 'b':22, 'c':23, 'e':24, 'j':25}


def _swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

def _swap_rows(arr, frm, to):
    arr[[frm, to],:] = arr[[to, frm],:]

def _swap_entries(arr, frm, to):
    arr[[frm, to]] = arr[[to, frm]]

def KW_transformation(Matrix, return_perm = False):
    """

    Function computing a permutation of a matrix s.t. one obtains zeros in the lower left corner.

    Returns
    -------

    M : array-like
        Permuted matrix.
    row_perm : array-like (if return_perm==True)
        Corresponding row permutation.
    rcol_perm : array-like (if return_perm==True)
        Corresponding column permutation.


    Parameters
    ----------

    Matrix : array-like
        Input matrix.
    return_perm : bool (optional)
        Return permutations (default=False).

    """
    # check each entry in {0,1}
    assert np.all(np.in1d(Matrix, [0,1]))
    M = Matrix.copy()
    col_perm = np.arange(M.shape[1])
    row_perm = np.arange(M.shape[0])
    sub_M = M
    for k in xrange(M.shape[1]):
        # find column with most zeros
        sizes = np.sum(sub_M, axis=0)
        ind = k + np.argmin(sizes)

        # put it in the front and track permutation
        _swap_cols(M, k, ind)
        _swap_entries(col_perm, k, ind)

        # put all the ones in that row on top
        for i in xrange(M.shape[0]-sub_M.shape[0], M.shape[0]):
            if M[i,k] == 0:
                for j in xrange(i+1,M.shape[0]):
                    if M[j,k] == 1:
                        _swap_rows(M, i, j)
                        _swap_entries(row_perm, i, j)
        # use sub matrix below the lowest one for next iteration
        lowest_one = np.max(np.argwhere(M[:,k] == 1))
        sub_M = M[lowest_one+1:,k+1:]

    if return_perm:
        return (M, row_perm, col_perm)
    else:
        return M

def find_cuts(Matrix, i_cuts=2, j_cuts=2):
    """

    Function finding the apropriate cuts separating the l-states in the adjacency matrix.

    Returns
    -------

    cuts : tuple
        Optimal cuts.


    Parameters
    ----------

    Matrix : array-like
        Input matrix.
    i_cuts : int (optional)
        Number of cuts on 0 axis. (default=0)
    j_cuts : int (optional)
        Number of cuts on 1 axis. (default=0)

    """
    # all possible cuts on 0- and 1-axis.
    no_i = i_cuts
    i_tuples = [x for x in itertools.product(np.arange(Matrix.shape[0]), repeat=no_i) if all([x[i] < x[i+1] for i in xrange(len(x)-1)])]
    no_j = j_cuts
    j_tuples = [x for x in itertools.product(np.arange(Matrix.shape[1]), repeat=no_j) if all([x[i] < x[i+1] for i in xrange(len(x)-1)])]

    # all possible cuts
    all_tuples = list(itertools.product(i_tuples, j_tuples))

    # determine optimal cut
    zero_count = np.zeros(len(all_tuples))
    for k, cuts in enumerate(all_tuples):
        # split sub matrices after cutting
        matrices_i_cut = np.split(Matrix,np.sort(np.array(cuts[0])))
        sub_matrices = []
        for mat in matrices_i_cut:
            matrices_j_cut = np.split(mat, np.sort(np.array(cuts[1])), axis=1)
            sub_matrices.append(matrices_j_cut)

        # check if non-zero values are at correct places
        nonzero_error = False
        zero_count_temp = 0
        for i,j in itertools.product(xrange(no_i+1), xrange(no_j+1)):
            if (not (i==j or i==j-1)):
                if np.sum(sub_matrices[i][j])!=0:
                    nonzero_error = True
                else:
                    # if so, count zeros
                    zero_count_temp += (sub_matrices[i][j]).size
        # store nans for non-zero values in wrong sub-matrices
        if nonzero_error:
            zero_count[k] = np.nan
        else:
            zero_count[k] = zero_count_temp
    # return optimal cut
    return all_tuples[np.nanargmax(zero_count)]

def spectral_bipartivity_edges(G):
    """

    Function calculating the spectral bipartivity of all edges in a graph using as defined by Estrada et.al. For more details see https://doi.org/10.1103/PhysRevE.72.046105 .

    Returns
    -------

    edge_bipartivity : dict
        Dictionary containing the bipartivity (values) for all edges (keys).


    Parameters
    ----------

    G : NetworkX graph

    """
    #bip = {edge: (1.0 - nx.bipartite.spectral_bipartivity((G.copy()).remove_edge(*edge)) + nx.bipartite.spectral_bipartivity(G)) for edge in G.edges()}
    edge_bipartivity = {}
    b = nx.bipartite.spectral_bipartivity(G)
    G2 = G.copy()
    for edge in G.edges_iter():
        G2.remove_edge(*edge)
        a = nx.bipartite.spectral_bipartivity(G2)
        G2.add_edge(*edge)
        diff = a - b
        value = 1.0 - diff
        edge_bipartivity[edge] = value
    return edge_bipartivity

# Not yet tested!
def cut_leaves(G):
    """

    Function cutting all nodes with degree 1 or less from a NetworkX graph.


    Parameters
    ----------

    G : NetworkX graph

    """
    ctr = 0
    while True:
        ctr += 1
        degrees = G.degree()
        if (0 or 1) in degrees.values():
            leaves = { x for x in G.nodes() if degrees[x] <= 1}
            for node in leaves:
                G.remove_node(node)
        else:
            break

# Cut poorly connected nodes: degree much smaller than their neighbors
def cut_quasi_leaves(G, threshold=0.3, iterations=2 , weight_exp=1.0):
    for i in range(iterations):
        degree = nx.degree(G)
        for node in G.nodes():
            deg_node = degree[node]
            if deg_node == 0:
                G.remove_node(node)
            else:
                deg_neighbor = (np.sum([(degree[neighbor])**weight_exp for neighbor in G.neighbors(node)])/deg_node)**(1.0/weight_exp)
                value = deg_node/deg_neighbor
                if value<threshold:
                    G.remove_node(node)

def spectroscopic_network(ion_type, line_data_path='../data/ASD54_lines.csv', level_data_path='../data/ASD54_levels.csv', weighted=True, check_accur=False, check_wl=False, check_obs_wl=False, check_calc_wl=False, dictionaries=False, alt_read_in=False):
    """

    Function loading a graph from line and level data stored in .csv files.

    Returns
    -------

    G : NetworkX graph
        If dictionaries==False.
    (G, elevel, term, j) : tuple
        If dictionaries==True.


    Parameters
    ----------

    ion_type : str
        String specifying the ion for which thdatae data is to be loaded. The string needs to contain both the element name as well as the ionisation state sc in one combined string; e.g. 'H1.0' for hydrogen.
    line_data_path : str (optional)
        Path to the line data file.(default='../data/ASD54_lines.csv')
    level_data_path : str (optional)
        Path to the line data file.(default='../data/ASD54_levels.csv')
    weighted : bool (optional)
        Whether or not to include weights. Both intensity and wavelength are added as weights to the NetworkX graph. (default=True)
    check_accur : bool (optional)
        Check accuracies of intensities and only take lines with 'D' or better. (default=False)
    check_wl : bool (optional)
        Filter out transitions where no wavelength value (calculated or observed) is given. (default=False)
    check_obs_wl : bool (optional)
        Filter out transitions with empty observed wavelength value. (default=False)
    check_calc_wl : bool (optional)
        Filter out transitions with empty calculated wavelength value. (default=False)
    dictionaries : bool (optional)
        If true, dictionaries containing the energylevels, j, and terms are returned as well. (default=False)
    alt_read_in : bool (optional)
        If True, uses an alternative way to read in the data file (hardcoded column number). (default=False)

    Notes
    -----

    One should only use the checks that are actually necessary for the application at hand, as any additional check may decrease the number of available lines in the graph significantly.
    The n number attribute works only for ions with one electron.

    """
    ### Get the Data of Lines
    # if csv reader does not work, use alternative read in mode
    if alt_read_in==True:
        with open(line_data_path) as line_file:
            line_reader = csv.reader(line_file)
            line_data_temp = np.array(list(line_reader))
            row_count = len(line_data_temp)

        # create empty 2d array and fill it with the data. 33 is the number of columns *hardcoded!
        line_data = np.empty((row_count,33),dtype='|S64')
        for i in range(len(line_data_temp)):
           for k in range(33):
               line_data[i,k] = line_data_temp[i][k]
    else:
        # use csv reader
        with open(line_data_path) as line_file:
            line_reader = csv.reader(line_file)
            line_data = np.array(list(line_reader))
            row_count = len(line_data)

    # types of atoms atoms[i] gives atom+ionization
    atoms   = np.empty(row_count-1, dtype='|S8')
    for i,row in enumerate(line_data[1:]):
        atoms[i] = row[0]+row[1]

    # Index that is true for the correct atom and false otherwise
    atom_index = np.concatenate(([False],(atoms == ion_type)))

    # if there are no lines in the data return empty graph
    if np.all(np.logical_not(atom_index)):
        G = nx.empty_graph()
        if dictionaries:
            return (G, None, None, None)
        else:
            return G

    # lower and upper levels, intensities and frequencies of transitions
    low_levels = line_data[atom_index,25]
    upp_levels = line_data[atom_index,26]
    trans_type = line_data[atom_index,22] # transition type
    theo_ref   = line_data[atom_index,23] # theoretical reference (sometimes books etc.)
    exp_ref    = line_data[atom_index,24] # measurement reference (sometimes books etc.)
    if weighted:
        intensity  = line_data[atom_index,17] # The Einstein A coefficient
        obs_wl     = line_data[atom_index,2]  # observed wavelength
        calc_wl    = line_data[atom_index,5]  # calculated wavelength

        """
        Combine calculated and observed wavelength. If both are available for a given line, choose observed.
        """
        wavelength = np.asarray([obs_wl[i] if not obs_wl[i]=='' else calc_wl[i] for i in xrange(len(obs_wl))])

    # Delete edges with no intensity information if Graph should be weighted
    if weighted:
        if check_accur:
            accur_list = np.array(['AAA','AA','A+','A','B+','B','C+','C','D+','D']) #These accuracies for the intensity are ok
            accur      = np.array(line_data[atom_index,21])
            accur_ok   = np.in1d(accur, accur_list)
            low_levels = low_levels[accur_ok]
            upp_levels = upp_levels[accur_ok]
            intensity  = intensity[accur_ok]
            wavelength = wavelength[accur_ok]
            obs_wl     = obs_wl[accur_ok]
            calc_wl    = calc_wl[accur_ok]
        if check_obs_wl:
            obs_wl_ok  = (obs_wl != '')
            low_levels = low_levels[obs_wl_ok]
            upp_levels = upp_levels[obs_wl_ok]
            intensity  = intensity[obs_wl_ok]
            wavelength = wavelength[obs_wl_ok]
            obs_wl     = obs_wl[obs_wl_ok]
            calc_wl    = calc_wl[obs_wl_ok]
        if check_calc_wl:
            calc_wl_ok = (calc_wl != '')
            low_levels = low_levels[calc_wl_ok]
            upp_levels = upp_levels[calc_wl_ok]
            intensity  = intensity[calc_wl_ok]
            wavelength = wavelength[calc_wl_ok]
            obs_wl     = obs_wl[calc_wl_ok]
            calc_wl    = calc_wl[calc_wl_ok]
        if check_wl:
            wl_ok = (wavelength != '')
            low_levels = low_levels[wl_ok]
            upp_levels = upp_levels[wl_ok]
            intensity  = intensity[wl_ok]
            wavelength = wavelength[wl_ok]
            obs_wl     = obs_wl[wl_ok]
            calc_wl    = calc_wl[wl_ok]

        # convert empty entries to NaN's
        if weighted:
            intensity[intensity == '']      = 'NaN'
            wavelength[wavelength == '']    = 'NaN'
            obs_wl[obs_wl == '']            = 'NaN'
            calc_wl[calc_wl == '']          = 'NaN'
            intensity  = intensity.astype(float)
            wavelength = wavelength.astype(float)
            obs_wl     = obs_wl.astype(float)
            calc_wl    = calc_wl.astype(float)

    # Create the Graph
    G       = nx.Graph()
    G.name  = ion_type
    if weighted:
        G.add_weighted_edges_from(zip(low_levels, upp_levels,intensity), weight = 'intensity')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,(1e-8)*intensity*wavelength**3.0), weight = 'matrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.log((1e-8)*intensity*wavelength**3.0)), weight = 'logarithmicMatrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,(-1e-8)*intensity*wavelength**3.0), weight = 'negativeMatrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,(-1)*np.log((1e-8)*intensity*wavelength**3.0)), weight = 'negativeLogarithmicMatrixElement')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(intensity)), weight = 'inverseIntensity')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,wavelength), weight = 'wavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(wavelength)), weight = 'inverseWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,obs_wl), weight = 'observedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(obs_wl)), weight = 'inverseObservedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,calc_wl), weight = 'calculatedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(calc_wl)), weight = 'inverseCalculatedWavelength')
    else:
        G.add_edges_from(zip(low_levels,upp_levels))

    # set edge attributes
    nx.set_edge_attributes(G, name='type', values=dict(zip(zip(low_levels, upp_levels),trans_type)))
    nx.set_edge_attributes(G, name='expref', values=dict(zip(zip(low_levels, upp_levels),exp_ref)))
    nx.set_edge_attributes(G, name='theo_ref', values=dict(zip(zip(low_levels, upp_levels),theo_ref)))

    ### Characterization of nodes
    with open(level_data_path) as level_file:
        level_reader = csv.reader(level_file)
        level_data = np.array(list(level_reader))
        j = dict()
        conf = dict()
        term = dict()
        parity = dict()
        n_dict = dict()
        l = dict()
        modelName = dict()
        dummyCommunity = dict()
        energy = dict()
        # leading_configurations = dict()
        ### dictionary for l-states, used in calculation of modelName
        L_dictionary = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4, 'h':5, 'i':6, 'k':7, 'l':8, 'm':9, 'n':10, 'o':11, 'q':12, 'r':13, 't':14, 'u':15, 'v':16, 'w':17, 'x':18, 'y':19, 'z':20, 'a':21, 'b':22, 'c':23, 'e':24, 'j':25}

        for n in nx.nodes(G):

            # total J
            j_value = (level_data[level_data[:,24] == n,8])
            if j_value.size == 0:
                j[n] = ''
            else:
                j[n] = j_value[0]

            # energy
            energy_value = (level_data[level_data[:,24] == n,10])
            if energy_value.size == 0:
                energy[n] = ''
            elif energy_value[0] == '':
                energy[n] = ''
            else:
                energy[n] = float(energy_value[0])

            # configuration
            conf_value = (level_data[level_data[:,24] == n,5])
            if conf_value.size == 0:
                conf[n] = ''
            else:
                conf[n] = conf_value[0]

            # # leading configurations
            # j_value = (level_data[level_data[:,24] == n,8])
            # if j_value.size == 0:
            #     j[n] = ''
            # else:
            #     j[n] = j_value[0]

            # n
            # only makes sense for H1.0
            n_dict[n] = conf[n][:-1]

            # term [2S+1]LJ
            try:
                term[n] = (level_data[level_data[:,24] == n,6])[0]+j[n]
                term[n] = term[n].replace('*', '')
            except:
                term[n] = ''

            # l
            try:
                for i, char in enumerate(term[n]):
                    if not char.isdigit():
                        index = i
                        break
                l[n] = term[n][index]
            except:
                l[n] = ''

            # parity
            parity_value = (level_data[level_data[:,24] == n ,7])
            if parity_value.size == 0:
                parity[n] = ''
            else:
                parity[n] = parity_value[0]


            # only works for 1 electron ions
            try:
                if int(j[n][:-2]) == 2*L_dictionary[l[n].lower()] + 1:
                    upper_true = True
                    lower_true = False
                else:
                    upper_true = False
                    lower_true = True

                modelName[n] = n_dict[n] + upper_true*l[n] + lower_true*l[n].lower()
            except:
                modelName[n] = ''

            # dummy community for commonneighbours link prediction algorithm, which needs a community
            dummyCommunity[n] = 0


    # Give attributes to nodes
    nx.set_node_attributes(G,name='conf',values=conf)
    nx.set_node_attributes(G,name='term',values=term)
    nx.set_node_attributes(G,name='J',values=j)
    nx.set_node_attributes(G,name='n',values=n_dict)
    nx.set_node_attributes(G,name='l',values=l)
    nx.set_node_attributes(G,name='parity',values=parity)
    nx.set_node_attributes(G,name='modelName',values=modelName)
    nx.set_node_attributes(G,name='dummyCommunity',values=dummyCommunity)
    nx.set_node_attributes(G,name='energy',values=energy)

    if dictionaries:
        return (G, conf, term, j)
    else:
        return G

def _create_ion_list(line_data_path='../Data/ASD54_lines.csv', return_string=False):
    """
    Function creating a list of all ions in the data.

    Returns
    -------
    ions : array-like
        If return_string==False.
    ions : str
        If return_string==True.

    Parameters
    ----------
    line_data_path : str (optional)
        Path to the line data file.(default='../Data/ASD54_lines.csv')

    """
    ### Get the Data of Lines
    with open(line_data_path) as line_file:
        line_reader = csv.reader(line_file)
        line_data = np.array(list(line_reader))
        row_count = len(line_data)

        # types of atoms atoms[i] gives atom+ionization
        atoms   = np.empty(row_count-1, dtype='|S8')
        for i,row in enumerate(line_data[1:]):
            atoms[i] = row[0]+row[1]

        ions = np.unique(atoms)
        if return_string:
            return np.array2string(ions, separator=',', max_line_width=np.inf )
        else:
            return ions

def _create_n_electron_list(n,level_data_path='../Data/ASD54_levels.csv', return_string=False, order_by_Z=True):
    """
    Function creating a list of all ions with n electrons.

    Returns
    -------
    ions : array-like
        If return_string==False.
    ions : str
        If return_string==True.

    Parameters
    ----------
    line_data_path : str (optional)
        Path to the line data file.(default='../Data/ASD54_lines.csv')

    """
    ### Get the Data of Lines
    with open(level_data_path) as level_file:
        level_reader = csv.reader(level_file)
        level_data = np.array(list(level_reader))

        # types of atoms atoms[i] gives atom+ionization
        atoms = np.array([level_data[i,0] for i in xrange(1,len(level_data)) if level_data[i,3]==str(float(n))])
        sc = np.array([level_data[i,1] for i in xrange(1,len(level_data)) if level_data[i,3]==str(float(n))])

        # concaatenate atoms and sc and find unique entries
        ions, ind = np.unique(np.core.defchararray.add(atoms, sc), return_index=True)

        # order by number of protons
        if order_by_Z:
            sc = sc[ind]
            ions = ions[np.argsort(sc.astype(float))]
        if return_string:
            return np.array2string(ions, separator=',', max_line_width=np.inf )
        else:
            return ions

def spectral_bipartivity_sparse(G, nodes=None, weight='weight'):
    """Returns the spectral bipartivity.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list or container  optional(default is all nodes)
      Nodes to return value of spectral bipartivity contribution.

    weight : string or None  optional (default = 'weight')
      Edge data key to use for edge weights. If None, weights set to 1.

    Returns
    -------
    sb : float or dict
       A single number if the keyword nodes is not specified, or
       a dictionary keyed by node with the spectral bipartivity contribution
       of that node as the value.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> bipartite.spectral_bipartivity(G)
    1.0

    Notes
    -----
    This implementation uses Numpy (dense) matrices which are not efficient
    for storing large sparse graphs.

    See Also
    --------
    color

    References
    ----------
    .. [1] E. Estrada and J. A. Rodríguez-Velázquez, "Spectral measures of
       bipartivity in complex networks", PhysRev E 72, 046105 (2005)
    """
    try:
        import scipy.sparse.linalg
    except ImportError:
        raise ImportError('spectral_bipartivity() requires SciPy: ',
                          'http://scipy.org/')
    nodelist = G.nodes() # ordering of nodes in matrix

    s1 = time.time()
    A = nx.to_scipy_sparse_matrix(G, nodelist, weight=weight, format='csc')
    e1 = time.time()
    print 'runtime1:', e1-s1, 's'
    s2 = time.time()
    expA = scipy.sparse.linalg.expm(A)
    e2 = time.time()
    print 'runtime2:', e2-s2, 's'
    s3 = time.time()
    expmA = scipy.sparse.linalg.expm(-A)
    e3 = time.time()
    print 'runtime3:', e3-s3, 's'
    s4 = time.time()
    coshA = 0.5 * (expA + expmA)
    e4 = time.time()
    print 'runtime4:', e4-s4, 's'
    if nodes is None:
        # return single number for entire graph
        return coshA.diagonal().sum() / expA.diagonal().sum()
    else:
        # contribution for individual nodes
        index = dict(zip(nodelist, range(len(nodelist))))
        sb = {}
        for n in nodes:
            i = index[n]
            sb[n] = coshA[i, i] / expA[i, i]
        return sb

def spectral_bipartivity(G, nodes=None, weight='weight'):
    """Returns the spectral bipartivity.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list or container  optional(default is all nodes)
      Nodes to return value of spectral bipartivity contribution.

    weight : string or None  optional (default = 'weight')
      Edge data key to use for edge weights. If None, weights set to 1.

    Returns
    -------
    sb : float or dict
       A single number if the keyword nodes is not specified, or
       a dictionary keyed by node with the spectral bipartivity contribution
       of that node as the value.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> bipartite.spectral_bipartivity(G)
    1.0

    Notes
    -----
    This implementation uses Numpy (dense) matrices which are not efficient
    for storing large sparse graphs.

    See Also
    --------
    color

    References
    ----------
    .. [1] E. Estrada and J. A. Rodríguez-Velázquez, "Spectral measures of
       bipartivity in complex networks", PhysRev E 72, 046105 (2005)
    """
    try:
        import scipy.linalg
    except ImportError:
        raise ImportError('spectral_bipartivity() requires SciPy: ',
                          'http://scipy.org/')
    nodelist = G.nodes() # ordering of nodes in matrix

    s1 = time.time()
    A = nx.to_numpy_matrix(G, nodelist, weight=weight)
    e1 = time.time()
    print 'runtime1:', e1-s1, 's'
    s2 = time.time()
    expA = scipy.linalg.expm(A)
    e2 = time.time()
    print 'runtime2:', e2-s2, 's'
    s3 = time.time()
    expmA = scipy.linalg.expm(-A)
    e3 = time.time()
    print 'runtime3:', e3-s3, 's'
    s4 = time.time()
    coshA = 0.5 * (expA + expmA)
    e4 = time.time()
    print 'runtime4:', e4-s4, 's'
    if nodes is None:
        # return single number for entire graph
        return coshA.diagonal().sum() / expA.diagonal().sum()
    else:
        # contribution for individual nodes
        index = dict(zip(nodelist, range(len(nodelist))))
        sb = {}
        for n in nodes:
            i = index[n]
            sb[n] = coshA[i, i] / expA[i, i]
        return sb

def random_imperfect_bipartite(n, l, p, q, seed=None, directed=False, label_imperfections=True):
    """
    Return a random graph G_{n,l,p,q} where n and l are the sizes of the partitions, p is the probability for each possible edge to exist between them and p is the probability for each possible edge to exist among them.

    Parameters
    ----------
    n : int
        The number of nodes in partitition 1.
    l : int
        The number of nodes in partitition 2.
    p : float
        Probability for edge creation between partitions.
    q : float
        Probability for edge creation among partitions.
    seed : int, optional
        Seed for random number generator (default=None).
    directed : bool, optional (default=False)
        If True return a directed graph
    label_imperfections : bool, optional (default=True)
        If True add edge labels indicating the edges between partitions.

    """
    assert(0.0<=p<=1.0)
    assert(0.0<=q<=1.0)

    if directed:
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_nodes_from(range(n+l))
    G.name="gnlpq(%s,%s)"%(n,p)
#    if p<=0:
#        return G
#    if p>=1:
#        return complete_graph(n,create_using=G)

    if not seed is None:
        random.seed(seed)

    # edges between partititons
    if G.is_directed():
        inter_edges=itertools.chain(itertools.product(range(n), range(n,n+l)), itertools.product(range(n,n+l), range(n)) )
    else:
        inter_edges=itertools.product(range(n),range(n,n+l))

    for e in inter_edges:
        if random.random() < p:
            G.add_edge(*e)

    if label_imperfections:
        imperfection_list = []
    # edges in first partition
    if G.is_directed():
        intra_edges_1=itertools.permutations(range(n),2)
    else:
        intra_edges_1=itertools.combinations(range(n),2)

    for e in intra_edges_1:
        if random.random() < q:
            G.add_edge(*e)
            if label_imperfections:
                imperfection_list.append(e)

    # edges in second partition
    if G.is_directed():
        intra_edges_2=itertools.permutations(range(n,n+l),2)
    else:
        intra_edges_2=itertools.combinations(range(n,n+l),2)

    for e in intra_edges_2:
        if random.random() < q:
            G.add_edge(*e)
            if label_imperfections:
                imperfection_list.append(e)
    if label_imperfections:
        imperfections = {}
        for e in G.edges_iter():
            if e in imperfection_list:
                imperfections[e] = True
            else:
                imperfections[e] = False
        nx.set_edge_attributes(G, 'imperfection', imperfections)
    return G

def find_non_bipartite_edges(G):
    """
    Return edges of G which prevent it from being bipartite by iteratively eliminating the edge with the least spectral bipartivity.

    Returns
    -------
    non_bipartite_edges : array-like

    Parameters
    ----------
    G : NetworkX graph
    """
    Gc = G.copy()
    non_bipartite_edges = []
    while not nx.bipartite.is_bipartite(Gc):
        # measure bipartivity
        bip_edges = spectral_bipartivity_edges(Gc)

        # remove worst edge
        worst_edge = min(bip_edges, key=bip_edges.get)
        Gc.remove_edge(*worst_edge)
        non_bipartite_edges.append(worst_edge)
    return non_bipartite_edges

def degree_distr(G):
    """
    Return degree distribution of undirected graph.

    Returns
    -------
    distr : dict

    Parameters
    ----------
    G : NetworkX graph
    """
    import collections
    degree_sequence = np.array(sorted([d for d in G.degree().values()], reverse=True)) # degree sequence
    hist, bin_edges = np.histogram(degree_sequence)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    distr = dict(zip(deg,cnt))
    return distr

def only_largest_component(G):
    no_conn = nx.number_connected_components(G)
    if no_conn==1:
        return G
    else:
        Gc = G.copy()
        components = sorted(nx.connected_components(Gc), key = len, reverse=True)
        # take only largest component
        for comp in components[1:]:
            Gc.remove_nodes_from(comp)
        return Gc

def Newman_community_detection(G, weight='weight', refinement_step=False):
    from scipy.sparse import linalg
    if len(G) == 0:
        raise nx.NetworkXException('Empty graph.')

    # create modularity matrix from adjacency and null model
    A = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight=weight,
                                  dtype=float)
    degrees = np.array(G.degree().values())
    m = G.number_of_edges()
    P = np.outer(degrees, degrees)/float(2*m)
    B = A - P

    # calculate eigenvectors with most negative eigenvalue
    eigenvalues, eigenvectors = linalg.eigs(B.T, k=3, which='SR')
    smallest = eigenvectors[:,0].flatten().real

    # get index vector
    s = smallest
    s[s >= 0] = 1
    s[s < 0]  = -1
    s1 = np.argwhere(s>=0).flatten()
    s2 = np.argwhere(s<0).flatten()

    # calculate modularity
    Q = (0.25/m)*np.dot(s, np.dot(B,s).T)[0,0]
    print 'Q before refinement', Q

    if refinement_step:
        group1_size = np.sum(s>=0)
        group2_size = np.sum(s<0)

        # list of index vectors and corresponding Q's
        s_list = np.zeros((len(s),min(group1_size, group2_size)+1))
        Q_list = np.zeros(min(group1_size, group2_size)+1)

        s_list[:,0] = s
        Q_list[0]   = Q

        s_test = np.copy(s)

        # find swap with minimal Q after swap
        for i in xrange(min(group1_size, group2_size)):
            min_swap_nodes = []
            min_Q = np.inf

            # check all pairs and choose the one with lowest Q after swap
            for n1, n2 in itertools.product(s1, s2):
#                print 's_test\n', s_test
#                print 'n1', n1
#                print 'n2', n2
#                print 's1\n', s1
                _swap_entries(s_test, n1, n2) # swap nodes in index vector
                Q_swap = (0.25/m)*np.dot(s_test, np.dot(B,s_test).T)
                if Q_swap < min_Q:
                    min_swap_nodes = [n1, n2]
                    min_Q = Q_swap
                else:
                    _swap_entries(s_test, n1, n2) # swap back

            # remove best pair from set of possible pairs
            s1 = s1[s1!=min_swap_nodes[0]]
            s2 = s2[s2!=min_swap_nodes[1]]

            # add index vector and Q to lists
            s_list[:,i+1] = s_test
            Q_list[i+1]   = min_Q

        print 's_list\n', s_list
        print 'Q_list\n', Q_list

        # choose s corresponding to lowest Q
        s = s_list[:,np.argmin(Q_list)]
        print 'Q before refinement', Q
        print 'Q after refinement', np.min(Q_list)

    # extract groups
    nodelist = np.array(G.nodes())
    group1 = nodelist[s >= 0]
    group2 = nodelist[s < 0]
#    print 'smallest', smallest
    print 'eigenvalues', eigenvalues
#    print 'nodes', G.nodes()
    print 'group1\n', group1
    print 'group2\n', group2
    return [group1, group2]


def model_network(Z=1, datafolder='../data', set1=True, set2=True, max_n=None, E1=False, E2=False, E3=False, M1=False, M2=False, M3=False):
    """
    Return model network for hydrogen.

    Returns
    -------
    G : NetworkXgraph

    Parameters
    ----------
    datafolder : str, optional
        Folder containing the datafiles. The files must be called "jitrik-bunge-e1-set1.csv", "jitrik-bunge-e1-set2.csv", "jitrik-bunge-e2-set1.csv", "jitrik-bunge-e2-set2.csv", "jitrik-bunge-m1-set1.csv" and "jitrik-bunge-m1-set2.csv". (default='../data')
    set1 : bool, optional
        Include data from set 1. (default=True)
    set2 : bool, optional
        Include data from set 2. (default=True)
    max_n : int or None, optional
        Maximum value for principal quantum number n to keep in network. (default=None)
    E1 : bool, optional
        Include electric dipole transitions. (default=False)
    E2 : bool, optional
        Include electric quadrupole transitions. (default=False)
    E3 : bool, optional
        Include electric octupole transitions. (default=False)
    M1 : bool, optional
        Include magnetic dipole transitions. (default=False)
    M2 : bool, optional
        Include magnetic quadrupole transitions. (default=False)
    M3 : bool, optional
        Include magnetic octupole transitions. (default=False)

    Notes
    -----
    Still missing 'conf' node attributes as in spectroscopic_network.o

    To-Do
    -----
    Check scaling factors (Wellnitz)!!!!
    Check Dipole Element rescaling.
    """
    E1_M2 = False
    if E1 and M2:
        E1_M2 = True

    M2_E3 = False
    if M2 and E3:
        M2_E3 = True

    M1_E2 = False
    if M1 and E2:
        M1_E2 = True

    E2_M3 = False
    if E2 and M3:
        E2_M3 = True

    E1_M2_E3 = False
    if E1 and M2 and E3:
        E1_M2_E3 = True

    M1_E2_M3 = False
    if M1 and E2 and M3:
        M1_E2_M3 = True

    G = nx.Graph()
    G.name = 'th. H until n = ' + str(max_n)

    datapath_E1_set1 = datafolder + '/jitrik_bunge_data/set1/E1_Aki_Z_' + str(Z) + '.csv'
    datapath_E1_set2 = datafolder + '/jitrik_bunge_data/set2/E1_Aki_Z_' + str(Z) + '.csv'
    datapath_E2_set1 = datafolder + '/jitrik_bunge_data/set1/E2_Aki_Z_' + str(Z) + '.csv'
    datapath_E2_set2 = datafolder + '/jitrik_bunge_data/set2/E2_Aki_Z_' + str(Z) + '.csv'
    datapath_E3_set1 = datafolder + '/jitrik_bunge_data/set1/E3_Aki_Z_' + str(Z) + '.csv'
    datapath_E3_set2 = datafolder + '/jitrik_bunge_data/set2/E3_Aki_Z_' + str(Z) + '.csv'
    datapath_M1_set1 = datafolder + '/jitrik_bunge_data/set1/M1_Aki_Z_' + str(Z) + '.csv'
    datapath_M1_set2 = datafolder + '/jitrik_bunge_data/set2/M1_Aki_Z_' + str(Z) + '.csv'
    datapath_M2_set1 = datafolder + '/jitrik_bunge_data/set1/M2_Aki_Z_' + str(Z) + '.csv'
    datapath_M2_set2 = datafolder + '/jitrik_bunge_data/set2/M2_Aki_Z_' + str(Z) + '.csv'
    datapath_M3_set1 = datafolder + '/jitrik_bunge_data/set1/M3_Aki_Z_' + str(Z) + '.csv'
    datapath_M3_set2 = datafolder + '/jitrik_bunge_data/set2/M3_Aki_Z_' + str(Z) + '.csv'

    datapath_E1_M2_set1 = datafolder + '/jitrik_bunge_data/set1/E1_M2_Aki_Z_' + str(Z) + '.csv'
    datapath_E1_M2_set2 = datafolder + '/jitrik_bunge_data/set2/E1_M2_Aki_Z_' + str(Z) + '.csv'
    datapath_M2_E3_set1 = datafolder + '/jitrik_bunge_data/set1/M2_E3_Aki_Z_' + str(Z) + '.csv'
    datapath_M2_E3_set2 = datafolder + '/jitrik_bunge_data/set2/M2_E3_Aki_Z_' + str(Z) + '.csv'
    datapath_M1_E2_set1 = datafolder + '/jitrik_bunge_data/set1/M1_E2_Aki_Z_' + str(Z) + '.csv'
    datapath_M1_E2_set2 = datafolder + '/jitrik_bunge_data/set2/M1_E2_Aki_Z_' + str(Z) + '.csv'
    datapath_E2_M3_set1 = datafolder + '/jitrik_bunge_data/set1/E2_M3_Aki_Z_' + str(Z) + '.csv'
    datapath_E2_M3_set2 = datafolder + '/jitrik_bunge_data/set2/E2_M3_Aki_Z_' + str(Z) + '.csv'

    datapath_E1_M2_E3_set1 = datafolder + '/jitrik_bunge_data/set1/E1_M2_E3_Aki_Z_' + str(Z) + '.csv'
    datapath_E1_M2_E3_set2 = datafolder + '/jitrik_bunge_data/set2/E1_M2_E3_Aki_Z_' + str(Z) + '.csv'
    datapath_M1_E2_M3_set1 = datafolder + '/jitrik_bunge_data/set1/M1_E2_M3_Aki_Z_' + str(Z) + '.csv'
    datapath_M1_E2_M3_set2 = datafolder + '/jitrik_bunge_data/set2/M1_E2_M3_Aki_Z_' + str(Z) + '.csv'

    paths_set1  = (   E1*[datapath_E1_set1]
                    + E2*[datapath_E2_set1]
                    + E3*[datapath_E3_set1]
                    + M1*[datapath_M1_set1]
                    + M2*[datapath_M2_set1]
                    + M3*[datapath_M3_set1]
                    + E1_M2*[datapath_E1_M2_set1]
                    + M2_E3*[datapath_M2_E3_set1]
                    + M1_E2*[datapath_M1_E2_set1]
                    + E2_M3*[datapath_E2_M3_set1]
                    + E1_M2_E3*[datapath_E1_M2_E3_set1]
                    + M1_E2_M3*[datapath_M1_E2_M3_set1] )
    paths_set2  = (   E1*[datapath_E1_set2]
                    + E2*[datapath_E2_set2]
                    + E3*[datapath_E3_set2]
                    + M1*[datapath_M1_set2]
                    + M2*[datapath_M2_set2]
                    + M3*[datapath_M3_set2]
                    + E1_M2*[datapath_E1_M2_set2]
                    + M2_E3*[datapath_M2_E3_set2]
                    + M1_E2*[datapath_M1_E2_set2]
                    + E2_M3*[datapath_E2_M3_set2]
                    + E1_M2_E3*[datapath_E1_M2_E3_set2]
                    + M1_E2_M3*[datapath_M1_E2_M3_set2] )
    # transition label
    labels_set1 = (   E1*['E1']
                    + E2*['E2']
                    + E3*['E3']
                    + M1*['M1']
                    + M2*['M2']
                    + M3*['M3']
                    + E1_M2*['E1_M2*']
                    + M2_E3*['M2_E3']
                    + M1_E2*['M1_E2']
                    + E2_M3*['E2_M3']
                    + E1_M2_E3*['E1_M2_E3']
                    + M1_E2_M3*['M1_E2_M3'] )
    labels_set2 = (   E1*['E1']
                    + E2*['E2']
                    + E3*['E3']
                    + M1*['M1']
                    + M2*['M2']
                    + M3*['M3']
                    + E1_M2*['E1_M2*']
                    + M2_E3*['M2_E3']
                    + M1_E2*['M1_E2']
                    + E2_M3*['E2_M3']
                    + E1_M2_E3*['E1_M2_E3']
                    + M1_E2_M3*['M1_E2_M3'] )
    # Einstein A muliplication factor
    """
    All files including E1 lines have a factor of 1.0
    """
    factors_set1    = (   E1*[1.0]
                        + E2*[1.0e-8]
                        + E3*[1.0e-8]
                        + M1*[1.0e-8]
                        + M2*[1.0e-8]
                        + M3*[1.0e-8]
                        + E1_M2*[1.0]
                        + M2_E3*[1.0e-8]
                        + M1_E2*[1.0e-8]
                        + E2_M3*[1.0e-8]
                        + E1_M2_E3*[1.0]
                        + M1_E2_M3*[1.0e-8] )
    factors_set2    = factors_set1

    # gain weights (for evaluation)
    # TODO: find better weight factors
    gain_weights_set1 = (   E1*[6.0]
                          + E2*[5.0]
                          + E3*[4.0]
                          + M1*[5.0]
                          + M2*[4.0]
                          + M3*[3.0]
                          + E1_M2*[5.0]
                          + M2_E3*[4.0]
                          + M1_E2*[3.5]
                          + E2_M3*[4.0]
                          + E1_M2_E3*[4.0]
                          + M1_E2_M3*[2.0] )
    gain_weights_set2 = gain_weights_set1

    paths   = set1*paths_set1   + set2*paths_set2
    labels  = set1*labels_set1  + set2*labels_set2
    factors = set1*factors_set1 + set2*factors_set2
    gain_weights = set1*gain_weights_set1 + set2*gain_weights_set2

    l_even  = ['s', 'd', 'g', 'i', 'l', 'n', 'q', 't', 'v', 'x', 'z', 'b', 'e']
    l_odd   = ['p', 'f', 'h', 'k', 'm', 'o', 'r', 'u', 'w', 'y', 'a', 'c', 'j']
    l_all   = np.asarray(['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'e', 'j'])

    matrix_element_minimum = 1
    matrix_element_maximum = 1
    for j in xrange(len(paths)):
        ### Get the Data of Lines
        with open(paths[j]) as line_file:
            line_reader = csv.reader(line_file)
            reader = np.array(list(line_reader))

            # find end of header
            end_found = False
            k = 0
            data_start = 0
            while not end_found:
                split = reader[k][0].split()
                if not split:
                    k += 1
                elif split[:1][0] == 'No.':
                    data_start = k + 1
                    end_found = True
                else:
                    k += 1

            line_data_dummy = reader[data_start:]
            row_count = len(line_data_dummy)

        # split all lines on whitespaces
        line_data = []
        for line in line_data_dummy:
            split = line[0].split()

            # combine upper and lower state of transition if they were separated
            # i.e. if split[2] starts with a minus
            if split[2][0] == '-':
                split[1] = split[1] + split[2]
                del split[2]
            line_data.append(split)

        ### Line data into lists
        trans = [line_data[i][1] for i in range(row_count)]
        # Einstein A matrix
        EinA = [line_data[i][7] for i in range(row_count)]
        for i in range(len(EinA)):
            if EinA[i][-1] == '*':
                EinA[i] = EinA[i][:-1]
        EinA = np.array(EinA, dtype = 'float')*factors[j]

        # wavelength
        wavelength = np.array([line_data[i][2] for i in range(row_count)],dtype = float)

        # Lower and upper levels
        low_level = np.empty(row_count, dtype='|S8')
        up_level  = np.empty(row_count, dtype='|S8')

        # Split transitions into two levels
        for i in range(row_count):
            low_level[i], up_level[i] = trans[i].split('-')

        # A/omega^3
        DipoleElement = EinA*wavelength**3

        # get minimum of matrix elements (throughout all sets)
        if np.amin(np.log(DipoleElement)) < matrix_element_minimum:
            matrix_element_minimum = np.amin(np.log(DipoleElement))
        # get maximum of matrix elements (throughout all sets)
        if np.amax(np.log(DipoleElement)) > matrix_element_maximum:
            matrix_element_maximum = np.amin(np.log(DipoleElement))

        # gain weights
        gain_weights_arr = np.ones_like(EinA, dtype = 'float')*gain_weights[j]

        # Add edges to Graph
        G.add_weighted_edges_from(zip(low_level, up_level,EinA), weight = 'EinsteinA')
        G.add_weighted_edges_from(zip(low_level, up_level,wavelength), weight = 'wavelength')
        G.add_weighted_edges_from(zip(low_level, up_level,DipoleElement), weight = 'dipoleElement'+labels[j], label=labels[j])
        G.add_weighted_edges_from(zip(low_level, up_level,DipoleElement), weight = 'matrixElement')
        G.add_weighted_edges_from(zip(low_level, up_level,np.log(DipoleElement)), weight = 'logarithmicMatrixElement')
        G.add_weighted_edges_from(zip(low_level, up_level,(-1)*DipoleElement), weight = 'negativeMatrixElement')
        G.add_weighted_edges_from(zip(low_level, up_level,(-1)*np.log(DipoleElement)), weight = 'negativeLogarithmicMatrixElement')
        G.add_weighted_edges_from(zip(low_level, up_level,gain_weights_arr), weight = 'gainValue')

        transitionType        = {}
        rescaledMatrixElement = {}
        for ll, ul in zip(low_level, up_level):
            transitionType[(ll,ul)] = labels[j]
            # TODO: find correct rescaling
            rescaledMatrixElement[(ll,ul)] = np.divide( ( G[ll][ul]['logarithmicMatrixElement'] - matrix_element_minimum ), (matrix_element_maximum - matrix_element_minimum) )
        nx.set_edge_attributes(G, name='transitionType', values=transitionType)
        nx.set_edge_attributes(G, name='rescaledMatrixElement', values=rescaledMatrixElement) # Add rescaled matrix element weight to edges


    ## Node attributes
    levels = G.nodes()
    # parity
    parity = {}
    for level in levels:
        if level[-1].lower() in l_even:
            parity[level] = 'e'
        elif level[-1].lower() in l_odd:
            parity[level] = 'o'
        else:
            parity[level] = ''

    n = {}
    term = {}
    j = {}
    l = {}
    dummyCommunity = {}
    for level in levels:
        #n
        n[level] = "".join(itertools.takewhile(str.isdigit, level))

        #term
        t1 = '2'
        t2 = level[-1].upper()
        l_number = np.argwhere(l_all == level[-1].lower())[0,0]
        t3 = ( str(2*l_number-1) if level[-1].islower() else str(2*l_number+1) ) + '/2'
        term[level] = t1 + t2 + t3

        #j
        j[level] = t3

        # dummy community for common neighbours link prediction algorithm, which needs a community
        dummyCommunity[level] = 0

        # l
        try:
            for i, char in enumerate(term[level]):
                if not char.isdigit():
                    index = i
                    break
            l[level] = term[level][index]
        except:
            l[level] = ''



    nx.set_node_attributes(G, name='parity', values=parity)
    nx.set_node_attributes(G, name='term', values=term)
    nx.set_node_attributes(G, name='J', values=j)
    nx.set_node_attributes(G, name='n', values=n)
    nx.set_node_attributes(G, name='l', values=l)
    nx.set_node_attributes(G, name='dummyCommunity', values=dummyCommunity)

    if max_n != None:
        nodes_over_max_n = [node for (node,n_number) in n.iteritems() if int(n_number) > max_n]
        G.remove_nodes_from(nodes_over_max_n)

    return G


# Requires that for each node the strongest Dipole line is at least by a factor of e stronger than the strongest non dipole line
# goes through the logarithmic dipole elements
def only_dipole_transitions(G, return_edge_sets=False):
    """
    Returns graph only containing dipole transitions. This method iteratively cuts the edge with smallest weight until
    the graph is bipartitie, and then restores cut edges which do not violate bipartivity in reverse order.
    - VERY Inefficient

    Parameters
    ----------
    G : NetworkX graph

    return_edge_sets : bool  (optional)

    Returns
    -------
    Gc : NetworkX graph
        If return_edge_sets==False.
    (Gc, ActuallyCut, CutEdgeSet, Dipole) : tuple
         If return_edge_sets==True.

    """
    CutEdgeSet = []
    false_positive = [] #Contains edges that will be readded later
    n = min([e[2] for e in G.edges(data = 'logarithmicMatrixElement')])
    Gc = G.copy()
    # Cut all edges from the weakest to the strongest until Graph is bipartite
    while not nx.is_bipartite(Gc):
        TheoEdgeSet = Gc.edges()
        for (u,v) in TheoEdgeSet:
            if G[u][v]['logarithmicMatrixElement'] < n+1.0 and G[u][v]['logarithmicMatrixElement'] >= n:
                Gc.remove_edge(u,v)
                # Add the edge back again if the connectivity is destroyed otherwise
                if not nx.is_connected(Gc) and nx.is_connected(G):
                    print 'disconnect', u,v
                    Gc.add_weighted_edges_from([(u,v,G[u][v]['logarithmicMatrixElement'])], weight = 'logarithmicMatrixElement')
                else:
                    CutEdgeSet.append((u,v,G[u][v]))
        n+=1
    even, odd = nx.bipartite.sets(Gc)
    ActuallyCut = [] # Edges that will be left cut
    # Readd edges that are between even and odd states to the Graph
    for (u,v,w) in CutEdgeSet:
        if (u in even and v in odd) or (v in even and u in odd):
            false_positive.append((u,v,w))
        else:
            ActuallyCut.append((u,v))
    Gc.add_edges_from(false_positive)
    if return_edge_sets:
        return Gc, ActuallyCut, CutEdgeSet, false_positive
    else:
        return Gc


# Given a graph G of a and its groundstate this function creates a dictionary assigning states to their angular momentum
def find_l(G, groundstate):
    # Compute distance of node from ground state
    dist = nx.shortest_path_length(G, source = groundstate)
    # Mean distance of neighbors from ground state
    neighbordist = dict()
    for u in G.nodes():
        distsum = 0.0
        for v in G.neighbors(u):
            distsum += dist[v]
        neighbordist[u] = distsum/float(len(G.neighbors(u)))
    l = dict()
    # assume that all p-states are connected to the groundstate
    for u in G.nodes():
        if dist[u] == 2:
            if  neighbordist[u] < 1.1:
                l[u] = 0
            else:
                l[u] = 2
        else:
            l[u] = dist[u]
    return l

l_list = ['s','p','d','f','g','h','i','k','l','m','n','o','q']
color = ['r','g','b','y','m','c']
# finds the groundstate in an experimental network
def ground_state(G):
    for n in G.nodes():
        if n.split('.')[-1] == '000001':
            groundstate = n
    return groundstate

def draw_groups(G, groups, labels=None, pos=None):
    """
    Function for drawing NetworkX graphs with different node colors indicating their group membership.

    Parameters
    ----------
    G : NetworkX graph
    groups : dictionary
        {nodes: group} dictionary with nodes as keys and an integer indicating the group it belongs to. The nodes should be of the same format as in G. group should be an integer {0,...,k}, where k is the number of groups.
    labels : dictionary, optional
        Node labels in a dictionary keyed by node of text labels. (default=None)
    pos : dictionary or None (optional)
        A dictionary with nodes as keys and positions as values. If not specified a spring layout positioning will be computed. (default=None)

    Notes
    -----
    For k larger than 19, the colors might not be unique, i.e. two different groups might be assigned the same color.

    """
    import matplotlib.pyplot as plt

    n = [] # nodes
    g = [] # assigned groups
    for node, group in groups.iteritems():
        n.append(node)
        g.append(group)
    n = np.asarray(n)
    g = np.asarray(g)

    # check set of group integer values
    gmin, gmax = g.min(), g.max()
    if gmin != 0:
        warnings.warn('Minimum group assignment number is not 0, this may result in multiple groups having the same color.',
                      RuntimeWarning)

    # choose colormap
    if gmax <= 9:
        colors = (plt.get_cmap('Vega10'))(np.linspace(0.0,1.0,num=gmax+1))
    elif gmax <= 19:
        colors = (plt.get_cmap('Vega20'))(np.linspace(0.0,1.0,num=gmax+1))
    else:
        colors = (plt.get_cmap('prism'))(np.linspace(0.0,1.0,num=gmax+1))
        warnings.warn('Maximum group assignment number is larger than 19, this may result in multiple groups having the same color.',
                      RuntimeWarning)

    # set position layout
    if not pos:
        pos=nx.spring_layout(G)

    # draw network
    for i in xrange(gmax+1):
        nodelist = list(n[g==i])
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, node_color=colors[i])
    nx.draw_networkx_edges(G, pos=pos)
    if labels != None:
        nx.draw_networkx_labels(G, pos=pos, labels=labels)


def remove_attributes(G):
    """
    This function returns a version of G where all the node and edge attributes have been removed.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    G2 : NetworkX graph

    """
    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes())
    G2.add_edges_from(G.edges())
    return G2


##
## @brief      Removes nodes from network that have an empty specified
##             attribute.
##
## @param      Graph      The networkx graph
## @param      attribute  The attribute (string)
##
## @return     Graph
##
def remove_empty_levels(Graph, attribute):
    # remove nodes with empty attribute
    temp_list=nx.get_node_attributes(Graph, attribute)
    for i, (node_ID, attribute) in enumerate(temp_list.items()):
        if not attribute: #if empty string, delete entries
            Graph.remove_node(node_ID)

    return Graph


##
## @brief      Removes a node with quantum number 'n' greater than the treshold
##
## @param      Graph      The graph
## @param      max_n  The max_n
##
## @return     Graph
##
def remove_n_greater(Graph, max_n):
    # remove nodes with empty attribute
    temp_list=nx.get_node_attributes(Graph, 'n')
    if max_n != None:
        for i, (node_ID, attribute) in enumerate(temp_list.items()):
            if int(attribute) > max_n: #if n > max_n
                Graph.remove_node(node_ID)

    return Graph


def load_network(ion='H1.0', experimental=True, only_dipole=False, n_limit=False, max_n=None, alt_read_in=False, weighted=True,  check_accur=False, check_obs_wl=False, check_calc_wl=False, check_wl=False, print_info=False):
    """

    Function loading a graph from line and level data stored in .csv files.

    Returns
    -------
    G : NetworkX graph

    Parameters
    ----------
    ion : str
        Ion ID. (default='H1.0')
    experimental : bool (optional)
        If True: use NIST data, if False: Jitrik model data. (default=True)
    only_dipole : bool (optional)
        If True: use only dipole transitions, if False: all transitions. (default=False)
    n_limit : bool (optional)
        If True one can set an upper n limit. (default=False)
    max_n : int (optional)
        Upper limit of the n quantum number. (default=None)
    alt_read_in : bool (optional)
        If True, uses an alternative way to read in the data file (hardcoded column number). (default=False)
    weighted : bool (optional)
        Whether to read in weights of networkx graph. (default=True)
    check_obs_wl : bool (optional)
        Whether to invoke the check_obs_wl option from the spectroscopic_networks function (only used for NIST data). (default=False)
    check_calc_wl : bool (optional)
        Whether to invoke the check_calc_wl option from the spectroscopic_networks function (only used for NIST data). (default=False)
    check_wl : bool (optional)
        Whether to invoke the check_wl option from the spectroscopic_networks function (only used for NIST data). (default=False)
    print_info : bool (optional)
        Whether to print some basic info about the chosen setting and the network to the console. (default=False)

    Notes
    -----

    References
    ----------
    Experimental data by NIST https://www.nist.gov/
    Theoretical data by Jitrik, O., & Bunge, C. F. (2004). Transition probabilities for hydrogen-like atoms. Journal of Physical and Chemical Reference Data, 33(4), 1059-1070. https://doi.org/10.1063/1.1796671 (check accompanying website).


    """
    if experimental == False:
        # load theoretical data (Jitrik-Bunge)
        print 'Model Data'

        # get Z
        for i, char in enumerate(ion.split('.')[0]):
            if char.isdigit():
                index = i
                break
        Z = int(ion.split('.')[0][index:])

        if only_dipole== True:
            Graph = model_network(Z=Z, E1=True, max_n=max_n) #only dipole lines up to max_n
        else:
            Graph = model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=max_n) #all lines
    else:
        # load NIST data
        print 'NIST Data'
        if only_dipole == True:
            Graph = spectroscopic_network(ion, weighted=weighted, alt_read_in=alt_read_in, check_accur=check_accur, check_obs_wl=check_obs_wl, check_calc_wl=check_calc_wl, check_wl=check_wl)
            Graph = remove_empty_levels(Graph, 'term') #remove nodes with empty term entry
            Graph = remove_n_greater(Graph, max_n=max_n) # maximal n considered
            Graph = only_dipole_transitions_parity(Graph) #only dipole lines
            Graph = only_largest_component(Graph)
        else:
            Graph = spectroscopic_network(ion, weighted=weighted, alt_read_in=alt_read_in, check_accur=check_accur, check_obs_wl=check_obs_wl, check_calc_wl=check_calc_wl, check_wl=check_wl)
            Graph = remove_empty_levels(Graph, 'term') #remove nodes with empty term entry
            Graph = remove_n_greater(Graph, max_n=max_n) # maximal n considered
            Graph = only_largest_component(Graph)

    if print_info:
        # print info to terminal
        print 'Data type: ', experimental*'NIST' + (not experimental)*'Model Data (Jitrik)'
        print 'max n: ', max_n
        print 'only dipole: ', only_dipole
        print 'check_accur', check_accur
        print 'check_obs_wl', check_obs_wl
        print 'check_calc_wl', check_calc_wl
        print 'check_wl', check_wl
        print 'weighted', weighted

        print 'Number of edges: ', nx.number_of_edges(Graph)
        print 'Number of nodes: ', nx.number_of_nodes(Graph)
        print 'Max missing edges: ', nx.number_of_nodes(Graph) * (nx.number_of_nodes(Graph) - 1) / 2- nx.number_of_edges(Graph)

    return Graph




def only_dipole_transitions_parity(G):
    """
    Return spectroscopic network only keeping the links between states of different parity.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Gc : NetworkX graph
        Graph containing only links between states of different parity
    """
    Gc = G.copy()
    for edge in G.edges_iter():
        p0 = nx.get_node_attributes(Gc, 'parity')[edge[0]]
        p1 = nx.get_node_attributes(Gc, 'parity')[edge[1]]
        if p0 == p1:
            Gc.remove_edge(*edge)
    return Gc


def _estimate(G, MCSWEEPS=110, sweep_cut=10, save_gml=False):
    """
    Estimation of group number based on the stochastic block model (SBM) by Monte Carlo (MC) maximization of the model likelyhood.

    Parameters
    ----------
    G : NetworkX graph
        Graph to do estimation on.
    MCSWEEPS : int
        Number of Monte Carlo sweeps used in the likelyhood maximisation algorithm.
    sweep_cut : int
        Number of sweeps at the start of the algorithm which are not to be used for evaluation.
    save_gml : bool
        Save .gml-file produced.

    Returns
    -------
    sweep : array-like
        MC sweep number.
    k_values : array-like
        Number of groups.
    logp_values : array-like
        Log-probability of SBM.
    g_values : np.ndarray
        Group vectors.
    node_labels : dict
        Dict {index: node_label} containing the node labels.

    Notes
    -----
    This uses a slightly modified script compared to the one provided by the authors [1]. It was modified to also write the group vectors to a file.

    References
    ----------
    .. [1] Newman, M. E. J., & Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7).
    """
    from subprocess import Popen, PIPE
    from os import getcwd, remove

    # create gml-file of G
    gml_path = getcwd() + '/' + G.name + '.gml'
    nx.write_gml(G, gml_path)
    gml_file = open(gml_path)

    # call ./estimate
    args = ["./estimate", str(MCSWEEPS)]
    a = Popen(args, stdin=gml_file, stdout=PIPE)

    # get node labels from gml-file
    G2 = nx.read_gml(gml_path, label='id')
    node_labels =  nx.get_node_attributes(G2, 'label')
    warnings.warn('Node labels are given in unicode strings. This might be a problem when matching them to graphs loaded in sepctroscopic_network.')

    # remove gml-file
    if not save_gml:
        remove(gml_path)

    # read output from ./estimate
    output = a.stdout.read()
    output_arr = np.fromstring(output, dtype=float, sep= ' ')
    cols = G.number_of_nodes()+3
    rows = len(output_arr)/cols
    output_arr = output_arr.reshape((rows, cols))

    sweep = output_arr[:,0].astype(int)
    keep_ind = np.argwhere(sweep>=sweep_cut).flatten()

    # cut all data below sweep_cut
    sweep = sweep[keep_ind]
    k_values = output_arr[keep_ind,1].astype(int)
    logp_values = output_arr[keep_ind,2]
    g_values = output_arr[keep_ind,3:].astype(int)

    return sweep, k_values, logp_values, g_values, node_labels


def newman_estimation_from_graph(G, MCSWEEPS=110, sweep_cut=10, save_gml=False):
    """
    Estimation of group number based on the stochastic block model (SBM) by Monte Carlo (MC) maximization of the model likelyhood.

    Parameters
    ----------
    G : NetworkX graph
        Graph to do estimation on.
    MCSWEEPS : int
        Number of Monte Carlo sweeps used in the likelyhood maximisation algorithm.
    sweep_cut : int
        Number of sweeps at the start of the algorithm which are not to be used for evaluation.
    save_gml : bool
        Save .gml-file produced.

    Returns
    -------
    g : array-like
        Group vector with the highest likelyhood SBM.

    References
    ----------
    .. [1] Newman, M. E. J., & Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7).
    """
    G = remove_attributes(G)
    estimation_results = _estimate(G,
                                   MCSWEEPS=MCSWEEPS,
                                   sweep_cut=sweep_cut,
                                   save_gml=save_gml)
    sweep, k_values, logp_values, g_values, node_labels = estimation_results

    max_ind = np.argmax(logp_values)
    k = k_values[max_ind]
    g_id = g_values[max_ind, :]

    """
    map g_id values to {0, 1, 2, 3, ..} s.t. values start at 0 and no values are skipped
    """
    mapping = np.unique(g_id)
    g_id = map(lambda x: np.argwhere(mapping == x)[0,0], g_id)

    g = {}
    for i in xrange(len(g_id)):
        g[node_labels[i]] = g_id[i]

    # check whether k at max(logp) is also the most occuring value in the MC, if not raise warning
    counts = np.bincount(k_values)
    k_hist = np.argmax(counts)
    if k != k_hist:
        warnings.warn('The group number k at the maximum log probability is not the value that occurs most frequently in the MC simulation.')

    return g


def SBM_calculate_omega(G, g):
    """
    Function calculating the group link probabilities for a stochastic block model.

    To Do
    -----
    Include check whether groups are in {0,1,2,...,k}
    """
    g_unique = np.sort(np.unique(g.values()))
    omega = np.zeros((len(g_unique), len(g_unique)), dtype=float)
    for g1, g2 in itertools.combinations(g_unique, 2):
        nodes1 = np.asarray([node for node, group in g.iteritems() if group == g1])
        nodes2 = np.asarray([node for node, group in g.iteritems() if group == g2])

        edge_count = np.sum(map(lambda e: G.has_edge(*e), itertools.product(nodes1, nodes2)))
        omega[g1,g2] = omega[g2,g1] = float(edge_count)/(len(nodes1)*len(nodes2))

    return omega


class LinkPrediction:
    G_original         = None # original graph
    G_training         = None # training graph
    G_probe            = None # probe graph
    G_predict          = None # predicition graph

    P                  = 0 # positive samples
    N                  = 0 # negative samples

    prediction_list    = [] # list of tuples (edge, probability) (hast to be ordered by descending probability)
    gain_list          = [] # list of gain values
    probe_list         = [] # list of probe edges
    probe_rank         = [] # list of probe edge ranks

    is_correct         = None # array of bools

    # ROC
    ROC                = None # (2, # non-observed links) - array containing TPR and FPR
    FPR                = None
    TPR                = None

    # gain evaluation measures (arrays)
    CG                 = None
    DCG                = None
    ICG                = None
    nCG                = None
    nDCG               = None

    # AUC value
    AUC                = 0

    #for HRG:
    strID_2_numID_dict = None # dict containing normal node ID and the node IDs used in the HRG prediction algorithm (which have to be digits)
    numID_2_strID_dict = None # opposite dictionary

    #########################FUNCTIONS#######################################

    def dropout(self, p):
        """
        This function creates the probe and training graphs self.G_probe and self.G_training by randomly dropping every node in the original graph with probability p.

        Parameters
        ----------
        p : float
            Dropout probability.

        To-Do
        -----
        Transcribe weights of G_original to G_probe and G_training.
        """
        # randomly select which edges to drop
        edge_list = np.asarray(self.G_original.edges())
        r = np.random.sample(len(edge_list))
        dropout_mask = r<p

        edges_dropped   = edge_list[dropout_mask]
        edges_kept      = edge_list[np.logical_not(dropout_mask)]

        # write kept edges to training graph
        self.G_training = nx.Graph()
        self.G_training.add_edges_from(edges_kept)

        # write dropped edges to probe graph
        self.G_probe = nx.Graph()
        self.G_probe.add_edges_from(edges_dropped)

        # store number of positive samples
        self.P = len(edges_dropped)

        # calculate number of negative samples
        n_nodes_training = self.G_training.number_of_nodes()
        n_edges_training = self.G_training.number_of_edges()
        self.N = n_nodes_training*(n_nodes_training-1)/2 - n_edges_training - len(edges_dropped)

        # store list of probe edges
        self.probe_list = self.get_dropout_list()

    def cross_validation(self, edge_list, train_index, validation_index):
        """
        This function creates the probe and training graphs self.G_probe and self.G_training.

        Parameters
        ----------scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
        edge_list :
            Edgelist of G_original.
        train_index :
            train_index array (output of **.split()).
        validation_index :
            validation_index array (output of **.split()).

        TO-DO
        ----------
        fix the routine using this function.
        """
        edges_kept      = edge_list[train_index]
        edges_dropped   = edge_list[validation_index]

        # print edges_kept

        # write kept edges to training graph
        self.G_training = nx.Graph()
        self.G_training.add_edges_from(edges_kept)

        # write dropped edges to probe graph
        self.G_probe = nx.Graph()
        self.G_probe.add_edges_from(edges_dropped)

        # store number of positive samples
        self.P = len(edges_dropped)

        # calculate number of negative samples
        n_nodes_training = self.G_training.number_of_nodes()
        n_edges_training = self.G_training.number_of_edges()
        self.N = n_nodes_training*(n_nodes_training-1)/2 - n_edges_training - len(edges_dropped)

        # store list of probe edges
        self.probe_list = self.get_dropout_list()


    def split_graph_into_sets(self, ion='H1.0', only_dipole=True, max_n=999):
        """
        Splits hydrogen network into training (experimental) and probe set. Overwrites G_training and G_probe.

        Parameters
        ----------
        only_dipole : boolean
            Boolean to choose only the dipole data.
        max_n : int
            The maximum n quantum number.
        """
        # Load networks
        print 'Loading network:', ion

        # get Z
        for i, char in enumerate(ion.split('.')[0]):
            if char.isdigit():
                index = i
                break
        Z = int(ion.split('.')[0][index:])

        exp_Graph = spectroscopic_network(ion, weighted=True, alt_read_in=False)
        exp_Graph = remove_empty_levels(exp_Graph, 'term')
        exp_Graph = remove_n_greater(exp_Graph, max_n)
        exp_Graph = only_largest_component(exp_Graph)
        if only_dipole==True:
            exp_Graph = only_dipole_transitions_parity(exp_Graph) #only dipole lines

        # relabel node IDs
        exp_modelName = nx.get_node_attributes(exp_Graph, 'modelName')  # ID-modelID dictionary
        nx.relabel_nodes(exp_Graph,exp_modelName,copy=False)

        # print 'Loading model network'
        if only_dipole==True:
            model_Graph = model_network(Z=Z, E1=True, E2=False, E3=False, M1=False, M2=False, M3=False, max_n=max_n) #only dipole lines
        else:
            model_Graph = model_network(Z=Z, E1=True, E2=True, E3=True, M1=True, M2=True, M3=True, max_n=None) #all lines

        # create two sets: training set used for calculating predictions and probe set for evaluating these predictions
        probe_set_graph = model_Graph
        training_set_graph = nx.Graph()

        # delete nodes from model graph that do not appear in exp graph (we wont be able to predict these anyways)
        probe_set_graph = probe_set_graph.subgraph(exp_Graph.nodes())

        for e in exp_Graph.edges():
                if probe_set_graph.has_edge(*e): #need "if", since some lines are missing in the model graph
                    probe_set_graph.remove_edge(*e)
                training_set_graph.add_edge(*e) #create training graph. note that there are only dipole lines in the probe set, so also in the training set.

        # write as class variable
        self.G_original = model_Graph
        self.G_training = training_set_graph
        self.G_probe = probe_set_graph


     #############################SBM################################
    @staticmethod
    def _SBM_prediction(G, n_samples=1000, sampling='random', MCSWEEPS=10000, sweep_cut=200, estimation_results=None):
        """
        Link prediction using the stochastic block model (SBM).

        Parameters
        ----------
        G : NetworkX graph
            Graph to do link prediction on.
        n_samples : int
            Number of SBM realisations sampled to calculate link probabilities.
        sampling : 'random' or 'best'
            How to select SBM realisations: randomly or using the best ones ordered by SBM likelyhood.
        MCSWEEPS : int
            Number of Monte Carlo sweeps used in the likelyhood maximisation algorithm.
        sweep_cut : int
            Number of sweeps at the start of the algorithm which are not to be used for evaluation.
        estimation_results : tuple
            The results of the SBM estimation can be calculated elsewhere and given as a parameter. The results are then used for analysis and are not calculated again.

        Returns
        -------
        p : np.ndarray
            Matrix of edge probabilities.
        node_labels : dict
            Dict {index: node_label} containing the node labels.

        References
        ----------
        .. [1] Newman, M. E. J., & Reinert, G. (2016). Estimating the Number of Communities in a Network. Physical Review Letters, 117(7).
        """
        from mpmath import mp, exp # package for arbitrarily large floats
        mp.dps = 20 # precision

        G = remove_attributes(G)

        # carry out estimation
        if estimation_results==None:
            estimation_results = _estimate(G,
                                           MCSWEEPS=MCSWEEPS,
                                           sweep_cut=sweep_cut,
                                           save_gml=False)
        sweep, k_values, logp_values, g_values, node_labels = estimation_results

        print g_values.shape

        # create adjacency matrix
        nodelist    = [node_labels[i] for i in xrange(len(node_labels))]
        A           = nx.to_numpy_matrix(G, nodelist=nodelist)

        p           = np.zeros( (len(G), len(G)) ) # link probability matrix
        omega_nodes = np.zeros( (len(G), len(G)) ) # group probability for node pairs
        norm        = 0.0 # normalization factor for probabilities

        # normalization factor in order to get back into the float range
        factor = 1.0/exp(logp_values[0])

        # choose index set of samples
        if sampling=='random':
            indices = np.random.choice(len(sweep), n_samples, replace=False)
        elif sampling=='best':
            indices = np.argsort(logp_values)[-n_samples:]
        else:
            raise ValueError('Possible values: "random", "best"')
        for l in indices:
            g = g_values[l,:] # group vector

            # calculate group probability matrix (omega)
            omega = np.zeros( (max(g)+1, max(g)+1) )
            for g1, g2 in itertools.combinations(np.unique(g), 2):
                node_group_1 = np.argwhere(g==g1).flatten()
                node_group_2 = np.argwhere(g==g2).flatten()

                # count edges between groups
                A_sub = (A[node_group_1])[:, node_group_2]
                omega[g1,g2] = omega_nodes[g2,g1] = np.sum(A_sub)/float(len(node_group_1)*len(node_group_2))

            # copy to omega_nodes
            for i,j in itertools.combinations(xrange(len(G)),2):
                omega_nodes[i,j] = omega_nodes[j,i] = omega[g[i], g[j]]

            # update link probability matrix
            p = p + float(factor*exp(logp_values[l]))*omega_nodes

            # update normalization factor
            norm += float(factor*exp(logp_values[l]))
        p = (p/norm).astype('float')
        return p, node_labels


    #########################SPM#####################
    def predict_SBM(self, n_samples=500, sampling='random', MCSWEEPS=10000, sweep_cut=100, estimation_results=None):
        """
        Predict links which are not observed in the training graph using the stochastic block model (SBM) method.

        Parameters
        ----------
        n_samples : int
            Number of SBM realisations sampled to calculate link probabilities.
        sampling : 'random' or 'best'
            How to select SBM realisations: randomly or using the best ones ordered by SBM likelyhood.
        MCSWEEPS : int
            Number of Monte Carlo sweeps used in the likelyhood maximisation algorithm.
        sweep_cut : int
            Number of sweeps at the start of the algorithm which are not to be used for evaluation.
        estimation_results : tuple
            The results of the SBM estimation can be calculated elsewhere and given as a parameter. The results are then used for analysis and are not calculated again.
        """
        import operator
        # run actual predicition
        p, node_labels = self._SBM_prediction(G=self.G_training,
                                             n_samples=n_samples,
                                             sampling=sampling,
                                             MCSWEEPS=MCSWEEPS,
                                             sweep_cut=sweep_cut,
                                             estimation_results=estimation_results)

        # create prediction graph
        self.G_predict = nx.Graph()
        self.G_predict.add_nodes_from(node_labels.values())

        # store edge probabilities and add edges to prediction graph
        probability_dict = {}
        prediction_list_dummy = []
        for ctr, (i, j) in enumerate(itertools.combinations(xrange(len(node_labels)),2)):
            node1 = node_labels[i]
            node2 = node_labels[j]
            if not self.G_training.has_edge(node1, node2):
                probability_dict[(node1, node2)] = p[i,j]
                self.G_predict.add_edge(node1, node2)

        self.prediction_list = prediction_list_dummy

        # store probabilities in prediction graph
        nx.set_edge_attributes(self.G_predict, 'probability', probability_dict)

        # sort predictions by probability
        prediction_list_dummy = sorted(probability_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        self.prediction_list = prediction_list_dummy


    @staticmethod
    def _SPM_delta_E_mat(V, E, M_cut):
        """
        Compute delta_E in terms of adjacency matrix indices.
        """
        # randomly select edges to cut
        choice = np.random.choice(len(E), size=M_cut, replace=False)
        delta_E = E[choice,:]

        # translate delta_E to indices in adjacency matrix
        delta_E_mat = np.asarray([(np.argwhere(V==edge1)[0,0], np.argwhere(V==edge2)[0,0]) for edge1, edge2 in delta_E])
        return delta_E_mat


    @staticmethod
    def _gt_delta_E_mat(V, E, M_cut):
        """
        Compute delta_E in terms of adjacency matrix indices.
        """
        # randomly select edges to cut
        choice = np.random.choice(len(E), size=M_cut, replace=False)
        delta_E_mat = E[choice,:2]

        # translate delta_E to indices in adjacency matrix
        # delta_E_mat = np.asarray([(np.argwhere(V==edge1)[0,0], np.argwhere(V==edge2)[0,0]) for edge1, edge2 in delta_E])
        return delta_E_mat


    @staticmethod
    def _SPM_A_tilde_A_R(A, V, delta_E_mat):
        """
        This function computes A_tilde and A_R for SPM prediction or structural consistency.

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix.
        V : array-like
            Vertex list by which the adjacency matrix is ordered.
        delta_E_mat : np.ndarray
            Cut edges in terms of matrix indices.
        """
        # create delta_A and A_R
        delta_A = np.zeros_like(A, dtype=int)
        delta_A[delta_E_mat[:,0], delta_E_mat[:,1]] = 1
        delta_A[delta_E_mat[:,1], delta_E_mat[:,0]] = 1
        A_R = A - delta_A

        # compute eigenvalues and eigenvectors
        lam0, v = np.linalg.eigh(A_R)
        lam = lam0.round(decimals=4)

        # calculate first order eigenvalue corrections and A_tilde
        delta_lam = np.zeros_like(lam)
        A_tilde = np.zeros_like(A, dtype=float)

        # no degenerate eigenvalues vs degenerate eigenvalues
        if len(lam)==len(np.unique(lam)):
            for i in xrange(len(lam)):
                vec = np.asarray(v[:,i]).flatten()
                delta_lam[i] = np.dot(vec, np.asarray(np.dot(delta_A, vec)).flatten())/np.dot(vec, vec)
                A_tilde += (lam[i] + delta_lam[i])*np.outer(vec, vec)
        else:
            for ew in np.unique(lam):
                counts = np.sum(lam==ew)
                if counts == 1:
                    i = np.argwhere(lam == ew)[0, 0]
                    vec = np.asarray(v[:, i]).flatten()
                    delta_lam[i] = np.dot(vec, np.asarray(np.dot(delta_A, vec)).flatten()) / np.dot(vec, vec)
                    A_tilde += (lam[i] + delta_lam[i]) * np.outer(vec, vec)
                else:
                    vecs = v[:, lam==ew]
                    W = np.dot(vecs.transpose(), np.dot(delta_A, vecs))
                    dlam, beta = np.linalg.eigh(W)
                    v_new = np.dot(vecs, beta)
                    for i, delta_ew in enumerate(dlam):
                        A_tilde += (ew + delta_ew) * np.outer(v_new[:, i], v_new[:, i])

        return A_tilde, A_R


    @staticmethod
    def _SPM_rank_mat(A_tilde, A):
        """
        Rank all adges not observed in A by the values in A_tilde. The edges are given in terms of matrix indices.
        """
        # rank non-observed links
        triu_ind = np.triu_indices(A_tilde.shape[0], k=1) # upper right triangular indices w/o diagonal
        triu_ind_flat = np.ravel_multi_index(triu_ind, dims=A_tilde.shape) # flattened

        ## sort upper right triangle
        sorted_ind = triu_ind_flat[np.asarray(np.argsort(A_tilde[triu_ind], axis=None)).flatten()][::-1]

        ## rank indices
        rank_ind = np.unravel_index(sorted_ind, dims=A_tilde.shape)
        rank_ind = np.stack((rank_ind[0], rank_ind[1]), axis=1)
        mask = np.asarray(A[rank_ind[:,0], rank_ind[:,1]] == 0).flatten() # remove edges that are already observed
        rank_mat = rank_ind[mask, :]

        return rank_mat


    @staticmethod
    def _SPM_prediction(G, p=0.1, n_selections=10):
        """
        This function predicts links in a network using the structural perturbation method (SPM).
        """

        # number of edges to cut rounded to next integer
        M = G.number_of_edges()
        M_cut = int(np.rint(p*M))

        # get adjacency matrix
        V = np.asarray(G.nodes()) # vertex list
        E = np.asarray(G.edges()) # edge list
        A = nx.to_numpy_matrix(G, nodelist=V).astype(int)

        delta_E_mat = LinkPrediction._SPM_delta_E_mat(V, E, M_cut)

        A_tilde_avg = np.zeros_like(A, dtype=float)
        for k in xrange(n_selections):
            A_tilde, A_R = LinkPrediction._SPM_A_tilde_A_R(A, V, delta_E_mat)
            A_tilde_avg += A_tilde

        A_tilde_avg /= n_selections # divide by number of selections

        rank_mat = LinkPrediction._SPM_rank_mat(A_tilde_avg, A)

        # translate rank_ind to edge labels
        # translate rank_ind to edge labels
        rank = [(V[ind1], V[ind2], A_tilde_avg[ind1, ind2]) for ind1, ind2 in rank_mat]
        return rank


    @staticmethod
    def SPM_structural_consistency(G, n_repeat=1, p=0.1):
        """
        Calculate the structural consistency, which is a measure of how predictable a network is.

        Parameters
        ----------
        G : NetworkX graph
            Graph for structural consistency calculation.
        n_repeat : int
            Number of repetitions of the calculation, take average at the end.
        p : float
            Cut fractistdon.

        To-Do
        -----
        The comparison of the two edge lists takes forever.

        References
        ----------
        .. [1] Lü, L., Pan, L., Zhou, T., Zhang, Y.-C., & Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
        """
        # number of edges to cut rounded to next integer
        M = G.number_of_edges()
        M_cut = int(np.rint(p*M))

        # get adjacency matrix
        V = np.asarray(G.nodes()) # vertex list
        E = np.asarray(G.edges()) # edge list
        A = nx.to_numpy_matrix(G, nodelist=V).astype(int)

        sigma_c_values = np.zeros(n_repeat)
        for k in xrange(n_repeat):
            delta_E_mat = LinkPrediction._SPM_delta_E_mat(V, E, M_cut) # cut edges
            A_tilde, A_R = LinkPrediction._SPM_A_tilde_A_R(A, V, delta_E_mat)
            rank_mat = LinkPrediction._SPM_rank_mat(A_tilde, A_R) # edge rank

            # intersection beween cut edges and first M_cut ranked edges
            intersection = filter(lambda x: x, [filter(lambda x: np.all(np.in1d(x, y)), delta_E_mat) for y in rank_mat[:M_cut]])

            # calculate structural consistency
            sigma_c_values[k] = float(len(intersection))/M_cut
        # calculate mean and standard deviation
        sigma_c = np.mean(sigma_c_values)
        std = np.std(sigma_c_values) if n_repeat>1 else np.nan
        return sigma_c, std

    @staticmethod
    def gt_structural_consistency(G, n_repeat=1, p=0.1):
        import graph_tool.all as gt
        """
        Calculate the structural consistency, which is a measure of how predictable a network is.

        Parameters
        ----------
        G : graph-tool graph
            Graph for structural consistency calculation.
        n_repeat : int
            Number of repetitions of the calculation, take average at the end.
        p : float
            Cut fraction.

        To-Do
        -----
        The comparison of the two edge lists takes forever.

        References
        ----------
        .. [1] Lue, L., Pan, L., Zhou, T., Zhang, Y.-C., & Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
        """
        # number of edges to cut rounded to next integer
        M = G.num_edges()
        M_cut = int(np.rint(p * M))

        # get adjacency matrix
        V = np.asarray(G.get_vertices())  # vertex list
        E = np.asarray(G.get_edges())  # edge list
        A = gt.adjacency(G).todense()

        sigma_c_values = np.zeros(n_repeat)
        for k in xrange(n_repeat):
            delta_E_mat = LinkPrediction._gt_delta_E_mat(V, E, M_cut)  # cut edges
            A_tilde, A_R = LinkPrediction._SPM_A_tilde_A_R(A, V, delta_E_mat)
            rank_mat = LinkPrediction._SPM_rank_mat(A_tilde, A_R)  # edge rank

            # intersection beween cut edges and first M_cut ranked edges
            intersection = filter(lambda x: x,
                                  [filter(lambda x: np.all(np.in1d(x, y)), delta_E_mat) for y in rank_mat[:M_cut]])

            # calculate structural consistency
            sigma_c_values[k] = float(len(intersection)) / M_cut
        # calculate mean and standard deviation
        sigma_c = np.mean(sigma_c_values)
        std = np.std(sigma_c_values) if n_repeat > 1 else np.nan
        return sigma_c, std


    def predict_SPM(self):
        """
        Predict links which are not observed in the training graph using the structural perturbation method (SBM).

        References
        ----------
        .. [1] Lü, L., Pan, L., Zhou, T., Zhang, Y.-C., & Stanley, H. E. (2015). Toward link predictability of complex networks. Proceedings of the National Academy of Sciences, 112(8), 201424644.
        """
        # calculate rank of non-observed links by SPM
        rank = self._SPM_prediction(self.G_training)

        # create prediction graph
        self.G_predict = nx.Graph()

        # store prediciton list and add edges to prediction graph
        prediction_list_dummy = []
        for node1, node2, score in rank:
            # set probabilities to score
            prediction_list_dummy.append(((node1, node2), score))

            # add edge to preediction graph
            self.G_predict.add_edge(node1, node2)

        self.prediction_list = prediction_list_dummy


    # ####################### nested SBM ############
    @staticmethod
    def _nested_SBM_prediction(nxG, force_niter=100, minimize_runs=10, mcmc_args=10, pred_list=None, cutoff=1.0):
        """
        Link Prediction using nested SBM

        Parameters
        ----------
        nxG : NetworkX graph
            Graph to do link prediction on.
        wait : int
            Number of iterations to wait for a record-breaking event.
        mcmc_args : int
            niter passed to graph-tool.mcmc_sweeps

        Returns
        -------
        probabilities : list
            list of tuples (node1, node2, link probability) sorted by link probability
        """
        try:
            import graph_tool.all as gt
        except:
            warnings.warn("Need graph-tool for nested SBM!")
        import time
        G = nx2gt(nxG)
        G = gt.Graph(G, directed=False, prune=True)

        ground_state_estimation_list = [gt.minimize_nested_blockmodel_dl(G) for i in range(minimize_runs)]
        ground_state_estimation_list = sorted(ground_state_estimation_list, key=lambda ground_state_estimation_list:
                                                  ground_state_estimation_list.entropy(), reverse=False)
        ground_state_estimation = ground_state_estimation_list[0]
        if pred_list is not None:
            potential_edges = [(v1, v2) for v1 in G.vertices() for v2 in G.vertices()
                               if (G.vp.nodeid[v1], G.vp.nodeid[v2]) in pred_list]
        else:
            adj = gt.adjacency(G)
            ones = np.triu(np.ones((G.num_vertices(), G.num_vertices())))
            potential_edges = np.argwhere(np.logical_not(adj + ones))

        # probs = np.zeros((wait, len(potential_edges)))
        probs = [[] for _ in range(len(potential_edges))]
        bs = ground_state_estimation.get_bs()
        bs += [np.zeros(1)] * (6 - len(bs))
        ground_state = ground_state_estimation.copy(bs=bs, sampling=True)
        run_number = [0]
        blockstatesvisited = []

        def collect_edge_probs(s):
            S = s.entropy()
            # only look at new blockstates
            if np.all(np.abs(np.array(blockstatesvisited)-S) > 0.0001):
                # only look at blockstates with high likelyhood
                if len(blockstatesvisited) == 0 or np.all((np.array(blockstatesvisited)-S) > -cutoff):
                    for i in range(len(potential_edges)):
                        p = s.get_edges_prob([potential_edges[i]], entropy_args=dict(partition_dl=False))
                        probs[i].append((np.exp(p), S))
                    blockstatesvisited.append(S)
            run_number[0] = run_number[0] + 1

        collect_edge_probs(ground_state)

        gt.mcmc_equilibrate(ground_state, force_niter=force_niter, mcmc_args=dict(niter=mcmc_args), callback=collect_edge_probs)

        def get_avg(p):
            p_avg = 0.0
            entropy_offset = p[0][1]
            norm = 0.0
            for prob in p:
                p_avg += prob[0]*np.exp(-prob[1] + entropy_offset)
                norm += np.exp(-prob[1] + entropy_offset)
            p_avg = p_avg/norm
            return p_avg

        probabilities = [(G.vp.nodeid[potential_edges[i][0]], G.vp.nodeid[potential_edges[i][1]], get_avg(probs[i])) for i in range(len(potential_edges))]
        probabilities = sorted(probabilities, key=lambda probabilities: probabilities[2], reverse=True)
        return probabilities


    def predict_nested_SBM(self, minimize_runs=10, force_niter=100, mcmc_args=10, pred_list=None, cutoff=1.0):
        '''
        Link Prediction using nested SBM

        Parameters
        ----------
        wait : int
            Number of iterations to wait for a record-breaking event.
        mcmc_args : int
            niter passed to graph-tool.mcmc_sweeps
            should be around 10, but makes the algorithm last long

        Returns
        -------
        updates self with prediction
        '''
        probabilities = self._nested_SBM_prediction(self.G_training, minimize_runs=10, force_niter=force_niter, mcmc_args=mcmc_args, pred_list=pred_list, cutoff=cutoff)
        self.G_predict = nx.Graph()
        self.G_predict.add_weighted_edges_from(probabilities, weight='likelyhood')
        self.prediction_list = [((probability[0], probability[1]), probability[2]) for probability in probabilities]


    #########################HRG#####################
    @staticmethod
    def _get_edgelist(Graph):
      temp = np.asarray(nx.to_edgelist(Graph))
      splits = np.split(temp, 3, axis=1)
      edgelist = np.hstack((splits[0], splits[1]))
      return edgelist


    ##
    ## @brief      Creates an npy array with normal node IDs and new ones which
    ##             will be used in the HRG prediction process. In case of the
    ##             model network, it generates numerical IDs from the order of
    ##             the nodes as an alternative to the string IDs (like '7S').
    ##
    ## @param      self  The object
    ##
    def create_HRG_ID_dict(self, experimental):
        # create array to store IDs and terms of nodes
        ID_array = np.ones((nx.number_of_nodes(self.G_original), 2), dtype='|S20')

        #fill array with the normal node ID and the node ID for the HRG prediction
        for i, node_ID in enumerate(self.G_original.nodes()):
            if experimental == False:
                ID_array[i,0] = node_ID                     # strID ('4D')
                ID_array[i,1] = i                           # numID/HRG_ID ('26')
            else:
                ID_array[i,0] = node_ID                     # strID ('0000.001.0033')
                ID_array[i,1] = str(int(node_ID[-4:]))      # numID/HRG_ID ('33')

        #create dictionaries from matching array, transalting between IDs and terms
        self.strID_2_numID_dict = dict(zip(ID_array[:,0], ID_array[:,1])) #keys: string ID ('7S'), values: numerical generated ID (saved as string: '33')
        #create second dict with reversed order ("decryption")
        self.numID_2_strID_dict = {y:x for x,y in self.strID_2_numID_dict.items()} #keys: numerical generated ID (saved as string: '33'), values: string ID ('7S')


    #TODO: doku
    ##
    ## @brief      Create an edgelist file exactly in the numerical format that
    ##             the Clauset/Newman c++ algorithm needs.
    ##
    ## @param      self          The object
    ## @param      ion           (String) The ion
    ## @param      label         (String) The label
    ## @param      directory     (String) The directory
    ## @param      label_switch  (Boolean) The label switch
    ##
    ## @return     { description_of_the_return_value }
    ##
    def write_edgelist(self, Graph, label='test', directory='./'):

        edgelist = self._get_edgelist(Graph)

        print 'Writing', directory + label + '.pairs'
        with open(directory + label + '.pairs', 'w') as edge_file:
            writer = csv.writer(edge_file, delimiter='\t')
            for i in range(edgelist.shape[0]):
                writer.writerow( [ self.strID_2_numID_dict[edgelist[i,0]] , self.strID_2_numID_dict[edgelist[i,1]] ] ) #save value of dict (numerical ID)
            edge_file.close()
        print '-> ' + directory + label + '.pairs', 'saved'


    ##
    ## @brief      Fit a dendrogram to the edgelist with the Clauset/Newman c++ algorithm.
    ##
    ## @param      label                     The label
    ## @param      directory                 The directory of the .pairs file
    ## @param      model_numID_2_strID_dict  The model number id 2 string id dictionary
    ##
    ## @return     No Return
    ##
    @staticmethod
    def fit_dendrogram(label, filedirectory='./HRG/', HRG_PATH='./HRG/'):
        import subprocess
        import shlex
        import time

        path_cpp = HRG_PATH + 'fitHRG/fitHRG'
        cmd = path_cpp + ' -f ' + filedirectory + label + '.pairs '
        args = shlex.split(cmd)

        print 'FITDENDRO'

        # run process
        process = subprocess.Popen(args)

        # run for 10 seconds
        time.sleep(10)

        # end process
        process.terminate()

    ##
    ## @brief      Predict links using the clauset newman algorithm. Needs a .pairs
    ##             as input, outputs a textfile with ordered link predictions
    ##
    ## @param      label          The label
    ## @param      filedirectory  The file directory or the .pairs file
    ## @param      HRG_PATH       The hrg path
    ##
    ## @return     No Return
    ##
    @staticmethod
    def HRG_predict_links(label, filedirectory='./HRG/', HRG_PATH='./HRG/'):
        import subprocess
        import shlex
        import time
        from shutil import copyfile

        # TODO: num_samples, num_bins? 10000, 25
        path_cpp = HRG_PATH + 'predictHRG/predictHRG_GPL/predictHRG'
        cmd = path_cpp + ' -f ' + filedirectory + label + '.pairs'
        args = shlex.split(cmd)

        # run process until it finishes
        # process = subprocess.Popen(args)
        subprocess.call(args)

    ##
    ## @brief      Loads predicted links from ranked.wpairs file created by the
    ##             Clauset C++ script.
    ##
    ## @param      ion            The ion
    ## @param      label          The label
    ## @param      label_switch   The label switch
    ## @param      filedirectory  The file directory
    ##
    ## @return     Returns array of predicted links ([u,v]) and a second array
    ##             with the corresponding probabilities.
    ##
    def load_predicted_links_from_wpairs_file(self, label, filedirectory='./HRG/'):
        # load predicted links
        with open(filedirectory +label + '-ranked.wpairs', 'r') as predict_file:
            line_reader = csv.reader(predict_file, delimiter='\t')
            line_data_temp = np.array(list(line_reader))


        # store edge probabilities and add edges to prediction graph
        predicted_links = np.empty((len(line_data_temp), 2), dtype=int)
        predicted_links_prob = np.empty((len(line_data_temp)), dtype=float)
        # probability_dict = {}
        prediction_list_dummy = []

        for i in range(len(line_data_temp)):
            for k in range(2):
                predicted_links[i,k] = int( line_data_temp[i][k] )
            predicted_links_prob[i] = float( line_data_temp[i][2] )
            prediction_list_dummy.append((( self.numID_2_strID_dict[line_data_temp[i][0]] , self.numID_2_strID_dict[line_data_temp[i][1]] ), predicted_links_prob[i] ))
            # probability_dict[(self.numID_2_strID_dict[line_data_temp[i][0]], self.numID_2_strID_dict[line_data_temp[i][1]])] = predicted_links_prob[i]

        # fill prediction list
        self.prediction_list = prediction_list_dummy




    #####################EVALUATION#####################
    def check_if_correct(self):
        """
        Check if predicted edges are present in original graph.

        TO-DO
        -----
        The gain value edge attribute only exists for jitrik data at the moment.
        """
        is_correct_dummy = []
        gain_list_dummy = []
        for (node1, node2), prob in self.prediction_list:
            if self.G_probe.has_edge(node1, node2):
                is_correct_dummy.append(True)
                try: # to not raise exception for NIST data
                    gain_list_dummy.append(self.G_original[node1][node2]['gainValue'])
                except:
                    gain_list_dummy.append(True)
            else:
                is_correct_dummy.append(False)
                gain_list_dummy.append(0)
        self.is_correct = np.asarray(is_correct_dummy)
        self.gain_list = gain_list_dummy


    def get_prediction_list(self):
        import operator
        probability_dict = nx.get_edge_attributes(self.G_predict, 'probability')
        prediction_list_dummy = sorted(probability_dict.iteritems(), key=operator.itemgetter(1), reverse=True)

        # check whether predictions are correct
        prediction_list = []
        for prediction in prediction_list_dummy:
            prediction_list.append([prediction[0], prediction[1], self.G_original.has_edge(*(prediction[0]))])
        return prediction_list

    def get_dropout_list(self):
        return np.asarray(self.G_probe.edges())

    def get_probe_rank(self):
        self.probe_rank = np.zeros(self.probe_list.shape[0], dtype=int)

        for i in xrange(len(self.probe_list)):
            for j in xrange(len(self.prediction_list)):
                if set(self.probe_list[i]) == set(self.prediction_list[j][0]):
                    self.probe_rank[i] = j
        return self.probe_rank

    def calculate_P_and_N(self):
        self.P = len(self.G_probe.edges())
        self.N = len(self.G_training.nodes())*(len(self.G_training.nodes())-1)/2 - len(self.G_training.edges()) - len(self.G_probe.edges())

    def calculate_ROC(self):
        self.ROC = np.zeros((len(self.prediction_list), 2))
        TP = 0
        FP = 0
        for i, correct in enumerate(self.is_correct):
            if correct:
                TP += 1
            else:
                FP += 1

            self.ROC[i,0] = float(TP)/self.P   # TPR
            self.ROC[i,1] = float(FP)/self.N   # FPR

        return self.ROC

    def calculate_ROC_2(self):
        import numpy as np
        from sklearn.metrics import roc_curve

        # ranks  = np.arange(0,len(self.is_correct))
        # scores = 1 - np.divide(ranks.astype(float),len(lp_SPM.is_correct))
        scores = np.linspace(1,0,num=len(self.is_correct))

        fpr, tpr, thresholds = roc_curve(self.is_correct, scores)
        print tpr
        print fpr
        print thresholds

        self.ROC = np.zeros((len(self.prediction_list), 2))

        self.TPR = tpr   # TPR
        self.FPR = fpr   # FPR

        return (self.TPR, self.FPR)

    def plot_ROC(self, fig=None, save_switch=False, name='test', plotdirectory='../plots/', plotlabel = ''):
        import matplotlib.pyplot as plt
        # if self.ROC==None:
            # self.calculate_ROC()
        TPR = self.ROC[:,0]
        FPR = self.ROC[:,1]
        if not fig:
            fig = plt.figure()
        ax = plt.axes()
        ax.scatter(FPR, TPR, marker='.', label = plotlabel)
        ax.plot((0, 1), (0, 1), 'r--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC')
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        if save_switch==True:
            fig.savefig(plotdirectory+name+'.png')
            plt.close(fig)
        else: fig.draw()

    # def plot_ROC_2(self, fig=None, save_switch=False, name='test', plotdirectory='../plots/', plotlabel = ''):
    #     import matplotlib.pyplot as plt
    #     if self.ROC==None:
    #         self.calculate_ROC()
    #     TPR = self.TPR
    #     FPR = self.FPR
    #     if not fig:
    #         fig = plt.figure()
    #     ax = plt.axes()
    #     ax.scatter(FPR, TPR, marker='.', label = plotlabel)
    #     ax.plot((0, 1), (0, 1), 'r--')
    #     ax.set_xlabel('False Positive Rate')
    #     ax.set_ylabel('True Positive Rate')
    #     ax.set_title('ROC')
    #     ax.set_xlim(xmin=0.0, xmax=1.0)
    #     ax.set_ylim(ymin=0.0, ymax=1.0)
    #     if save_switch==True:
    #         fig.savefig(plotdirectory+name+'.png')
    #         plt.close(fig)
    #     else: fig.draw()

    def calculate_gain_measures(self, base):
        """
        ## @brief      Calculates the information retrieval measures described by Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems, 20(4), 422–446. https://doi.org/10.1145/582415.582418
        ##
        ## @param      masked_ranks  The masked array of predictions
        ## @param      base          The base of the logarithm in the discount
        ##                           factor (hyperparameter)
        ##
        ## @return     A tuple consisting of (CG, DCG, ICG, nCG, nDCG):
        ##             Cumulated gain, Discounted cumulated gain, ideal
        ##             cumulated gain, normalised cumulated gain, normalised
        ##             discounted cumulated gain.
        """

        # gain vector
        # G = self.is_correct.flatten() #old
        G = np.asarray(self.gain_list).flatten()

        # cumulated gain vector
        self.CG = np.cumsum(G)

        # discounted cumulative gain vector
        temp_log_arg  = np.log10(np.arange( base,G.shape[0] ))
        temp_discount = np.divide ( temp_log_arg , np.log10(base) )
        temp          = np.concatenate(( G[:base], np.divide( G[base:] , temp_discount  ) ))
        self.DCG = np.cumsum(temp)

        # Ideal gain vector
        IG   = np.sort(G, axis=0)[::-1]
        # Ideal cumulated gain vector
        self.ICG  = np.cumsum(IG)

        # normalised cumulated gain
        self.nCG  = np.divide( self.CG.astype(float) , self.ICG.astype(float) )

        # normalised discounted cumulated gain
        self.nDCG = np.divide( self.DCG , self.ICG.astype(float) )


    def plot_gain_metrics(self):
        import matplotlib.pyplot as plt
        #TODO: not working
        # if self.CG == None: #TODO: check also self.DCG, self.ICG, self.nCG, self.nDCG
        #     self.calculate_gain_measures()
        # plot gain metrics
        plt.figure()
        plt.plot(self.CG[:], label='Cumulated Gain', marker='.')
        plt.plot(self.DCG[:], label='Discounted Cumulated Gain', marker='.')
        plt.plot(self.ICG[:], label='Ideal Cumulated Gain', marker='.')
        plt.xlabel('Rank')
        plt.ylabel('Gain')
        plt.title('Gain curves')
        # plt.legend(loc=8)
        plt.show()

        # plot normalised gain metrics
        plt.figure()
        plt.plot(self.nCG[:], label='Normalised Cumulated Gain', marker='.')
        plt.plot(self.nDCG[:], label='Normalised Discounted Cumulated Gain', marker='.')
        plt.title('Normalised Gain curves')
        plt.ylabel('Normalised Gain')
        plt.xlabel('Rank')
        plt.ylim([0,1])
        # plt.legend(loc=2)
        plt.show()


    # @brief      Calculates the AUC value.
    #
    # @param      predictions                  The predictions
    # @param      ranks                        The ranks
    # @param      ranks_considered_percentage  The ranks considered percentage
    #
    # @return     The AUC value.
    #
    # AUC: Lu, L., and Zhou, T. (2010). Link Prediction in Complex Networks: A
    # Survey. Physica A, 390(6), 1150-1170.
    # https://doi.org/10.1016/j.physa.2010.11.027
    #
    def calculate_AUC_MC(self, sample_size_percentage=0.3, ranks_considered_percentage=0.99):

        b = 0 #number of times the predicted link has a higher rank than a randomly chosen non-existent link
        w = 0 #number of times the predicted link has a equal or worse rank than a randomly chosen non-existent link

        #sampling
        #TODO: what is a sensible sample size?
        #TODO: ranks_considered_percentage = 1.0 funktioniert nicht?
        sample_size = int(sample_size_percentage * self.is_correct.shape[0])
        for e in range(sample_size):
            temp_true = np.asarray(np.argwhere(self.is_correct==True).flatten())
            rank_probe = np.random.choice( temp_true[np.where( temp_true < int(ranks_considered_percentage * self.is_correct.shape[0]) )], 1, replace=True) #choosing from E_p
            temp_false = np.asarray(np.argwhere(self.is_correct==False).flatten())
            rank_nonexist = np.random.choice( temp_false[np.where( temp_false < int(ranks_considered_percentage * self.is_correct.shape[0]) )], 1, replace=True) #choosing from non-existent links, = U-E

            if (rank_probe < rank_nonexist):
                b+=1
            elif (rank_probe == rank_nonexist):
                w+=1

        self.AUC = (b + 0.5 * w) / float(sample_size)
        # print 'b',b
        # print 'w',w
        # print 'sample size', sample_size

        return self.AUC

    def calculate_AUC(self):
        from sklearn.metrics import roc_auc_score

        y_true = self.is_correct
        scores = np.linspace(1,0,num=len(self.is_correct))
        # TODO fix exception problem: in case all entries of is_correct are either True or False, the next line rasises an exception
        self.AUC = roc_auc_score(y_true, scores)

        return self.AUC

    def check_selection_rules(self):
        j = nx.get_node_attributes(self.G_original, 'J')
        parity = nx.get_node_attributes(self.G_original, 'parity')
        is_correct_dummy = []
        for (n1, n2), prob in self.prediction_list:
            if '/' in j[n1]:
                if np.abs(float(j[n1][:-2])/2 - float(j[n2][:-2])/2) < 1.2:
                    if parity[n1] != parity[n2]:
                        is_correct_dummy.append(True)
                    else:
                        is_correct_dummy.append(False)
                else:
                    is_correct_dummy.append(False)
            else:
                if np.abs(float(j[n1]) - float(j[n2])) < 1.2:
                    if parity[n1] != parity[n2]:
                        is_correct_dummy.append(True)
                    else:
                        is_correct_dummy.append(False)
                else:
                    is_correct_dummy.append(False)

        self.is_correct = is_correct_dummy
        self.P = np.sum(np.array(is_correct_dummy))
        self.N = np.sum(1 - np.array(is_correct_dummy))

class node2vec:
    G = None    # graph
    node_list = [] #list of node names
    node_vec_array = None # array containing the feature vectors of the nodes


    def learn_features(self, name = 'test', path_main = './node2vec/src/main.py', input_folder = './node2vec/graph/', output_folder = './node2vec/emb/', dimensions = 20, p=1.0, q=1.0, save_file=False):
        """
        Calculate features vectors of nodes in the graph using the node2vec algorithm.

        Parameters
        ----------
        p : float
            Return hyperparameter. (default=1.0)
        q : float
            Inout hyperparameter. (default=1.0)
        name : string
            Name of the run.
        path_main : string
            Path of the node2vec main.py.
        input_folder : string
            Path where the edgelist file will be saved.
        output_folder : string
            Path where the result will be outputted.
        dimensions : int
            Number of dimensions of the feature vectors.
        p : float
            Return hyperparameter. Default is 1.
        q : float
            In-out hyperparameter. Default is 1.
        save_file : boolean
            Switch to keep edgelist and output files.

        To-Do
        -----
        Don't call program via subprocess, but include the main as a library

        References
        ----------
        .. [1] Grover, A., & Leskovec, J. (2016). node2vec. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’16 (pp. 855-864). httpoutput_filenames://doi.org/10.1145/2939672.2939754
        """
        import subprocess
        import shlex
        import os

        edgelist_filename = name + '.edgelist'
        output_filename   = name + '.emb'

        nx.write_edgelist(self.G, input_folder + edgelist_filename, data=['matrixElement'])


        # cmd = 'python ' + path_main + ' --help'
        cmd = 'python ' + path_main + ' --input ' + input_folder + edgelist_filename + ' --output ' + output_folder + output_filename + ' --dimensions ' + str(dimensions) + ' --weighted' + ' --p ' + str(p) + ' --p ' + str(q) # Command in command-line style which will passed to the program.
        args = shlex.split(cmd)
        # run
        subprocess.call(args)

        with open(output_folder + output_filename, 'r') as output_file:
            line_reader = csv.reader(output_file, delimiter=' ')
            header = next(line_reader, None)  # returns the headers or `None` if the input is empty
            line_data = np.array(list(line_reader))
            output_file.close()

        n_nodes = int(header[0]) # number of nodes
        d = int(header[1]) # dimension of feature vectors

        self.node_vec_array = np.empty((n_nodes,d),dtype='|S64')
        for i, element in enumerate(line_data[:]):
            self.node_list.append(element[0])
            self.node_vec_array[i:] = element[1:]
        self.node_vec_array = self.node_vec_array.astype(float)

        if not save_file:
            os.remove(input_folder+edgelist_filename)
            os.remove(output_folder+output_filename)


# Copied from https://bbengfort.github.io/snippets/2016/06/23/graph-tool-from-networkx.html


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    import graph_tool.all as gt

    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    import graph_tool.all as gt

    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'nodeid' -- modify as needed!
    gtG.vertex_properties['nodeid'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['nodeid'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG