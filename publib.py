#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Armin Kekic, Julian Heiss, David Wellnitz
@date: February 2018

Module transforming the spectral data into NetworkX graphs.
The data has to be downloaded from http://www.fisica.unam.mx/research/tables/spectra/1el/index.shtml and https://physics.nist.gov/PhysRefData/ASD/lines_form.html as .csv files.

"""

# + how to download
# + load_network() & dessen abhaengigkeiten?

import numpy as np
import networkx as nx
import itertools
import csv
import warnings


ions         = np.array(['10B1.0','10B2.0','11B1.0','11B2.0','198Hg1.0','3He1.0','Ac1.0','Ac2.0','Ac3.0','Ac4.0','Ag1.0','Ag2.0','Ag3.0','Al1.0','Al10.0','Al11.0','Al12.0','Al13.0','Al2.0','Al3.0','Al4.0','Al5.0','Al6.0','Al7.0','Al8.0','Al9.0','Am1.0','Am2.0','Ar1.0','Ar10.0','Ar11.0','Ar12.0','Ar13.0','Ar14.0','Ar15.0','Ar16.0','Ar17.0','Ar18.0','Ar2.0','Ar3.0','Ar4.0','Ar5.0','Ar6.0','Ar7.0','Ar8.0','Ar9.0','As1.0','As2.0','As3.0','As4.0','As5.0','At1.0','Au1.0','Au2.0','Au3.0','B1.0','B2.0','B3.0','B4.0','B5.0','Ba1.0','Ba10.0','Ba11.0','Ba12.0','Ba2.0','Ba20.0','Ba21.0','Ba22.0','Ba24.0','Ba25.0','Ba26.0','Ba27.0','Ba28.0','Ba29.0','Ba3.0','Ba30.0','Ba31.0','Ba32.0','Ba33.0','Ba34.0','Ba35.0','Ba36.0','Ba37.0','Ba38.0','Ba4.0','Ba40.0','Ba41.0','Ba42.0','Ba43.0','Ba44.0','Ba45.0','Ba46.0','Ba47.0','Ba48.0','Ba49.0','Ba5.0','Ba50.0','Ba52.0','Ba53.0','Ba54.0','Ba55.0','Ba56.0','Ba6.0','Ba7.0','Ba8.0','Ba9.0','Be1.0','Be2.0','Be3.0','Be4.0','Bi1.0','Bi2.0','Bi3.0','Bi4.0','Bi5.0','Bk1.0','Bk2.0','Br1.0','Br2.0','Br3.0','Br4.0','Br5.0','C1.0','C2.0','C3.0','C4.0','C5.0','C6.0','Ca1.0','Ca10.0','Ca11.0','Ca12.0','Ca13.0','Ca14.0','Ca15.0','Ca16.0','Ca2.0','Ca3.0','Ca4.0','Ca5.0','Ca6.0','Ca7.0','Ca8.0','Ca9.0','Cd1.0','Cd2.0','Cd3.0','Cd4.0','Ce1.0','Ce2.0','Ce3.0','Ce4.0','Ce5.0','Cf1.0','Cf2.0','Cl1.0','Cl10.0','Cl2.0','Cl3.0','Cl4.0','Cl5.0','Cl6.0','Cl7.0','Cl8.0','Cl9.0','Cm1.0','Cm2.0','Co1.0','Co10.0','Co11.0','Co12.0','Co13.0','Co14.0','Co15.0','Co16.0','Co17.0','Co18.0','Co19.0','Co2.0','Co20.0','Co21.0','Co22.0','Co23.0','Co24.0','Co25.0','Co26.0','Co27.0','Co3.0','Co4.0','Co5.0','Co8.0','Co9.0','Cr1.0','Cr10.0','Cr11.0','Cr12.0','Cr13.0','Cr14.0','Cr15.0','Cr16.0','Cr17.0','Cr18.0','Cr19.0','Cr2.0','Cr20.0','Cr21.0','Cr22.0','Cr23.0','Cr24.0','Cr3.0','Cr4.0','Cr5.0','Cr6.0','Cr7.0','Cr8.0','Cr9.0','Cs1.0','Cs10.0','Cs11.0','Cs19.0','Cs2.0','Cs20.0','Cs23.0','Cs25.0','Cs26.0','Cs27.0','Cs28.0','Cs29.0','Cs3.0','Cs34.0','Cs37.0','Cs39.0','Cs4.0','Cs40.0','Cs41.0','Cs42.0','Cs43.0','Cs44.0','Cs45.0','Cs46.0','Cs47.0','Cs48.0','Cs49.0','Cs5.0','Cs51.0','Cs52.0','Cs53.0','Cs54.0','Cs55.0','Cs6.0','Cs7.0','Cs8.0','Cs9.0','Cu1.0','Cu10.0','Cu11.0','Cu12.0','Cu13.0','Cu14.0','Cu15.0','Cu16.0','Cu17.0','Cu18.0','Cu19.0','Cu2.0','Cu20.0','Cu21.0','Cu22.0','Cu23.0','Cu24.0','Cu25.0','Cu26.0','Cu27.0','Cu28.0','Cu29.0','Cu3.0','Cu4.0','Cu5.0','D1.0','Dy1.0','Dy2.0','Er1.0','Er2.0','Er3.0','Es1.0','Es2.0','Eu1.0','Eu2.0','Eu3.0','F1.0','F2.0','F3.0','F4.0','F5.0','F6.0','F7.0','F8.0','Fe1.0','Fe10.0','Fe11.0','Fe12.0','Fe13.0','Fe14.0','Fe15.0','Fe16.0','Fe17.0','Fe18.0','Fe19.0','Fe2.0','Fe20.0','Fe21.0','Fe22.0','Fe23.0','Fe24.0','Fe25.0','Fe26.0','Fe3.0','Fe4.0','Fe5.0','Fe6.0','Fe7.0','Fe8.0','Fe9.0','Fr1.0','Ga1.0','Ga13.0','Ga14.0','Ga15.0','Ga16.0','Ga17.0','Ga18.0','Ga19.0','Ga2.0','Ga20.0','Ga21.0','Ga22.0','Ga23.0','Ga24.0','Ga25.0','Ga26.0','Ga29.0','Ga3.0','Ga30.0','Ga31.0','Ga4.0','Ga5.0','Ga6.0','Ga7.0','Gd1.0','Gd2.0','Gd3.0','Gd4.0','Ge1.0','Ge2.0','Ge3.0','Ge4.0','Ge5.0','H1.0','He1.0','He2.0','Hf1.0','Hf2.0','Hf3.0','Hf4.0','Hf5.0','Hg1.0','Hg2.0','Hg3.0','Ho1.0','Ho2.0','I1.0','I2.0','I3.0','I4.0','I5.0','In1.0','In2.0','In3.0','In4.0','In5.0','Ir1.0','Ir2.0','Ir4.0','K1.0','K10.0','K11.0','K12.0','K13.0','K14.0','K15.0','K16.0','K17.0','K18.0','K19.0','K2.0','K3.0','K4.0','K5.0','K6.0','K7.0','K8.0','K9.0','Kr1.0','Kr10.0','Kr18.0','Kr19.0','Kr2.0','Kr20.0','Kr21.0','Kr22.0','Kr23.0','Kr24.0','Kr25.0','Kr26.0','Kr27.0','Kr28.0','Kr29.0','Kr3.0','Kr30.0','Kr31.0','Kr32.0','Kr33.0','Kr34.0','Kr35.0','Kr36.0','Kr4.0','Kr5.0','Kr6.0','Kr7.0','Kr8.0','Kr9.0','La1.0','La2.0','La3.0','La4.0','La5.0','Li1.0','Li2.0','Li3.0','Lu1.0','Lu2.0','Lu3.0','Lu4.0','Lu5.0','Mg1.0','Mg10.0','Mg11.0','Mg12.0','Mg2.0','Mg3.0','Mg4.0','Mg5.0','Mg6.0','Mg7.0','Mg8.0','Mg9.0','Mn1.0','Mn10.0','Mn11.0','Mn12.0','Mn13.0','Mn14.0','Mn15.0','Mn16.0','Mn17.0','Mn18.0','Mn19.0','Mn2.0','Mn20.0','Mn21.0','Mn22.0','Mn23.0','Mn24.0','Mn25.0','Mn3.0','Mn4.0','Mn5.0','Mn6.0','Mn7.0','Mn8.0','Mn9.0','Mo1.0','Mo10.0','Mo11.0','Mo12.0','Mo13.0','Mo14.0','Mo15.0','Mo16.0','Mo17.0','Mo18.0','Mo2.0','Mo23.0','Mo24.0','Mo25.0','Mo26.0','Mo27.0','Mo28.0','Mo29.0','Mo3.0','Mo30.0','Mo31.0','Mo32.0','Mo33.0','Mo34.0','Mo35.0','Mo38.0','Mo39.0','Mo4.0','Mo40.0','Mo41.0','Mo42.0','Mo5.0','Mo6.0','Mo7.0','Mo8.0','Mo9.0','N1.0','N2.0','N3.0','N4.0','N5.0','N6.0','N7.0','Na1.0','Na10.0','Na11.0','Na2.0','Na3.0','Na4.0','Na5.0','Na6.0','Na7.0','Na8.0','Na9.0','Nb1.0','Nb2.0','Nb3.0','Nb4.0','Nb5.0','Nd1.0','Nd2.0','Ne1.0','Ne2.0','Ne3.0','Ne4.0','Ne5.0','Ne6.0','Ne7.0','Ne8.0','Ne9.0','Ni1.0','Ni10.0','Ni11.0','Ni12.0','Ni13.0','Ni14.0','Ni15.0','Ni16.0','Ni17.0','Ni18.0','Ni19.0','Ni2.0','Ni20.0','Ni21.0','Ni22.0','Ni23.0','Ni24.0','Ni25.0','Ni26.0','Ni27.0','Ni28.0','Ni3.0','Ni4.0','Ni5.0','Ni7.0','Ni9.0','Np1.0','O1.0','O2.0','O3.0','O4.0','O5.0','O6.0','O7.0','O8.0','Os1.0','Os2.0','P1.0','P10.0','P11.0','P12.0','P13.0','P2.0','P3.0','P4.0','P5.0','P6.0','P7.0','P8.0','P9.0','Pa1.0','Pa2.0','Pb1.0','Pb2.0','Pb3.0','Pb4.0','Pb5.0','Pd1.0','Pd2.0','Pd3.0','Pm1.0','Pm2.0','Po1.0','Pr1.0','Pr2.0','Pr3.0','Pr4.0','Pr5.0','Pt1.0','Pt2.0','Pt4.0','Pt5.0','Pu1.0','Pu2.0','Ra1.0','Ra2.0','Rb1.0','Rb10.0','Rb11.0','Rb12.0','Rb13.0','Rb19.0','Rb2.0','Rb20.0','Rb21.0','Rb22.0','Rb23.0','Rb24.0','Rb25.0','Rb26.0','Rb27.0','Rb28.0','Rb29.0','Rb3.0','Rb30.0','Rb31.0','Rb33.0','Rb34.0','Rb35.0','Rb36.0','Rb37.0','Rb4.0','Rb5.0','Rb6.0','Rb7.0','Rb8.0','Rb9.0','Re1.0','Re2.0','Rh1.0','Rh2.0','Rh3.0','Rn1.0','Ru1.0','Ru2.0','Ru3.0','S1.0','S10.0','S11.0','S12.0','S13.0','S14.0','S15.0','S16.0','S2.0','S3.0','S4.0','S5.0','S6.0','S7.0','S8.0','S9.0','Sb1.0','Sb2.0','Sb3.0','Sb4.0','Sb5.0','Sc1.0','Sc10.0','Sc11.0','Sc12.0','Sc13.0','Sc14.0','Sc15.0','Sc16.0','Sc17.0','Sc18.0','Sc19.0','Sc2.0','Sc20.0','Sc21.0','Sc3.0','Sc4.0','Sc5.0','Sc6.0','Sc7.0','Sc8.0','Sc9.0','Se1.0','Se2.0','Se3.0','Se4.0','Se5.0','Si1.0','Si10.0','Si11.0','Si12.0','Si13.0','Si2.0','Si3.0','Si4.0','Si5.0','Si6.0','Si7.0','Si8.0','Si9.0','Sm1.0','Sm2.0','Sn1.0','Sn2.0','Sn3.0','Sn4.0','Sn5.0','Sr1.0','Sr10.0','Sr11.0','Sr12.0','Sr13.0','Sr14.0','Sr2.0','Sr20.0','Sr21.0','Sr22.0','Sr23.0','Sr24.0','Sr25.0','Sr26.0','Sr27.0','Sr28.0','Sr29.0','Sr3.0','Sr30.0','Sr31.0','Sr32.0','Sr33.0','Sr34.0','Sr35.0','Sr36.0','Sr37.0','Sr38.0','Sr4.0','Sr5.0','Sr6.0','Sr7.0','Sr8.0','Sr9.0','T1.0','Ta1.0','Ta2.0','Ta4.0','Ta5.0','Tb1.0','Tb2.0','Tb4.0','Tc1.0','Tc2.0','Te1.0','Te2.0','Th1.0','Th2.0','Th3.0','Th4.0','Ti1.0','Ti10.0','Ti11.0','Ti12.0','Ti13.0','Ti14.0','Ti15.0','Ti16.0','Ti17.0','Ti18.0','Ti19.0','Ti2.0','Ti20.0','Ti21.0','Ti22.0','Ti3.0','Ti4.0','Ti5.0','Ti6.0','Ti7.0','Ti8.0','Ti9.0','Tl1.0','Tl2.0','Tl3.0','Tl4.0','Tm1.0','Tm2.0','Tm3.0','U1.0','U2.0','V1.0','V10.0','V11.0','V12.0','V13.0','V14.0','V15.0','V16.0','V17.0','V18.0','V19.0','V2.0','V20.0','V21.0','V22.0','V23.0','V3.0','V4.0','V5.0','V6.0','V7.0','V8.0','V9.0','W1.0','W14.0','W2.0','W28.0','W29.0','W3.0','W30.0','W31.0','W32.0','W33.0','W34.0','W35.0','W36.0','W37.0','W38.0','W39.0','W4.0','W40.0','W41.0','W42.0','W43.0','W44.0','W45.0','W46.0','W47.0','W48.0','W49.0','W5.0','W50.0','W51.0','W52.0','W53.0','W54.0','W55.0','W56.0','W57.0','W58.0','W59.0','W6.0','W60.0','W61.0','W62.0','W63.0','W64.0','W65.0','W66.0','W67.0','W68.0','W69.0','W7.0','W70.0','W71.0','W72.0','W73.0','W74.0','W8.0','Xe1.0','Xe10.0','Xe11.0','Xe19.0','Xe2.0','Xe25.0','Xe26.0','Xe27.0','Xe28.0','Xe29.0','Xe3.0','Xe4.0','Xe43.0','Xe44.0','Xe45.0','Xe5.0','Xe51.0','Xe52.0','Xe53.0','Xe54.0','Xe6.0','Xe7.0','Xe8.0','Xe9.0','Y1.0','Y2.0','Y3.0','Y4.0','Y5.0','Yb1.0','Yb2.0','Yb3.0','Yb4.0','Zn1.0','Zn2.0','Zn3.0','Zn4.0','Zr1.0','Zr2.0','Zr3.0','Zr4.0','Zr5.0','Zr6.0'])
one_electron = np.array(['H1.0','T1.0','D1.0','He2.0','Li3.0','Be4.0','C6.0','N7.0','O8.0','Na11.0','Mg12.0','Al13.0','S16.0','Ar18.0','K19.0','Sc21.0','Ti22.0','V23.0','Cr24.0','Mn25.0','Fe26.0','Co27.0','Ni28.0','Cu29.0','Ga31.0','Kr36.0','Rb37.0','Sr38.0','Mo42.0','Cs55.0','Ba56.0','W74.0'])
two_electron = np.array(['He1.0','Li2.0','Be3.0','B4.0','C5.0','N6.0','O7.0','F8.0','Ne9.0','Na10.0','Mg11.0','Al12.0','Si13.0','S15.0','Ar17.0','Sc20.0','Ti21.0','V22.0','Cr23.0','Mn24.0','Fe25.0','Co26.0','Ni27.0','Cu28.0','Ga30.0','Kr35.0','Sr37.0','Mo41.0','Cs54.0','Ba55.0'])



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
    intensity  = line_data[atom_index,17] # The Einstein A coefficient
    obs_wl     = line_data[atom_index,2]  # observed wavelength
    calc_wl    = line_data[atom_index,5]  # calculated wavelength
    trans_type = line_data[atom_index,22] # transition type

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
        G.add_weighted_edges_from(zip(low_levels, upp_levels,obs_wl), weight = 'obeservedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(obs_wl)), weight = 'inverseObeservedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,calc_wl), weight = 'calculatedWavelength')
        G.add_weighted_edges_from(zip(low_levels, upp_levels,np.reciprocal(calc_wl)), weight = 'inverseCalculatedWavelength')
    else:
        G.add_edges_from(zip(low_levels,upp_levels))

    # set edge attributes
    nx.set_edge_attributes(G, 'type', dict(zip(zip(low_levels, upp_levels),trans_type)))

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
        ### dictionary for l-states, used in calculation of modelName
        L_dictionary = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4, 'h':5, 'i':6, 'k':7, 'l':8, 'm':9, 'n':10, 'o':11, 'q':12, 'r':13, 't':14, 'u':15, 'v':16, 'w':17, 'x':18, 'y':19, 'z':20, 'a':21, 'b':22, 'c':23, 'e':24, 'j':25}

        for n in nx.nodes(G):

            # total J
            j_value = (level_data[level_data[:,24] == n,8])
            if j_value.size == 0:
                j[n] = ''
            else:
                j[n] = j_value[0]

            # configuration
            conf_value = (level_data[level_data[:,24] == n,5])
            if conf_value.size == 0:
                conf[n] = ''
            else:
                conf[n] = conf_value[0]

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
    nx.set_node_attributes(G,'conf',conf)
    nx.set_node_attributes(G,'term',term)
    nx.set_node_attributes(G,'J',j)
    nx.set_node_attributes(G,'n',n_dict)
    nx.set_node_attributes(G,'l',l)
    nx.set_node_attributes(G,'parity',parity)
    nx.set_node_attributes(G,'modelName',modelName)
    nx.set_node_attributes(G,'dummyCommunity',dummyCommunity)

    if dictionaries:
        return (G, conf, term, j)
    else:
        return G


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
                        + E2*[1.0e8]
                        + E3*[1.0e8]
                        + M1*[1.0e8]
                        + M2*[1.0e8]
                        + M3*[1.0e8]
                        + E1_M2*[1.0]
                        + M2_E3*[1.0e8]
                        + M1_E2*[1.0e8]
                        + E2_M3*[1.0e8]
                        + E1_M2_E3*[1.0]
                        + M1_E2_M3*[1.0e8] )
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
        nx.set_edge_attributes(G, 'transitionType', transitionType)
        nx.set_edge_attributes(G, 'rescaledMatrixElement', rescaledMatrixElement) # Add rescaled matrix element weight to edges


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



    nx.set_node_attributes(G, 'parity', parity)
    nx.set_node_attributes(G, 'term', term)
    nx.set_node_attributes(G, 'J', j)
    nx.set_node_attributes(G, 'n', n)
    nx.set_node_attributes(G, 'l', l)
    nx.set_node_attributes(G, 'dummyCommunity', dummyCommunity)

    if max_n != None:
        nodes_over_max_n = [node for (node,n_number) in n.iteritems() if int(n_number) > max_n]
        G.remove_nodes_from(nodes_over_max_n)

    return G