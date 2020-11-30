"""
@author: Julian Heiss
Description: Plots all wavelengths in the NIST Dataset (calculated and observed wavelengths)

"""
try:
	reload
except NameError:
	# Python 3
	from imp import reload
import nx2
reload(nx2)

import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import warnings



def plot_all_wavelengths(line_data_path):
	with open(line_data_path) as line_file:
		line_reader = csv.reader(line_file)
		line_data_temp = np.array(list(line_reader))
		row_count = len(line_data_temp)

	# create empty 2d array and fill it with the data. 33 is the number of columns *hardcoded!
	line_data = np.empty((row_count,33),dtype='|S64')
	for i in range(len(line_data_temp)):
	   for k in range(33):
		   line_data[i,k] = line_data_temp[i][k]

	
	calc_wl = np.array(line_data[1:,5]) # calculated wavelength
	calc_wl_ok  = (calc_wl != '')
	calc_wl     = calc_wl[calc_wl_ok]
	calc_wl = calc_wl.astype(float)
	calc_wl = calc_wl/10 #nm
	calc_wn = (1/(calc_wl/10))*1e7 #wavenumber in cm-1

	obs_wl = np.array(line_data[1:,2]) # observed wavelength
	obs_wl_ok  = (obs_wl != '')
	obs_wl     = obs_wl[obs_wl_ok]
	obs_wl = obs_wl.astype(float)
	obs_wl = obs_wl/10 #nm
	obs_wn = (1/(obs_wl/10))*1e7 #wavenumber in cm-1

	#wavelength plot
	fig = plt.figure(0)
	binwidth = 10
	highest_bin = 4000
	bins=np.arange(0, highest_bin + binwidth, binwidth)
	plt.hist(np.clip(calc_wl, bins[0], bins[-1]), bins=bins, normed=True, facecolor='blue', alpha=0.5, label='calc')
	plt.hist(np.clip(obs_wl, bins[0], bins[-1]), bins=bins, normed=True, facecolor='red', alpha=0.5, label='obs')
	plt.title("NIST wavelengths")
	plt.xlabel('wavelength [nm]')
	plt.ylabel('normalised occurrence')
	plt.legend(loc='upper right')
	plt.savefig('../plots/all_wavelengths.png')

	#wavenumber plot
	fig = plt.figure(1)
	binwidth = 10000
	highest_bin = 4000000
	bins=np.arange(0, highest_bin + binwidth, binwidth)
	plt.hist(np.clip(calc_wn, bins[0], bins[-1]), bins=bins, normed=True, facecolor='blue', alpha=0.5, label='calc')
	plt.hist(np.clip(obs_wn, bins[0], bins[-1]), bins=bins, normed=True, facecolor='red', alpha=0.5, label='obs')
	plt.title("NIST wavenumbers")
	plt.xlabel('wavenumber [cm^-1]')
	plt.ylabel('normalised occurrence')
	plt.legend(loc='upper right')
	plt.savefig('../plots/all_wavenumbers.png')



plot_all_wavelengths(line_data_path='../data/ASD54_lines.csv')