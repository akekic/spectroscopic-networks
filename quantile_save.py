"""
@author: Julian Heiss

Basic fitting routines for the degree distributions of networks. Network models considered are: Scale-free (Power-law) and Erdos Renyi (Poisson).
The exponent of the scale-free network is estimated by a fit of the degree distribution directly (binned fit) or via a fit of the quantile function.
The parameter of the Poisson distribution is estimated by a fit which uses the log likelihood of the distribution (thus not having to care about binning fits).
TODO: Error models, goodness of fits

"""
try:
	reload
except NameError:
	# Python 3
	from imp import reload
import nx2
reload(nx2)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.misc import factorial

class Parameter:
	def __init__(self, value):
			self.value = value

	def set(self, value):
			self.value = value

	def __call__(self):
			return self.value
# not used
# def fit(function, parameters, y, x = None):
#     def f(params):
#         i = 0
#         for p in parameters:
#             p.set(params[i])
#             i += 1
#         return y - function(x)

#     if x is None: x = np.arange(y.shape[0])
#     p = [param() for param in parameters]
#     return optimize.leastsq(f, p)

# quick auto-plot of the data
def plot_hist(data):
	fig = plt.figure()
	plt.hist(data, bins='auto')
	plt.show()

# calculating the negative log likelihood of the dsitribution using
# the function 'func' that has been passed
def negLogLikelihood(params, data, func):
	# """ the negative log-Likelohood-Function"""
	lnl = - np.sum(np.log(func(data, params[0], params[1])))
	return lnl

# Umkehrfunktion der Cdf von scale-free. prefac=1/b wenn cdf = b*x^(-(exp-1))
def nroot(x, exp, prefac):
	return (prefac*x) ** (1.0 / -(exp-1))

# simple power law, exp and prefac are the fitting parameters
def power_law(x, exp, prefac):
	return prefac*(x ** (- exp))

# poisson function, parameter lamb is the fit parameter
def poisson_manual(x, lamb):
	return (lamb**x/factorial(x)) * np.exp(-lamb)

#return all the necessary quantities to carry on with a fit
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


#####
# Scale-free / Power-law
# what about the logarithmic scale?
# fitfunc = lambda p, x:.... 
def scale_free_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('scale_free_quantile_fit') 
	#input
	parameter_init 	 = (0.5,1)
	sigma_func       = np.ones_like(data_points)  #sigma=0.1 for every value

	#fit
	parameters, cov_matrix = curve_fit(nroot, np.linspace(0,1,len(deg_dist)),
										data_points, p0=parameter_init, sigma=sigma_func)
	par_errors             = np.sqrt(np.diag(cov_matrix))

	parameters_results.append(parameters)
	cov_matrix_results.append(cov_matrix)
	parameters_errors_results.append(par_errors)
	print parameters
	print cov_matrix
	print par_errors
	 
	if plot == True:
		fig = plt.figure()
		x_axis= np.linspace(0,1,len(deg_dist)) #use as x_values
		plt.scatter(x_axis, data_points, label='data')
		plt.title("Degree inverse cumulative distribution - Scale-free")
		plt.plot(x_axis, nroot(np.linspace(0,1,len(deg_dist)), parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.legend(loc=8)
		plt.show()

def scale_free_deg_curve_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('scale_free_deg_curve_fit') 
	#input
	parameter_init 	 = (0.5,1)
	x_values         = np.linspace(0,len(data)-1,len(data)) #use as x_values
	sigma_func       = np.ones_like(x_values)  #sigma=1 for every value

	# print len(data)
	# print data
	# print x_values

	#get y values
	fig = plt.figure()
	binwidth = 1
	data, bin_edges, patches = plt.hist(data, bins=np.arange(0, len(data) + binwidth, binwidth),
										   facecolor='green', alpha=0.75)

	# calculate binmiddles
	bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

	#fit
	parameters, cov_matrix = curve_fit(power_law, bin_middles, data, p0=parameter_init, sigma=sigma_func)
	par_errors             = np.sqrt(np.diag(cov_matrix))
	parameters_results.append(parameters)
	cov_matrix_results.append(cov_matrix)
	parameters_errors_results.append(par_errors)
	print parameters
	print cov_matrix
	print par_errors
	 

	if plot == True:
		x_axis = x_values
		plt.title("Degree distribution - power-Law")
		plt.plot(x_axis, power_law(x_values, parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.legend(loc=8)
		plt.show()

# using the power law package
def power_law_test(data):
	import powerlaw

	#convert the degree values to a histogram
	fig = plt.figure()
	binwidth = 1
	data, bin_edges, patches = plt.hist(data, bins=np.arange(0, len(data) + binwidth, binwidth),
										   facecolor='green', alpha=0.75)

	# calculate binmiddles
	bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1]) #probleme bei berechnung
	# print data

	powerlaw.plot_pdf(data, color='b', linear_bins=True)
	results = powerlaw.Fit(data, discrete=True)
	print 'alpha: ', results.power_law.alpha
	print 'sigma: ', results.power_law.sigma
	print 'xmin: ', results.power_law.xmin
	R, p = results.distribution_compare('power_law', 'truncated_power_law') 
	print R
	print p
	plt.show()

#klappt nicht wirklich
# def scale_free_logliho_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results):

#     # minimize the negative log-Likelihood
# 	result = minimize(negLogLikelihood,  # function to minimize
#                   x0=(1,1.5),     # start values: exp, prefac
#                   args=(data, power_law),      # additional arguments for function
#                   method='BFGS',   # minimization method, see docs
#                   )
# 	# result is a scipy optimize result object, the fit parameters 
# 	# are stored in result.x
# 	print result

# 	# plot poisson-deviation with fitted parameter
# 	fig = plt.figure()
# 	x_plot= np.linspace(0,max(data)-1,1000) #use as x_values
# 	binwidth = 1

# 	plt.hist(data, bins=np.arange(min(data), max(data) + binwidth -0.5, binwidth), facecolor='green', alpha=0.75)
# 	plt.plot(x_plot, power_law(x_plot, result.x[0], result.x[1]), 'r-', lw=2, label='fit',)
# 	plt.title("Degree distribution - Power-Law")
# 	plt.legend(loc=8)
# 	plt.show()
######

###### Erdos Renyi -> Poisson
def erdos_deg_logliho_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('erdos_deg_logliho_fit') 
	# minimize the negative log-Likelihood
	result = minimize(negLogLikelihood,  # function to minimize
				  x0=np.ones(1),     # start value
				  args=(data,),      # additional arguments for function
				  method='Powell',   # minimization method, see docs
				  )
	# result is a scipy optimize result object, the fit parameters 
	# are stored in result.x
	print result

	# plot poisson-deviation with fitted parameter
	fig = plt.figure()
	x_values = np.linspace(0,len(data)-1,1000) #use as x_values
	binwidth = 1

	if plot == True:
		plt.hist(data, bins=np.arange(0, len(data) + binwidth -0.5, binwidth), normed=True, facecolor='green', alpha=0.75)
		x_axis = x_values
		plt.plot(x_axis, poisson_manual(x_values, result.x), 'r-', lw=2, label='fit',)
		plt.title("Degree distribution - Poisson")
		plt.legend(loc=8)
		plt.show()

# def erdos_deg(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results):

# def erdos_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results):
######




################## RUN ########################################
def main(plot_degree=False):
	print('Fits Routines')
# Take Model Network as input
	# G = nx2.Model_Network('../data/jitrik-bunge-e1-set1.csv', '../data/jitrik-bunge-e1-set2.csv', '../data/jitrik-bunge-e2-set1.csv', '../data/jitrik-bunge-e2-set2.csv', '../data/jitrik-bunge-m1-set1.csv', '../data/jitrik-bunge-m1-set2.csv')
	# E1 = [(u,v) for (u,v) in G0.edges() if G0[u][v]['label'] == 'E1']
	# G = nx.Graph()
	# G.add_edges_from(E1)


	# savefigs        = True
	draw_plots      = True

	ions_list       = nx2.one_electron[:4]
	empty_ions      = []

# loop through all ions in ions_list
	for ion in ions_list:
# create network with the NIST lines
		G = nx2.spectroscopic_network(ion, weighted=True, check_wl=True, dictionaries=False)
		if G.size() == 0:
			print('Empty graph.')
			empty_ions.append(ion)
			continue
		elif G.size() < 50:
			print('Less than 50 nodes in Graph.')
			empty_ions.append(ion)
			continue
		else:
			print '#######Ion:', ion
			deg_dist, log_deg_dist, normalised_deg_dist,  quantile, num_data_points = create_deg_dist(G)
			if plot_degree == True:
				plot_hist(deg_dist)
		#lists to eventually save the results
			parameters_results        = [] #list with fitted (optimised) parameters. Each entry is a list of parameters for the used fit function	
			cov_matrix_results        = [] #list of covariance matrices
			parameters_errors_results = [] #list of standard deviations
		 # which fit routines do you want to use?
			# scale_free_quantile_fit(quantile, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=draw_plots)
			print deg_dist
			# scale_free_deg_curve_fit(deg_dist, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot=draw_plots)
			# erdos_deg_logliho_fit(deg_dist, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot=draw_plots)
			# power_law_test(deg_dist)

if __name__ == '__main__':
  main()


######################################
#2nd version
"""
@author: Julian Heiss

Basic fitting routines for the degree distributions of networks. Network models considered are: Scale-free (Power-law) and Erdos Renyi (Poisson).
The exponent of the scale-free network is estimated by a fit of the degree distribution directly (binned fit) or via a fit of the quantile function.
The parameter of the Poisson distribution is estimated by a fit which uses the log likelihood of the distribution (thus not having to care about binning fits).

"""
try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.misc import factorial

class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value
# not used
# def fit(function, parameters, y, x = None):
#     def f(params):
#         i = 0
#         for p in parameters:
#             p.set(params[i])
#             i += 1
#         return y - function(x)

#     if x is None: x = np.arange(y.shape[0])
#     p = [param() for param in parameters]
#     return optimize.leastsq(f, p)

# quick auto-plot of the data
def plot_hist(data):
	fig = plt.figure()
	plt.hist(data, bins='auto')
	plt.show()

# calculating the negative log likelihood of the dsitribution using
# the function 'func' that has been passed
def negLogLikelihood(params, data, func):
   	# """ the negative log-Likelohood-Function"""
   	lnl = - np.sum(np.log(func(data, params[0], params[1])))
   	return lnl

# Umkehrfunktion der Cdf von scale-free. prefac=1/b wenn cdf = b*x^(-(exp-1))
def nroot(x, exp, prefac):
	return (prefac*x) ** (1.0 / -(exp-1))

# simple power law, exp and prefac are the fitting parameters
def power_law(x, exp, prefac):
	return prefac*(x ** (- exp))

# poisson function, parameter lamb is the fit parameter
def poisson_manual(x, lamb):
    return (lamb**x/factorial(x)) * np.exp(-lamb)

#return all the necessary quantities to carry on with a fit
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


#####
# Scale-free / Power-law
# what about the logarithmic scale?
# fitfunc = lambda p, x:.... 
def scale_free_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('scale_free_quantile_fit') 
	#input
	parameter_init 	 = (0.5,1)
	sigma_func       = np.ones_like(data_points)  #sigma=0.1 for every value

	#fit
	parameters, cov_matrix = curve_fit(nroot, np.linspace(0,1,len(deg_dist)),
										data_points, p0=parameter_init, sigma=sigma_func)
	par_errors             = np.sqrt(np.diag(cov_matrix))

	parameters_results.append(parameters)
	cov_matrix_results.append(cov_matrix)
	parameters_errors_results.append(par_errors)
	print parameters
	print cov_matrix
	print par_errors
	 
	if plot == True:
		fig = plt.figure()
		x_axis= np.linspace(0,1,len(deg_dist)) #use as x_values
		plt.scatter(x_axis, data_points, label='data')
		plt.title("Degree inverse cumulative distribution - Scale-free")
		plt.plot(x_axis, nroot(np.linspace(0,1,len(deg_dist)), parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.legend(loc=8)
		plt.show()

def scale_free_deg_curve_fit(data, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('scale_free_deg_curve_fit') 
	#input
	parameter_init 	     = (2,2)
	x_values = np.linspace(0,max(data)-1,max(data)) #use as x_values
	sigma_func = np.ones_like(x_values)  #sigma=1 for every value

	#get y values
	fig = plt.figure()
	binwidth = 1
	data, bin_edges, patches = plt.hist(data, bins=np.arange(0, max(data) + binwidth, binwidth),
										   facecolor='green', alpha=0.75)

	# calculate binmiddles
	bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1]) #probleme bei berechnung

	#fit
	parameters, cov_matrix = curve_fit(power_law, bin_middles, data, p0=parameter_init, sigma=sigma_func)
	par_errors             = np.sqrt(np.diag(cov_matrix))
	parameters_results.append(parameters)
	cov_matrix_results.append(cov_matrix)
	parameters_errors_results.append(par_errors)
	print parameters
	print cov_matrix
	print par_errors
	 

	if plot == True:
		x_axis = x_values
		plt.title("Degree distribution - power-Law")
		plt.plot(x_axis, power_law(x_values, parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.legend(loc=8)
		plt.show()


#klappt nicht wirklich
# def scale_free_logliho_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results):

#     # minimize the negative log-Likelihood
# 	result = minimize(negLogLikelihood,  # function to minimize
#                   x0=(1,1.5),     # start values: exp, prefac
#                   args=(data, power_law),      # additional arguments for function
#                   method='BFGS',   # minimization method, see docs
#                   )
# 	# result is a scipy optimize result object, the fit parameters 
# 	# are stored in result.x
# 	print result

# 	# plot poisson-deviation with fitted parameter
# 	fig = plt.figure()
# 	x_plot= np.linspace(0,max(data)-1,1000) #use as x_values
# 	binwidth = 1

# 	plt.hist(data, bins=np.arange(min(data), max(data) + binwidth -0.5, binwidth), facecolor='green', alpha=0.75)
# 	plt.plot(x_plot, power_law(x_plot, result.x[0], result.x[1]), 'r-', lw=2, label='fit',)
# 	plt.title("Degree distribution - Power-Law")
# 	plt.legend(loc=8)
# 	plt.show()
######

###### Erdos Renyi -> Poisson
def erdos_deg_logliho_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('erdos_deg_logliho_fit') 
    # minimize the negative log-Likelihood
	result = minimize(negLogLikelihood,  # function to minimize
                  x0=np.ones(1),     # start value
                  args=(data,),      # additional arguments for function
                  method='Powell',   # minimization method, see docs
                  )
	# result is a scipy optimize result object, the fit parameters 
	# are stored in result.x
	print result

	# plot poisson-deviation with fitted parameter
	fig = plt.figure()
	x_values = np.linspace(0,max(data)-1,1000) #use as x_values
	binwidth = 1

	if plot == True:
		plt.hist(data, bins=np.arange(min(data), max(data) + binwidth -0.5, binwidth), normed=True, facecolor='green', alpha=0.75)
		x_axis = x_values
		plt.plot(x_axis, poisson_manual(x_values, result.x), 'r-', lw=2, label='fit',)
		plt.title("Degree distribution - Poisson")
		plt.legend(loc=8)
		plt.show()

# def erdos_deg(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results):

# def erdos_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results):
######




################## RUN ########################################
def main(plot_degree=False):
 	print('Fits')

 	# create network with the NIST lines
 	G = nx2.spectroscopic_network('H1.0', weighted=True, check_wl=True, dictionaries=False)

 	# Take Model Network as input
 	# G = nx2.Model_Network('../data/jitrik-bunge-e1-set1.csv', '../data/jitrik-bunge-e1-set2.csv', '../data/jitrik-bunge-e2-set1.csv', '../data/jitrik-bunge-e2-set2.csv', '../data/jitrik-bunge-m1-set1.csv', '../data/jitrik-bunge-m1-set2.csv')
 	# E1 = [(u,v) for (u,v) in G0.edges() if G0[u][v]['label'] == 'E1']
 	# G = nx.Graph()
 	# G.add_edges_from(E1)

 	deg_dist, log_deg_dist, normalised_deg_dist,  quantile, num_data_points = create_deg_dist(G)
 	if plot_degree == True:
	 	plot_hist(deg_dist)

	#lists to eventually save the results
 	parameters_results        = [] #list with fitted (optimised) parameters. Each entry is a list of parameters for the used fit function	
 	cov_matrix_results        = [] #list of covariance matrices
 	parameters_errors_results = [] #list of standard deviations

 	# which fit routines do you want to make?
	scale_free_quantile_fit(quantile, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=False)
	scale_free_deg_curve_fit(deg_dist, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot=True)
	# erdos_deg_logliho_fit(deg_dist, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot=False)

if __name__ == '__main__':
  main()
"""


########################
3rd version
@author: Julian Heiss

Basic fitting routines for the degree distributions of networks. Network models considered are: Scale-free (Power-law) and Erdos Renyi (Poisson).
The exponent of the scale-free network is estimated by a fit of the degree distribution directly (binned fit) or via a fit of the quantile function.
The parameter of the Poisson distribution is estimated by a fit which uses the log likelihood of the distribution (thus not having to care about binning fits).
TODO: Error models, goodness of fits

"""
try:
	reload
except NameError:
	# Python 3
	from imp import reload
import nx2
reload(nx2)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.misc import factorial

class Parameter:
	def __init__(self, value):
			self.value = value

	def set(self, value):
			self.value = value

	def __call__(self):
			return self.value
# not used
# def fit(function, parameters, y, x = None):
#     def f(params):
#         i = 0
#         for p in parameters:
#             p.set(params[i])
#             i += 1
#         return y - function(x)

#     if x is None: x = np.arange(y.shape[0])
#     p = [param() for param in parameters]
#     return optimize.leastsq(f, p)

# quick auto-plot of the data
def plot_hist(data):
	fig = plt.figure()
	plt.hist(data, bins='auto')
	plt.show()

# calculating the negative log likelihood of the dsitribution using
# the function 'func' that has been passed
def negLogLikelihood(params, data, func):
	# """ the negative log-Likelohood-Function"""
	lnl = - np.sum(np.log(func(data, params[0], params[1])))
	return lnl

# Umkehrfunktion der Cdf von scale-free. prefac=1/b wenn cdf = b*x^(-(exp-1))
def nroot(x, exp, prefac):
	return (prefac*x) ** (1.0 / -(exp-1))

# simple power law, exp and prefac are the fitting parameters
def power_law(x, exp, prefac):
	return prefac*(x ** (- exp))

# poisson function, parameter lamb is the fit parameter
def poisson_manual(x, lamb):
	return (lamb**x/factorial(x)) * np.exp(-lamb)

#return all the necessary quantities to carry on with a fit
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


#####
# Scale-free / Power-law
# what about the logarithmic scale?
# fitfunc = lambda p, x:.... 
def scale_free_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('scale_free_quantile_fit') 
	#input
	parameter_init 	 = (0.5,1)
	sigma_func       = np.ones_like(data_points)  #sigma=0.1 for every value

	#fit
	parameters, cov_matrix = curve_fit(nroot, np.linspace(0,1,len(deg_dist)),
										data_points, p0=parameter_init, sigma=sigma_func)
	par_errors             = np.sqrt(np.diag(cov_matrix))

	parameters_results.append(parameters)
	cov_matrix_results.append(cov_matrix)
	parameters_errors_results.append(par_errors)
	print parameters
	print cov_matrix
	print par_errors
	 
	if plot == True:
		fig = plt.figure()
		x_axis= np.linspace(0,1,len(deg_dist)) #use as x_values
		plt.scatter(x_axis, data_points, label='data')
		plt.title("Degree inverse cumulative distribution - Scale-free")
		plt.plot(x_axis, nroot(np.linspace(0,1,len(deg_dist)), parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.legend(loc=8)
		plt.show()

def scale_free_deg_curve_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('scale_free_deg_curve_fit') 
	#input
	parameter_init 	 = (0.5,1)
	x_values         = np.linspace(0,len(data)-1,len(data)) #use as x_values
	sigma_func       = np.ones_like(x_values)  #sigma=1 for every value

	# print len(data)
	# print data
	# print x_values

	#get y values
	fig = plt.figure()
	binwidth = 1
	data, bin_edges, patches = plt.hist(data, bins=np.arange(0, len(data) + binwidth, binwidth),
										   facecolor='green', alpha=0.75)

	# calculate binmiddles
	bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

	#fit
	parameters, cov_matrix = curve_fit(power_law, bin_middles, data, p0=parameter_init, sigma=sigma_func)
	par_errors             = np.sqrt(np.diag(cov_matrix))
	parameters_results.append(parameters)
	cov_matrix_results.append(cov_matrix)
	parameters_errors_results.append(par_errors)
	print parameters
	print cov_matrix
	print par_errors
	 

	if plot == True:
		x_axis = x_values
		plt.title("Degree distribution - power-Law")
		plt.plot(x_axis, power_law(x_values, parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.legend(loc=8)
		plt.show()

# using the power law package
def power_law_test(data):
	import powerlaw

	#convert the degree values to a histogram
	fig = plt.figure()
	binwidth = 1
	data, bin_edges, patches = plt.hist(data, bins=np.arange(0, len(data) + binwidth, binwidth),
										   facecolor='green', alpha=0.75)

	# calculate binmiddles
	bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1]) #probleme bei berechnung
	# print data

	powerlaw.plot_pdf(data, color='b', linear_bins=True)
	results = powerlaw.Fit(data, discrete=True)
	print 'alpha: ', results.power_law.alpha
	print 'sigma: ', results.power_law.sigma
	print 'xmin: ', results.power_law.xmin
	R, p = results.distribution_compare('power_law', 'truncated_power_law') 
	print R
	print p
	plt.show()

#klappt nicht wirklich
# def scale_free_logliho_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results):

#     # minimize the negative log-Likelihood
# 	result = minimize(negLogLikelihood,  # function to minimize
#                   x0=(1,1.5),     # start values: exp, prefac
#                   args=(data, power_law),      # additional arguments for function
#                   method='BFGS',   # minimization method, see docs
#                   )
# 	# result is a scipy optimize result object, the fit parameters 
# 	# are stored in result.x
# 	print result

# 	# plot poisson-deviation with fitted parameter
# 	fig = plt.figure()
# 	x_plot= np.linspace(0,max(data)-1,1000) #use as x_values
# 	binwidth = 1

# 	plt.hist(data, bins=np.arange(min(data), max(data) + binwidth -0.5, binwidth), facecolor='green', alpha=0.75)
# 	plt.plot(x_plot, power_law(x_plot, result.x[0], result.x[1]), 'r-', lw=2, label='fit',)
# 	plt.title("Degree distribution - Power-Law")
# 	plt.legend(loc=8)
# 	plt.show()
######

###### Erdos Renyi -> Poisson
def erdos_deg_logliho_fit(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot):
	print('erdos_deg_logliho_fit') 
	# minimize the negative log-Likelihood
	result = minimize(negLogLikelihood,  # function to minimize
				  x0=np.ones(1),     # start value
				  args=(data,),      # additional arguments for function
				  method='Powell',   # minimization method, see docs
				  )
	# result is a scipy optimize result object, the fit parameters 
	# are stored in result.x
	print result

	# plot poisson-deviation with fitted parameter
	fig = plt.figure()
	x_values = np.linspace(0,len(data)-1,1000) #use as x_values
	binwidth = 1

	if plot == True:
		plt.hist(data, bins=np.arange(0, len(data) + binwidth -0.5, binwidth), normed=True, facecolor='green', alpha=0.75)
		x_axis = x_values
		plt.plot(x_axis, poisson_manual(x_values, result.x), 'r-', lw=2, label='fit',)
		plt.title("Degree distribution - Poisson")
		plt.legend(loc=8)
		plt.show()

# def erdos_deg(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results):

# def erdos_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results):
######




################## RUN ########################################
def main(plot_degree=False):
	print('Fits Routines')
# Take Model Network as input
	# G = nx2.Model_Network('../data/jitrik-bunge-e1-set1.csv', '../data/jitrik-bunge-e1-set2.csv', '../data/jitrik-bunge-e2-set1.csv', '../data/jitrik-bunge-e2-set2.csv', '../data/jitrik-bunge-m1-set1.csv', '../data/jitrik-bunge-m1-set2.csv')
	# E1 = [(u,v) for (u,v) in G0.edges() if G0[u][v]['label'] == 'E1']
	# G = nx.Graph()
	# G.add_edges_from(E1)


	# savefigs        = True
	draw_plots      = True

	ions_list       = nx2.one_electron[:4]
	empty_ions      = []

# loop through all ions in ions_list
	for ion in ions_list:
# create network with the NIST lines
		G = nx2.spectroscopic_network(ion, weighted=True, check_wl=True, dictionaries=False)
		if G.size() == 0:
			print('Empty graph.')
			empty_ions.append(ion)
			continue
		elif G.size() < 50:
			print('Less than 50 nodes in Graph.')
			empty_ions.append(ion)
			continue
		else:
			print '#######Ion:', ion
			deg_dist, log_deg_dist, normalised_deg_dist,  quantile, num_data_points = create_deg_dist(G)
			if plot_degree == True:
				plot_hist(deg_dist)
		#lists to eventually save the results
			parameters_results        = [] #list with fitted (optimised) parameters. Each entry is a list of parameters for the used fit function	
			cov_matrix_results        = [] #list of covariance matrices
			parameters_errors_results = [] #list of standard deviations
		 # which fit routines do you want to use?
			# scale_free_quantile_fit(quantile, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=draw_plots)
			print deg_dist
			# scale_free_deg_curve_fit(deg_dist, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot=draw_plots)
			# erdos_deg_logliho_fit(deg_dist, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results, plot=draw_plots)
			# power_law_test(deg_dist)

if __name__ == '__main__':
  main()
  