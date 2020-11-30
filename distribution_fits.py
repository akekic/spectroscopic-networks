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
# def negLogLikelihood(params, data, func):
	""" the negative log-Likelohood-Function"""
	# lnl = - np.sum(np.log(func(data, params[0], params[1])))
	# return lnl

# calculating the negative log likelihood of the dsitribution using
# the function 'func' that has been passed
# only one parameter
def negLogLikelihood_one_par(params, data, func):
	# """ the negative log-Likelohood-Function"""
	lnl = - np.sum(np.log(func(data, params[0])))
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

####################################################################

def create_dipole_model_graph(full_model_graph):
	E1 = [(u,v) for (u,v) in full_model_graph.edges() if full_model_graph[u][v]['label'] == 'E1']
	dipole_model_graph = nx.Graph()
	dipole_model_graph.add_edges_from(E1)
	return dipole_model_graph


#return all the necessary quantities to carry on with a fit
def create_deg_dist(G):
	# read all degrees into an numpy array
	degrees            = np.array(G.degree().values())

	degree_sequence = np.array(sorted([d for d in G.degree().values()], reverse=True)) # degree sequence

	# deg_dist, bin_edges, patches = plt.hist(degree_sequence, bins=bins, facecolor='green', alpha=0.75)
	# print(deg_dist, len(deg_dist))
	hist = np.histogram(degrees, bins=np.arange(1, max(degree_sequence)+1+1, 1), density=False)
	deg_dist = hist[0].astype(float)
	# print deg_dist

	# number of data points
	num_data_points     = len(deg_dist)

	# logarithm of the degree distribution
	# log_deg_dist        = np.log(deg_dist[1:])
	log_deg_dist        = 0

	# formula to normalise to range [a,b]: (b-a)*( (x - min(x)/(max(x) - min(x)) +a )
	normalised_deg_dist = ( deg_dist.astype(np.float) - deg_dist.astype(np.float).min() ) /( deg_dist.astype(np.float).max() - deg_dist.astype(np.float).min() )

	# X2 = np.sort(Z)
	# F2 = np.array(range(N))/float(N)

	# calculate quantile function (inverse cumulative distribution function)
	# TODO: calculate with normalised degree distribution?
	quantile            = np.percentile(normalised_deg_dist, np.linspace(0,100,num_data_points))
	# quantile          = np.percentile(deg_dist, np.linspace(0,100,num_data_points))

	return degrees, deg_dist, log_deg_dist, normalised_deg_dist, quantile, num_data_points

#######################################################################

# Scale-free / Power-law what about the logarithmic scale? fitfunc = lambda
# p,x:....
#
# @brief      Description
#
# @param      data_points                The data points: the percentile values
#                                        of the degree distribution
# @param      deg_dist                   The degree distribution
# @param      parameters_results         The global parameters results list
# @param      cov_matrix_results         The global cov matrix results list
# @param      parameters_errors_results  The global parameters errors results
#                                        list
# @param      plot                       bool: If False, doesnt do any plot
#                                        whatsoever.
# @param      savefig                    bool: Only important if plot=True:
#                                        specifies if plot should be saved to
#                                        file or simply outputted.
# @param      ion                        string: The name of the ion to which
#                                        the distribution belongs, if not
#                                        specified, it is assumed to be the
#                                        model network.
# @param      savepar  bool: If false, does not append the fit results to the
#                      global lists, merely prints them.
#
# @return     { description_of_the_return_value }
#
#TODO ist diese funktion korrekt??
#cumulative?
def scale_free_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=True, savepar=False, ion='not specified'):
	#input
	parameter_init 	 = (0.5,1)
	sigma_func       = np.ones_like(data_points)  #sigma=0.1 for every value

	print(len(data_points))
	print(len(deg_dist))

	#fit
	parameters, cov_matrix = curve_fit(nroot, np.linspace(0,1,max(deg_dist)),
										data_points, p0=parameter_init, sigma=sigma_func, absolute_sigma=True)
	par_errors             = np.sqrt(np.diag(cov_matrix))

	if savepar==True:
		parameters_results.append(parameters)
		cov_matrix_results.append(cov_matrix)
		parameters_errors_results.append(par_errors)
	else:
		print('scale_free_quantile_fit') 
		print parameters
		print cov_matrix
		print par_errors
	 
	if plot == True:
		fig = plt.figure()
		x_axis= np.linspace(0,1,max(deg_dist)) #use as x_values
		plt.scatter(x_axis, data_points, label='data')
		plt.title("Degree inverse cumulative distribution - Scale-free")
		plt.plot(x_axis, nroot(np.linspace(0,1,max(deg_dist)), parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.legend(loc=1)
		if savefig==True:
			fig.savefig('../plots/'+ion+'_scale_free_quantile_fit.png')
		else:
			plt.show()


#
#
# @brief      Fits a scale free function to the degree ditribution (binned #
#             histogram)
#
# @param      data                       The data, hsould be a degree #
#                                        distribution #
# @param      parameters_results         The global parameters results list #
# @param      cov_matrix_results         The global cov matrix results list #
# @param      parameters_errors_results  The global parameters errors results #
#                                        list #
# @param      plot                       bool: If False, doesnt do any plot #
#                                        whatsoever #
# @param      savefig                    bool: Only important if plot=True: #
#                                        specifies if plot should be saved to #
#                                        file or simply outputted #
# @param      savepar                    bool: If false, does not append the #
#                                        fit results to the global lists, #
#                                        merely prints them #
# @param      ion                        string: The name of the ion to which #
#                                        the distribution belongs, if not #
#                                        specified, it is assumed to be the #
#                                        model network # #
#
# @return     { description_of_the_return_value } #
#
def scale_free_deg_curve_fit(degrees, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=True, savepar=False, ion='not specified'):
	print('scale_free_deg_curve_fit') 

	# print(degrees, len(degrees))
	# print(deg_dist, len(deg_dist))

	#input
	parameter_init 	 = (0.5,1)
	x_values         = np.linspace(1,max(degrees),max(degrees)) #use as x_values
	# TODO 
	# sigma_func       = np.ones_like(x_values)  #sigma=1 for every value
	sigma_func       = np.sqrt(deg_dist)  #use square root of value as sigma
	sigma_func[sigma_func == 0] = 1  #replace the zeros by error of 1
	# print('sigma', sigma_func)

	# print ('xvalues', x_values, len(x_values))


	# calculate binmiddles
	# bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
	# print(bin_middles, len(bin_middles))

	# do the fit via curve_fit
	parameters, cov_matrix = curve_fit(power_law, x_values, deg_dist, p0=parameter_init, sigma=sigma_func)
	par_errors             = np.sqrt(np.diag(cov_matrix))

	# residuals
	# r = ydata - f(xdata, *popt)
	resid = deg_dist - power_law(x_values, parameters[0], parameters[1])
	# print resid

	# residual sum of squares
	ss_res = np.sum( resid ** 2)
	# print ss_res
	
	# total sum of squares
	ss_tot = np.sum((deg_dist - np.mean(deg_dist)) ** 2)
	# print np.mean(deg_dist)
	# print ss_tot

	# r-squared
	r2 = 1 - (ss_res / ss_tot)
	# print r2

	# chi-squared
	RChi2 = np.sum( ( ( resid )** 2) / sigma_func ** 2 ) / (len(deg_dist)-len(parameter_init))
	# chi2 = np.sum( ( ( resid )** 2) / sigma_func ** 2 )
	print('reduced chi2', RChi2)

	 
	if savepar==True:
		parameters_results.append(parameters)
		cov_matrix_results.append(cov_matrix)
		parameters_errors_results.append(par_errors)
	else:
		print parameters
		print cov_matrix
		print par_errors
		print r2

	if plot == True:
		fig = plt.figure()
		binwidth = 1
		bins = np.arange(1, max(degrees)+binwidth, binwidth)
		x_axis = x_values
		plt.hist(degrees, bins=bins, facecolor='green', alpha=0.75)
		plt.plot(x_axis, power_law(x_values, parameters[0], parameters[1]), 'r-', label='fit', lw=2)
		plt.title(ion+' - Degree distribution - Power-Law/Scale-free fit. RChi2={:.4f}'.format(RChi2))
		plt.ylabel('occurrence')
		plt.xlabel('degree k')
		plt.legend(loc=1)
		if savefig==True:
			fig.savefig('../plots/fits/'+ion+'_scale_free_deg_curve_fit.png')
			plt.close()
		else:
			plt.show()



# using the power law package
#
# @param      data  The degree distribution
#
# @return     { description_of_the_return_value }
#
def power_law_test(degrees):
	import powerlaw

	#convert the degree values to a histogram
	fig = plt.figure()
	binwidth = 1
	data, bin_edges, patches = plt.hist(degrees, bins=np.arange(0, max(degrees) + binwidth, binwidth),
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
# 	plt.legend(loc=1)
# 	plt.show()
######



#-------------------------------------------------------------------------------
## Erdos Renyi -> Poisson
##
## @brief      Fits a poisson distribution to the degree distribution
##
## @param      data                       The degree distribution
## @param      parameters_results         The global parameters results list
## @param      cov_matrix_results         The global cov matrix results list
## @param      parameters_errors_results  The global parameters errors results
##                                        list
## @param      plot                       bool: If False, doesnt do any plot
##                                        whatsoever
## @param      savefig                    bool: Only important if plot=True:
##                                        specifies if plot should be saved to
##                                        file or simply outputted
## @param      savepar                    bool: If false, does not append the
##                                        fit results to the global lists,
##                                        merely prints them
## @param      ion                        string: The name of the ion to which
##                                        the distribution belongs, if not
##                                        specified, it is assumed to be the
##                                        model network
##
## @return     { description_of_the_return_value }
##
## binning?
def erdos_deg_logliho_fit(degrees, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=True, savepar=False, ion='not specified'):
	print('erdos_deg_logliho_fit') 

	# print('degrees', degrees, len(degrees))
	# print('deg_dist', deg_dist, len(deg_dist))
	# print deg_dist.dtype

	#input
	x_values         = np.linspace(1,max(degrees),max(degrees)) #use as x_values
	# print('xvalues', x_values, len(x_values))

	# print('sigma', sigma_func)
	# sigma_func       = np.ones_like(x_values)  #sigma=1 for every value
	sigma_func       = np.sqrt(deg_dist)  #use square root of value as sigma
	sigma_func[sigma_func == 0] = 1  #replace the zeros by error of 1
	# print sigma_func


	# minimize the negative log-Likelihood
	result = minimize(negLogLikelihood_one_par,  # function to minimize
				  		x0=np.ones(1),   # start value
				  		args=(degrees,poisson_manual),    # additional arguments for function
				  		# method='BFGS',     # minimization method, see docs
				  		)
	# result is a scipy optimize result object, the fit parameters 
	# are stored in result.x
	# print result

	# residuals
	# r = ydata - f(xdata, *popt)
	# use the probability density function in the bin
	hist = np.histogram(degrees, bins=np.arange(1, max(degrees)+1+1, 1), density=True)
	deg_dist_pdf = hist[0].astype(float)
	resid = deg_dist_pdf - poisson_manual(x_values, result.x)
	# print resid

	# residual sum of squares
	ss_res = np.sum( resid ** 2)
	# print ss_res
	
	# total sum of squares
	ss_tot = np.sum((deg_dist_pdf - np.mean(deg_dist_pdf)) ** 2)
	# print np.mean(deg_dist_pdf)
	# print ss_tot

	# r-squared
	r2 = 1 - (ss_res / ss_tot)
	# print r2

	# sigma_func       = np.ones_like(x_values)  #sigma=1 for every value
	sigma_func       = np.sqrt(deg_dist) / len(degrees)  #use square root of scaled value as sigma
	sigma_func[sigma_func == 0] = 1.0 / len(degrees)  #replace the zeros by error of 1, scaled down to fit pdf
	# print sigma_func
	# chi-squared
	RChi2 = np.sum( ( ( resid )** 2) / sigma_func ** 2) /(len(deg_dist_pdf)-1)
	print('reduced chi2', RChi2)


	if plot == True:
		# plot poisson-deviation with fitted parameter
		fig = plt.figure()
		binwidth = 1
		plt.hist(degrees, bins=np.arange(1, max(degrees) + binwidth, binwidth), normed=True, facecolor='green', alpha=0.75)
		x_axis = np.linspace(1,max(degrees)-1,1000)
		plt.plot(x_axis, poisson_manual(x_axis, result.x), 'r-', lw=2, label='fit',)
		plt.title(ion+' - Degree distribution - Poisson/Erdos-Renyi. RChi2={:.4f}'.format(RChi2))
		plt.ylabel('occurrence')
		plt.xlabel('degree k')
		plt.legend(loc=1)
		if savefig==True:
			fig.savefig('../plots/fits/'+ion+'_erdos_deg_logliho_fit.png')
			plt.close()
		else:
			plt.show()




# def erdos_deg(data, num_data_points, parameters_results, cov_matrix_results, parameters_errors_results):

# def erdos_quantile_fit(data_points, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results):
######






################## RUN ########################################
print('Fit Routines')

plot_degree = False
model_fit  = True #if false, this is skipping the model fit. be careful: model fit stores his result also in global parameter list (eg. if True, last result is the result of the model fit)

# ions_list   = [] #to not do anything
ions_list   = nx2.one_electron[:10]
# ions_list   = ['C6.0']
empty_ions  = []

#lists to eventually save the results
parameters_results        = [] #list with fitted (optimised) parameters. Each entry is a list of parameters for the used fit function	
cov_matrix_results        = [] #list of covariance matrices
parameters_errors_results = [] #list of standard deviations

# loop through all ions in ions_list
for ion in ions_list:
# create network with the NIST lines
	G = nx2.spectroscopic_network(ion, weighted=True, check_wl=True, dictionaries=False)
	if G.size() == 0:
		print('Empty graph.')
		empty_ions.append(ion)
		continue
	elif G.size() < 1:
		print('Less than 50 nodes in Graph.')
		empty_ions.append(ion)
		continue
	else:
		print '#######Ion:', ion
		degrees, deg_dist, log_deg_dist, normalised_deg_dist,  quantile, num_data_points = create_deg_dist(G)
		if plot_degree == True:
			plot_hist(deg_dist)
	# which fit routines do you want to use?
		# print deg_dist
		scale_free_deg_curve_fit(degrees, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=True, savepar=False, ion=ion)
		erdos_deg_logliho_fit(degrees, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=True, savepar=False, ion=ion)
		# scale_free_quantile_fit(quantile, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=False, savepar=False, ion=ion)
		# power_law_test(deg_dist)



if model_fit==True:
	# Take Model Network as input
	G_model = nx2.Model_Network('../data/jitrik-bunge-e1-set1.csv', '../data/jitrik-bunge-e1-set2.csv', '../data/jitrik-bunge-e2-set1.csv', '../data/jitrik-bunge-e2-set2.csv', '../data/jitrik-bunge-m1-set1.csv', '../data/jitrik-bunge-m1-set2.csv')
	degrees, deg_dist, log_deg_dist, normalised_deg_dist,  quantile, num_data_points = create_deg_dist(G_model)
	scale_free_deg_curve_fit(degrees, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=True, savepar=False, ion='model')
	erdos_deg_logliho_fit(degrees, deg_dist, parameters_results, cov_matrix_results, parameters_errors_results, plot=True, savefig=True, savepar=False, ion='model')

