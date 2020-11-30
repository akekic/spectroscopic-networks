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

import scipy
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import poisson, powerlaw
from scipy.misc import factorial


########################################
# Styling
import seaborn as sns
sns.set(color_codes=True)
sns.set_context("poster")

sns.set_palette("Set2")
# sns.set_palette("colorblind")
current_palette = sns.color_palette()
sns.palplot(current_palette)
# plt.show()
########################################




########################################
class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value

# quick auto-plot of the data
def plot_hist(data):
    fig = plt.figure()
    plt.hist(data, bins='auto')
    plt.show()

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

# def power_law_cdf(x, exp, prefac, xmin):
    # return (x/xmin) ** (- exp + 1)

# poisson function, parameter lamb is the fit parameter
def poisson_manual(x, lamb):
    return (lamb**x/factorial(x)) * np.exp(-lamb)
####################################################################

#return all the necessary quantities to carry on with a fit
def create_deg_dist(G):
    # read all degrees into an numpy array
    degrees            = np.array(G.degree().values())
    print degrees

    degree_sequence = np.array(sorted([d for d in G.degree().values()], reverse=True)) # degree sequence
    print degree_sequence

    # deg_dist, bin_edges, patches = plt.hist(degree_sequence, bins=bins, facecolor='green', alpha=0.75)
    # print(deg_dist, len(deg_dist))
    # hist = np.histogram(degrees, bins=np.arange(1, max(degree_sequence)+1+1, 1), density=False)
    hist = scipy.stats.histogram(degree_sequence, numbins=max(degree_sequence), defaultlimits=(0, max(degree_sequence)))

    deg_dist = hist[0].astype(float)
    # print deg_dist

    x_values = np.linspace(1,max(degrees),max(degrees))-0.5 #use as x_values of histogram for fits

    # number of data points
    num_data_points     = len(deg_dist)

    # logarithm of the degree distribution
    # log_deg_dist        = np.log(deg_dist[1:])
    # log_deg_dist        = 0

    # formula to normalise to range [a,b]: (b-a)*( (x - min(x)/(max(x) - min(x)) +a )
    normalised_deg_dist = ( deg_dist.astype(np.float) - deg_dist.astype(np.float).min() ) /( deg_dist.astype(np.float).max() - deg_dist.astype(np.float).min() )

    # X2 = np.sort(Z)
    # F2 = np.array(range(N))/float(N)

    # calculate quantile function (inverse cumulative distribution function)
    # TODO: calculate with normalised degree distribution?
    quantile            = np.percentile(normalised_deg_dist, np.linspace(0,100,num_data_points))
    # quantile          = np.percentile(deg_dist, np.linspace(0,100,num_data_points))

    return degrees, deg_dist, x_values, normalised_deg_dist, quantile, num_data_points

#######################################################################


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
# @param      save_plot                    bool: Only important if draw_plot=True: #
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
def scale_free_deg_curve_fit(degrees, deg_dist, x_values, parameters_results, cov_matrix_results, parameters_errors_results, draw_plot=True, save_plot=False, savepar=False, ion='not specified'):
    print('scale_free_deg_curve_fit')



    ## quick plot by sns
    sns.distplot(degrees, bins=max(degrees)+1, kde=False, fit=scipy.stats.powerlaw)

    # print(degrees, len(degrees))
    print(deg_dist, len(deg_dist))

    ## paremeter input
    parameter_init = (0.5,1)

    ## specify how to calculate the errors
    # sigma_func       = np.sqrt(deg_dist)  #use square root of value as sigma
    # sigma_func[sigma_func == 0] = 1  #replace the zeros by error of 1
    # sigma_func       = np.ones_like(x_values)  #sigma=1 for every value
    sigma_func = None


    ## Exclude first few bins in fit
    if ion == 'C1.0':
        lower_bound = 7
    elif ion == 'Fe1.0':
        lower_bound = 10
    else:
        lower_bound = 0


    ## for later plotting
    x_values_unbounded = x_values
    deg_dist_unbounded = deg_dist
    # sigma_func_unbounded = sigma_func

    x_values = x_values[lower_bound:]
    deg_dist = deg_dist[lower_bound:]
    # sigma_func = sigma_func[lower_bound:]

    ## do the fit via curve_fit
    parameters, cov_matrix = curve_fit(power_law, x_values, deg_dist, p0=parameter_init, sigma=sigma_func)
    par_errors             = np.sqrt(np.diag(cov_matrix))

    # ## residuals
    # # r = ydata - f(xdata, *popt)
    # resid = deg_dist - power_law(x_values, parameters[0], parameters[1])
    # ## residual sum of squares
    # ss_res = np.sum( resid ** 2)
    # # print ss_res
    # ## total sum of squares
    # ss_tot = np.sum((deg_dist - np.mean(deg_dist)) ** 2)
    # # print ss_tot
    # ## r-squared
    # r2 = 1 - (ss_res / ss_tot)
    # # print r2

    # ## chi-squared
    # RChi2 = np.sum( ( ( resid )** 2) / sigma_func ** 2 ) / (len(deg_dist)-len(parameter_init))
    # # chi2 = np.sum( ( ( resid )** 2) / sigma_func ** 2 )
    # print('Reduced Chi2: ', RChi2)


    if savepar:
        parameters_results.append(parameters)
        cov_matrix_results.append(cov_matrix)
        parameters_errors_results.append(par_errors)
    else:
        print parameters
        print cov_matrix
        print par_errors
        # print r2

    ## Plotting
    fig = plt.figure()
    ax = plt.axes()


    ## Data bins
    binwidth = 1
    bins = np.arange(0, max(degrees)+binwidth, binwidth)
    # bins = np.logspace(np.log10(0.1),np.log10(max(degrees)), 50) #logarithmic binning
    # plt.hist(degrees, bins=bins, alpha=0.5, facecolor='y')
    plt.hist(degrees, bins=bins, facecolor='blue', alpha=0.5)
    # ax = sns.distplot(degrees, bins=bins, kde=False, rug=False, ax=ax, color='y', hist_kws=dict(edgecolor="k", linewidth=0))

    # plt.errorbar(x_values_unbounded, deg_dist_unbounded, xerr=0.5, yerr=sigma_func_unbounded, color='green', fmt='o', markersize=1.2, linewidth=0.8)
    # plt.errorbar(x_values_unbounded, deg_dist_unbounded,  yerr=sigma_func_unbounded, color='green', fmt='o', markersize=1, linewidth=0.8)

    ## Fit Function
    plt.plot(x_values, power_law(x_values, parameters[0], parameters[1]), label='PL Fit', color='red')

    ## Title / Legend / Axes
    # plt.title(ion[:-3] + ' - Degree distribution - Power-Law Fit // RChi2={:.4f}'.format(RChi2))
    plt.ylabel('Frequency')
    plt.xlabel('Degree k')
    # ax.set_xlim(xmin=1.0)
    # ax.set_ylim(ymin=0.1)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.legend(loc=1)

    if save_plot:
        fig.savefig(figure_concept_path+ion + (not experimental_global)*'_Jitrik' + only_dipole_global*'_dipole'+'_scale_free_deg_curve_fit.png', dpi=300)
        plt.close()
    if draw_plot:
        plt.show()



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
## @param      save_plot                    bool: Only important if draw_plot=True:
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
def erdos_deg_logliho_fit(degrees, deg_dist, x_values, parameters_results, cov_matrix_results, parameters_errors_results, draw_plot=False, save_plot=False, savepar=False, ion='not specified'):
    print('erdos_deg_logliho_fit')


    # sigma_func       = np.ones_like(x_values)  #sigma=1 for every value
    # sigma_func       = np.sqrt(deg_dist)  #use square root of value as sigma
    # sigma_func[sigma_func == 0] = 1  #replace the zeros by error of 1


    # minimize the negative log-Likelihood
    result = minimize(negLogLikelihood_one_par,  # function to minimize
                        x0=np.ones(1),   # start value
                        args=(degrees,poisson_manual),    # additional arguments for function
                        # method='BFGS',     # minimization method, see docs
                        ) # result is a scipy optimize result object, the fit parameters are stored in result.x
    # print result

    # residuals
    # r = ydata - f(xdata, *popt)
    # use the probability density function in the bin
    # hist = np.histogram(degrees, bins=np.arange(0, max(degrees)+1, 1), density=True) ######TODO
    hist = scipy.stats.histogram(degrees, numbins=max(degrees), defaultlimits=(0, max(degrees)))
    deg_dist_pdf = hist[0].astype(float)/sum(hist)

    # resid = deg_dist_pdf - poisson_manual(x_values, result.x)
    # # print resid

    # # residual sum of squares
    # ss_res = np.sum( resid ** 2)
    # # print ss_res

    # # total sum of squares
    # ss_tot = np.sum((deg_dist_pdf - np.mean(deg_dist_pdf)) ** 2)
    # # print np.mean(deg_dist_pdf)
    # # print ss_tot

    # # r-squared
    # r2 = 1 - (ss_res / ss_tot)
    # # print r2

    # # scale down errors to match normalised distribution
    # sigma_func       = np.sqrt(deg_dist) / len(degrees)  #use square root of scaled value as sigma
    # sigma_func[sigma_func == 0] = 1.0 / len(degrees)  #replace the zeros by error of 1, scaled down to fit pdf

    # # chi-squared
    # RChi2 = np.sum( ( ( resid )** 2) / sigma_func ** 2) /(len(deg_dist_pdf)-1)
    # print('Reduced Chi2: ', RChi2)


    # plot poisson-deviation with fitted parameter
    fig = plt.figure()
    binwidth = 1
    # bins=np.arange(1, max(degrees) + binwidth, binwidth) #??
    bins = np.arange(0, max(degrees)+binwidth, binwidth)

    plt.hist(degrees, bins=bins, normed=True, facecolor='blue', alpha=0.5)
    # plt.errorbar(x_values, deg_dist_pdf, xerr=0.5, yerr=sigma_func, color='green', fmt='o', markersize=1.2, linewidth=0.8) #vers3
    # plt.hist(degrees, bins=np.arange(0, max(degrees) + binwidth, binwidth), normed=True, alpha=0.5)
    # plt.scatter(x_values, deg_dist, color='green', s=2)

    x_axis = np.linspace(1,max(degrees)-1,1000)
    # plt.plot(x_axis, poisson_manual(x_axis, result.x), 'r-', lw=2, label='fit',)
    plt.plot(x_axis, poisson_manual(x_axis, result.x), label='ER Fit', color='red')

    # plt.title(ion+' - Degree distribution - Poisson/Erdos-Renyi. RChi2={:.4f}'.format(RChi2))
    plt.ylabel('Occurrence')
    plt.xlabel('Degree k')
    plt.legend(loc=1)

    if save_plot:
        fig.savefig(figure_concept_path+ion + (not experimental_global)*'_Jitrik' + only_dipole_global*'_dipole'+'_erdos_deg_logliho_fit.png', dpi=500)
        plt.close()
    if draw_plot:
        plt.show()



################## RUN ########################################
print('Fit Routines')

ions_list       = ['H1.0', 'He1.0', 'C1.0', 'Fe1.0']
ions_list       = ['H1.0', 'He1.0']
# ions_list       = ['H1.0']
# ions_list       = ['He1.0']
# ions_list       = ['C1.0']
# ions_list       = ['Fe1.0']

experimental_global  = True              # use the experimental NIST data or the theoretical Jitrik-Bunge data
only_dipole_global   = False              # global switch for taking the full network or only dipole lines
n_limit              = False              # limit is only to be used for one electron ions (hydrogenic ions)
weighted_global      = True              # if True, load the weighted graphs
check_obs_wl_global  = False
check_calc_wl_global = False
print_info_global    = True
if n_limit==False:
    max_n_global = None
else:
    max_n_global   = 8                 # maximal n considered in network

draw_plot = True    #whether to draw the plot
save_plot = False    #whether to save the plot to file
figure_concept_path = '../../thesis/figure_concept/pics/'


#lists to eventually save the results
parameters_results        = [] #list with fitted (optimised) parameters. Each entry is only_largest_componenta list of parameters for the used fit function
cov_matrix_results        = [] #list of covariance matrices
parameters_errors_results = [] #list of standard deviations


for ion in ions_list: # loop through all ions in ions_list
    print '#######Ion:', ion

    ## load network
    Graph = nx2.load_network(ion=ion, experimental=experimental_global, only_dipole=only_dipole_global, n_limit=n_limit, max_n=max_n_global, alt_read_in=False, weighted=weighted_global, check_obs_wl=check_obs_wl_global, check_calc_wl=check_calc_wl_global, print_info=print_info_global)


    ## get histograms etc.
    degrees, deg_dist, x_values, normalised_deg_dist, quantile, num_data_points = create_deg_dist(Graph)

    # # which fit routines do you want to use?
    scale_free_deg_curve_fit(degrees, deg_dist, x_values, parameters_results, cov_matrix_results, parameters_errors_results, draw_plot=False, save_plot=True, savepar=False, ion=ion)
    # erdos_deg_logliho_fit(degrees, deg_dist, x_values, parameters_results, cov_matrix_results, parameters_errors_results, draw_plot=False, save_plot=True, savepar=False, ion=ion)





