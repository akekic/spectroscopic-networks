import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# from matplotlib2tikz import save as tikz_save

# Styling
# import seaborn as sns
# sns.set(color_codes=True)
# sns.set_context("paper")
# sns.set_palette("Set1")
# sns.set_style("ticks")
# current_palette = sns.color_palette()
# sns.palplot(current_palette)
# plt.show()
##############################################


ion_list = ['H1.0_Jitrik', 'H1.0_NIST', 'He1.0', 'C1.0', 'Fe1.0']
# ion_list = ['H1.0_Jitrik', 'H1.0_NIST', 'He1.0', 'C1.0']
dropout_list = [0.1, 0.3, 0.5]
# dropout_list = [0.0]
method_list = ['SPM', 'HRG', 'nSBM','AA', 'PA', 'JC', 'RA']
# method_list = ['SPM', 'HRG','AA', 'PA', 'JC', 'RA']
# method_list = ['AA']

###################

import brewer2mpl

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
bmapcolors = bmap.mpl_colors

bmap_set3 = brewer2mpl.get_map('Set3', 'qualitative', 8)
bmapcolors_set3 = bmap_set3.mpl_colors

bmap_dark2 = brewer2mpl.get_map('Dark2', 'qualitative', 8)
bmapcolors_dark2 = bmap_dark2.mpl_colors


## create colors for markers
cmap = plt.get_cmap('Set1') # choose colormap
# colors=[bmapcolors[6]] #choose three colors by hand
# colors=[cmap(0), cmap(2/float(7)), bmapcolors[3], bmapcolors[6], bmapcolors[2], bmapcolors[1]] #choose colors by hand
# colors=[cmap(0), bmapcolors[0], bmapcolors[3], cmap(6/float(9)), bmapcolors[2]] #choose colors by hand
# colors=[cmap(0), bmapcolors[0], bmapcolors[3], bmapcolors[2]] #choose colors by hand
colors=[bmapcolors_dark2[0], bmapcolors_dark2[1], bmapcolors_dark2[2], bmapcolors_dark2[3]] #choose colors by hand


params = {
   'axes.labelsize': 14,
   'font.size': 14,
   'font.family': 'serif',
   'legend.fontsize': 14,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'text.usetex': False,
   'figure.figsize': [11, 5]
   }
plt.rcParams.update(params)




def paper_plot(method_list, dropout, ion_list, inset=False):
    ##
    ## fuer jede methode und jeden dropout werden jeweils alle ionen geplottet
    ##

    ##plotting
    fig = plt.figure()

    for j, ion in enumerate(ion_list):

        axes[ion] = fig.add_subplot(1,2,j+1)
        ax = axes[ion]

        ## add text labels
        fig.text(0.02, 0.97, "A", weight="bold", horizontalalignment='left', verticalalignment='center')
        fig.text(0.51, 0.97, "B", weight="bold", horizontalalignment='left', verticalalignment='center')

        ## spacing between subplots
        fig.subplots_adjust(left=0.09, bottom=0.2, right=0.99, top=0.98, wspace=0.1)


        x_values = np.linspace(0,1,num=1000**2)
        for i, method in enumerate(method_list):
            print ion

            ## import csv data
            data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/'+ion+'/' +method+ '/'+ion+'_dropout_'+method+'_ROC_full_NIST_dropout_value_'+str(dropout)+'.csv', delimiter=',')
            ion_label = '$\mathrm{' + ion[:-3] + '}$'
            method_label = '$\mathrm{' + method + '}$'

            ## read data into array
            FPR = data[:,0]
            TPR = data[:,1]
            TPR_err = data[:,2]
            print data.shape

            ## interpolate TPR values
            TPR_interp = np.zeros(len(x_values))
            TPR_interp = np.interp(x_values, FPR, TPR)

            ## calculate errors
            # 'TODO'
            TPR_err = np.interp(x_values, FPR, TPR_err) #fehler interpolieren?
            # oder
            # TPR_err = np.std(TPR_interp,axis=0)

            ## use interpolated TPR data and x_values as FPR to make ROC plot
            TPR = TPR_interp
            FPR = x_values

            ## get colors for markers
            # color = cmap(i / float(len(method_list)))
            color=colors[i]

            ##grid
            ax.grid(color="0.95", linestyle='-', linewidth=1.2, alpha=0.6)

            ## plot ROC curves
            ax.scatter(FPR, TPR, label=method_label, color=color, s=0.4)
            ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color) # plot error shading


        # font = {'family': 'serif','color': 'black','weight': 'normal','size': 16,}
        # ax.set_xlabel('False Positive Rate', fontdict=font)
        # ax.set_ylabel('True Positive Rate', fontdict=font)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        # ax.set_title('ROC')

        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.009)

        ax.plot((0, 1), (0, 1), linestyle='--', color=bmapcolors[7], alpha=0.8)

        # ax.tick_params(labelsize=9, pad=0.8, labelbottom=False, labeltop=True, labelleft=True, labelright=False)
        ax.tick_params(pad=0.8, labelbottom=False, labeltop=True, labelleft=True, labelright=False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # if j==1:
            # ax.spines['left'].set_visible(False)
            # ax.tick_params(axis='y', length=0)
            # ax.set_yticklabels([])
            # ax.set_ylabel('')

        # offset the spines
        for spine in ax.spines.values():
          spine.set_position(('outward', 5))
        # put the grid behind

        ax.set_axisbelow(True)

        # handles, labels = ax.get_legend_handles_labels()

        # ## simple legend (for poster plots)
        # legend = ax.legend(loc = 'lower right', fancybox=False, shadow=False, scatterpoints=1, markerscale=13,
        #             ncol=1, labelspacing=0.6, handlelength=0.9)
        # frame = legend.get_frame()
        # frame.set_facecolor('1.0')
        # frame.set_edgecolor('1.0')

        ## plot legend in additional file
        import pylab
        # fig = pylab.figure()
        legend_fig = pylab.figure()
        legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(),
                    loc = 'center', fancybox=True, shadow=False, scatterpoints=1, markerscale=13,
                    ncol=1, labelspacing=0.8, handlelength=0.9)
        # legend.get_frame().set_color('0.70') # make grey background
        legend_fig.canvas.draw()
        legend_fig.savefig('/home/julian/qd-networks/svn/paper/pictures/legend_boxed'+'.pdf', bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()), dpi=800)


    ## spacing between subplots
    # fig.subplots_adjust(left=0.09, right=0.99)
    fig.tight_layout()


    # # # Shrink current axis's height by 10% on the bottom
    # for ion in ion_list:
    #     ax = axes[ion]
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # ax = fig.add_subplot(1,3,3)
    # box = ax.get_position()
    # ax.set_position([box.x0 + box.width * 0.1, box.y0, box.width + box.width * 0.5, box.height])

    # ## Put a legend below current axis
    # legend = axes['He1.0'].legend(loc='upper center', bbox_to_anchor=(0, 0),
    #           fancybox=False, shadow=False, scatterpoints=1, markerscale=12,
    #            ncol=1, labelspacing=0.3, handlelength=0.6)
    # frame = legend.get_frame()
    # frame.set_facecolor('1.0')
    # frame.set_edgecolor('1.0')


    plt.draw()
    ## show figure
    plt.show()

    ## save figure to file
    fig.savefig('/home/julian/qd-networks/svn/paper/pictures/ROC_paper'+'.png', dpi=900)




######################################################
## RUN
######################################################

axes = {} #emtpy dict


paper_plot(method_list=['SPM', 'nSBM', 'PA', 'AA'], dropout=0.1, ion_list=['He1.0', 'Th2.0'])
# paper_plot(method_list=['SPM', 'nSBM'], dropout=0.1, ion_list=['He1.0'])


######################################################