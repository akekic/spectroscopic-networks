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

# ion = 'He1.0'
# # ion_list = ['H1.0', 'He1.0', 'Fe1.0'] # poster

# dropout_list = [0.1, 0.3, 0.5]
# dropout_list = [0.0]
# dropout_list = [0.1]

# # method_list = ['SPM', 'HRG', 'nSBM','AA', 'PA', 'JC', 'RA']
# method_list = ['AA']
# # method_list = ['SPM', 'nSBM','AA', 'PA', 'JC', 'RA']

###################

def ion_dropout_method(ion_list, dropout_list, method_list):
    ##
    ## fuer jedes ion und jeden dropout werden alle methoden in einem plot gezeichnet
    ##
    for dropout in dropout_list: #paper
    # for method, dropout in zip(method_list,dropout_list): #poster

        FPR_curves     = [] # to store the result for each dropout fraction
        TPR_curves     = []
        TPR_err_curves = []

        x_values = np.linspace(0,1,num=1000**2)
        # x_values = np.linspace(0,1,num=10**3) #fuer np

        # for i,dropout in enumerate(dropout_list):
        for i, method in enumerate(method_list): #paper
        # for i, ion in enumerate(ion_list): #poster

            # import csv data for different methods
            # data = genfromtxt('/home/julian/qd-networks/thesis/figure_concept/pics/'+ion+'.csv', delimiter=',')
            # data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/'+ion+'/' +method+ '/'+ion+'_dropout_'+method+'_ROC_full_NIST_dropout_value_'+str(dropout)+'.csv', delimiter=',')
            # data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/'+ion+'_Jitrik'+'/' +method+ '/'+ion+'_dropout_'+method+'_ROC_full_Jitrik_dropout_value_'+str(dropout)+'.csv', delimiter=',')
            data = genfromtxt('/home/julian/qd-networks/svn/plots/'+method+'/'+ion+'/'+ion+'_dropout_'+method+'_ROC_full_NIST_dropout_value_'+str(dropout)+'.csv', delimiter=',')

            ## for poster
            # if ion=='H1.0':
            #     # data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/'+ion+'_Jitrik'+'/' +method+ '/'+ion+'_dropout_'+method+'_ROC_only_dipole_Jitrik_dropout_value_'+str(dropout)+'.csv', delimiter=',')
            #     data = genfromtxt('/media/julian/DAVID/NP_data_results/'+ion+'_Jitrik'+'/'+'ROC_adj'+ion+'_Jitrik'+'.dat', delimiter=' ')
            # else:
            #     # data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/'+ion+'/' +method+ '/'+ion+'_dropout_'+method+'_ROC_full_NIST_dropout_value_'+str(dropout)+'.csv', delimiter=',')
            #     data = genfromtxt('/media/julian/DAVID/NP_data_results/'+ion+'/'+'ROC_adj'+ion+'.dat', delimiter=' ')
            #     # data = genfromtxt('/home/julian/qd-networks/svn/plots/'+method+'/'+ion+'/'+ion+'_dropout_'+method+'_ROC_full_NIST_dropout_value_'+str(dropout)+'.csv', delimiter=',')


            FPR = data[:,0]
            TPR = data[:,1]
            TPR_err = data[:,2]
            print data.shape

            TPR_interp = np.zeros(len(x_values))
            TPR_interp = np.interp(x_values, FPR, TPR)
            TPR_avg = np.mean(TPR_interp,axis=0)

            # 'TODO'
            TPR_err = np.interp(x_values, FPR, TPR_err) #fehler interpolieren?
            # oder
            # TPR_err = np.std(TPR_interp,axis=0)

            #FPR_curves.append(data[:,0])
            TPR_curves.append(TPR_interp)
            TPR_err_curves.append(TPR_err)
            ###############################################################



        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

    # create colors for markers
    # color=cmap(i / float(len(method_list)))
        fig = plt.figure()
        cmap = plt.get_cmap('Set1') # choose colormap

        # colors=[cmap(0), cmap(0.3), cmap(0.7)] #choose three colors

        # from matplotlib.colors import ListedColormap
        # cmap = ListedColormap(sns.color_palette("Set1", desat=.5).as_hex())


        ax = plt.axes()

        inset_axes = zoomed_inset_axes(ax, 6, # zoom
                                       loc=4)


        # for i,dropout in enumerate(dropout_list):
        for i,method in enumerate(method_list):
        # for i,ion in enumerate(ion_list): #poster
            TPR     = TPR_curves[i]
            TPR_err = TPR_err_curves[i]
            FPR     = x_values

            # create colors for markers
            color = cmap(i / float(len(method_list))) #paper
            # color = colors[i] #choose own colours

            # plot ROC curves
            # ax.plot(FPR, TPR, color=color, linewidth=3) #changed / poster
            # ax.scatter(FPR[1], TPR[1], label=method, color=color, s=0.01)

            # ax.scatter(FPR, TPR, label=ion[:-3], color=color, s=0.4) #poster
            ax.scatter(FPR, TPR, label=method, color=color, s=0.4) # paper
            ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color) # plot error shading


            # inset_axes.plot(FPR, TPR, color=color, linewidth=3) #poster
            inset_axes.scatter(FPR, TPR, label=method, color=color, s=0.4) #paper
            inset_axes.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color)


        ax.plot((0, 1), (0, 1), 'r--')
        font = {'family': 'serif','color': 'black','weight': 'normal','size': 16,}
        # ax.tick_params(labelsize=9, pad=0.7, labelbottom=False, labeltop=True, labelleft=True, labelright=False)
        # ax.set_xlabel('FPR', fontdict=font)
        ax.set_xlabel('False Positive Rate', fontdict=font)
        # ax.set_ylabel('TPR', fontdict=font)
        ax.set_ylabel('True Positive Rate', fontdict=font)
        # ax.set_title('ROC')
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)



        # sub region of the original image
        x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
        inset_axes.set_xlim(x1, x2)
        inset_axes.set_ylim(y1, y2)

        # position of the bbox in original image (where to plot)
        ip = InsetPosition(ax, [0.45, 0.05, 0.5, 0.5])
        inset_axes.set_axes_locator(ip)

        # achsenbeschriftung von box in box
        x_sub_ticks = ['0.0', '0.1', '0.2', '0.3']
        y_sub_ticks = ['0.7', '0.8', '0.9', '1.0']
        inset_axes.set_xticks((0, 0.1, 0.2, 0.3))
        inset_axes.set_yticks((0.7, 0.8, 0.9, 1.0))
        inset_axes.tick_params(labelsize=9, pad=0.8, labelbottom=False, labeltop=True, labelleft=True, labelright=False, direction='out')
        plt.xticks(visible=True, rotation='horizontal')
        inset_axes.set_xticklabels(x_sub_ticks)
        plt.yticks(visible=True, rotation='horizontal')
        inset_axes.set_yticklabels(y_sub_ticks)

        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, inset_axes, loc1=1, loc2=3, fc="none", ec="0.4", ls='solid')


        ## simple legend (for poster plots)
        # ax.legend(loc = 'upper right', bbox_to_anchor=(0.99, 0.95), fancybox=True, shadow=True, scatterpoints=1, markerscale=13*6.3,
                    # ncol=1, labelspacing=0.6, handlelength=0.9)

        # # old version of legend
        # # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])

        # # Put a legend below current axis
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, shadow=True, scatterpoints=1, markerscale=10,
        #            ncol=7, labelspacing=0.3, handlelength=0.6)

        # # plot legend in additional file
        # import pylab
        # # fig = pylab.figure()
        # legend_fig = pylab.figure()
        # legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(),
        #             loc = 'center', fancybox=True, shadow=True, scatterpoints=1, markerscale=13,
        #             ncol=1, labelspacing=0.6, handlelength=0.9)
        # # legend.get_frame().set_color('0.70') # make grey background
        # legend_fig.canvas.draw()
        # # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/figure_concept/'+ion+'_legend_cropped.png', bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
        # # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'_legend_original.png')
        # # legend_fig.savefig('/home/julian/qd-networks/svn/reports/DPG2018/poster'+ion+'_legend_original.png')

        # # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-NIST'+'_legend_original.png')
        # # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik-Dipole'+'_legend_original.png')
        # # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik'+'_legend_original.png')

        plt.draw()
        plt.show()
        # fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'_'+str(dropout)+'.png')
        # fig.savefig('/home/julian/qd-networks/thesis/figure_concept/pics/'+ion+'_'+str(dropout)+'.png', dpi=300)
        # fig.savefig('/home/julian/qd-networks/svn/reports/DPG2018/poster/NP_ROC'+'_'+str(dropout)+'.png', dpi=500)
        # fig.savefig('/home/julian/qd-networks/svn/reports/DPG2018/poster/NP_ROC'+'_'+str(dropout)+'.pdf')

        # fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-NIST_'+str(dropout)+'.png') #fuer H1.0
        # fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik-Dipole_'+str(dropout)+'.png') #fuer H1.0
        # fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik_'+str(dropout)+'.png') #fuer H1.0
        # fig.savefig('/home/julian/qd-networks/svn/plots/'+ion+'_'+str(dropout)+'.png', dpi=300)


        #save to tikz
        # tikz_save('/home/julian/qd-networks/thesis/figure_concept/pics/'+ion+'.tex')


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################



ion_list = ['H1.0_Jitrik', 'H1.0_NIST', 'He1.0', 'C1.0', 'Fe1.0']
ion_list = ['H1.0_Jitrik', 'H1.0_NIST']
dropout_list = [0.1, 0.3, 0.5]
method_list = ['SPM', 'HRG', 'nSBM','AA', 'PA', 'JC', 'RA']
method_list = ['SPM']

def method_dropout_ion(method_list, dropout_list, ion_list):
    ##
    ## fuer jede methode und jeden dropout werden jeweils alle ionen geplottet
    ##
    for method, dropout in zip(method_list,dropout_list):

        FPR_curves     = [] # to store the result for each dropout fraction
        TPR_curves     = []
        TPR_err_curves = []

        x_values = np.linspace(0,1,num=1000**2)

        for i, ion in enumerate(ion_list): #poster
            ## import csv data for different methods
            if ion=='H1.0_Jitrik':
                data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/H1.0_Jitrik/' +method+ '/H1.0_dropout_'+method+'_ROC_full_Jitrik_dropout_value_'+str(dropout)+'.csv', delimiter=',')
                ion_label = 'H (Jitrik)'
            elif ion=='H1.0_NIST':
                data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/H1.0_NIST/' +method+ '/H1.0_dropout_'+method+'_ROC_full_NIST_dropout_value_'+str(dropout)+'.csv', delimiter=',')
                ion_label = 'H (NIST)'
            else:
                data = genfromtxt('/home/julian/qd-networks/svn/data/ROC_data_results/'+ion+'/' +method+ '/'+ion+'_dropout_'+method+'_ROC_full_NIST_dropout_value_'+str(dropout)+'.csv', delimiter=',')
                ion_label = ion[:-3]


            FPR = data[:,0]
            TPR = data[:,1]
            TPR_err = data[:,2]
            print data.shape

            TPR_interp = np.zeros(len(x_values))
            TPR_interp = np.interp(x_values, FPR, TPR)
            # TPR_avg = np.mean(TPR_interp,axis=0)

            # 'TODO'
            TPR_err = np.interp(x_values, FPR, TPR_err) #fehler interpolieren?
            # oder
            # TPR_err = np.std(TPR_interp,axis=0)

            #FPR_curves.append(data[:,0])
            TPR_curves.append(TPR_interp)
            TPR_err_curves.append(TPR_err)
            ###############################################################
            ## (read-in finished)


        ##plotting
        fig = plt.figure()
        ax = plt.axes()

        ## create colors for markers
        # color=cmap(i / float(len(method_list)))
        cmap = plt.get_cmap('Set1') # choose colormap
        # colors=[cmap(0), cmap(0.3), cmap(0.7)] #choose three colors


        inset_axes = zoomed_inset_axes(ax, 6, # zoom
                                       loc=4)

        for i,ion in enumerate(ion_list): #poster
            TPR     = TPR_curves[i]
            TPR_err = TPR_err_curves[i]
            FPR     = x_values

            # create colors for markers
            color = cmap(i / float(len(ion_list))) #paper
            # color = colors[i] #use previously chosen colours

            ## plot ROC curves
            ax.scatter(FPR, TPR, label=ion, color=color, s=0.4) # paper
            ax.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color) # plot error shading

            ## plot inset ROC
            inset_axes.scatter(FPR, TPR, label=ion, color=color, s=0.4) #paper
            inset_axes.fill_between(FPR, TPR+0.5*TPR_err, TPR-0.5*TPR_err, alpha=0.3, color=color)


        ax.plot((0, 1), (0, 1), 'r--')
        font = {'family': 'serif','color': 'black','weight': 'normal','size': 16,}
        # ax.tick_params(labelsize=9, pad=0.7, labelbottom=False, labeltop=True, labelleft=True, labelright=False)
        # ax.set_xlabel('FPR', fontdict=font)
        ax.set_xlabel('False Positive Rate', fontdict=font)
        # ax.set_ylabel('TPR', fontdict=font)
        ax.set_ylabel('True Positive Rate', fontdict=font)
        # ax.set_title('ROC')
        ax.set_xlim(xmin=0.0, xmax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)

        ## sub region of the original image
        x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
        inset_axes.set_xlim(x1, x2)
        inset_axes.set_ylim(y1, y2)

        ## position of the bbox in original image (where to plot)
        ip = InsetPosition(ax, [0.45, 0.05, 0.5, 0.5])
        inset_axes.set_axes_locator(ip)

        ## achsenbeschriftung von box in box
        x_sub_ticks = ['0.0', '0.1', '0.2', '0.3']
        y_sub_ticks = ['0.7', '0.8', '0.9', '1.0']
        inset_axes.set_xticks((0, 0.1, 0.2, 0.3))
        inset_axes.set_yticks((0.7, 0.8, 0.9, 1.0))
        inset_axes.tick_params(labelsize=9, pad=0.8, labelbottom=False, labeltop=True, labelleft=True, labelright=False, direction='out')
        plt.xticks(visible=True, rotation='horizontal')
        inset_axes.set_xticklabels(x_sub_ticks)
        plt.yticks(visible=True, rotation='horizontal')
        inset_axes.set_yticklabels(y_sub_ticks)

        ## draw a bbox of the region of the inset axes in the parent axes and
        ## connecting lines between the bbox and the inset axes area
        mark_inset(ax, inset_axes, loc1=1, loc2=3, fc="none", ec="0.4", ls='solid')


        ## -------------------------------------- LEGEND
        ## simple legend (for poster plots)
        # ax.legend(loc = 'upper right', bbox_to_anchor=(0.99, 0.95), fancybox=True, shadow=True, scatterpoints=1, markerscale=13*6.3,
                    # ncol=1, labelspacing=0.6, handlelength=0.9)

        ## old version of legend
        # # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])

        ## Put a legend below current axis
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, shadow=True, scatterpoints=1, markerscale=10,
        #            ncol=7, labelspacing=0.3, handlelength=0.6)

        ## plot legend in additional file
        import pylab
        # fig = pylab.figure()
        legend_fig = pylab.figure()
        legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(),
                    loc = 'center', fancybox=True, shadow=True, scatterpoints=1, markerscale=13,
                    ncol=1, labelspacing=0.6, handlelength=0.9)
        # legend.get_frame().set_color('0.70') # make grey background
        legend_fig.canvas.draw()
        # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/figure_concept/'+ion+'_legend_cropped.png', bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
        legend_fig.savefig('/home/julian/qd-networks/thesis/figure_concept/pics/'+'('+method+'_'+str(dropout)+')'+'_legend_boxed'+'.png', dpi=400)
        ## -------------------------------------- LEGEND


        ## show figure
        plt.draw()
        plt.show()

        ## save figure to file
        fig.savefig('/home/julian/qd-networks/thesis/figure_concept/pics/'+'('+method+'_'+str(dropout)+')'+'.png', dpi=400)



######################################################
## RUN
######################################################
method_dropout_ion(method_list, dropout_list, ion_list)


# ion_dropout_method(ion_list, dropout_list, method_list)



######################################################