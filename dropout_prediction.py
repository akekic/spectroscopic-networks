try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

import matplotlib.pyplot as plt



def _plot_auc(auc_list, fraction_kept_list):
    ##
    ## @brief      Plots the AUC value over the fraction of observed links. Hard-coded x axis
    ##
    ## @param      auc_list                 The auc list
    ## @param      fraction_kept_list   The list of the fraction of kept edges
    ##
    ## @return     Save a figure to file.
    ##

    fig = plt.figure()
    plt.plot(fraction_kept_list, auc_list)
    plt.title('AUC curve for '+ion+' Network')
    plt.ylabel('AUC')
    plt.xlabel('Fraction of kept links ( 1 - dropout )')
    # plt.xlabel('Fraction of edges observed')
    plt.legend(loc=8)
    plt.ylim((0.4,1.0))
    plt.plot((0.0, 1.0), (0.5, 0.5), 'b--')
    plt.savefig('../plots/HRG/AUC_'+ion+'_'+label+'.png')
    plt.show()

def run_whole(ion, label, dropout_fraction_list, rank_percentage):
    print 'Dropout fractions: ', dropout_fraction_list
    fraction_kept_list = [1-x for x in dropout_fraction_list]
    auc_list = []

    fig = plt.figure(0)
    plt.title('HRG Prediction Model Network only dipole max_n=8 ROC curve')
    plt.ylabel('True Positive rate')
    plt.xlabel('False Positive rate')

    # #calculate the mean auc out of 10 runs for each dropout fraction
    # m = 1 #sampling size
    # for dropout_fraction in dropout_fraction_list:
    #   sum_auc = 0
    #   for i in range(m):
    #       print '#########################################################', dropout_fraction
    #       auc, roc_array = single_run(dropout_fraction)
    #       sum_auc += auc
    #   mean = sum_auc / float(m)
    #   auc_list.append(mean)
    #   fig = plt.figure(0)
    #   plt.plot(roc_array[:,1], roc_array[:,0], label='dropout p =' + str(dropout_fraction), linewidth=1.5)
    #   plt.legend(loc=8)
    #   plt.savefig('../plots/HRG/ROC_'+ion+'_'+label+'.png')

    # for dropout_fraction in dropout_fraction_list:
    for fraction_kept in fraction_kept_list:
        dropout_fraction = 1 - fraction_kept
        print '######################################################### Dropout: ', dropout_fraction
        auc, roc_array = nx2.hierarchical_random_graph_prediction(ion, label, dropout_fraction, rank_percentage)
        auc_list.append(auc)
        fig = plt.figure(0)
        plt.plot(roc_array[:,1], roc_array[:,0], label='dropout p =' + str(dropout_fraction), linewidth=1.5)
        plt.legend(loc=8)
        plt.savefig('../plots/HRG/ROC_'+ion+'_'+label+'.png')


    #plot the auc graph
    _plot_auc(auc_list, fraction_kept_list)
####################################################################



ion = 'model'
label = 'dropout_full'
dropout_fraction_list = [0.5,0.3,0.1]
rank_percentage = 0.5 #percentage of predictions used for the sampling in the determination of the auc value
#TODO 1.0 funktioniert nicht?
run_whole(ion, label, dropout_fraction_list, rank_percentage)