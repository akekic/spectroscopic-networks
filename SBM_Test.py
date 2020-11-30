"""
@author: David Wellnitz
Description: ...

"""
import networkx as nx
import numpy as np
import graph_tool.all as gt
from sklearn.metrics import adjusted_rand_score
import time
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

try:
    reload
except NameError:
    # Python 3
    from imp import reload
import nx2
reload(nx2)

# H = nx2.model_network(E1=True, max_n=8)
# H = nx2.spectroscopic_network('He1.0', weighted=False)
# H = nx2.only_largest_component(H)
H = nx2.spectroscopic_network('H1.0', weighted=False)
H = nx2.only_largest_component(H)
dropout = 0.1
Graphs = [[H, 'Fe_Prediction_Length']]
force_niter_list = [100]
cutoff_list = [1]
pdict = []
probability_list = []
runs = 1


for G in Graphs:
    # start = time.time()
    lp_nSBM = nx2.LinkPrediction()
    lp_nSBM.G_original = G[0]
    # lp_nSBM.predict_nested_SBM()
    # lp_nSBM.check_if_correct()
    # duration = time.time() - start
    # print "This took ", duration, "seconds"
    # AUC = lp_nSBM.calculate_AUC()
    # print G[1], ' AUC: ', AUC
    # name = 'ROC_nSBM_' + G[1] + '_dropout' + str(dropout)
    # lp_nSBM.plot_ROC(name=name, save_switch=True, plotlabel=name)

    edge_list = np.asarray(lp_nSBM.G_original.edges())
    rs = ShuffleSplit(n_splits=1, test_size=dropout)  # fuer ShuffleSplit ist
    # n_splits so etwas wie unser n_runs http://scikit-learn.org/stable/modules
    # /generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    AUC_avg = np.zeros(len(cutoff_list))
    AUC_std = np.zeros(len(cutoff_list))
    runtime_avg = np.zeros(len(cutoff_list))
    runtime_std = np.zeros(len(cutoff_list))
    for train_index, validation_index in rs.split(edge_list):
        # get sets / do dropout
        lp_nSBM.cross_validation(edge_list, train_index, validation_index)
        # Add all right and as many wrong edges to prediction_list
        # n1list = np.random.choice(lp_nSBM.G_original.nodes(), size=len(lp_nSBM.probe_list))
        # n2list = np.random.choice(lp_nSBM.G_original.nodes(), size=len(lp_nSBM.probe_list))
        # false_edges = [(n1list[i], n2list[i]) for i in range(len(n1list)) if n1list[i] is not n2list[i]
        # pred_list = [(a[0], a[1]) for a in lp_nSBM.probe_list] + false_edges
        # print "pred_list", len(pred_list)
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
        ax = axs[0]
        ax.set_title('AUC')
        ax.axis(xmin=-1, xmax=cutoff_list[-1]+1)
        ax.set_xlabel('cutoff')
        ax.set_ylabel('AUC')
        axruntime = axs[1]
        axruntime.axis(xmin=-1, xmax=cutoff_list[-1]+1)
        axruntime.set_xlabel('cutoff')
        axruntime.set_ylabel('runtime')
        axruntime.set_title('runtime')
        for i, n in enumerate(force_niter_list):
            for j, cutoff in enumerate(cutoff_list):
                runtime = np.zeros(runs)
                AUC = np.zeros(runs)
                for run in range(runs):
                    start = time.time()
                    lp_nSBM.predict_nested_SBM(force_niter=n, cutoff=cutoff)
                    runtime[run] = time.time() - start
                    # pdict.append({p[0]: p[1] for p in lp_nSBM.prediction_list})
                    # probability_list.append(np.array([pdict[i][e] for e in pred_list]))
                    # np.savetxt('linkprobabilities_iron.txt', np.array(probability_list))
                    # np.savetxt('Order_iron.txt', np.array(pred_list), fmt="%s")
                    lp_nSBM.check_if_correct()
                    AUC[run] = lp_nSBM.calculate_AUC()
                    print AUC[run]
                AUC_avg[j] = AUC.mean()
                AUC_std[j] = AUC.std()
                runtime_avg[j] = runtime.mean()
                runtime_std[j] = runtime.std()
            ax.errorbar(cutoff_list, AUC_avg, yerr=AUC_std, fmt='o', label=n)
            ax.legend()
            axruntime.errorbar(cutoff_list, runtime_avg, yerr=runtime_std, fmt='o', label=n)
            axruntime.legend()
        plt.show()
