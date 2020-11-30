import sys
sys.path.append('../code')

import nx2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pylab
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import re
import graph_tool.all as gt
from sbm_thesis_plots import *


# plt.style.use('bmh')
params = {
   'axes.labelsize': 14,
   'font.size': 14,
   'font.family': 'serif',
   'legend.fontsize': 14,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'text.usetex': False,
   }
plt.rcParams.update(params)


# 1/2, 3/2, ..., 15/2, unknown
th2_colors = {'1/2': r'#e077e0', '3/2': r'#ff9499', '5/2': r'#ffbf87', '7/2': r'#ffe566', '9/2': r'#8dd96c',
              '11/2': r'#45e6c8', '13/2': r'#4391e0', '15/2': r'#884dc2', 'unknown': r'#404040'}


# ------------------------------- Plot Energy of all nodes ----------------------

def plot_energies(h, wavelength=False):
    from matplotlib.cm import get_cmap
    tab20b = get_cmap('Dark2')
    g = h.g
    if wavelength:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    else:
        fig, ax = plt.subplots()
        ax = [ax]
    group_number = g.new_vp('int')
    group_number.a = np.argsort(np.argsort(h.get_levels()[1].get_blocks().a))[h.project_level(0).get_blocks().a]
    for group in range(h.project_level(1).get_B()):
        blocks = h.project_level(1).get_blocks()
        vertices = [v for v in g.get_vertices() if blocks[v] == group]
        ax[0].scatter(group_number.a[vertices], g.vp.energy.a[vertices],
                   label=group, color=tab20b(float(group)/float(h.project_level(1).get_B())))
    xticks = h.project_level(0).get_blocks().a[np.unique(group_number.a, return_index=True)[1]]
    # Just increasing numbers
    # xticks = [''] * h.get_levels()[1].get_N()
    # for i in range(h.get_levels()[1].get_N()):
    #     if i%5 == 0:
    #         xticks[i] = i
    ax[0].set_xticks(range(h.get_levels()[1].get_N()))
    ax[0].set_xticklabels(xticks, fontdict={'fontsize': 7.})
    ax[0].xaxis.grid(True, alpha=0.5, color='k', linestyle='-')
    ax[0].set_xlabel('Community')
    ax[0].set_ylabel('State Energy in cm$^{-1}$')
    ax[0].set_ylim(0)
    ax[0].set_xlim(-1)
    # ax[0].legend(loc='best')
    if wavelength:
        ax[0].set_xticklabels(['']*len(xticks))
        ax[1].hist(g.ep.wavelength.a**(-1)*10**8, bins=50)
        ax[1].set_xlim(0)
        ax[1].set_xlabel('Transition Energy in cm$^{-1}$')
        ax[1].set_ylabel('Number of Transitions')
        fig.subplots_adjust(left=0.08)
    else:
        fig.subplots_adjust(left=0.15)
    return fig, ax


def plot_energies_configurations(h, wavelength=False):
    from matplotlib.cm import get_cmap
    tab20b = get_cmap('Dark2')
    g = h.g
    if wavelength:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = [ax]
    group_number = g.new_vp('int')
    group_number.a = np.argsort(np.argsort(h.get_levels()[1].get_blocks().a))[h.project_level(0).get_blocks().a]
    conf_values = np.unique([g.vp.configuration[v] for v in g.vertices()
                             if h.get_levels()[0].get_blocks()[v] in [8,10,28]])
    for group, c in enumerate(conf_values):
        vertices = [v for v in g.get_vertices() if g.vp.configuration[v] == c
                    if h.get_levels()[0].get_blocks()[v] in [25, 9, 24]]
        ax[0].scatter(g.vp.energy.a[vertices], group_number.a[vertices],
                   label=c, color=tab20b(float(group)/float(len(conf_values)+1)))
    # xticks = h.project_level(0).get_blocks().a[np.unique(group_number.a, return_index=True)[1]]
    # Just increasing numbers
    # xticks = [''] * h.get_levels()[1].get_N()
    # for i in range(h.get_levels()[1].get_N()):
    #     if i%5 == 0:
    #         xticks[i] = i
    # ax[0].set_xticks(range(h.get_levels()[1].get_N()))
    # ax[0].set_xticklabels(xticks, fontdict={'fontsize': 7.})
    # ax[0].xaxis.grid(True, alpha=0.5, color='k', linestyle='-')
    ax[0].set_ylabel('Community')
    ax[0].set_xlabel('State Energy in cm$^{-1}$')
    ax[0].legend(loc='upper left')
    if wavelength:
        ax[1].hist(g.ep.wavelength.a**(-1)*10**8)
        ax[1].set_xlabel('Transition Energy in cm$^{-1}$')
        ax[1].set_ylabel('Number of Transitions')
    return fig, ax


def plot_energies_by_jpar(h, wavelength=False):
    from matplotlib.cm import get_cmap
    tab20b = get_cmap('Dark2')
    g = h.g
    if wavelength:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    else:
        fig, ax = plt.subplots()
        ax = [ax]
    group_number = g.new_vp('int')
    group_number.a = np.argsort(np.argsort(h.get_levels()[1].get_blocks().a))[h.project_level(0).get_blocks().a]
    jp_map = concat_pmaps([g.vp.J, g.vp.parity])
    jp_values = sorted(np.unique([jp_map[v] for v in g.vertices()]),
                       key=lambda x: int(x[-1]=='e')*100 + int(x.split('/')[0]))
    for group, jp in enumerate(jp_values):
        vertices = [v for v in g.get_vertices() if jp_map[v] == jp]
        if re.findall('or', jp) == []:
            ax[0].scatter(group_number.a[vertices], g.vp.energy.a[vertices],
                          label=jp, color=tab20b(float(group)/float(len(jp_values))))
        else:
            ax[0].scatter(group_number.a[vertices], g.vp.energy.a[vertices], color='k')
    xticks = [''] * h.get_levels()[1].get_N()
    for i in range(h.get_levels()[1].get_N()):
        if i%5 == 0:
            xticks[i] = i
    ax[0].set_xticks(range(h.get_levels()[1].get_N()))
    ax[0].set_xticklabels(xticks)
    ax[0].xaxis.grid(True, alpha=0.5, color='k', linestyle='-')
    ax[0].set_xlabel('Community')
    ax[0].set_ylabel('State Energy in cm$^{-1}$')
    ax[0].set_xlim(-1, 45)
    ax[0].legend(loc='best')
    ax[0].set_ylim(0)
    if wavelength:
        ax[1].hist(g.ep.wavelength.a**(-1)*10**8)
        ax[1].set_xlabel('Transition Energy in cm$^{-1}$')
        ax[1].set_ylabel('Number of Transitions')
    else:
        fig.subplots_adjust(left=0.15)
    return fig, ax


def plot_energies_by_j(h, wavelength=False):
    from matplotlib.cm import get_cmap
    tab20b = get_cmap('Dark2')
    g = h.g
    if wavelength:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    else:
        fig, ax = plt.subplots()
        ax = [ax]
    group_number = g.new_vp('int')
    group_number.a = np.argsort(np.argsort(h.get_levels()[1].get_blocks().a))[h.project_level(0).get_blocks().a]
    j_values = sorted(np.unique([g.vp.J[v] for v in g.vertices()]),
                       key=lambda x: int(x.split('/')[0]))
    for group, j in enumerate(j_values):
        vertices = [v for v in g.get_vertices() if g.vp.J[v] == j]
        if re.findall('or', j) == []:
            ax[0].scatter(group_number.a[vertices], g.vp.energy.a[vertices],
                          label=j, color=tab20b(float(group)/float(len(j_values))))
        else:
            ax[0].scatter(group_number.a[vertices], g.vp.energy.a[vertices], color='k', alpha=0.3)
        odd_vs = [v for v in vertices if g.vp.parity[v] == 'o']
        ax[0].scatter(group_number.a[odd_vs], g.vp.energy.a[odd_vs], color='k', marker='x', s=8)
    ax[0].scatter([], [], color='k', alpha=0.3, label='unknown')
    # Just increasing number
    # xticks = [''] * h.get_levels()[1].get_N()
    # for i in range(h.get_levels()[1].get_N()):
    #     if i%5 == 0:
    #         xticks[i] = i
    xticks = h.project_level(0).get_blocks().a[np.unique(group_number.a, return_index=True)[1]]
    ax[0].set_xticks(range(h.get_levels()[1].get_N()))
    ax[0].set_xticklabels(xticks, fontdict={'fontsize': 7.})
    ax[0].xaxis.grid(True, alpha=0.5, color='k', linestyle='-')
    ax[0].set_xlabel('Community')
    ax[0].set_ylabel('State energy in cm$^{-1}$')
    ax[0].set_xlim(-1, 45)
    ax[0].set_ylim(0)
    ax[0].legend(loc='best')
    if wavelength:
        ax[1].hist(g.ep.wavelength.a**(-1)*10**8)
        ax[1].set_xlabel('Transition energy in cm$^{-1}$')
        ax[1].set_ylabel('Number of transitions')
    else:
        fig.subplots_adjust(left=0.15)
    return fig, ax


def plot_prop_by_j(h, prop_name):
    from matplotlib.cm import get_cmap
    tab20b = get_cmap('Dark2')
    g = h.g
    fig, ax = plt.subplots()
    ax = [ax]
    group_number = g.new_vp('int')
    group_number.a = np.argsort(np.argsort(h.get_levels()[1].get_blocks().a))[h.project_level(0).get_blocks().a]
    j_values = sorted(np.unique([g.vp.J[v] for v in g.vertices()]),
                       key=lambda x: int(x.split('/')[0]))
    for group, j in enumerate(j_values):
        vertices = [v for v in g.get_vertices() if g.vp.J[v] == j]
        if re.findall('or', j) == []:
            ax[0].scatter(group_number.a[vertices], g.vp[prop_name].a[vertices],
                          label=j, color=tab20b(float(group)/float(len(j_values))))
        else:
            ax[0].scatter(group_number.a[vertices], g.vp[prop_name].a[vertices], color='k', alpha=0.3)
        odd_vs = [v for v in vertices if g.vp.parity[v] == 'o']
        ax[0].scatter(group_number.a[odd_vs], g.vp[prop_name].a[odd_vs], color='k', marker='x', s=8)
    ax[0].scatter([], [], color='k', alpha=0.3, label='unknown')
    # Just increasing number
    # xticks = [''] * h.get_levels()[1].get_N()
    # for i in range(h.get_levels()[1].get_N()):
    #     if i%5 == 0:
    #         xticks[i] = i
    xticks = h.project_level(0).get_blocks().a[np.unique(group_number.a, return_index=True)[1]]
    ax[0].set_xticks(range(h.get_levels()[1].get_N()))
    ax[0].set_xticklabels(xticks, fontdict={'fontsize': 7.})
    ax[0].xaxis.grid(True, alpha=0.5, color='k', linestyle='-')
    ax[0].set_xlabel('Community')
    ax[0].set_ylabel('State energy in cm$^{-1}$')
    ax[0].set_xlim(-1, 45)
    ax[0].legend(loc='best')
    fig.subplots_adjust(left=0.15)
    return fig, ax


def plot_th2_energies_by_j(h, wavelength=False):
    g = h.g
    if wavelength:
        fig, ax = plt.subplots(ncols=3, figsize=(26, 5))
    else:
        fig, ax = plt.subplots(ncols=2, figsize=(14, 5))
    group_number = g.new_vp('int')
    even_groups = np.unique(h.project_level(0).get_blocks().a[np.array([g.vp.parity[v] == 'e' for v in g.vertices()],
                                                                       dtype='bool')])
    odd_groups = np.unique(h.project_level(0).get_blocks().a[np.array([g.vp.parity[v] == 'o' for v in g.vertices()],
                                                                       dtype='bool')])
    # group : position in plot so that increasing by J and increasing energy
    # even_positions = {group: np.argsort(np.argsort(h.get_levels()[1].get_blocks().a[even_groups]))[i]
    #                   for i, group in enumerate(even_groups)}
    odd_positions = {group: np.argsort(np.argsort(h.get_levels()[1].get_blocks().a[odd_groups]))[i]
                      for i, group in enumerate(odd_groups)}
    even_positions = {31: 0, 21: 1,
                      10: 3, 28: 4, 8: 5,
                      26: 7, 22: 8, 6: 9,
                      25: 11, 24: 12, 9: 13,
                      27: 15, 23: 16, 5: 17, 11: 18, 30: 19}
    odd_positions = {3: 0, 4: 1, 17: 2, 13: 3, 15: 4,
                     19: 6, 2: 7, 29: 8, 12: 9,
                     20: 11, 7: 12, 16: 13,
                     0: 15, 1: 16, 18: 17, 14: 18}

    # group_number.a = np.argsort(np.argsort(h.get_levels()[1].get_blocks().a))[h.project_level(0).get_blocks().a]
    j_values = sorted(np.unique([g.vp.J[v] for v in g.vertices()]), key=lambda x: int(x.split('/')[0]))

    for group, j in enumerate(j_values):
        vertices = [v for v in g.get_vertices() if g.vp.J[v] == j and g.vp.parity[v] == 'e']
        if not re.findall('or', j):
            ax[0].scatter([even_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp.energy.a[vertices], label=j, color=th2_colors[j], s=35)
        else:
            ax[0].scatter([even_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp.energy.a[vertices], color=th2_colors['unknown'], s=35)
    ax[0].scatter([], [], color=th2_colors['unknown'], label='unknown')
    ax[0].set_title('even')

    for group, j in enumerate(j_values):
        vertices = [v for v in g.get_vertices() if g.vp.J[v] == j and g.vp.parity[v] == 'o']
        if not re.findall('or', j):
            ax[1].scatter([odd_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp.energy.a[vertices], label=j, color=th2_colors[j], s=35)
        else:
            ax[1].scatter([odd_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp.energy.a[vertices], color=th2_colors['unknown'], s=35)
    ax[1].scatter([], [], color=th2_colors['unknown'], label='unknown')
    ax[1].set_title('odd')
    # Just increasing number
    # xticks = [''] * h.get_levels()[1].get_N()
    # for i in range(h.get_levels()[1].get_N()):
    #     if i%5 == 0:
    #         xticks[i] = i
    # xticks = h.project_level(0).get_blocks().a[np.unique(group_number.a, return_index=True)[1]]
    # ax[0].set_xticks(range(h.get_levels()[1].get_N()))
    # ax[0].set_xticklabels(xticks, fontdict={'fontsize': 7.})
    ax[0].set_xlabel('Community')
    ax[0].set_ylabel('State energy in cm$^{-1}$')
    ax[0].set_ylim(-800, 70000)
    ax[0].set_xlim(-0.2, 20)
    for x in [2, 6, 10, 14]:
        ax[0].plot((x, x), ax[0].get_ylim(), alpha=1., color='k', linestyle='--', lw=1.4)

    ax[1].set_xlabel('Community')
    ax[1].set_ylim(-800, 70000)
    ax[1].set_xlim(-0.2, 19)
    ax[1].set_yticklabels(['']*len(ax[1].get_yticks()))
    ax[1].tick_params(length=0.)
    for x in [5, 10, 14]:
        ax[1].plot((x, x), ax[1].get_ylim(), alpha=1., color='k', linestyle='--', lw=1.4)
    for axis in ax:
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.tick_params(length=0.)
        # for spine in axis.spines.values():
        #   spine.set_position(('outward', 5))
        axis.yaxis.grid(color="k", linestyle='-', lw=0.6, alpha=0.3)
        axis.set_xticks([])
    ax[1].spines['left'].set_visible(False)
    # ax[0].legend(loc='best')
    if wavelength:
        ax[-1].hist(g.ep.wavelength.a**(-1)*10**8)
        ax[-1].set_xlabel('Transition energy in cm$^{-1}$')
        ax[-1].set_ylabel('Number of transitions')
    else:
        fig.subplots_adjust(left=0.15)
    return fig, ax


def plot_th2_prop_by_j(h, prop_name):
    g = h.g
    fig, ax = plt.subplots(ncols=2, figsize=(14, 5))
    group_number = g.new_vp('int')
    even_groups = np.unique(h.project_level(0).get_blocks().a[np.array([g.vp.parity[v] == 'e' for v in g.vertices()],
                                                                       dtype='bool')])
    odd_groups = np.unique(h.project_level(0).get_blocks().a[np.array([g.vp.parity[v] == 'o' for v in g.vertices()],
                                                                       dtype='bool')])
    # group : position in plot so that increasing by J and increasing energy
    # even_positions = {group: np.argsort(np.argsort(h.get_levels()[1].get_blocks().a[even_groups]))[i]
    #                   for i, group in enumerate(even_groups)}
    odd_positions = {group: np.argsort(np.argsort(h.get_levels()[1].get_blocks().a[odd_groups]))[i]
                      for i, group in enumerate(odd_groups)}
    even_positions = {31: 0, 21: 1,
                      10: 3, 28: 4, 8: 5,
                      26: 7, 22: 8, 6: 9,
                      25: 11, 24: 12, 9: 13,
                      27: 15, 23: 16, 5: 17, 11: 18, 30: 19}
    odd_positions = {3: 0, 4: 1, 17: 2, 13: 3, 15: 4,
                     19: 6, 2: 7, 29: 8, 12: 9,
                     20: 11, 7: 12, 16: 13,
                     0: 15, 1: 16, 18: 17, 14: 18}

    # group_number.a = np.argsort(np.argsort(h.get_levels()[1].get_blocks().a))[h.project_level(0).get_blocks().a]
    j_values = sorted(np.unique([g.vp.J[v] for v in g.vertices()]), key=lambda x: int(x.split('/')[0]))

    for group, j in enumerate(j_values):
        vertices = [v for v in g.get_vertices() if g.vp.J[v] == j and g.vp.parity[v] == 'e']
        if not re.findall('or', j):
            ax[0].scatter([even_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp[prop_name].a[vertices], label=j, color=th2_colors[j], s=35)
        else:
            ax[0].scatter([even_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp[prop_name].a[vertices], color=th2_colors['unknown'], s=35)
    ax[0].scatter([], [], color=th2_colors['unknown'], label='unknown')
    ax[0].set_title('even')

    for group, j in enumerate(j_values):
        vertices = [v for v in g.get_vertices() if g.vp.J[v] == j and g.vp.parity[v] == 'o']
        if not re.findall('or', j):
            ax[1].scatter([odd_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp[prop_name].a[vertices], label=j, color=th2_colors[j], s=35)
        else:
            ax[1].scatter([odd_positions[comm] for comm in h.get_levels()[0].get_blocks().a[vertices]],
                          g.vp[prop_name].a[vertices], color=th2_colors['unknown'], s=35)
    ax[1].scatter([], [], color=th2_colors['unknown'], label='unknown')
    ax[1].set_title('odd')
    # Just increasing number
    # xticks = [''] * h.get_levels()[1].get_N()
    # for i in range(h.get_levels()[1].get_N()):
    #     if i%5 == 0:
    #         xticks[i] = i
    # xticks = h.project_level(0).get_blocks().a[np.unique(group_number.a, return_index=True)[1]]
    # ax[0].set_xticks(range(h.get_levels()[1].get_N()))
    # ax[0].set_xticklabels(xticks, fontdict={'fontsize': 7.})
    ax[0].set_xlabel('Community')
    ax[0].set_ylabel('State energy in cm$^{-1}$')
    ax[0].set_xlim(-0.2, 20)
    for x in [2, 6, 10, 14]:
        ax[0].plot((x, x), ax[0].get_ylim(), alpha=1., color='k', linestyle='--', lw=2.)

    ax[1].set_xlabel('Community')
    ax[1].set_xlim(-0.2, 19)
    ax[1].set_yticklabels(['']*len(ax[1].get_yticks()))
    ax[1].tick_params(length=0.)
    for x in [5, 10, 14]:
        ax[1].plot((x, x), ax[1].get_ylim(), alpha=1., color='k', linestyle='--', lw=2.)
    for axis in ax:
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['right'].set_visible(False)
        for spine in axis.spines.values():
          spine.set_position(('outward', 5))
        axis.yaxis.grid(color="k", linestyle=':', lw=0.6, alpha=0.8)
        axis.set_xticks([])
    ax[1].spines['left'].set_visible(False)
    # ax[0].legend(loc='best')
    fig.subplots_adjust(left=0.15)
    return fig, ax


def plot_some_energies():
    trans_data = pd.read_csv('node-statistics.csv')
    trans_data = trans_data.set_index('Atom')
    trans_data.sort_values('Edges', ascending=False, inplace=True)
    for atom in trans_data.head(10).index.values[4:]:
        nxg = nx2.load_network(atom)
        g = nx2.nx2gt(nxg)
        h = gt.minimize_nested_blockmodel_dl(g)
        fig, ax = plot_energies(h, wavelength=True)
        fig.savefig(atom[:-2]+'-energies.pdf')


def plot_feth_energies():
    import pickle
    trans_data = pd.read_csv('node-statistics.csv')
    trans_data = trans_data.set_index('Atom')
    trans_data.sort_values('Edges', ascending=False, inplace=True)
    for atom in trans_data.head(4).index.values:
        with open(atom + '-hierarchies.pickle') as f:
            hs = pickle.load(f)
        h = hs[0]
        fig, ax = plot_energies(h, wavelength=True)
        fig.savefig(atom[:-2]+'-energies.pdf')


# ------------------------------ Import the data --------------------------------
def load_lp_data(atoms, methods, dropouts):
    roc_data = pd.DataFrame()
    for atom in atoms:
        for method in methods:
            for dropout in dropouts:
                fname = '../data/ROC_data_results/' + atom + '/' + method + '/' + atom + '_dropout_' + method + \
                        '_ROC_full_NIST_dropout_value_' + str(dropout) + '.csv'
                single_df = pd.read_csv(fname, header=None, comment='#',
                                           names=[atom+method+str(dropout)+'-FPR', atom+method+str(dropout)+'-TPR',
                                                  atom+method+str(dropout)+'-TPR_err'])
                print atom, method, dropout, len(single_df)
                roc_data = pd.concat([roc_data, single_df], axis=1)
    return roc_data


def load_np_data(atoms, methods):
    roc_data = pd.DataFrame()
    for atom in atoms:
        for method in methods:
            fname = '../data/np-data/' + atom + '/' + method + '/' + atom + method + '-roc-dropout10.csv'
            single_df = pd.read_csv(fname, header=None, comment='#',
                                       names=[atom+method+'-FPR', atom+method+'-TPR', atom+method+'-TPR_err'])
            print atom, method, len(single_df)
            roc_data = pd.concat([roc_data, single_df], axis=1)
    return roc_data



# ------------------------------ Plot the data ----------------------------------
def plot_rocs(df, atoms, methods, dropouts=[''], fname=None):
    fig, ax = plt.subplots()
    # inset_axes = zoomed_inset_axes(ax, 6, loc=4)
    colors = ['#1b9e77', '#d95f02']
    for i, atom in enumerate(atoms):
        for method in methods:
            for dropout in dropouts:
                FPR = df[atom+method+str(dropout)+'-FPR']
                TPR = df[atom+method+str(dropout)+'-TPR']
                TPR_err = df[atom+method+str(dropout)+'-TPR_err']
                if len(FPR)>10000:
                    TPR = np.interp(np.linspace(0., 1., 1000), FPR, TPR)
                    TPR_err = np.interp(np.linspace(0., 1., 1000), FPR, TPR_err)
                    FPR = np.linspace(0., 1., 1000)
                lines = ax.plot(FPR, TPR, label=atom[:-3], color=colors[i])
                ax.fill_between(FPR, TPR + 0.5 * TPR_err, TPR - 0.5 * TPR_err, alpha=0.3, color=lines[0].get_color())
                # lines = inset_axes.plot(FPR, TPR, label=method)
                # inset_axes.fill_between(FPR, TPR + 0.5 * TPR_err, TPR - 0.5 * TPR_err, alpha=0.3,
                #                         color=lines[0].get_color())
    # ax.legend(loc=(0.607, 0.65))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC Curves for ' + '-'.join([atom[:-3] + (int(atom[-3]) - 1)*'+' for atom in atoms]))
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3, color='k', linestyle='-')
    ax.yaxis.grid(True, alpha=0.3, color='k', linestyle='-')
    # x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
    # ip = InsetPosition(ax, [0.45, 0.05, 0.5, 0.5])
    # inset_axes.set_axes_locator(ip)
    # inset_axes.set_xlim(x1, x2)
    # inset_axes.set_ylim(y1, y2)
    # x_sub_ticks = ['0.0', '0.1', '0.2', '0.3']
    # y_sub_ticks = ['0.7', '0.8', '0.9', '1.0']
    # inset_axes.set_xticks((0, 0.1, 0.2, 0.3))
    # inset_axes.set_yticks((0.7, 0.8, 0.9, 1.0))
    # inset_axes.tick_params(labelsize=9, pad=0.8, labelbottom=False, labeltop=True, labelleft=True, labelright=False)
    # plt.xticks(visible=True, rotation='horizontal')
    # inset_axes.set_xticklabels(x_sub_ticks)
    # plt.yticks(visible=True, rotation='horizontal')
    # inset_axes.set_yticklabels(y_sub_ticks)
    # legend_fig = pylab.figure()
    # legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(),
    #                          loc='center', fancybox=True, shadow=True, scatterpoints=1, markerscale=13,
    #                          ncol=1, labelspacing=0.6, handlelength=0.9)
    # legend_fig.canvas.draw()
    if fname:
        fig.savefig(fname)
        # lname = re.split(r'(\.)', fname)
        # lname.insert(-2, '-legend')
        # legend_fig.savefig(''.join(lname))
    return ax


if __name__ == '__main__':
    load_lp_data(['Fe1.0'],['nSBM'],[0.1])
    data = load_lp_data(['Fe1.0', 'C1.0'], ['nSBM', 'SPM', 'HRG'], [0.1])
    plot_rocs(data, ['Fe1.0'], ['nSBM', 'SPM', 'HRG'], [0.1])
    plt.show()
    plot_rocs(data, ['C1.0'], ['nSBM', 'SPM', 'HRG'], [0.1])
    plt.show()
    np_data = load_np_data(['Th2.0'], ['adjacency', 'groups', 'laplace', 'laplacian'])
    plot_rocs(np_data, ['Th2.0'], ['adjacency', 'groups', 'laplace', 'laplacian'])


# for method in methods:
#     roc_curves = np.array(roc[method][i])  # (n_runs x n_datapoints) array of tuples TPR, FPR
#     x_values = np.linspace(0, 1, num=10 ** 6)
#     TPR_interp = np.zeros((n_runs, 10 ** 6))
#     for j in range(n_runs):
#         TPR_interp[j] = np.interp(x_values, roc_curves[j][:, 0], roc_curves[j][:, 1])
#     TPR_values = np.mean(TPR_interp, axis=0)
#     TPR_errors = np.std(TPR_interp, axis=0) / np.sqrt(float(n_runs))
#     roc_data[method] = np.array(zip(x_values, TPR_values, TPR_errors))
#     np.savetxt('../plots/ROC_' + method + G[1] + '.dat', roc_data[method])
#
#
#
# # create colors for markers
# # color=cmap(i / float(len(method_list)))
# fig = plt.figure()
# # cmap = plt.get_cmap('Set1') # choose colormap
# cmap = plt.get_cmap('Set1')  # choose colormap
# ax = plt.axes()
#
# inset_axes = zoomed_inset_axes(ax, 6,  # zoom
#                                loc=4)
#
# # for i,dropout in enumerate(dropout_list):
# for j, method in enumerate(methods):
#     TPR = roc_data[method][:, 1]
#     TPR_err = roc_data[method][:, 2]
#     FPR = roc_data[method][:, 0]
#
#     # create colors for markers
#     color = cmap(j / float(len(methods)))
#
#     # plot ROC curves
#     ax.scatter(FPR, TPR, label=method, color=color, s=0.4)
#     ax.fill_between(FPR, TPR + 0.5 * TPR_err, TPR - 0.5 * TPR_err, alpha=0.3, color=color)  # plot error shading
#
#     inset_axes.scatter(FPR, TPR, label=method, color=color, s=0.4)
#     inset_axes.fill_between(FPR, TPR + 0.5 * TPR_err, TPR - 0.5 * TPR_err, alpha=0.3, color=color)
#
# ax.plot((0, 1), (0, 1), 'r--')
# font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
# # ax.tick_params(labelsize=9, pad=0.7, labelbottom=False, labeltop=True, labelleft=True, labelright=False)
# ax.set_xlabel('FPR', fontdict=font)
# ax.set_ylabel('TPR', fontdict=font)
# # ax.set_title('ROC')
# ax.set_xlim(xmin=0.0, xmax=1.0)
# ax.set_ylim(ymin=0.0, ymax=1.0)
#
# # sub region of the original image
# x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
# inset_axes.set_xlim(x1, x2)
# inset_axes.set_ylim(y1, y2)
#
# # position of the bbox in original image (where to plot)
# ip = InsetPosition(ax, [0.45, 0.05, 0.5, 0.5])
# inset_axes.set_axes_locator(ip)
#
# # achsenbeschriftung von box in box
# x_sub_ticks = ['0.0', '0.1', '0.2', '0.3']
# y_sub_ticks = ['0.7', '0.8', '0.9', '1.0']
# inset_axes.set_xticks((0, 0.1, 0.2, 0.3))
# inset_axes.set_yticks((0.7, 0.8, 0.9, 1.0))
# inset_axes.tick_params(labelsize=9, pad=0.8, labelbottom=False, labeltop=True, labelleft=True, labelright=False)
# plt.xticks(visible=True, rotation='horizontal')
# inset_axes.set_xticklabels(x_sub_ticks)
# plt.yticks(visible=True, rotation='horizontal')
# inset_axes.set_yticklabels(y_sub_ticks)
#
# # draw a bbox of the region of the inset axes in the parent axes and
# # connecting lines between the bbox and the inset axes area
# mark_inset(ax, inset_axes, loc1=1, loc2=3, fc="none", ec="0.4", ls='solid')
#
# # # old version of legend
# # # Shrink current axis's height by 10% on the bottom
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0 + box.height * 0.1,
# #                  box.width, box.height * 0.9])
#
# # # Put a legend below current axis
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
# #           fancybox=True, shadow=True, scatterpoints=1, markerscale=10,
# #            ncol=7, labelspacing=0.3, handlelength=0.6)
#
# # plot legend in additional file
# import pylab
#
# # fig = pylab.figure()
# legend_fig = pylab.figure()
# legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(),
#                          loc='center', fancybox=True, shadow=True, scatterpoints=1, markerscale=13,
#                          ncol=1, labelspacing=0.6, handlelength=0.9)
# # legend.get_frame().set_color('0.70') # make grey background
# legend_fig.canvas.draw()
# # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/figure_concept/'+ion+'_legend_cropped.png', bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
# legend_fig.savefig('../plots/ROC' + G[1] + 'legend.png')
# # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-NIST'+'_legend_original.png')
# # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik-Dipole'+'_legend_original.png')
# # legend_fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik'+'_legend_original.png')
#
# plt.draw()
# # plt.show()
# # fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-NIST_'+str(dropout)+'.png') #fuer H1.0
# # fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik-Dipole_'+str(dropout)+'.png') #fuer H1.0
# # fig.savefig('/home/julian/qd-networks/svn/reports/paper_draft/pics/'+ion+'-Jitrik_'+str(dropout)+'.png') #fuer H1.0
# fig.savefig('../plots/ROC' + G[1] + '.png')
# # fig.savefig('/home/julian/qd-networks/svn/plots/'+ion+'_'+str(dropout)+'.png', dpi=300)
