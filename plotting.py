#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.linalg import block_diag
from sklearn.metrics.cluster import contingency_matrix
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import silhouette_score

import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns
from utils import perf_measure, hcluster_silhouette, clustering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import matplotlib
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]


# =============================================================================
#                     Plot atlas
# =============================================================================
def plot_simulation_atlas(atlas,
                          gs,
                          args,
                          W, W_hat,
                          H, H_hat,
                          t, e,
                          beta,
                          labels_true,
                          affinity='euclidean',
                          normalize = True,
                          weighted = False):
    
    if normalize: # row normalization on W
        W_hat = (W_hat.T / np.linalg.norm(W_hat, axis = 1)).T
    # W_Tilde
    W_Tilde = W_hat * beta
    
    f3_ax3 = atlas.add_subplot(gs[2:, 2])
    f3_ax3.set_title(r'(C) $\bm{\bar{W}}$', loc = 'left')
    f3_ax3.axis('off')
    
    T_sorted, sil_score, roworder = clustering(args, W_hat, beta, 'hclust')
    
    colorder = np.argsort(beta)
    beta_ordered = beta[colorder]
    W_Tilde_ordered = W_Tilde[roworder,:][:,colorder]
    labels_true_ordered = labels_true[roworder]
    cluster_list = np.arange(args.K_hat) # [0, 1, 2, ..., K_hat-1]
    
# =============================================================================
#     labels_true cluster
# =============================================================================
    f3_ax1 = atlas.add_subplot(gs[2:, 0])
    clusters = f3_ax1.imshow(labels_true.reshape(-1,1), aspect='auto', cmap = 'rainbow', interpolation='none')
    f3_ax1.get_yaxis().set_visible(False)
    f3_ax1.get_xaxis().set_visible(False)
    alphabetics = ['C' + str(i+1) for i in range(len(np.unique(labels_true)))]
    for x in range(len(np.unique(labels_true))):
        f3_ax1.axhline(y = np.max(np.where(labels_true == x)[0]), linewidth=0.5, color='w')
        txt = f3_ax1.text(-0.8, np.mean(np.where(labels_true == x)[0]), '%s' % alphabetics[int(np.unique(labels_true)[x])],\
                 horizontalalignment='right',\
                 verticalalignment='center')
    f3_ax1.set_title('(B)', loc = 'left')
# =============================================================================
#     W
# =============================================================================
    f3_ax2 = atlas.add_subplot(gs[2:, 1])
    f3_ax2.set_title(r'Ground truth $\bm{W}$', loc = 'left')
    cmap = plt.cm.bwr
    cmap.set_bad(color='white')
    maxval = np.max(W)
    im_W = f3_ax2.imshow(W, cmap=cmap, vmin=-maxval, vmax=maxval, interpolation='none')
    f3_ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    f3_ax2.set_xticklabels(np.arange(W.shape[1]+1))
    f3_ax2.axis('auto')
    f3_ax2.get_yaxis().set_visible(False)
    atlas.colorbar(im_W, ax=f3_ax2, orientation='horizontal')
# =============================================================================
#     W_hat
# =============================================================================
    # W_hat
    f3_ax4 = atlas.add_subplot(gs[2:, 3])
    f3_ax5 = atlas.add_subplot(gs[2:, 4])
       
    
    maxval = np.max(np.abs(W_Tilde_ordered))
    im_W_Tilde = f3_ax4.imshow(W_Tilde_ordered, cmap = cmap, vmin=-maxval, vmax=maxval, interpolation='none') # norm = matplotlib.colors.LogNorm()
    f3_ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
    f3_ax4.set_xticklabels(np.arange(W_Tilde_ordered.shape[1]+1))
    f3_ax4.axis('auto')
    f3_ax4.get_yaxis().set_visible(False)
    atlas.colorbar(im_W_Tilde, ax=f3_ax4, orientation='horizontal')
    
    """
    Note: All "negative" terms below stands for the negative value in W_Tilde,
    not the negative association to survival! Actually, the "negative" value 
    in W_Tilde suggests a positive association to survival. Don't get confused!
    """
    tau1, tau2 = 0, 0
    if 'univariate' in args.simulation_type:
        if 'better' in args.simulation_type:
            tau1 = 1
        if 'worse' in args.simulation_type:
            tau2 = 1
            
    if 'multivariate' in args.simulation_type:
        if 'better' in args.simulation_type:
            tau1 = 2
        if 'worse' in args.simulation_type:
            tau2 = 2
        
        
        
    for i in range(args.K_hat):
        if i < tau1:
            color = 'b'
            weight = 'bold'
        elif i >= args.K_hat-tau2:
            color = 'r'
            weight = 'bold'
        else:
            color = 'grey'
            weight = 'normal'
        f3_ax4.text(i, -0.01*W.shape[0], '%.2f' % beta_ordered[i],\
                 horizontalalignment='center',\
                 verticalalignment='bottom', rotation=90, \
                 color = color, weight = weight)

    
    
    for tau in range(tau1):
        cluster_neg_in_T = np.argmin([np.mean(W_Tilde_ordered[T_sorted == c, tau]) for c in cluster_list])
        y_negative_pred = [1 if ts == cluster_neg_in_T else 0 for ts in T_sorted]
        x_start = tau-0.5
        y_start = np.where(y_negative_pred)[0][0]
        x_len = 1
        y_len = sum(y_negative_pred)
        rect_min = f3_ax4.add_patch(patches.Rectangle((x_start,y_start),x_len,y_len, linewidth=2, edgecolor='b', facecolor='none'))

    for tau in range(1,tau2+1):
        cluster_pos_in_T = np.argmax([np.mean(W_Tilde_ordered[T_sorted == c, -tau]) for c in cluster_list])
        y_positive_pred = [1 if ts == cluster_pos_in_T else 0 for ts in T_sorted]
        x_start = args.K_hat-tau-0.5
        y_start = np.where(y_positive_pred)[0][0]
        x_len = 1
        y_len = sum(y_positive_pred)
        rect_max = f3_ax4.add_patch(patches.Rectangle((x_start,y_start),x_len,y_len, linewidth=2, edgecolor='r', facecolor='none'))

        
    f3_ax5.imshow(labels_true_ordered.reshape(-1,1), aspect='auto', cmap = 'rainbow')
    f3_ax5.get_yaxis().set_visible(False)
    f3_ax5.get_xaxis().set_visible(False)
    f3_ax5.set_title('(D)', loc = 'left')
    
# =============================================================================
#     H
# =============================================================================
    # survival time
    f3_ax7 = atlas.add_subplot(gs[0, :])
    sortind = np.argsort(t)
    e_sorted = e[sortind]
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='black')
    f3_ax7.plot(np.arange(len(t)), t[sortind], color = 'red')
    f3_ax7.set_xlim(0, len(t))
    f3_ax7.get_xaxis().set_visible(False)
    f3_ax7.set_title(r'(A) Survival time and $\hat{\bm{H}}$', loc = 'left')
    # H
    
    vmax = np.max(H_hat)
    H_hat_show = H_hat[colorder,:][:,sortind]
    f3_ax8 = atlas.add_subplot(gs[1, :])
    im = f3_ax8.imshow(H_hat_show, cmap = cmap, vmin=0, vmax=vmax)
    f3_ax8.get_xaxis().set_visible(False)
    f3_ax8.get_yaxis().set_visible(False)
    f3_ax8.set_ylabel('CoxNMF')
    f3_ax8.set_yticklabels([])
    f3_ax8.axis('auto')
    atlas.colorbar(im, ax=f3_ax8,aspect=40, orientation="horizontal")
    
    return atlas


def plot_TCGA_atlas(atlas,
                    gs,
                    args,
                    X,
                    W, W_hat,
                    H, H_hat,
                    t, e,
                    beta,
                    labels_true,
                    # sil_rerror,
                    # relative_error_threshold,
                    affinity='euclidean',
                    normalize = True,
                    weighted = False):
    
    if normalize: # row normalization on W
        W_hat = (W_hat.T / np.linalg.norm(W_hat, axis = 1)).T
    # W_Tilde
    
    if args.specific_row is None:
        W_Tilde = W_hat * beta
    else:
        W_Tilde = W_hat
    
# =============================================================================
#     hierarchical clustering dendrogram
# =============================================================================
    f_all_0 = atlas.add_subplot(gs[:,0])
    f_all_0.set_title(r'(A) $\bm{\tilde{W}}$', loc = 'left')
    f_all_0.axis('off')

    
    T_sorted, sil_score, roworder = clustering(args, W_hat, beta, 'hclust')
    
    
    if args.specific_row is None:
        colorder = np.argsort(beta)
        beta_ordered = beta[colorder]
    else:
        colorder = np.arange(W_Tilde.shape[1])
        beta_ordered = np.zeros(W_Tilde.shape[1])
        beta_ordered[args.specific_row] = beta
        
    W_Tilde_ordered = W_Tilde[roworder,:][:,colorder]
    cluster_list = np.arange(args.K_hat) # [0, 1, 2, ..., K_hat-1]
    X_sorted = X.iloc[roworder,:]
    
    gene_clusters = {}
    for x in range(len(np.unique(T_sorted))):
        gid = np.where(T_sorted == x)[0]
        genes = X_sorted.index[gid].astype(str)
        gene_clusters['C%02d' % (x+1)] = list(genes)
    
# =============================================================================
#     label bar
# =============================================================================
    f_all_1 = atlas.add_subplot(gs[:,1])
    clusters = f_all_1.imshow(T_sorted.reshape(-1,1), aspect='auto', cmap = 'rainbow', interpolation='none')
    f_all_1.get_xaxis().set_visible(False)
    
    for x in range(len(np.unique(T_sorted))):
        loc = x
        f_all_1.axhline(y = np.max(np.where(T_sorted == loc)[0]), linewidth=0.5, color='w')
        txt = f_all_1.text(-0.8, np.mean(np.where(T_sorted == loc)[0]), 'C%d' % (loc+1),\
                 horizontalalignment='right',\
                 verticalalignment='center')
    f_all_1.axis('off')
    f_all_1.text(0, 0, r'$\hat{\beta}=$',\
             horizontalalignment='center',\
             verticalalignment='bottom', fontsize=14)

# =============================================================================
#     W_hat
# =============================================================================
    f_all_2 = atlas.add_subplot(gs[:, 2])
    maxval = np.max(W_hat)
    cmap = plt.cm.bwr
    cmap.set_bad(color='white')
    maxval = np.max(np.abs(W_Tilde_ordered))
    im_W_Tilde = f_all_2.imshow(W_Tilde_ordered, cmap = cmap, vmin=-maxval, vmax=maxval, interpolation='none') # norm = matplotlib.colors.LogNorm()
    f_all_2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    f_all_2.set_xticklabels(np.arange(W_Tilde_ordered.shape[1]+1))
    f_all_2.axis('auto')
    f_all_2.get_yaxis().set_visible(False)
    atlas.colorbar(im_W_Tilde, ax=f_all_2, orientation='horizontal')
    
    """
    Note: All "negative" terms below stands for the negative value in W_Tilde,
    not the negative association to survival! Actually, the "negative" value 
    in W_Tilde suggests a positive association to survival. Don't get confused!
    """
    
        
    cluster_ID = {'positve (red)': [],
                  'negative (blue)': []}
    cluster_in_T_list = []
    for k in range(args.K_hat):
        if beta_ordered[k] < 0:
            color = 'b'
            weight = 'bold'
            key = 'negative (blue)'
        elif beta_ordered[k] >= 0:
            color = 'r'
            weight = 'bold'
            key = 'positve (red)'
        else:
            color = 'k'
            weight = 'normal'
        f_all_2.text(k, -0.01*W_hat.shape[0], '%.2f' % beta_ordered[k],\
                 horizontalalignment='center',\
                 verticalalignment='bottom', rotation=90, \
                 color = color, weight = weight)
        
        if beta_ordered[k] < 0:
            cluster_in_T = np.argmin([np.mean(W_Tilde_ordered[T_sorted == c, k]) for c in cluster_list])
        elif beta_ordered[k] >= 0:
            cluster_in_T = np.argmax([np.mean(W_Tilde_ordered[T_sorted == c, k]) for c in cluster_list])
        cluster_in_T_list.append(cluster_in_T)
        y_pred = [1 if ts == cluster_in_T else 0 for ts in T_sorted]
        x_start = k-0.5
        y_start = np.where(y_pred)[0][0]
        x_len = 1
        y_len = sum(y_pred)
        rect = f_all_2.add_patch(patches.Rectangle((x_start,y_start),x_len,y_len, linewidth=2, edgecolor=color, facecolor='none'))
        cluster_ID[key].append(cluster_in_T)
        
# =============================================================================
#     H
# =============================================================================
    # survival time
    f_0_3 = atlas.add_subplot(gs[0, 3])
    sortind = np.argsort(t)
    e_sorted = e[sortind]
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='black')
    f_0_3.plot(np.arange(len(t)), t[sortind], color = 'red')
    f_0_3.set_xlim(0, len(t))
    f_0_3.get_xaxis().set_visible(False)
    f_0_3.set_title(r'(B) Survival time and $\hat{\bm{H}}$', loc = 'left')
    
    vmax = np.max(H_hat)
    H_hat_show = H_hat[colorder,:][:,sortind]
    f_1_3 = atlas.add_subplot(gs[1, 3])
    im = f_1_3.imshow(H_hat_show, cmap = cmap, vmin=0, vmax=vmax, interpolation='none')
    f_1_3.get_xaxis().set_visible(False)
    f_1_3.get_yaxis().set_visible(False)
    f_1_3.set_ylabel('CoxNMF')
    f_1_3.set_yticklabels([])
    f_1_3.axis('auto')
    # f_1_3.set_title('(B)', loc = 'left')
    atlas.colorbar(im, ax=f_1_3,aspect=40, orientation="horizontal")
    
# =============================================================================
#     Bottom right panel: Correlation plot
# =============================================================================

    f_3_3 = atlas.add_subplot(gs[2, 3])
    f_3_3.set_title('(C) Correlation plot', loc = 'left')
    X_corr = spearmanr(X.values.T)[0]
    X_corr_ordered = X_corr[roworder,:][:,roworder]
    cmap = plt.cm.bwr
    cmap.set_bad(color='white')
    X_corr_im = f_3_3.imshow(X_corr_ordered, vmin = -1, vmax = 1,
                             aspect='equal',
                             cmap=cmap, interpolation='none')
    for c in range(args.K_hat):
        where = np.where(T_sorted == c)[0]
        x_start=y_start=where[0]
        x_len=y_len=where[-1]-where[0]
        lw=1
        color='grey'
        rect = f_3_3.add_patch(patches.Rectangle((x_start,y_start),x_len,y_len,
                                                linewidth=lw, edgecolor=color,
                                                facecolor='none'))

        txt = f_3_3.text(x_start+x_len/2, y_start+x_len/2,
                       'C%d' % (c+1),
                       horizontalalignment='center',
                       verticalalignment='center',
                       color = color,
                       bbox=dict(facecolor='white', alpha=0.6, edgecolor=color, boxstyle='round,pad=0.2'))
        

    atlas.colorbar(X_corr_im, ax=f_3_3,aspect=40, orientation="horizontal")
    return atlas, cluster_ID, gene_clusters

def feature_analysis(fig1, axes1, fig2, axes2, W, W_hat, cph_hat_params, model, labels_true, simulation,\
                     affinity='euclidean', linkage='average', normalize = True, weighted = False):
    
    basis_norm = W_hat[model]
    cph_params = cph_hat_params[model].values
    if normalize:
        basis_norm = (basis_norm.T / np.linalg.norm(x = basis_norm, axis = 1)).T
    basis_norm_weighted = basis_norm * cph_params

    
    cmap = plt.cm.bwr
    cmap.set_bad(color='white')
    vmin = min(np.min(W), np.min(basis_norm), np.min(basis_norm_weighted))
    vmax = max(np.max(W), np.max(basis_norm), np.max(basis_norm_weighted))
    maxval = max(np.abs(vmin), np.abs(vmax))
    im = axes1[0].imshow(W, cmap = cmap, vmin = -maxval, vmax = maxval)
    axes1[0].axis('auto')
    axes1[0].set_title('W (ground truth)')
    axes1[0].get_yaxis().set_visible(False)
    axes1[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes1[0].set_xticklabels(np.arange(W_hat[model].shape[1]+1))
    im = axes1[1].imshow(basis_norm, cmap = cmap, vmin = -maxval, vmax = maxval)
    axes1[1].axis('auto')
    axes1[1].set_title('W (%s)' % model)
    axes1[1].get_yaxis().set_visible(False)
    axes1[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes1[1].set_xticklabels(np.arange(W_hat[model].shape[1]+1))
    im = axes1[2].imshow(basis_norm_weighted, cmap = cmap, vmin = -maxval, vmax = maxval)
    axes1[2].axis('auto')
    axes1[2].set_title('W (%s) * Cox Weights' % model)
    axes1[2].get_yaxis().set_visible(False)
    axes1[2].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes1[2].set_xticklabels(np.arange(W_hat[model].shape[1]+1))
    fig1.subplots_adjust(right=0.8, wspace=0.2, hspace=0.2)
    cbar_ax = fig1.add_axes([0.85, 0.2, 0.01, 0.6])
    fig1.colorbar(im, cax=cbar_ax)
    
    # Hierarchical clustering    
    n_samples = len(basis_norm)
    n_clusters = basis_norm.shape[1]
    if weighted:
        d = sch.distance.pdist(basis_norm_weighted)
    else:
        d = sch.distance.pdist(basis_norm)
    Z = sch.linkage(d, method=linkage)
    T = sch.fcluster(Z, n_clusters, 'maxclust')
    
    # calculate labels
    labels=list('' for i in range(n_samples))
    for i in range(n_samples):
        labels[i]=str(i)+ ',' + str(T[i])
    
    # calculate color threshold
    ct = Z[-(n_clusters-1),2]
    
    #plot
    R = sch.dendrogram(Z, ax = axes2[2], labels=labels, orientation='left', \
                       count_sort = 'ascending', color_threshold=ct, \
                       above_threshold_color='b', \
                       no_labels = True)
    
    axes2[2].set_title('Dendrogram\nn_cluster = %d' % n_clusters)
    axes2[2].axis('off')
    
    

    maxval = np.max(np.abs(basis_norm_weighted))
    roworder = np.array(R['leaves'])[::-1]
    matrix2show = basis_norm_weighted[roworder,:]
    im = axes2[3].imshow(matrix2show, cmap = cmap, vmin = -maxval, vmax = maxval) # norm = matplotlib.colors.LogNorm()
    axes2[3].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes2[3].set_xticklabels(np.arange(matrix2show.shape[1]+1))
    axes2[3].axis('auto')
    axes2[3].get_yaxis().set_visible(False)
    
    axes2[3].text(-1.5, W_hat[model].shape[0]*1.05, 'Cox Weight:',\
             horizontalalignment='right',\
             verticalalignment='center', weight = 'bold')
    
    for i in range(W_hat[model].shape[1]):
        if cph_params[i] == min(cph_params):
            color = 'b'
            weight = 'bold'
        elif cph_params[i] == max(cph_params):
            color = 'r'
            weight = 'bold'
        else:
            color = 'k'
            weight = 'normal'
        axes2[3].text(i, W_hat[model].shape[0]*1.05, '%.2f' % cph_params[i],\
                 horizontalalignment='center',\
                 verticalalignment='top', rotation=45, \
                 color = color, weight = weight)
    
    # highlight strongest positive and negative group by mean values
    T_sorted = T[roworder]
    globalminval, globalmaxval = np.inf, -np.inf
    
    block_meanval = pd.DataFrame(columns = ['cluster', 'mean value of smallest beta column', 'mean value of highest beta column'])
    
    for cluster in np.unique(T_sorted):
        meanval_min = np.mean(matrix2show[T_sorted == cluster, np.argmin(cph_params)]) # mean value of block in smallest beta column
        meanval_max = np.mean(matrix2show[T_sorted == cluster, np.argmax(cph_params)]) # mean value of block in smallest beta column
        block_meanval = block_meanval.append({'cluster': cluster,
                                              'mean value of smallest beta column': meanval_min,
                                              'mean value of highest beta column': meanval_max},
                                              ignore_index = True)
        
    cluster_position_min1 = block_meanval.sort_values('mean value of smallest beta column').iloc[0,0]
    cluster_position_min2 = block_meanval.sort_values('mean value of smallest beta column').iloc[1,0]
    cluster_position_max1 = block_meanval.sort_values('mean value of highest beta column').iloc[-1,0]
    cluster_position_max2 = block_meanval.sort_values('mean value of highest beta column').iloc[-2,0]

    x_start = np.argmin(cph_params)-0.5
    y_start = np.where(T_sorted == cluster_position_min1)[0][0]
    x_len = 1
    y_len = sum(T_sorted == cluster_position_min1)
    rect_min = axes2[3].add_patch(patches.Rectangle((x_start,y_start),x_len,y_len, linewidth=2, edgecolor='b', facecolor='none'))
        
    x_start = np.argmax(cph_params)-0.5
    y_start = np.where(T_sorted == cluster_position_max1)[0][0]
    x_len = 1
    y_len = sum(T_sorted == cluster_position_max1)
    rect_max = axes2[3].add_patch(patches.Rectangle((x_start,y_start),x_len,y_len, linewidth=2, edgecolor='r', facecolor='none'))

    axes2[3].set_title('Sorted W\nSmallest cluster (blue) \nand highest cluster (red)\n by mean are highlighted.')
    
    axes2[4].imshow(labels_true[roworder].reshape(-1,1), aspect='auto', cmap = 'rainbow')
    axes2[4].get_yaxis().set_visible(False)
    axes2[4].get_xaxis().set_visible(False)
    axes2[4].set_title('True\nlabel\nPosition')
    axes2[1].axis('off')
    
    clusters = axes2[0].imshow(labels_true.reshape(-1,1), aspect='auto', cmap = 'rainbow')
    axes2[0].get_yaxis().set_visible(False)
    axes2[0].get_xaxis().set_visible(False)
    alphabetics = ['C' + str(i+1) for i in range(len(np.unique(labels_true)))]
    for x in range(len(np.unique(labels_true))):
        axes2[0].axhline(y = np.max(np.where(labels_true == x)[0]), linewidth=0.5, color='w')
        axes2[0].text(0, np.mean(np.where(labels_true == x)[0]), '%s' % alphabetics[int(np.unique(labels_true)[x])],\
                 horizontalalignment='center',\
                 verticalalignment='center')
    axes2[0].set_title('Label\nReference')
    
    # quantitative measure
    
    unique_labels = np.sort(np.unique(labels_true))
    labels_true_ordered = labels_true[roworder]
    
    performance = pd.DataFrame(index = ['Negative', 'Positive', 'Total'],
                               columns = ['Model', 'tp', 'fp', 'tn', 'fn', 'sensitivity', 'specificity', \
                                   'ppv', 'npv', 'hitrate', 'f1', 'precision', 'recall', 'acc'])
    performance['Model'] = model
    
    
    if 'feature_dependent' in simulation:
        negative_cluster_id = [cluster_position_min1]
        y_negative_pred = np.array([1 if ts in negative_cluster_id else 0 for ts in T_sorted]).astype(int)
        
        positive_cluster_id = [cluster_position_max1]
        y_positive_pred = np.array([1 if ts in positive_cluster_id else 0 for ts in T_sorted]).astype(int)
        
        y_negative_true = (np.array([l in unique_labels[:2] for l in labels_true_ordered])).astype(int)
        y_positive_true = (np.array([l in unique_labels[-2:] for l in labels_true_ordered])).astype(int)
        
        
    else:
        negative_cluster_id = [cluster_position_min1]
        y_negative_pred = np.array([1 if ts in negative_cluster_id else 0 for ts in T_sorted]).astype(int)
        
        positive_cluster_id = [cluster_position_max1]
        y_positive_pred = np.array([1 if ts in positive_cluster_id else 0 for ts in T_sorted]).astype(int)
        
        
        y_negative_true = (labels_true_ordered == unique_labels[0]).astype(int)
        y_positive_true = (labels_true_ordered == unique_labels[-1]).astype(int)
        
    
    y_pred = np.concatenate((y_negative_pred, y_positive_pred))
    y_true = np.concatenate((y_negative_true, y_positive_true))
    
    tp, fp, tn, fn, sensitivity, specificity, ppv, npv, hitrate, f1, precision, recall, acc, iou, dice = perf_measure(y_true, y_pred)
    performance.loc['Total', ['tp', 'fp', 'tn', 'fn', 'sensitivity', 'specificity', \
                                   'ppv', 'npv', 'hitrate', 'f1', 'precision', 'recall', 'acc', 'iou', 'dice']] \
                    = tp, fp, tn, fn, sensitivity, specificity, ppv, npv, hitrate, f1, precision, recall, acc, iou, dice
    
    
    fig2.subplots_adjust(right=0.8, wspace=0, hspace=0.2)
    cbar_ax = fig2.add_axes([0.85, 0.2, 0.01, 0.6])
    fig2.colorbar(im, cax=cbar_ax)
    fig2.suptitle('Hierarchical Agglomerative Clustering (Affinity=%s; Linkage=%s)' % (affinity, linkage))
    
    
    return fig1, fig2, performance
