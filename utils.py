#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from scipy.stats import spearmanr
from scipy.linalg import block_diag
from sklearn.metrics.cluster import contingency_matrix
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.patches as patches
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import silhouette_score

import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns


def perf_measure(y_actual, y_hat, average = 'macro'):
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           tp += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           fp += 1
        if y_actual[i]==y_hat[i]==0:
           tn += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           fn += 1
    if (tp+fn) == 0: sensitivity = np.nan
    else: sensitivity = tp/(tp+fn) # recall
    if (tn+fp) == 0: specificity = np.nan
    else: specificity = tn/(tn+fp)
    if (tp+fp) == 0: ppv = np.nan
    else: ppv = tp/(tp+fp) # precision or positive predictive value (PPV)
    if (tn+fn) == 0: npv = np.nan
    else: npv = tn/(tn+fn) # negative predictive value (NPV)
    if (tp+tn+fp+fn) == 0: hitrate = np.nan
    else: hitrate = (tp+tn)/(tp+tn+fp+fn) # accuracy (ACC)
    f1 = f1_score(y_actual, y_hat, average = average)
    precision = precision_score(y_actual, y_hat, average = average)
    recall = recall_score(y_actual, y_hat, average = average)
    acc = accuracy_score(y_actual, y_hat)
    
    intersect = np.sum(np.array(y_actual) & np.array(y_hat))
    union = np.sum(np.array(y_actual) | np.array(y_hat))
    if union > 0:
        iou = intersect/union
    else:
        print('Warning: [IoU calculation] union = 0.')
        iou = np.nan
    if (2*tp + fp + fn) > 0:
        dice = 2*tp / (2*tp + fp + fn)
    else:
        dice = np.nan
    return tp, fp, tn, fn, sensitivity, specificity, ppv, npv, hitrate, f1, precision, recall, acc, iou, dice


def spearman_corr(X, W, labels):
    W_corr, p_values = spearmanr(W.T)
    X_corr, p_values = spearmanr(X.T)
    X_corr = X_corr[np.argsort(labels)]
    X_corr = X_corr[:, np.argsort(labels)]
    
    cmap = plt.cm.bwr
    cmap.set_bad(color='white')
    
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    mat0 = axes[0].matshow(X_corr, cmap=cmap, vmin=-1, vmax=1)
    axes[0].set_title("X")
    #    axes[0].axis('off')
    axes[0].axis('auto')
    axes[0].set_aspect('equal')
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    #        plt.show()
    
    mat1 = axes[1].matshow(W_corr, cmap=cmap, vmin=-1, vmax=1)
    axes[1].set_title("W")
    #    axes[1].axis('off')
    axes[1].axis('auto')
    axes[1].set_aspect('equal')
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6]) # rect = [x0, y0, width, height]
    fig.colorbar(mat1, cax=cbar_ax)
    return fig


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cmat = contingency_matrix(y_true, y_pred)
    # return purity
    amax = np.amax(cmat, axis=0)
#    amax = amax[[0,-1]] # fist and last 
    return np.sum(amax) / np.sum(cmat)


def purity_score_truncated(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cmat = contingency_matrix(y_true, y_pred)
    # return purity
    amax = np.amax(cmat, axis=0)
    amax = amax[[0,-1]] # fist and last 
    return np.sum(amax) / np.sum(cmat[[0,-1]])


def hcluster_silhouette(args,
                        W_hat,
                        beta,
                        ax = None,
                        affinity='euclidean',
                        normalize = True,
                        weighted = False):
    
    if normalize: # row normalization on W
        W_hat = (W_hat.T / np.linalg.norm(W_hat, axis = 1)).T
    
    # W_Tilde
    W_Tilde = W_hat * beta
    
    # Hierarchical clustering    
    n_samples = len(W_hat)
    n_clusters = args.K_hat
    if weighted:
        d = sch.distance.pdist(W_Tilde)
    else:
        d = sch.distance.pdist(W_hat)
    Z = sch.linkage(d, method=args.linkage)
    T = sch.fcluster(Z, n_clusters, 'maxclust')
    
    # calculate labels
    labels=list('' for i in range(n_samples))
    for i in range(n_samples):
        labels[i]=str(i)+ ',' + str(T[i])
    
    # calculate color threshold
    ct = Z[-(n_clusters-1),2]
    #plot
    R = sch.dendrogram(Z, ax = None, labels=labels, orientation='left', \
                       count_sort = 'ascending', color_threshold=ct, \
                       above_threshold_color='b', \
                       no_labels = True)
    
    sil_score = silhouette_score(W_hat, T)
    return R, T, sil_score

    
def clustering(args, W_hat, beta, method='hclust'):
    if method == 'hclust':
        R, T, sil_score = hcluster_silhouette(args,
                                            W_hat,
                                            beta,
                                            ax = None,
                                            affinity='euclidean',
                                            normalize = True,
                                            weighted = False)
        roworder = np.array(R['leaves'])[::-1]
        T_sorted = T[roworder] - 1 # start from 0
        
    return T_sorted, sil_score, roworder



def feature_analysis_no_fig(args,
                            W_hat,
                            H_hat,
                            beta,
                            labels_true=None, # None if dataset == 'TCGA'
                            affinity='euclidean',
                            normalize = True,
                            weighted = False):
    
    
    if normalize: # row normalization on W
        W_hat = (W_hat.T / np.linalg.norm(W_hat, axis = 1)).T
    # W_Tilde
    W_Tilde = W_hat * beta
    
    
    
    method = 'hclust'
    T_sorted, sil_score, roworder = clustering(args, W_hat, beta, method)
    
    
    colorder = np.argsort(beta)
    W_Tilde_ordered = W_Tilde[roworder,:][:,colorder]
    cluster_list = np.arange(args.K_hat) # [0, 1, 2, ..., K_hat-1]
    if labels_true is not None:
        labels_true_ordered = labels_true[roworder]
        
    # highlight strongest positive and negative group by mean values.
    
    """
    Note: All "negative" terms below stands for the negative value in W_Tilde,
    not the negative association to survival! Actually, the "negative" value 
    in W_Tilde suggests a positive association to survival. Don't get confused!
    """
    tau1, tau2 = 1, 1
    if args.dataset == 'simulation':
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
        
    y_negative_pred, y_positive_pred = [], []
    if labels_true is not None:
        y_negative_true, y_positive_true = [], []
    
    for tau in range(tau1):
        cluster_neg_in_T = np.argmin([np.mean(W_Tilde_ordered[T_sorted == c, tau]) for c in cluster_list])
        y_negative_pred += [1 if ts == cluster_neg_in_T else 0 for ts in T_sorted]
        if labels_true is not None:
            y_negative_true += [int(l in cluster_list[0:tau+1]) for l in labels_true_ordered]

    for tau in range(1,tau2+1):
        cluster_pos_in_T = np.argmax([np.mean(W_Tilde_ordered[T_sorted == c, -tau]) for c in cluster_list])
        y_positive_pred += [1 if ts == cluster_pos_in_T else 0 for ts in T_sorted]
        if labels_true is not None:
            y_positive_true += [int(l in cluster_list[-tau:]) for l in labels_true_ordered]

    
    # Get performance
    if labels_true is not None:
        y_pred = np.concatenate((y_positive_pred, y_negative_pred))
        y_true = np.concatenate((y_positive_true, y_negative_true))
        performance = pd.DataFrame(index = ['Negative', 'Positive', 'Total'],
                                   columns = ['Model', 'tp', 'fp', 'tn', 'fn', 'sensitivity', 'specificity', \
                                       'ppv', 'npv', 'hitrate', 'f1', 'precision', 'recall', 'acc', 'iou', 'dice'])
        tp, fp, tn, fn, sensitivity, specificity, ppv, npv, hitrate, f1, precision, recall, acc, iou, dice = perf_measure(y_true, y_pred)
        performance.loc['Total', ['tp', 'fp', 'tn', 'fn', 'sensitivity', 'specificity', \
                                       'ppv', 'npv', 'hitrate', 'f1', 'precision', 'recall', 'acc', 'iou', 'dice']] \
                        = tp, fp, tn, fn, sensitivity, specificity, ppv, npv, hitrate, f1, precision, recall, acc, iou, dice
        
    else:
        acc, precision, recall, f1, iou, dice = None, None, None, None, None, None
    
    return acc, precision, recall, f1, iou, dice, sil_score
