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

import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns


from _TCGA import TCGA
from _filter import expression_filter

def getdata(args):
    if args.dataset == 'simulation':
        X, W, H, t, e, labels = \
            generate_synthetic_data(n_samples=args.N,
                                    death_rate=args.death_rate,
                                    K=args.K,
                                    simulation_type=args.simulation_type,
                                    W_group_size=args.W_group_size,
                                    W_noise=args.W_noise,
                                    H_noise=args.H_noise,
                                    X_noise=args.X_noise,
                                    W_distribution=args.W_distribution,
                                    time_distribution=args.time_distribution,
                                    random_seed=args.random_seed)
    else:
        data = TCGA(args.cancer)
        
        pid1 = [p[:12] for p in data.mRNAseq.columns]
        pid2 = data.clinical.index.values
        xy, x_ind, y_ind = np.intersect1d(pid1, pid2, return_indices=True)
        
        X = data.mRNAseq.iloc[:,x_ind]
        t = data.overall_survival_time[y_ind]
        e = data.overall_survival_event[y_ind]
        
        #remove patients with NaN value in t or e
        valididx = np.isfinite(t) & np.isfinite(e)
        if np.sum(valididx) < len(t):
            print('%d patients with NaN values in T or E were removed.' % (len(t)-np.sum(valididx)))
            X = X.iloc[:, valididx]
            t = t[valididx]
            e = e[valididx]
        
        W = None
        H = None
        labels = None
        
        X = expression_filter(X, args.mean_quantile, args.var_quantile)
        X = np.log2(X+1) # gene expression log-scale pseudocount conversion
        X = X.fillna(0)
    return X, W, H, t, e, labels


def generate_synthetic_data(n_samples,
                            death_rate,
                            K,
                            simulation_type,
                            W_group_size,
                            W_noise,
                            H_noise,
                            X_noise,
                            W_distribution,
                            time_distribution,
                            random_seed):
    
    np.random.seed(random_seed)
    
    H = np.zeros((K, n_samples))
    
    for i in range(K):
        H[i,:] = np.random.uniform(0,1, n_samples)
    
    if simulation_type == 'univariate_better':
        H[0,:] = np.arange(n_samples)/n_samples/5+0.5
    if simulation_type == 'univariate_worse':
        H[-1,:] = np.arange(n_samples)[::-1]/n_samples/5+0.5
    if simulation_type == 'univariate_better_worse':
        H[0,:] = np.arange(n_samples)/n_samples/5+0.5
        H[-1,:] = np.arange(n_samples)[::-1]/n_samples/5+0.5
    elif simulation_type == 'multivariate_better':
        H[1,:] = np.random.uniform(0,1, n_samples)
        H[0,:] = np.arange(n_samples)/n_samples/5+0.5 - H[1,:]/10
    elif simulation_type == 'multivariate_worse':
        H[-2,:] = np.random.uniform(0,1, n_samples)
        H[-1,:] = np.arange(n_samples)[::-1]/n_samples/5+0.5 - H[-2,:]/10
    elif simulation_type == 'multivariate_better_worse':
        H[1,:] = np.random.uniform(0,1, n_samples)
        H[0,:] = np.arange(n_samples)/n_samples/5+0.5 - H[1,:]/10
        
        H[-2,:] = np.random.uniform(0,1, n_samples)
        H[-1,:] = np.arange(n_samples)[::-1]/n_samples/5+0.5 - H[-2,:]/10
    
    b_diag = []
    for group in range(K):
        if W_distribution.lower() == 'exponential':
            block = np.random.exponential(scale = 1, size = W_group_size).reshape(-1,1)
        b_diag += [block]
    W = block_diag(*b_diag)
    
    if W_noise > 0:
        W += np.random.exponential(scale = W_noise, size = K*W_group_size * K).reshape(-1,K)
    
    if H_noise > 0:
        H += np.random.exponential(scale = H_noise, size = K*n_samples).reshape(K,-1)
        
    X = W.dot(H)
    
    if X_noise > 0:
        X += np.random.exponential(scale = X_noise, size = X.shape[0]*X.shape[1]).reshape(X.shape)

    labels = np.repeat(np.arange(K), W_group_size)
    
    if time_distribution == 'exponential':
        t = np.sort(np.random.exponential(scale = 100, size = n_samples))
    else:
        t = np.arange(n_samples) + 1
    e = (np.random.uniform(0, 1, n_samples) > (1-death_rate)).astype(int)
    return X, W, H, t, e, labels


