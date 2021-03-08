#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 13:43:35 2021

@author: zhihuan
"""


import sys, os, platform, copy
import logging, re
import pandas as pd
import numpy as np
import itertools
import time, random
from tqdm import tqdm
tqdm.pandas()

from conf import getworkdir, conf
from models import run_model, CoxNMF_initialization
from utils import feature_analysis_no_fig
from dataset import getdata
from plotting import plot_simulation_atlas, plot_TCGA_atlas
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from main import prepare_results
from models import run_model_and_save_logs

from gen_performance import find_TCGA_args, get_final_perfomance, model_selection


if __name__ == '__main__':
    workdir = getworkdir()
    args, logger = conf() 
    args.dataset = 'TCGA'

    args.result_folder = os.path.join(workdir, 'Results','20210131_hclust_Cancer')
    
    
    for cancer in ['BLCA','BRCA','COAD','KIRC','KIRP','LIHC','LUAD','LUSC','OV','PAAD']:
        args.cancer = cancer
        args.seed_list = [1]
        start = 10
        stop = 31
        args.random_seed = 1
        
        args.K_hat_list = np.arange(start=start, stop=stop)
        X, W, H, t, e, labels = getdata(args)
        args.P, args.N = X.shape[0], X.shape[1]
        _, df_seed, _, _ = get_final_perfomance(args, metric='CIndex')
        
        
        args.alpha = 5
        args.penalizer = 0.01
        args.W_H_initialization = 'random'
        df_final = df_seed.loc[(df_seed['Model'] == 'CoxNMF') & \
                               (df_seed['alpha'] == args.alpha) & \
                               (df_seed['penalizer'] == args.penalizer) & \
                               (df_seed['W_H_initialization'] == args.W_H_initialization),:]
        
        silouette_score = [m for (m,v) in df_final['silhouette_score']]
        relative_error = [m for (m,v) in df_final['relative_error']]
        
        args.K_hat = int(args.K_hat_list[np.argmax(silouette_score)])
        
        W_init, H_init = CoxNMF_initialization(args, X.values)
        res = run_model(X.values,t,e,args,W_init,H_init,model='CoxNMF',logger=logger,verbose=1)
        acc, p, r, f1, iou, dice, silhouette_score = feature_analysis_no_fig(args, res['W_hat'], res['H_hat'],\
                                                                            res['beta'], labels,\
                                                                            affinity='euclidean',\
                                                                            normalize = True,\
                                                                            weighted = False)
        
        
        # =============================================================================
        #         Plot CoxNMF atlas
        # =============================================================================
        args.output_dir = os.path.join(args.result_folder, 'Cancer=%s' % args.cancer,
                                       'meanq=%.2f_varq=%.2f_P=%d_N=%d_citol=%.2f_maxiter=%d' % \
                                       (args.mean_quantile, args.var_quantile, args.P, args.N, args.ci_tol, args.max_iter),
                                       'CoxNMF_results_alpha=%.2f_penalizer=%.2f_init=%s' % \
                                       (args.alpha, args.penalizer, args.W_H_initialization))
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        
        atlas = plt.figure(figsize=(10,8), constrained_layout=True)
        gs = atlas.add_gridspec(nrows=3, ncols=4, width_ratios=[0.5,0.1,1.3,2], height_ratios=[3,15,40])
        atlas, cluster_ID, gene_clusters = plot_TCGA_atlas(atlas,
                                                            gs,
                                                            args,
                                                            X,
                                                            W, res['W_hat'],
                                                            H, res['H_hat'],
                                                            t, e,
                                                            res['beta'], labels,\
                                                            affinity='euclidean',
                                                            normalize = True, weighted = False)
        atlas.savefig(os.path.join(args.output_dir, 'atlas.png'), dpi = 600)
        
        
        stat = pd.DataFrame.from_dict({'error': res['error'],
                                       'relative_error': res['relative_error'],
                                       'cindex': res['cindex'],
                                       'running_time': res['running_time'],
                                       'silhouette_score': silhouette_score,
                                       'W_H_initialization': args.W_H_initialization,
                                       'random_seed': args.random_seed,
                                       'alpha': args.alpha,
                                       'ci_tol': args.ci_tol,
                                       'K_hat': args.K_hat,
                                       'beta_normalization': args.beta_normalization,
                                       'H_normalization': args.H_normalization,
                                       'penalizer': args.penalizer}, orient='index')
        
        stat.to_csv(os.path.join(args.output_dir, 'results.csv'))
        # =============================================================================
        #         Save What, Hhat, and gene modules
        # =============================================================================
        
        
        cluster_ID_df = pd.DataFrame(index = cluster_ID.keys(), columns = ['Cluster'])
        for k in cluster_ID:
            cluster_ID_df.loc[k, 'Cluster'] = ['C%02d' % (v+1) for v in cluster_ID[k]]
        cluster_ID_df.to_csv(os.path.join(args.output_dir, 'cluster_ID.csv'))
        
        
        W_hat = pd.DataFrame(res['W_hat'], index=X.index)
        H_hat = pd.DataFrame(res['H_hat'], columns=X.columns)
        beta_hat = pd.DataFrame(res['beta'])
        W_hat.to_csv(os.path.join(args.output_dir, 'W_hat.csv'))
        H_hat.to_csv(os.path.join(args.output_dir, 'H_hat.csv'))
        beta_hat.to_csv(os.path.join(args.output_dir, 'beta_hat.csv'))
        
        
        gene_clusters_list = []
        key_sorted = []
        for i in range(len(gene_clusters)):
            key='C%02d'%(i+1)
            key_sorted.append(key)
            gene_clusters_list.append(gene_clusters[key])
        gene_clusters_df = pd.DataFrame(gene_clusters_list, index = key_sorted)
        gene_clusters_df.to_csv(os.path.join(args.output_dir, 'gene_clusters.csv'))
        
        for ax in atlas.get_axes(): ax.clear()
        
            
        
        
