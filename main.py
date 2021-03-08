#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import time
import datetime
import copy
import warnings
from conf import conf
from tqdm import tqdm
from utils import *
from dataset import getdata, generate_synthetic_data
from models import run_model, run_model_and_save_logs, CoxNMF_initialization
from utils import feature_analysis_no_fig


def prepare_results(X, args):
    columns_basic_info = ['Model','random_seed','K','P','N','K_hat','alpha',
                          'penalizer','l1_ratio','hclust_linkage','max_iter']
    columns_simulation_info = ['death_rate','W_distribution','X_noise','W_noise','H_noise','simulation_type']
    columns_cancer_info = ['mean_quantile','var_quantile','cancer']
    columns_CoxNMF = ['W_H_initialization','W_normalization','H_normalization','beta_normalization']
    columns_performance_basic = ['Fnorm','relative_error','CIndex','Runtime','silhouette_score']
    columns_performance_simulation = ['Accuracy','precision','recall','F1_score','IoU','Dice coefficient']
    
    if args.dataset == 'simulation':
        args.results_dir = os.path.join(args.results_dir,
                                        'Simulation=%s' % args.simulation_type,
                                        'deathrate=%.2f_Wdistribution=%s_citol=%.2f_maxiter=%d' %
                                        (args.death_rate, args.W_distribution, args.ci_tol, args.max_iter), # not likely to change
                                        'K=%d_P=%d_N=%d' % (args.K, X.shape[0], X.shape[1]),
                                        'Noise_X=%.2f_W=%.2f_H=%.2f' % (args.X_noise, args.W_noise, args.H_noise),
                                        )
        dfcolumns = columns_basic_info +\
                    columns_simulation_info +\
                    columns_CoxNMF +\
                    columns_performance_basic +\
                    columns_performance_simulation
    else: # real cancer data
        args.results_dir = os.path.join(args.results_dir,
                                        'Cancer=%s' % args.cancer,
                                        'meanq=%.2f_varq=%.2f_P=%d_N=%d_citol=%.2f_maxiter=%d' %
                                        (args.mean_quantile, args.var_quantile, X.shape[0], X.shape[1], args.ci_tol, args.max_iter) # not likely to change
                                        )
        dfcolumns = columns_basic_info +\
                    columns_cancer_info +\
                    columns_performance_basic
    try:
    	if not os.path.exists(args.results_dir): os.makedirs(args.results_dir)
    except:
        print('path just existed.')
    result_df = pd.DataFrame(columns = dfcolumns)
    
    return args, result_df


def run_simulation(args,
                   X,
                   W,
                   H,
                   t,
                   e,
                   labels,
                   logger=None):
    
    args, result_df = prepare_results(X, args)
    print(args)
    starttime = time.time()
    
    penalizer_list = [0, 0.01, 0.1, 1]
    K_hat_list = args.K + np.array([-2,-1,0,1,2])
    for K_hat in K_hat_list:
        args.K_hat = K_hat
# =============================================================================
#         # General dimensionality reduction method and vanilla NMF
        alpha_list = [0.1,1,10,100,1000] # for SparsePCA only
        model_list = ['TruncatedSVD','PCA','SparsePCA','FactorAnalysis','NMF (CD)','NMF (MU)', 'NNDSVD']
        for model in model_list:
            for p in penalizer_list:
                print('Model: %s, penalizer: %.4f' % (model, p))
                args.penalizer = p
                if model == 'SparsePCA':
                    for a in alpha_list:
                        args.alpha = a
                        result_df = run_model_and_save_logs(X,None,None,t,e,labels,args,model,result_df,logger)
                else:
                    result_df = run_model_and_save_logs(X,None,None,t,e,labels,args,model,result_df,logger)
# =============================================================================
        
#         # CoxNMF
        W_init, H_init = {}, {}
        for init in ['CD','MU','nndsvd','random']:
            args.W_H_initialization = init
            W_init[init], H_init[init] = CoxNMF_initialization(args, X)
            
        alpha_list = [0.5,1,2,5,10,20]
        for a in alpha_list:
            args.alpha = a
            for p in penalizer_list:
                args.penalizer = p
                for init in ['CD','MU','nndsvd','random']:
                    args.W_H_initialization = init
                    result_df = run_model_and_save_logs(X,W_init[init],H_init[init],t,e,labels,
                                                        args,'CoxNMF',result_df,logger)
# =============================================================================
#         # SNMF
# =============================================================================
        SNMF_hparam_list = [0.001,0.01,0.1]
        for a in SNMF_hparam_list:
            args.alpha = a
            for p in penalizer_list:
                args.penalizer = p
                result_df = run_model_and_save_logs(X,None,None,t,e,labels,args,'SNMF',result_df,logger)

    total_runtime = str(datetime.timedelta(seconds=time.time() - starttime))
    print('Total running time: %s' % total_runtime)

    return args, result_df


def run_TCGA(args,
             X,
             t,
             e,
             logger=None):
    
    args, result_df = prepare_results(X, args)
    print(args)
    starttime = time.time()
    penalizer_list = [0, 0.01, 0.1, 1]
    # CoxNMF
    
    W_init, H_init = {}, {}
    for init in ['CD','MU','nndsvd','random']:
        args.W_H_initialization = init
        W_init[init], H_init[init] = CoxNMF_initialization(args, X)

    alpha_list = [0.5,1,2,5,10,20]
    for a in alpha_list:
        args.alpha = a
        for p in penalizer_list:
            args.penalizer = p
            for init in ['CD','MU','nndsvd','random']:
                args.W_H_initialization = init
                result_df = run_model_and_save_logs(X,W_init[init],H_init[init],t,e,None,
                                                    args,'CoxNMF',result_df,logger)

    total_runtime = str(datetime.timedelta(seconds=time.time() - starttime))
    print('Total running time: %s' % total_runtime)

    return args, result_df

if __name__ == '__main__':
    args, logger = conf()
    X, W, H, t, e, labels = getdata(args)
    warnings.filterwarnings("ignore")
    
    """
    Hyper-parameters should get from the bash_generator.
    Hyper-parameters for simulation:
        random_seed (e.g., 1,2,3,4,5)
        simulation_type (e.g., univariate_pos_neg, multivariate_pos_neg)
        death_rate (e.g., 1)
        W_distribution (e.g., exponential)
        max_iter (e.g., 500)
        K (e.g., 7,8,9)
        X_noise (e.g., 0,0.05,0.1)
        W_noise (e.g., 0,0.05,0.1)
        H_noise (e.g., 0,0.05,0.1)
    Hyper-parameters for cancer:
        cancer (e.g., BRCA, KIRC)
        K_hat (e.g., 10,20,30)
        mean_quantile (e.g., 0,0.2)
        var_quantile (e.g., 0,0.2)
        max_iter (e.g., 500)
    """


    if args.dataset == 'simulation':
        args, result_df = run_simulation(args, X, W, H, t, e, labels, logger)
        result_df.to_csv(os.path.join(args.results_dir,
                                      'result_df_seed=%d.csv' % args.random_seed))


    if args.dataset == 'TCGA':
        args, result_df = run_TCGA(args, X.values, t, e, logger)
        result_df.to_csv(os.path.join(args.results_dir,
                                      'result_df_Khat=%d_seed=%d.csv' % (args.K_hat, args.random_seed)) )
        
        
    
    
    
