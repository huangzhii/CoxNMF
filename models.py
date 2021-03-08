#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import copy
import warnings
from lifelines import CoxPHFitter
from math import sqrt
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.decomposition import NMF
from sklearn.decomposition import randomized_svd, TruncatedSVD
from sklearn.decomposition import PCA, SparsePCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from SNMF import NNDSVD, Supernmf_pg
from CoxNMF._nmf import CoxNMF, _initialize_nmf
from utils import feature_analysis_no_fig


def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))

def NNDSVD_NMF(X,
               n_components,
               eps=1e-6,
               max_iter=100,
               random_state=None):
    """
    Computes the non-negative rank k matrix approximation for X: X = WH
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    n_samples, n_features = X.shape
    
    # Random initialization
    avg = np.sqrt(X.mean() / n_components)
    H = avg * np.random.randn(n_components, n_features).astype(X.dtype,
                                                         copy=False)
    W = avg * np.random.randn(n_samples, n_components).astype(X.dtype,
                                                        copy=False)
    np.abs(H, out=H)
    np.abs(W, out=W)
    # NNDSVD
    U, S, V = randomized_svd(X,
                             n_components,
                             n_iter=max_iter,
                             random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0
    return W, H


def CoxNMF_initialization(args,
                          X):

    print('Performing initialization for W0 and H0 with %s ...' % args.W_H_initialization)
    if args.W_H_initialization in ['CD', 'MU']:
        model = NMF(n_components=args.K_hat,
                                init='random',
                                solver=args.W_H_initialization.lower(),
                                tol=1e-6,
                                max_iter=args.max_iter,
                                random_state=args.random_seed)
        W_init = model.fit_transform(X)
        H_init = model.components_
    elif args.W_H_initialization in ['random', 'nndsvd']:
        W_init, H_init = _initialize_nmf(X, args.K_hat,
                                         init = args.W_H_initialization,
                                         random_state=args.random_seed)
    print('Done initialization.')
    return W_init, H_init
    
def evaluate_result(args, X, W, H, T, E):
    cph = CoxPHFitter(baseline_estimation_method="breslow",
                      penalizer=args.penalizer,
                      l1_ratio=args.l1_ratio)
    df = copy.deepcopy(pd.DataFrame(H.T))
    df['t'], df['e'] = T, E
    cph.fit(df, 't', event_col='e')
    error = np.linalg.norm(X - W.dot(H), ord='fro')
    cindex = cph.concordance_index_
    return cph, error, cindex

def run_model(X,
              T,
              E,
              args,
              W_init = None,
              H_init = None,
              model='CoxNMF',
              verbose=0,
              logger=None):
    original_error = np.linalg.norm(X, ord='fro')
    print('current model: %s' % model)
    n_components = args.K_hat
    start_time = time.time()
    
    result_dict = {}
    
    if model == 'TruncatedSVD':
        svd = TruncatedSVD(n_components=n_components,
                           algorithm='arpack',
                           random_state=args.random_seed)
        W_hat = svd.fit_transform(X)
        H_hat = svd.components_
        
        cph, error, cindex = evaluate_result(args, X, W_hat, H_hat, T, E)
        beta = cph.params_.values
        
    if model == 'PCA':
        pca = PCA(n_components=n_components,
                  svd_solver='arpack',
                  random_state=args.random_seed)
        pca.fit(X)
        W_hat = pca.fit_transform(X)
        H_hat = pca.components_
        cph, error, cindex = evaluate_result(args, X-pca.mean_, W_hat, H_hat, T, E)
        beta = cph.params_.values

    if model == 'SparsePCA':
        pca = SparsePCA(n_components=n_components,
                        alpha = args.alpha, # default: 1. Higher values lead to sparser components.
                        max_iter = args.max_iter,
                        random_state=args.random_seed)
        pca.fit(X)
        W_hat = pca.fit_transform(X)
        H_hat = pca.components_
        cph, error, cindex = evaluate_result(args, X - pca.mean_, W_hat, H_hat, T, E)
        beta = cph.params_.values
        
    
    if model == 'FactorAnalysis':
        transformer = FactorAnalysis(n_components=n_components,
                                    svd_method='randomized',
                                    iterated_power=3,
                                    max_iter=args.max_iter,
                                    random_state=args.random_seed)
        W_hat = transformer.fit_transform(X)
        H_hat = transformer.components_
        cph, error, cindex = evaluate_result(args, X - transformer.mean_, W_hat, H_hat, T, E)
        beta = cph.params_.values
        
    if model == 'NNDSVD':
        W_hat, H_hat = NNDSVD_NMF(X,
                                   n_components,
                                   eps=1e-6,
                                   max_iter=args.max_iter,
                                   random_state=args.random_seed)
        cph, error, cindex = evaluate_result(args, X, W_hat, H_hat, T, E)
        beta = cph.params_.values
    
    if model in ['NMF (CD)', 'NMF (MU)']:
        if model == 'NMF (CD)': solver = 'cd'
        elif model == 'NMF (MU)': solver = 'mu'
        model = NMF(n_components=n_components,
                                init='random',
                                solver=solver,
                                tol=1e-6,
                                max_iter=args.max_iter,
                                random_state=args.random_seed)
        W_hat = model.fit_transform(X)
        H_hat = model.components_
        
        cph, error, cindex = evaluate_result(args, X, W_hat, H_hat, T, E)
        beta = cph.params_.values
        
    elif model == 'CoxNMF':
        
        W_init_in = copy.deepcopy(W_init)
        H_init_in = copy.deepcopy(H_init)
        W_hat, H_hat, n_iter, error_list, cindex_list, res = \
            CoxNMF(X = X,
                   t = T,
                   e = E,
                   W_init = W_init_in,
                   H_init = H_init_in,
                   n_components = n_components,
                   alpha = args.alpha,

                   penalizer = args.penalizer,
                   l1_ratio = args.l1_ratio,
                   ci_tol = args.ci_tol,
                   max_iter = args.max_iter,
                   solver = 'mu',
                   update_rule = 'projection',
                   tol = 1e-6,
                   random_state = args.random_seed,
                   update_H = True,
                   update_beta = True,
                   W_normalization = args.W_normalization, # False
                   H_normalization = args.H_normalization, # False
                   beta_normalization = args.beta_normalization, # True
                   logger=logger,
                   verbose = verbose)
        W_hat = res['W']
        H_hat = res['H']
        error = res['error']
        cindex = res['cindex']
        beta = res['beta']
        result_dict['n_iter'] = res['n_iter']
        
        
    elif model == 'SNMF':
        print('Use alpha as hyperparameter.')
        hparam = args.alpha
        Winit = np.random.rand(n_components).reshape(-1,1)
        binit = np.random.rand(1)
        U_init, V_init = NNDSVD.NNDSVD(X.T, n_components) # note: U_init is actually H
        Y = T.reshape(-1,1)
        apha, bta, gma = hparam/Winit.shape[0], hparam/n_components, hparam/(X.shape[1]*n_components)
        snmf_res = Supernmf_pg.snmfpg(X=X.T, Y=Y, Uinit=U_init, Vinit=V_init,
                    Winit = Winit, binit = binit,
                    lambda1 = apha, lambda2 = bta,
                    lambda3 = gma, lambda4 = 0,
                    tol=1e-6, timelimit=np.inf, maxiter=args.max_iter)
        U, V, weight, bias, loss_nmf, loss_logit, loss_wb, loss_U, loss_V, obj_total_lst, obj_U, obj_W, obj_b, obj_V = snmf_res
        W_hat, H_hat = V.T, U.T
        cph, error, cindex = evaluate_result(args, X, W_hat, H_hat, T, E)
        beta = cph.params_.values
        
    running_time = time.time() - start_time
    relative_error = error/original_error
    
    result_dict['W_hat'] = W_hat
    result_dict['H_hat'] = H_hat
    result_dict['error'] = error
    result_dict['relative_error'] = relative_error
    result_dict['cindex'] = cindex
    result_dict['running_time'] = running_time
    result_dict['beta'] = beta
    
    return result_dict





def run_model_and_save_logs(X,
                            W_init,
                            H_init,
                            t,
                            e,
                            labels, # None if dataset == 'TCGA'
                            args,
                            model,
                            result_df,
                            logger,
                            verbose=0):

    result_dict = {'Model': model,
                    'random_seed': args.random_seed,
                    'K': args.K,
                    'K_hat': args.K_hat,
                    'P': X.shape[0],
                    'N': X.shape[1],
                    'alpha': args.alpha,
                    'penalizer':args.penalizer,
                    'l1_ratio':args.l1_ratio,
                    'hclust_linkage':args.linkage,
                    'max_iter':args.max_iter,
                    }
    if args.dataset == 'simulation':
        result_dict.update({
            'death_rate': args.death_rate,
            'W_distribution': args.W_distribution,
            'X_noise': args.X_noise,
            'W_noise': args.W_noise,
            'H_noise': args.H_noise,
            'simulation_type': args.simulation_type,
            'specific_row': args.specific_row,
            })
    else:
        result_dict.update({
            'mean_quantile':args.mean_quantile,
            'var_quantile':args.var_quantile,
            'cancer':args.cancer,
            'simulation_type': args.simulation_type,
            'specific_row': args.specific_row,
            })
    if model == 'CoxNMF':
        result_dict.update({
            'W_H_initialization':args.W_H_initialization,
            'W_normalization':args.W_normalization,
            'H_normalization':args.H_normalization,
            'beta_normalization':args.beta_normalization
            })
    try:
    # if True:
        acc,p,r,f1 = None, None, None, None
        res = run_model(X,t,e,args,W_init,H_init,model=model,logger=logger,verbose=verbose)
                
        W_hat = res['W_hat']
        H_hat = res['H_hat']
        error = res['error']
        relative_error = res['relative_error']
        cindex = res['cindex']
        beta = res['beta']
        running_time = res['running_time']
            
        acc, p, r, f1, iou, dice, silhouette_score = feature_analysis_no_fig(args,
                                                                            W_hat,
                                                                            H_hat,
                                                                            beta,
                                                                            labels, # None if dataset == 'TCGA'
                                                                            affinity='euclidean',
                                                                            normalize = True,
                                                                            weighted = False)

                                        
        print('Cindex = %.4f, Fnorm = %.4f, percentage = %.4f%%, runtime = %.4fs' % (cindex, error, relative_error*100, running_time))
        
        result_dict.update({
            'Fnorm':error,
            'relative_error':relative_error,
            'CIndex':cindex,
            'Runtime':running_time,
            'silhouette_score':silhouette_score,
            'Accuracy':acc,
            'precision':p,
            'recall':r,
            'F1_score':f1,
            'IoU':iou,
            'Dice coefficient':dice,
            'Success': True,
            })
        
    except Exception as e:
        print(e)
        result_dict.update({
            'Success': False,
            })
        
    result_df = result_df.append(result_dict,ignore_index=True)
    return result_df
        
