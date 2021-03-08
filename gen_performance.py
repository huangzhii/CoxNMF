#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

def average_by_seeds(r):
    if np.sum(r['Success'].astype(bool)) < len(r):
        # One of the seeds doesn't generate results (failed to converge). Abandon all seeds.
        return None
    rmean = r.mean(skipna=True)
    rstd = r.std(skipna=True)
    
    r2 = r.iloc[0,:]
    r2['random_seed'] = 'averaged'
    for c in ['Fnorm', 'relative_error','CIndex','Runtime','silhouette_score',
              'Accuracy','precision','recall','F1_score','IoU','Dice coefficient']:
        r2.loc[c] = (rmean.loc[c], rstd.loc[c])
    return r2

def find_k_hat(x):
    imax = np.argmax([m for (m, v) in x['silhouette_score']])
    x2 = x.iloc[imax,:]
    return x2

def model_selection(x, metric='CIndex'):
    #This is all possible combination of hyperparameters for one model
    if metric == 'CIndex' or metric == 'silhouette_score':
        idx = np.argmax([m for (m, v) in x[metric]])
    x2 = x.iloc[idx,:]
    return x2


def get_final_perfomance(args, metric='CIndex'):
    # st = time.time()
    if args.dataset == 'simulation':
        folder = os.path.join(args.result_folder,
                              'Simulation=%s' % args.simulation_type,
                              'deathrate=%.2f_Wdistribution=%s_citol=%.2f_maxiter=%d' % (args.death_rate, args.W_distribution, args.ci_tol, args.max_iter),
                              'K=%d_P=%d_N=%d' % (args.K, args.P, args.N),
                              'Noise_X=%.2f_W=%.2f_H=%.2f' % (args.X_noise, args.W_noise, args.H_noise)
                              )
    elif args.dataset == 'TCGA':
        folder = os.path.join(args.result_folder,
                              'Cancer=%s' % args.cancer,
                              'meanq=%.2f_varq=%.2f_P=%d_N=%d_citol=%.2f_maxiter=%d' % \
                                  (args.mean_quantile, args.var_quantile, args.P, args.N, args.ci_tol, args.max_iter)
                              )
        
    if not os.path.exists(folder): print('Folder not exists!')
    df = None
    if args.dataset == 'simulation':
        for seed in args.seed_list:
            df_dir = os.path.join(folder, 'result_df_seed=%d.csv' % seed)
            if not os.path.exists(df_dir):
                print(folder)
                raise Exception('All: Seed = %d path not exists!' % seed)
                continue
            if df is None:
                df = pd.read_csv(df_dir, index_col = 0)
            else:
                df = df.append(pd.read_csv(df_dir, index_col = 0),ignore_index=True)
                
            
    elif args.dataset == 'TCGA':
        for K_hat in args.K_hat_list:
            for seed in args.seed_list:
                df_dir = os.path.join(folder, 'result_df_Khat=%d_seed=%d.csv' % (K_hat, seed))
                if not os.path.exists(df_dir):
                    print(folder)
                    raise Exception('K_hat = %d Seed = %d path not exists!' % (K_hat, seed))
                    continue
                if df is None:
                    df = pd.read_csv(df_dir, index_col = 0)
                else:
                    df = df.append(pd.read_csv(df_dir, index_col = 0),ignore_index=True)
        
        
    df = df.fillna(-1)
        
    # =============================================================================
    #     Average by seeds
    # =============================================================================
    # print('Averaging results by seeds ...')
    df_seed = df.groupby(['Model','K', 'K_hat','alpha','penalizer',
               'W_H_initialization','W_normalization','H_normalization','beta_normalization',
               ]).apply(lambda r: average_by_seeds(r)).reset_index(drop=True)
    df_seed.dropna(axis=0, how='any', inplace=True)
    
    # =============================================================================
    #     This step help each model find their own best alpha & penalizer w.r.t. CIndex/relative_error
    # =============================================================================
    
    df_best_mdl = df_seed.groupby(['Model','K', 'K_hat', # For every K_hat, we select the best model w.r.t. CIndex/relative_error
                      ]).apply(lambda x: model_selection(x, metric)).reset_index(drop=True)
        
    # =============================================================================
    #     Upon their optimal parameter determined, find their optimal K_hat final solutions.
    # =============================================================================
    # print('Finding optimal K_hat w.r.t. CIndex ...')
    df_khat = df_best_mdl.groupby(['Model',
                        ]).apply(lambda x: find_k_hat(x)).reset_index(drop=True)
    # print('Total time elapsed: %.2f seconds.' % (time.time() - st))
    return df, df_seed, df_best_mdl, df_khat


def generate_main_table(args,
                        table,
                        metrics,
                        models,
                        X_noise,
                        metric='CIndex'):
    
    coxnmf_params = pd.DataFrame(index = args.K_list, columns = ['Simulation type','X_noise','K','K_hat','alpha','penalizer','initialization'])
    
    for K in args.K_list:
        print('K = %.2f' % K)
        args.K = K
        args.X_noise = X_noise
        X, W, H, t, e, labels = getdata(args)
        args.P, args.N = X.shape
        _, _, _, df_khat = get_final_perfomance(args, metric=metric)
        for mo in models:
            row = df_khat.loc[df_khat['Model'] == mo]
            for me in metrics:
                try:
                    value = row.loc[:,me].values[0]
                except:
                    print('No metric %s found for %s' % (me, mo))
                if isinstance(value, tuple):
                    value = '%.4f±%.2f' % (value[0],value[1])
                table.loc[(me, mo),('X_noise=%.2f' % X_noise, '%d' % K)] = value
            
            
        alpha = df_khat.loc[df_khat['Model'] == 'CoxNMF','alpha'].values[0]
        penalizer = df_khat.loc[df_khat['Model'] == 'CoxNMF','penalizer'].values[0]
        initialization = df_khat.loc[df_khat['Model'] == 'CoxNMF','W_H_initialization'].values[0]
        K_hat = df_khat.loc[df_khat['Model'] == 'CoxNMF','K_hat'].values[0]
        coxnmf_params.loc[K, :] = args.simulation_type, args.X_noise, K, K_hat, alpha, penalizer, initialization
        
    return table, coxnmf_params

def generate_table_supplementary(args,
                                 table,
                                 metrics,
                                 models,
                                 metric='CIndex'):
    alphalist = []
    for K in args.K_list:
        print('K = %.2f' % K)
        for X_noise in args.X_noise_list:
            print('    X_noise = %.2f' % X_noise)
            args.K = K
            args.X_noise = X_noise
            X, W, H, t, e, labels = getdata(args)
            args.P, args.N = X.shape
            _, _, _, df_khat = get_final_perfomance(args, metric=metric)
            
            alphalist += list(df_khat.iloc[:4,6].values)
            for mo in models:
                if mo == '':
                    table.loc[(K,mo),:] = ''
                    continue
                if 'CoxNMF' in mo:
                    row = df_khat.loc[df_khat['Model'] == mo]
                else:
                    row = df_khat.loc[df_khat['Model'] == mo]
                for me in metrics:
                    if me == '':
                        table.loc[:,('X_noise=%.2f' % X_noise, me)] = ''
                        continue
                    value = row.loc[:,me].values[0]
                    if isinstance(value, tuple):
                        value = '%.4f±%.2f' % (value[0],value[1])
                    table.loc[(K,mo),('X_noise=%.2f' % X_noise, me)] = value
    return table
                    
def generate_table_massive_performances(args,
                                        metric='CIndex'):
    table = None
    for K in args.K_list:
        print('K = %.2f' % K)
        for X_noise in args.X_noise_list:
            print('    X_noise = %.2f' % X_noise)
            args.K = K
            args.X_noise = X_noise
            X, W, H, t, e, labels = getdata(args)
            args.P, args.N = X.shape
            _, df_seed, _, _ = get_final_perfomance(args, metric=metric)
            if table is None:
                table = df_seed
            else:
                table = table.append(df_seed, ignore_index=True)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            value = table.iloc[r,c]
            if isinstance(value, tuple):
                value = '%.4f±%.2f' % (value[0],value[1])
            table.iloc[r,c] = value
    return table


def get_simulation_performance(args,
                               table,
                               metric='CIndex'):
# =============================================================================
#     Table 2-3: Univariate Multivariate of accuracy and relative error
# =============================================================================
    models = ['TruncatedSVD','PCA','SparsePCA','NNDSVD','FactorAnalysis',
              'NMF (CD)','NMF (MU)','SNMF','CoxNMF']
    metrics = ['CIndex','relative_error','Accuracy','IoU','Dice coefficient']
    
    coxnmf_params_all = pd.DataFrame()
    for X_noise in args.X_noise_list:
        print('X_noise = %.2f' % X_noise)
        col1 = ['X_noise=%.2f' % X_noise]*len(args.K_list)
        col2 = args.K_list
        columns = np.array([col1, col2])
        indices = [(me,mo) for me in metrics for mo in models]
        for args.simulation_type in args.all_simulation_types:
            print(args.simulation_type)
            curr_table = pd.DataFrame(index = pd.MultiIndex.from_tuples(indices, names=['Metrics', 'Model']),
                                      columns = pd.MultiIndex.from_arrays(columns))
            
                
            curr_table, coxnmf_params = generate_main_table(args,
                                                            curr_table,
                                                            metrics,
                                                            models,
                                                            X_noise,
                                                            metric=metric)
            
            # =============================================================================
            #         Find optimal results and bold it in LaTeX!
            curr_table_textbf = copy.deepcopy(curr_table)
            for m in metrics:
                part = curr_table.loc[(m,),]
                for c in part.columns:
                    val2compare = part.loc[:,c]
                    val2compare_mean = np.array([float(s.split('±')[0]) for s in val2compare.values])
                    if m == 'relative_error':
                        optimalval = np.min(val2compare_mean)
                    else:
                        optimalval = np.max(val2compare_mean)
                    idx = part.index[val2compare_mean == optimalval]
                    for i in idx:
                        curr_table_textbf.loc[(m,i),c] =  '\\bf{' + curr_table_textbf.loc[(m,i),c] + '}'
            # =============================================================================
                        
            with open(os.path.join(args.output_folder,
                                    '%s_main_Xnoise=%.2f.txt' % (args.simulation_type, X_noise)), "w") as f:
                tbl_latex = curr_table_textbf.to_latex().replace('±','$\pm$')
                tbl_latex = tbl_latex.replace('\\textbackslash bf','\\bf')
                tbl_latex = tbl_latex.replace('\\{','{')
                tbl_latex = tbl_latex.replace('\\}','}')
                f.write(tbl_latex)
            table[args.simulation_type]['main_Xnoise=%.2f' % X_noise] = curr_table_textbf
            
            coxnmf_params_all = coxnmf_params_all.append(coxnmf_params, ignore_index=True)
    coxnmf_params_all.to_csv(os.path.join(args.output_folder, 'CoxNMF_params.csv'))
    
# =============================================================================
#     Table Supplementary
# =============================================================================
    
    metrics = ['K_hat','relative_error','CIndex','Accuracy','Dice coefficient','Runtime','']
    col1 = sum([['X_noise=%.2f' % X_noise]*len(metrics) for X_noise in args.X_noise_list], [])
    col2 = metrics*3
    columns = np.array([col1, col2])
    models = ['TruncatedSVD','PCA','SparsePCA','NNDSVD','FactorAnalysis',
              'NMF (CD)','NMF (MU)','SNMF','CoxNMF','']
    indices = [(k,m) for k in args.K_list for m in models]
    
    
    for args.simulation_type in args.all_simulation_types:
        curr_table = pd.DataFrame(index = pd.MultiIndex.from_tuples(indices, names=['K', 'Model']),
                                  columns = pd.MultiIndex.from_arrays(columns))
        curr_table = generate_table_supplementary(args,
                                                curr_table,
                                                metrics,
                                                models)
        
        # =============================================================================
        #         Find optimal results and bold it in LaTeX!
        curr_table_textbf = copy.deepcopy(curr_table)
        for k in args.K_list:
            part = curr_table.loc[(k,),]
            for c in part.columns:
                if c[1] == '': continue
                if c[1] == 'K_hat':
                    val2compare = part.loc[:,c]
                    idx = part.index[val2compare.values == k]
                    for i in idx:
                        curr_table_textbf.loc[(k,i),c] =  '\\bf{%d}' % curr_table_textbf.loc[(k,i),c]
                else:
                    val2compare = part.loc[:,c][:-1]
                    val2compare_mean = np.array([float(s.split('±')[0]) for s in val2compare.values])
                    if c[1] == 'relative_error' or c[1] == 'Runtime':
                        optimalval = np.min(val2compare_mean)
                    else:
                        optimalval = np.max(val2compare_mean)
                    idx = part[:-1].index[val2compare_mean == optimalval]
                    for i in idx:
                        curr_table_textbf.loc[(k,i),c] =  '\\bf{' + curr_table_textbf.loc[(k,i),c] + '}'
        # =============================================================================
        
        
        
        with open(os.path.join(args.output_folder,
                                '%s_supplementary.txt' % args.simulation_type), "w") as f:
            tbl_latex = curr_table_textbf.to_latex().replace('±','$\pm$')
            tbl_latex = tbl_latex.replace('\\textbackslash bf','\\bf')
            tbl_latex = tbl_latex.replace('\\{','{')
            tbl_latex = tbl_latex.replace('\\}','}')
            f.write(tbl_latex)
        table[args.simulation_type]['supplementary'] = curr_table
    
    
#     Generate massive performances table

    table_performances = generate_table_massive_performances(args)
    table_performances.to_csv(os.path.join(args.output_folder,
                              'table_performances.csv'))
    return table
    
         
def generate_simulation_atlas(args,X,W,H,t,e,labels, metric='CIndex'):
    _, _, _, df_khat = get_final_perfomance(args,metric=metric)
    df_khat.index = df_khat['Model']
    
    args.K_hat,\
    args.W_H_initialization,\
    args.W_normalization,\
    args.H_normalization,\
    args.beta_normalization,\
    args.linkage,\
    args.alpha,\
    args.penalizer,\
    args.l1_ratio = df_khat.loc['CoxNMF', ['K_hat',
                                            'W_H_initialization',
                                            'W_normalization',
                                            'H_normalization',
                                            'beta_normalization',
                                            'hclust_linkage',
                                            'alpha',
                                            'penalizer',
                                            'l1_ratio'
                                            ]].values
    args.K_hat = int(args.K_hat)
    W_init, H_init = CoxNMF_initialization(args, X)
    res = run_model(X,t,e,args,W_init,H_init,model='CoxNMF',logger=None,verbose=1)
    acc, p, r, f1, iou, dice, silhouette_score = feature_analysis_no_fig(args, res['W_hat'], res['H_hat'],\
                                                                            res['beta'], labels,\
                                                                            affinity='euclidean',\
                                                                            normalize = True,\
                                                                            weighted = False)
    print('Cindex = %.4f, Fnorm = %.4f, percentage = %.4f%%, runtime = %.4fs' % (res['cindex'], res['error'], res['relative_error']*100, res['running_time']))
    print('Accuracy: %.4f' % acc)
    print('F-1 score: %.4f' % f1)
    print('Precision: %.4f' % p)
    print('Recall: %.4f' % r)
    print('IoU: %.4f' % iou)
    print('Dice: %.4f' % dice)
    
    
    stat = pd.DataFrame.from_dict({'error': res['error'],
                                   'relative_error': res['relative_error'],
                                   'cindex': res['cindex'],
                                   'running_time': res['running_time'],
                                   'silhouette_score': silhouette_score,
                                   'Accuracy': acc,
                                   'F-1 score': f1,
                                   'Precision': p,
                                   'Recall': r,
                                   'IoU': iou,
                                   'Dice coefficient': dice,
                                   'W_H_initialization': args.W_H_initialization,
                                   'random_seed': args.random_seed,
                                   'alpha': args.alpha,
                                   'ci_tol': args.ci_tol,
                                   'K_hat': args.K_hat,
                                   'beta_normalization': args.beta_normalization,
                                   'H_normalization': args.H_normalization,
                                   'penalizer': args.penalizer}, orient='index')
    
    temp_folder = os.path.join(args.output_folder,
                              '%s_CoxNMF_K=%d_Xnoise=%.2f_seed=%d_init=%s_alpha=%.2f_penalizer=%.2f'%\
                              (args.simulation_type,args.K,args.X_noise,args.random_seed,args.W_H_initialization,\
                               args.alpha,args.penalizer))
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)
        
    stat.to_csv(os.path.join(temp_folder, 'stats.csv'))
    
    
    atlas = plt.figure(figsize=(5,7), constrained_layout=True)
    gs = atlas.add_gridspec(nrows=3, ncols=5, width_ratios=[0.15, 1, 0.2, 1, 0.15], height_ratios=[2,10,40])
    atlas = plot_simulation_atlas(atlas, gs, args, W, res['W_hat'], H, res['H_hat'], t, e, res['beta'], labels,\
                                    affinity='euclidean', normalize = True, weighted = False)
    atlas.savefig(os.path.join(temp_folder, 'atlas.pdf'), dpi = 600)
    for ax in atlas.get_axes(): ax.clear()
    return

def prepare_figure(args,
                   metric='CIndex'):
    df_ci = None
    
    all_combinations = itertools.product(
                                        args.all_simulation_types,
                                        args.X_noise_list,
                                        args.K_list)
    
    for combination in tqdm(list(all_combinations)):
        args.simulation_type, X_noise, K = combination
        
        args.K = K
        args.X_noise = X_noise
        X, W, H, t, e, labels = getdata(args)
        args.P, args.N = X.shape
        _, _, _, df_khat = get_final_perfomance(args, metric=metric)
        df_temp = pd.DataFrame()
        df_temp['Model'] = df_khat['Model']
        df_temp['K'] = args.K
        df_temp['simulation_type'] = args.simulation_type
        df_temp['X_noise'] = X_noise
        df_temp['CIndex'] = [m for m,s in df_khat['CIndex']]
        df_temp['Accuracy'] = [m for m,s in df_khat['Accuracy']]
        df_temp['relative error'] = [m for m,s in df_khat['relative_error']]
        df_temp['IoU'] = [m for m,s in df_khat['IoU']]
        df_temp['Dice coefficient'] = [m for m,s in df_khat['Dice coefficient']]
        if df_ci is None:
            df_ci = df_temp
        else:
            df_ci = df_ci.append(df_temp, ignore_index=True)
    return df_ci

def get_CIndex_vs_(args, models, df_ci):
    
    colors = {'CoxNMF':1,
                'TruncatedSVD':2,
                'PCA':3,
                'SparsePCA':4,
                'NNDSVD':5,
                'FactorAnalysis':6,
                'NMF (CD)':7,
                'NMF (MU)':8,
                'SNMF':9}
    for Cindex_vs_ in ['Accuracy', 'relative error','IoU','Dice coefficient']:
        sizelist = (1.5**(np.arange(len(args.K_list))+1))*10
        fig, axes = plt.subplots(ncols=len(args.X_noise_list),
                                      nrows=len(args.all_simulation_types),
                                      sharey=True, sharex=True,
                                      figsize=(10,4), constrained_layout=True)
    
        scatter = {}
        for i, st in enumerate(args.all_simulation_types):
            for j, X_noise in enumerate(args.X_noise_list):
                axes[i,j].grid(True, linestyle='dotted')
                axes[i,j].set_axisbelow(True)
                axes[i,j].set_xscale('logit')
                axes[i,j].set_xlim((0.5,1.0001))
                axes[i,j].axvline(x=0.99, linestyle=':', linewidth=1, color='red')
                axes[i,j].tick_params(axis='x', labelrotation=30)
                if i==1: axes[i,j].set_xlabel(r'CIndex ($\varepsilon=%.2f$)' % X_noise, fontsize=12)
                if j==0: axes[i,j].set_ylabel('%s\n%s'%(st.split('_')[0].capitalize(),Cindex_vs_), fontsize=12)
                
                xl,yl,cl,sl = [],[],[],[]
                for K in args.K_list:
                    df_temp = df_ci[(df_ci['simulation_type'] == st) & \
                                    (df_ci['X_noise'] == X_noise) & \
                                    (df_ci['K'] == K)]
                    for mo in models:
                        ci = df_temp.loc[df_temp['Model'] == mo, 'CIndex'].values[0]
                        if ci == 1: ci -= 0.00001
                        xl.append(ci)
                        yl.append(df_temp.loc[df_temp['Model'] == mo, Cindex_vs_].values[0])
                        cl.append(colors[mo])
                        sl.append(sizelist[K-min(args.K_list)])
                scatter[(i,j)] = axes[i,j].scatter(x=xl, y=yl, c=cl, s=sl,
                                            marker='o', alpha=0.6, cmap='Set1') # set1 has 9 colors
                
                
        # Create the legend
        legend1 = axes[0,2].legend(*scatter[(0,2)].legend_elements(),     # The line objects
                    loc="upper left",   # Position of legend
                    ncol=2,
                    bbox_to_anchor=(1.05, 1), # (x, y, width, height)
                    borderaxespad=0.1,    # Small spacing around legend box
                    title="Models"  # Title for the legend
                    )
        for i, mo in enumerate(colors.keys()):
            legend1.get_texts()[i].set_text(mo)
        fig.add_artist(legend1)
        
        handles, labels = scatter[(1,2)].legend_elements(prop="sizes", alpha=0.6)
        legend2 = axes[1,2].legend(handles,
                                   labels,     # The line objects
                                    loc="lower left",   # Position of legend
                                    ncol=2,
                                    bbox_to_anchor=(1.05, 0), # (x, y, width, height)
                                    borderaxespad=0.1,    # Small spacing around legend box
                                    title=r"$K$"  # Title for the legend
                                    )
        for i, k in enumerate(args.K_list):
            legend2.get_texts()[i].set_text(k)
        fig.add_artist(legend2)
        fig.savefig(os.path.join(args.result_folder,
                              'Result_Tables_and_Figs',
                              'CIndex_vs_%s.pdf' % Cindex_vs_),
                              dpi = 600)
        
    return



    
def generate_CoxNMF_logs(args, metric='CIndex'):
    df_hparam = None
    for i, args.simulation_type in enumerate(args.all_simulation_types):
        for j, X_noise in enumerate(args.X_noise_list):
            print('X_noise = %.2f' % X_noise)
            for K in tqdm(args.K_list):
                args.K = K
                args.X_noise = X_noise
                X, W, H, t, e, labels = getdata(args)
                args.P, args.N = X.shape
                _, _, _, df_khat = get_final_perfomance(args, metric=metric)
                df_khat = df_khat.loc[df_khat['Model'] == 'CoxNMF',:]
                if df_hparam is None:
                    df_hparam = df_khat
                else:
                    df_hparam = df_hparam.append(df_khat, ignore_index=True)
    df_hparam.to_csv(os.path.join(args.convergence_folder, 'CoxNMF_all_hparams.csv'))
    
    for i in tqdm(range(len(df_hparam))):
        args.K, args.K_hat,\
        args.W_H_initialization,\
        args.W_normalization,\
        args.H_normalization,\
        args.beta_normalization,\
        args.linkage,\
        args.alpha,\
        args.penalizer,\
        args.X_noise,\
        args.simulation_type,\
        args.l1_ratio = df_hparam.loc[i, ['K','K_hat',
                                                'W_H_initialization',
                                                'W_normalization',
                                                'H_normalization',
                                                'beta_normalization',
                                                'hclust_linkage',
                                                'alpha',
                                                'penalizer',
                                                'X_noise',
                                                'simulation_type',
                                                'l1_ratio'
                                                ]].values
        args.K_hat = int(args.K_hat)
        args.K = int(args.K)
        for args.random_seed in args.seed_list:
            X, W, H, t, e, labels = getdata(args)
            args.P, args.N = X.shape
            W_init, H_init = CoxNMF_initialization(args, X)
            outfile =  os.path.join(args.convergence_folder,
                                    '%s_K=%d_Xnoise=%.2f_seed=%d.log' % \
                                    (args.simulation_type, args.K, args.X_noise, args.random_seed))
            # =============================================================================
            #     Logger
            # =============================================================================
            TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime()) + str(i) + str(args.random_seed)
            logger = logging.getLogger(TIMESTRING)
            logger.setLevel(logging.DEBUG)
            # create file handler which logs even debug messages
            fh = logging.FileHandler(outfile, mode='w')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            logger.log(logging.INFO, "Arguments: %s" % args)
            res = run_model(X,t,e,args,W_init,H_init,model='CoxNMF',logger=logger,verbose=0)
    return

def parse_log(args):
    df_iteration = pd.DataFrame(columns = ['K','simulation_type','X_noise','seed','Epoch','Relative error', 'C-Index'])
    for K in tqdm(args.K_list):
        for args.simulation_type in args.all_simulation_types:
            for X_noise in args.X_noise_list:
                for seed in args.seed_list:
                    logfile =  os.path.join(args.convergence_folder,
                                            '%s_K=%d_Xnoise=%.2f_seed=%d.log' % \
                                            (args.simulation_type, K, args.X_noise, seed))
                    with open(logfile,'r') as f: lines = [line.rstrip() for line in f]
                    for line in lines:
                        if not line.startswith('Epoch'): continue
                        m = re.search('Epoch (.+?) error: (.+?), relative_error: (.+?), concordance index: (.*)', line)
                        epoch = int(m.group(1))
                        relative_error = float(m.group(3))
                        CIndex = float(m.group(4))
                        df_iteration = df_iteration.append({'K': K,
                                                            'simulation_type': args.simulation_type,
                                                            'X_noise': X_noise,
                                                            'seed': seed,
                                                            'Epoch': epoch,
                                                            'Relative error': relative_error,
                                                            'C-Index': CIndex}, ignore_index=True)
    return df_iteration

def plot_convergence(args, df_iteration):
    
    df_iteration_averaged = df_iteration.groupby(['K','simulation_type','X_noise','Epoch']).progress_apply(lambda x: x.mean())
    df_iteration_averaged['simulation_type'] = df_iteration_averaged.index.get_level_values('simulation_type')
    df_iteration_averaged.reset_index(drop=True, inplace=True)
    df_iteration_averaged['X_noise'] = df_iteration_averaged['X_noise'].round(4)
    
    
    for args.simulation_type in args.all_simulation_types:
        fig, axes = plt.subplots(ncols=len(args.K_list),
                                nrows=len(args.X_noise_list),
                                sharey=True, sharex=False,
                                figsize=(12,6), constrained_layout=True)
        for j, K in tqdm(enumerate(args.K_list)):
            for i, X_noise in enumerate(args.X_noise_list):
                df = df_iteration_averaged.loc[(df_iteration_averaged['simulation_type'] == args.simulation_type)&\
                                          (df_iteration_averaged['K'] == K)&\
                                          (df_iteration_averaged['X_noise'] == X_noise),:]
                idxmax = df['C-Index'].reset_index(drop=True).idxmax()
                df = df.iloc[:idxmax+1,:]
                axes[i,j].grid(True, linestyle='dotted')
                axes[i,j].set_axisbelow(True)
                axes[i,j].set_xlim((0,df['Epoch'].max()))
                axes[i,j].set_ylim((0,1))
                if i==2: axes[i,j].set_xlabel(r'Epoch ($K=%d$)' % K, fontsize=12)
                if j==0: axes[i,j].set_ylabel(r'$\varepsilon=%.2f$'%(X_noise), fontsize=12)
                lines = axes[i,j].plot(df['Epoch'].values, df['C-Index'].values, color='blue')
                lines = axes[i,j].plot(df['Epoch'].values, df['Relative error'].values, color='orange')
        
        fig.suptitle(args.simulation_type.split('_')[0].capitalize() + ' simulation convergence plot',fontsize=14)
        
        fig.savefig(os.path.join(args.result_folder,
                              'Result_Tables_and_Figs',
                              'CoxNMF_convergence_%s.pdf' % args.simulation_type),
                              dpi = 600)
    return

def find_TCGA_args(args, metric='CIndex'):
    _, _, df_best_mdl, _ = get_final_perfomance(args, metric=metric)
    
    for idx in args.K_hat_list:
        if idx not in df_best_mdl.K_hat.values:
            print('[  ***  Warning  ***  ]: K=%d not in df_best_mdl.' % idx)
    
    silhouette_scores = pd.DataFrame([m for m,v in df_best_mdl['silhouette_score']],
                                     index = df_best_mdl.K_hat, columns = ['silhouette_score'])
    relative_errors = pd.DataFrame([m for m,v in df_best_mdl['relative_error']],
                                   index = df_best_mdl.K_hat, columns = ['relative_error'])
    Cindex = pd.DataFrame([m for m,v in df_best_mdl['CIndex']],
                                   index = df_best_mdl.K_hat, columns = ['CIndex'])
    metrics = pd.concat([silhouette_scores, relative_errors, Cindex], axis=1)
    metrics_selected = metrics.loc[metrics['relative_error'] < args.RELATIVE_ERROR_THRESHOLD,:]
    best_K_hat = metrics_selected['silhouette_score'].idxmax()
    
    print('=====================================')
    print('%s: Optimal K_hat = %d' % (args.cancer, best_K_hat))
    print('=====================================')
    df_K_hat = df_best_mdl.loc[df_best_mdl['K_hat'] == best_K_hat,:]
    args.best_K_hat = best_K_hat
    # CoxNMF
    args.K_hat,\
    args.W_H_initialization,\
    args.W_normalization,\
    args.H_normalization,\
    args.beta_normalization,\
    args.linkage,\
    args.alpha,\
    args.penalizer,\
    args.l1_ratio = df_K_hat.loc[:, ['K_hat','W_H_initialization',
                                    'W_normalization',
                                    'H_normalization',
                                    'beta_normalization',
                                    'hclust_linkage',
                                    'alpha',
                                    'penalizer',
                                    'l1_ratio'
                                    ]].values[0]
    args.K_hat = int(args.K_hat)
    
    args.output_folder = os.path.join(args.result_folder, 'Cancer=%s' % args.cancer,
                                      'meanq=%.2f_varq=%.2f_P=%d_N=%d_citol=%.2f_maxiter=%d' %
                                    (args.mean_quantile, args.var_quantile, args.P, args.N, args.ci_tol, args.max_iter),
                                    'CoxNMF_results_threshold=%.2f_K_hat=%d' % (args.RELATIVE_ERROR_THRESHOLD, args.K_hat))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    return args, df_best_mdl
    
if __name__ == '__main__':
    workdir = getworkdir()
    args, logger = conf()
    args.dataset = 'simulation'
    
    if args.dataset == 'simulation':
        args.result_folder = os.path.join(workdir, 'Results','20210131_hclust')
        
        args.output_folder = os.path.join(args.result_folder, 'Result_Tables_and_Figs')
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        
        args.all_simulation_types = ['univariate_better_worse', 'multivariate_better_worse']
        args.death_rate = 1
        args.W_distribution = 'exponential'
        args.max_iter = 500
        args.W_noise = 0
        args.H_noise = 0
        args.seed_list = [1,2,3,4,5]
        args.X_noise_list = [0,0.05,0.1]
        args.K_list=[6,7,8,9,10,11,12]
        
        # Generate table performances
        table = {}
        for st in args.all_simulation_types: table[st] = {}
        table = get_simulation_performance(args, table)
        
            
        # Plot atlas
        args.K = 10
        args.simulation_type = 'multivariate_better_worse'
        # args.simulation_type = 'univariate_better_worse'
        args.X_noise = 0.05
        args.random_seed = 3
        X, W, H, t, e, labels = getdata(args)
        args.P, args.N = X.shape[0], X.shape[1]
        generate_simulation_atlas(args,X,W,H,t,e,labels)
    
    
        # Figure: Concordance Index v.s. Accuracy (Univariate and Multivariate)
        models = ['TruncatedSVD','PCA','SparsePCA','NNDSVD','FactorAnalysis',
                  'NMF (CD)','NMF (MU)','SNMF','CoxNMF']
        
        df_ci = prepare_figure(args)
        get_CIndex_vs_(args, models, df_ci)
        
        
        # convergence plot, use massive performances table
        args.convergence_folder = os.path.join(args.output_folder, 'CoxNMF_convergences')
        if not os.path.exists(args.convergence_folder): os.makedirs(args.convergence_folder)
        generate_CoxNMF_logs(args) # This step can take several minutes (~10 minutes)
        df_iteration = parse_log(args)
        plot_convergence(args, df_iteration)
            
    
    elif args.dataset == 'TCGA':
        args.result_folder = os.path.join(workdir, 'Results','20210131_hclust_Cancer')
        
        args.cancer = 'KIRC'
        
        '''
        Note:
            To properly find the desired best model w.r.t. each K_hat,
            the minimum value of relative_error was chosen.
        '''
        
        X, W, H, t, e, labels = getdata(args)
        args.P, args.N = X.shape[0], X.shape[1]
        args.seed_list = [1]
            
        start = 10
        stop = 30+1
        args.K_hat_list = np.arange(start=start, stop=stop)
        args, df_best_mdl = find_TCGA_args(args)
        
        
        fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(10,4))
        val, feq = np.unique(df_best_mdl['alpha'], return_counts=True)
        axes[0].bar(val.astype('str'),feq)
        axes[0].set_xlabel('Alpha')
        val, feq = np.unique(df_best_mdl['penalizer'], return_counts=True)
        axes[1].bar(val.astype('str'),feq)
        axes[1].set_xlabel('Penalizer')
        val, feq = np.unique(df_best_mdl['W_H_initialization'], return_counts=True)
        axes[2].bar(val.astype('str'),feq)
        axes[2].set_xlabel('Initialization')
        
        fig
        
        
        
        
        
        
        
        
