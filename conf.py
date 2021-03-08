#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import logging

def getworkdir():
    workdir = 'Your_Folder'
    return workdir


def conf():
    workdir = getworkdir()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='simulation', choices=['simulation','TCGA'])
    
    # simulation
    parser.add_argument('--simulation_type', type=str, default='univariate_better_worse',
                        choices=['univariate_better_worse', 'multivariate_better_worse'])
    parser.add_argument('--K', type=int, default=7, help="code dimension")
    parser.add_argument('--K_hat', type=int, default=7, help="code dimension")
    parser.add_argument('--W_noise', type=float, default=0, help='float value from 0 to 1. 0: no noise.')
    parser.add_argument('--H_noise', type=float, default=0, help='float value from 0 to 1. 0: no noise.')
    parser.add_argument('--X_noise', type=float, default=0, help='float value from 0 to 1. 0: no noise.')
    parser.add_argument('--W_distribution', type=str, default='exponential', help='exponential, weibull')
    parser.add_argument('--W_group_size', type=int, default=50)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--time_distribution', type=str, default='exponential')
    parser.add_argument('--death_rate', type=float, default=1, help='1: all patient will dead (default)')
        
    # cancer data
    parser.add_argument('--cancer', type=str, default='BRCA', help="TCGA cancer dataset")
    parser.add_argument('--mean_quantile', type=float, default=0.2, help="")
    parser.add_argument('--var_quantile', type=float, default=0.2, help="")
    
    # hyper-parameters
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--linkage', type=str, default='ward')
    parser.add_argument('--penalizer', type=float, default=1)
    parser.add_argument('--l1_ratio', type=float, default=1)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--ci_tol', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=10, help="alpha for cox. range: 1 (no Cox) to infinite (more Cox)")
    parser.add_argument('--cph_max_steps', type=float, default=1, help="max steps for coxph")
    parser.add_argument('--W_H_initialization', type=str, default='CD')
    parser.add_argument('--W_normalization', type=bool, default=False)
    parser.add_argument('--H_normalization', type=bool, default=False)
    parser.add_argument('--beta_normalization', type=bool, default=True)
    parser.add_argument('--results_dir', default=workdir + 'Results/20201220/', help="results dir")
    args = parser.parse_args()
    
    # TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())

    plt.ioff()
    np.random.seed(args.random_seed)
# # =============================================================================
# #     Logger
# # =============================================================================
#     logger = logging.getLogger(TIMESTRING)
#     logger.setLevel(logging.DEBUG)
#     # create file handler which logs even debug messages
#     fh = logging.FileHandler(os.path.join(writedir,'logging_file.log'), mode='w')
#     fh.setLevel(logging.DEBUG)
#     logger.addHandler(fh)
#     logger.log(logging.INFO, "Arguments: %s" % args)
    logger = None
    if args.simulation_type != 'specific_row':
        args.specific_row = None
    return args, logger





