# -*- coding: utf-8 -*-

"""
Created on Sat Sep  5 18:14:51 2020

@author: zhihuan
"""



import sys, os, platform, copy
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from conf import getworkdir

workdir = getworkdir()


# =============================================================================
# Simulation
# =============================================================================
bash_folder = os.path.join(workdir,'Model/bash_simulation')
if not os.path.exists(bash_folder): os.makedirs(bash_folder)
print('remove all files in bash_folder')
filelist = [f for f in os.listdir(bash_folder)]
for f in filelist: os.remove(os.path.join(bash_folder, f))

n = 60 # total nodes

simulation_type_list = ['univariate_better_worse', 'multivariate_better_worse']
K_list = [6,7,8,9,10,11,12]
X_noise_list = [0, 0.05, 0.1]
W_distribution_list = ['exponential']
death_rate_list = [1]
max_iter_list = [500]
ci_tol_list = [0.2]
random_seed_list = [1,2,3,4,5]

all_combinations = itertools.product(
                                    simulation_type_list,
                                    K_list,
                                    X_noise_list,
                                    W_distribution_list,
                                    death_rate_list,
                                    max_iter_list,
                                    ci_tol_list,
                                    random_seed_list)

text_list = []
for combination in all_combinations:
    simulation_type, K, X_noise, W_distribution, death_rate, max_iter, ci_tol, random_seed = combination
    bashtext = 'python ' + workdir + 'Model' + \
                '/main.py' + \
                ' --dataset simulation' + \
                ' --K ' + str(K) + \
                ' --X_noise ' + str(X_noise) + \
                ' --W_distribution ' + str(W_distribution) + \
                ' --death_rate ' + str(death_rate) + \
                ' --simulation_type ' + str(simulation_type) + \
                ' --max_iter ' + str(max_iter) + \
                ' --ci_tol ' + str(ci_tol) + \
                ' --random_seed ' + str(random_seed) + \
                ' --results_dir ' + workdir + 'Results/20210131_hclust/'
    text_list.append(bashtext)
print('Total text list size = %d' % len(text_list))


n_text_list = np.array_split(np.array(text_list), n)

for i, tl in enumerate(n_text_list):
    np.savetxt(os.path.join(bash_folder, 'run_bash_%d.sh' % (i+1)), tl, fmt = "%s")
    
    
    jobscript = " #!/bin/bash \n" + \
                " #PBS -k o  \n" + \
                " #PBS -l nodes=1:ppn=2,vmem=8gb,walltime=5:59:00 \n" + \
                " #PBS -M zhihuan@iu.edu \n" + \
                " #PBS -m abe  \n" + \
                " #PBS -N run_%d  \n" % (i+1) + \
                " #PBS -j oe  \n" + \
                "bash %s/run_bash_%d.sh" % (bash_folder ,i+1)
    with open(os.path.join(bash_folder, 'job_%d.script' % (i+1)), 'w+') as f:
        f.write(jobscript)


qsub = []
for i in range(n):
    qsub.append('qsub job_%d.script' % (i+1))
  
np.savetxt(os.path.join(bash_folder, '_qsub_all.sh'), qsub, fmt = "%s")






# =============================================================================
# TCGA
# =============================================================================

import sys, os, platform, copy
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from conf import getworkdir

workdir = getworkdir()

bash_folder = os.path.join(workdir,'Model/bash_TCGA')
if not os.path.exists(bash_folder): os.makedirs(bash_folder)
# remove all files in bash_folder
filelist = [f for f in os.listdir(bash_folder)]
for f in filelist: os.remove(os.path.join(bash_folder, f))

n = 182 # total nodes

cancer_list = ['BLCA','BRCA','COAD','KIRC','KIRP','LIHC','LUAD','LUSC','OV','PAAD']
K_hat_list = np.arange(start=5,stop=31)
max_iter_list = [500]
random_seed_list = [1]

all_combinations = itertools.product(
                                    cancer_list,
                                    max_iter_list,
                                    random_seed_list,
                                    K_hat_list)

text_list = []
for combination in all_combinations:
    cancer, max_iter, random_seed, K_hat = combination

    results_folder = '20210131_hclust_Cancer'


    results_dir = os.path.join(workdir,'Results', results_folder,'Cancer=%s' % cancer)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if len(os.listdir(results_dir))>0: # If has some results already
        results_dir = os.path.join(results_dir, os.listdir(results_dir)[0])
        output_fname = os.path.join(results_dir, 'result_df_Khat=%d_seed=%d.csv' % (K_hat, random_seed))
        if os.path.exists(output_fname): continue
    
    bashtext = 'python ' + workdir + 'Model' + \
                '/main.py' + \
                ' --dataset TCGA' + \
                ' --K_hat ' + str(K_hat) + \
                ' --cancer ' + cancer + \
                ' --max_iter ' + str(max_iter) + \
                ' --random_seed ' + str(random_seed) + \
                ' --results_dir ' + os.path.join(workdir, 'Results', results_folder)
    text_list.append(bashtext)
print('Total text list size = %d' % len(text_list))


n_text_list = np.array_split(np.array(text_list), n)

for i, tl in enumerate(n_text_list):
    np.savetxt(os.path.join(bash_folder, 'run_bash_%d.sh' % (i+1)), tl, fmt = "%s")
    
    
    jobscript = " #!/bin/bash \n" + \
                " #PBS -k o  \n" + \
                " #PBS -l nodes=1:ppn=4,vmem=16gb,walltime=5:59:00 \n" + \
                " #PBS -M zhihuan@iu.edu \n" + \
                " #PBS -m abe  \n" + \
                " #PBS -N run_%d  \n" % (i+1) + \
                " #PBS -j oe  \n" + \
                "bash %s/run_bash_%d.sh" % (bash_folder ,i+1)
    with open(os.path.join(bash_folder, 'job_%d.script' % (i+1)), 'w+') as f:
        f.write(jobscript)


qsub = []
for i in range(n):
    qsub.append('qsub job_%d.script' % (i+1))
  
np.savetxt(os.path.join(bash_folder, '_qsub_all.sh'), qsub, fmt = "%s")



