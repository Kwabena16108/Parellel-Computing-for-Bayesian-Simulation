#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:26:21 2021

@author: dicksonnkwantabisa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:30:02 2021

@author: dicksonnkwantabisa

Parallel algorithm to simulate probabilities from a Gamma distribution

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import math
import random
import time
from scipy.stats import weibull_min
import multiprocessing
import time as time

lam = pd.read_csv("grp4_lambda.csv")
kappa = pd.read_csv("grp4_kappa.csv")
theta = pd.read_csv("grp4_theta.csv")

lam = lam.loc[:, ['qtr1','qtr2','qtr3','qtr4']]
kappa = kappa.loc[:, ['x']]
theta = theta.loc[:, ['x']]

totalloss=np.array([93980862,203573187,82438511,58207494,98789767,203784637,84601207,48266955,112214807,405363680,30273264,180974006,
                    37739274,651461972,53756681,143107731,36090613,84401541,588539114,36781837,130601067,75331736,286491004,83436015,
                    190295693,109419766,188648779,37859589,239306986,77273365,778381182,60095502,34004513,85415402,163555540,1719550417,
                    992335953,134950418,28929260,248717795,40185538,216417898,132177018,573134416,38945825,53035617,158020158,83606360,
                    273221882,132571995,218272961,35560468,3884408659,163731979,79632374,111151588,132945745,835223966,29211447,140578276,
                    119570081,174425978,46653607,147798727,163938682,101706921,177644518,69793493,106730574,60693231,164691525,142759633,
                    121663769,100681909,353522100,775291475,82083862,53310706,251714530,327836544,295925561,100576311,163783790,153859331,
                    88008553,249893480,141914000,522868000,1306405000,219446000,17201000]) / 10e06 # aggr claim sizes


lam = lam.to_numpy()
lam = lam[::3, ]


kappa = kappa.to_numpy()
kappa = kappa[::3, ]


theta = theta.to_numpy()
theta = theta[::3, ]

t = np.linspace(0., 1., 100) 
T = 1.

D = np.linspace(np.mean(totalloss)*3, np.mean(totalloss)*3, 100)
#D = np.tile(D, 20)
    
def prob(sft, D):
    return np.mean(sft <= D * 10e06)



def sim(t,T, D, lam, kappa, theta):
    sft = np.empty((int(lam.shape[1]),int(lam.shape[0]))) # declare sft
    for i in range(0,lam.shape[1]):
        for j in range(0,lam.shape[0]):
            # draw random number from poisson distribution
            nft = np.random.poisson(lam[j,i] * (T-t)) 
            # draw nft random numbers from a gamma distribution
            sft[i,j] = np.random.gamma(shape = (kappa[j] * nft), scale = (1 / theta[j]), size=1)
    #print(sft)
    p = prob(sft, D)
                  
    return p
    
# Parallelized simulations--------------------

# run len(t)*len(D) simulations
def parallelized_loop(i, j1, j2, prob_vec, t, T, D):
    for j in range (j1, j2): # for every D
        prob_vec[j] = sim(t,T, D[j-i*len(D)], lam, kappa, theta) # Determine the (i*len(D)+j)th element
        




start = time.perf_counter()

ncpu = multiprocessing.cpu_count()
print('ncpu:', ncpu)

pv = multiprocessing.Array('d', len(t)*len(D))

if __name__ == '__main__':
    for i in range (0, len(t)): # for every t
        
        # for every D:
        processes = []
        if (ncpu < len(D)):
            D_per_processor = int(len(D)/(ncpu-1))
            D_last_processor = int(len(D)%(ncpu-1))
            
            # Assign loops over D for the first n-1 processors
            for n in range(0, ncpu-1):
                j1 = i*len(D) + n * D_per_processor
                j2 = j1 + D_per_processor
                #print(i, "main-1", j1, j2)
                proc = multiprocessing.Process(target=parallelized_loop, 
                                             args=(i, j1, j2, pv, t[i], T, D))
                proc.start()
                processes.append(proc)
                
            # Assign loops over D for the last processor
            j1 = i*len(D) +(ncpu-1) * D_per_processor
            j2 = i*len(D) + len(D)
            #print(i, "main-2", j1, j2)
            proc = multiprocessing.Process(target=parallelized_loop, 
                                             args=(i, j1, j2, pv, t[i], T, D))
            proc.start()
            processes.append(proc)
        else:
            for n in range(0, len(D)):
                j1 = i*len(D) + n
                j2 = j1 + 1
                #print("main-3", j1, j2)
                proc = multiprocessing.Process(target=parallelized_loop, 
                                             args=(i, j1, j2, pv, t[i], T, D))
                proc.start()
                processes.append(proc)
        
        for process in processes:
            process.join()
    
    finish = time.perf_counter()
    
    prob_vec = pv[:]
    
    dtime = (finish - start) / 3600
    print(dtime)
    #print('probability vector:', prob_vec)
    
    prob_mtx2=np.array(prob_vec).reshape(100,100)
    pd.DataFrame(prob_mtx2).to_csv('prob_mtx_gamma_single_thrshld_grp4_1year.csv')
    pd.DataFrame(np.array(prob_vec)).to_csv('prob_vec_gamma_single_thrshld_grp4_1year.csv')






