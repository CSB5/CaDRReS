__author__ = 'Nok'
import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import sys
import os
import argparse

import pyjags
pyjags.load_module('glm')
import pymc


#scale to new range [a, b]
def scaler(x, min, max, a, b):
    return ((b-a)*(x-min)/(max-min)) + a

n_iter = 5000
burn_in = 500
n_chains = 3
n_threads = 5
p = 95

code = '''
model {
    for ( i in 1:Ntotal ) {
        # y[i] ~ dt( beta0[s[i]] + beta1[s[i]] * x[i] , 1/sigma^2 , nu )
        temp[i] <- (beta0[s[i]] - x[i]) * beta1[s[i]]
        y[i] ~ dt( 1/(1+2^temp[i]) , 1/sigma^2 , nu )
    }
    for ( j in 1:Nsubj ) {
        beta0[j] ~ dnorm( beta0mu , 1/(beta0sigma)^2 )
        # beta0[j] ~ dnorm( 0 , 1/(10)^2 )
        # beta1[j] ~ dnorm( beta1mu , 1/(beta1sigma)^2 )
        beta1[j] ~ dgamma(beta1lambda, beta1r)
        # beta1[j] ~ dgamma(1, 2)
    }
    # Priors vague on standardized scale:
    beta0mu ~ dnorm( 0 , 1/(10)^2 )
    beta0sigma ~ dunif( 1.0E-3 , 1.0E+3 )
    # beta1mu ~ dnorm( 1 , 1/(10)^2 )   # not in used
    beta1r ~ dnorm( 2 , 1/(10)^2 )
    beta1lambda ~ dnorm( 1 , 1/(10)^2 )
    # beta1sigma ~ dunif( 1.0E-3 , 1.0E+3 ) # not in used
    sigma ~ dunif( 1.0E-3 , 1.0E+3 )
    nu ~ dexp(1/30.0)
}
'''


################################
##### Read and select data #####

parser = argparse.ArgumentParser()
parser.add_argument("ccle_fname")
parser.add_argument("out_dir")
args = parser.parse_args()

ccle_df = pd.read_csv(args.ccle_fname, sep='\t')
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

cl_list = sorted(list(set(ccle_df['CCLE Cell Line Name'])))

for cl_name in cl_list:
    print cl_name, '...'

    cl_df = ccle_df[ccle_df['CCLE Cell Line Name'] == cl_name]

    ##########################
    ##### transform data #####

    temp_df = pd.DataFrame(columns=[0, 1, 2])
    for ri, drug_df in cl_df.iterrows():
        dosages = np.array(drug_df['Doses (uM)'].split(','), dtype=float)
        responses = np.array(drug_df['Activity Data (median)'].split(','), dtype=float)
        log2_dosage = np.log2(dosages)
        n_dosages = len(dosages)
        res_0to1 = scaler(responses, 0.0, -100.0, 0.0, 1.0)
        temp_df = temp_df.append(pd.DataFrame([log2_dosage, res_0to1, [drug_df['Compound']] * n_dosages]).transpose())

    temp_df.columns = ['X', 'Y', 'Subj']
    cl_df = temp_df

    ###############################
    ##### Construct JAGS model #####

    xName = "X"
    yName = "Y"
    sName = "Subj"

    RopeMuDiff = (-0.5, 0.5)
    RopeSdDiff = (-0.5, 0.5)
    RopeEff = (-0.1, 0.1)

    y = cl_df[yName]
    x = cl_df[xName]
    s = cl_df[sName]
    s_list = list(set(s))
    n_group = len(s_list)

    # map subject to [1,n_group]
    s_map_new_old = {}
    for si, s_name in enumerate(s_list):
        s_map_new_old[si+1] = s_name

    s_map_old_new = {}
    for si, s_name in enumerate(s_list):
        s_map_old_new[s_name] = si+1

    s = np.array([s_map_old_new[si] for si in s])

    data = dict(
        y = y,
        x = x,
        s = s,
        Nsubj = n_group,
        Ntotal = len(y)
    )

    model = pyjags.Model(code, data=data, chains=n_chains, adapt=burn_in, threads=n_threads)
    parameters = ["beta0", "beta1", "beta0mu", "sigma", "nu"]
    samples = model.sample(n_iter, vars=parameters)

    #########################
    ##### Calculate HDI #####

    results = []
    for i in range(n_group):
        d_name = s_map_new_old[i+1]
        beta0 = samples['beta0'][i].reshape(n_iter * n_chains, )
        beta0_mode = np.mean(pymc.utils.hpd(beta0, 0.99))
        beta0_HDI = pymc.utils.hpd(beta0, 1 - (p / 100.))
        beta0_ESS = pymc.diagnostics.effective_n(samples['beta0'][i])
        beta1 = samples['beta1'][i].reshape(n_iter * n_chains, )
        beta1_mode = np.mean(pymc.utils.hpd(beta1, 0.99))
        beta1_HDI = pymc.utils.hpd(beta1, 1-(p/100.))
        results += [[d_name, beta0_mode, beta0_HDI[0], beta0_HDI[1], beta0_ESS, beta1_mode, beta1_HDI[0], beta1_HDI[1]]]

    fout_name = os.path.join(out_dir, '{:s}.csv'.format(cl_name))
    results_df = pd.DataFrame(results, columns=['drug', 'abs_ic50', 'abs_ic50_low', 'abs_ic50_high', 'abs_ic50_ess', 'slope', 'slope_low', 'slope_high'])
    results_df.to_csv(fout_name, index=False)
    print 'Saved'
