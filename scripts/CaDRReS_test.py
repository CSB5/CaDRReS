__author__ = 'C. Suphavilai'

import pandas as pd
import numpy as np
import sys, os
import time
import pickle
import random
from scipy import stats

import argparse
np.set_printoptions(precision=3, suppress=True)

##############
# Parameters #
##############

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
parser.add_argument("ss_test_name")
parser.add_argument("cl_feature_fname")
parser.add_argument("out_dir")

# args = parser.parse_args(['../output/10D/seed0/lr0-01/CaDRReS_model.pickle', '../input/ccle_all_abs_ic50_bayesian_sigmoid.csv', '../input/ccle_cellline_pcor_ess_genes.csv', '../output'])
args = parser.parse_args()

##### Read the model #####
mdict = pickle.load(open(args.model_name, 'r'))

WP = mdict['WP']
WQ = mdict['WQ']
mu = mdict['mu']
b_p = mdict['b_p'] 
b_q = mdict['b_q']
err_list = mdict['err_list']
P_list = mdict['P_list']
Q_list = mdict['Q_list']

f = WP.shape[1]

out_dir = args.out_dir

##### Read data #####

ss_test_df = pd.read_csv(args.ss_test_name, index_col=0)
cl_features_df = pd.read_csv(args.cl_feature_fname, index_col=0)
cl_features_df.index = cl_features_df.index.astype(str)
cl_list = list(cl_features_df.index.astype(str))
ss_test_df = ss_test_df[ss_test_df.index.isin(cl_list)]

##############
# Prediction #
##############

new_out_fname = os.path.join(out_dir, 'CaDRReS_pred.csv')
new_out_dict_fname = os.path.join(out_dir, 'CaDRReS_pred.pickle')

P_test = list(ss_test_df.index)
n_test = len(P_test)
m_test = len(b_q)

X_test = np.matrix(cl_features_df.loc[P_test, P_list])
Y_test = np.matrix(np.identity(m_test))

##### Estimate b_p_test #####
num_seen_cl = len(set(P_list).intersection(ss_test_df.index))
if num_seen_cl == n_test:
    P_train = np.array(P_list)
    b_p_test = np.zeros(ss_test_df.shape[0])
    for u, cl in enumerate(ss_test_df.index):
        if cl in P_train:
            cl_train_idx = np.argwhere(P_train == cl)[0][0]
            b_p_test[u] = b_p[cl_train_idx]
        else:
            print 'ERROR: Unseen cell line, have to estimate b_p'
            sys.exit(1)
# if not all cell lines are seen, then estimate biases for every cell line
else:
    print 'Estimating biases for unseen samples'
    b_p_test = np.matrix(b_p) * X_test.T

##### Calculate prediction #####
Q_mat_test = Y_test * WQ
P_mat_test = X_test * WP

temp = mu + (Q_mat_test * P_mat_test.T).T
temp = temp + b_q
pred_mat = (temp.T + b_p_test).T

out_dict = {}
out_dict['P'] = P_mat_test
out_dict['Q'] = Q_mat_test
out_dict['mu'] = mu
out_dict['b_p'] = b_p_test
out_dict['b_q'] = b_q
pickle.dump(out_dict, open(new_out_dict_fname, 'w'))

pred_df = pd.DataFrame(pred_mat, columns=Q_list, index=ss_test_df.index)
# convert sensitivity score to IC50
pred_df *= -1
pred_df.to_csv(new_out_fname)
print 'Saved to', new_out_fname