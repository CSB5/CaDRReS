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
parser.add_argument("ss_name")
parser.add_argument("ss_test_name")
parser.add_argument("cl_feature_fname")
parser.add_argument("drug_list_fname")
parser.add_argument("out_dir")
parser.add_argument("f")
parser.add_argument("max_iterations")
parser.add_argument("l_rate")
parser.add_argument("seed")

# args = parser.parse_args(['../input/ccle_all_abs_ic50_bayesian_sigmoid.csv', '../input/ccle_all_abs_ic50_bayesian_sigmoid.csv', '../input/ccle_cellline_pcor_ess_genes.csv', '../misc/ccle_drugMedianGE0.txt', '../output', '10', '50', '0.01', '0'])
args = parser.parse_args()

print args
out_dict = {}
out_dict['args'] = args

f = int(args.f)
l_rate = args.l_rate
seed = int(args.seed)

ss_name = args.ss_name
ss_test_name = args.ss_test_name
cl_feature_fname = args.cl_feature_fname
drug_list_fname = args.drug_list_fname

###########################
# Create output directory #
###########################

out_name = "{:s}/{:d}D/seed{:d}/lr{:s}".format(args.out_dir, f, seed, str(l_rate).replace('.', '-'))
if os.path.isdir(out_name):
    pass
else:
    os.makedirs(out_name)
print out_name

##################
# Read drug list #
##################

if os.path.isfile(drug_list_fname):
    selected_drugs = list(pd.read_csv(drug_list_fname, header=None)[0].values.astype(str))
else:
    print 'ERROR: invalid drug list file'
    sys.exit(1)

#############################
# Read train and test files #
#############################

cl_features_df = pd.read_csv(cl_feature_fname, index_col=0)

##### Training data #####

ss_df = pd.read_csv(ss_name, index_col=0)
ss_df.index = ss_df.index.astype(str)

# Convert IC50 to sensitivity score
ss_df *= -1

drug_list = list(ss_df.columns)
drug_list = [d for d in drug_list if d in selected_drugs]

cl_features_df.index = cl_features_df.index.astype(str)
cl_list = list(cl_features_df.index.astype(str))
# select only predictable cell lines and drugs
ss_df = ss_df[ss_df.index.isin(cl_list)][drug_list]

##### Testing data #####

ss_test_df = pd.read_csv(ss_test_name, index_col=0)
ss_test_df.index = ss_test_df.index.astype(str)
# select only predictable cell lines and drugs
ss_test_df = ss_test_df[ss_test_df.index.isin(cl_list)][ss_df.columns]

print ss_df.shape, ss_test_df.shape

#######################################
# Remove cell lines with no drug info #
#######################################


print 'TRAIN: cell lines =', ss_df.shape[0], ' # after remove empty rows =', ss_df.dropna(how='all').shape[0], ' (for subset of drugs)'
print 'TEST: cell lines =', ss_test_df.shape[0], ' # after remove empty rows =', ss_test_df.dropna(how='all').shape[0], ' (for subset of drugs)'

ss_df = ss_df.dropna(how='all')
ss_test_df = ss_test_df.dropna(how='all')

##### Get features for training cell lines #####

cl_features_df = cl_features_df[ss_df.index]

print 'start training a model ...'

##############
# Initiation #
##############

P = list(ss_df.index)   # cell lines
n = len(P)
Q = list(ss_df.columns) # drugs
m = len(Q)

R = np.matrix(ss_df)

n_K = np.sum(~np.isnan(R))
X = np.matrix(cl_features_df.loc[P])
xdim = X.shape[1] # number of features
# Use identity matrix as default drug features (aka. learn directly to q_i)
Y = np.matrix(np.identity(m))
ydim = Y.shape[1]

prng = np.random.RandomState(seed)
WP = (np.matrix(prng.rand(xdim, f))-0.5) / 10.
prng = np.random.RandomState(seed)
WQ = (np.matrix(prng.rand(ydim, f))-0.5) / 10.

# weigt of each sample
WE = np.matrix(np.ones(R.shape))

mu = np.nanmean(R)
b_p = np.zeros(n)
b_q = np.zeros(m)

out_dict['WP_init'] = WP.copy()
out_dict['WQ_init'] = WQ.copy()
out_dict['mu_init'] = mu
out_dict['b_p_init'] = b_p.copy()
out_dict['b_q_init'] = b_q.copy()

##################
# Util functions #
##################

def score_to_exact_rank(s):
    return (-1*s).argsort().argsort()

def cal_exact_avg_ndcg(pred, R):
    all_ndcg = []
    for u in range(R.shape[0]):
        test_drug_bool = ~np.isnan(R[u,:])
        s_u = R[u,:][test_drug_bool]
        r_u = score_to_exact_rank(s_u)
        s_u_pred = pred[u,:][test_drug_bool]
        r_u_pred = score_to_exact_rank(s_u_pred)
        G_u_max = np.sum((np.power(2, s_u)) / np.log(r_u + 2))
        G_u = np.sum((np.power(2, s_u)) / np.log(r_u_pred + 2))
        # print G_u, G_u_max, G_u / G_u_max
        all_ndcg += [G_u / G_u_max]
    return np.mean(all_ndcg)

def save_and_predict(ss_df, ss_test_df, cl_features_df, X, WP, WQ, mu, b_p, b_q, err_list, epoch, out_name, suffix):       

    out_dict['WP'] = WP
    out_dict['WQ'] = WQ
    out_dict['mu'] = mu
    out_dict['b_p'] = b_p
    out_dict['b_q'] = b_q
    out_dict['err_list'] = err_list
    out_dict['P_list'] = list(ss_df.index)
    out_dict['Q_list'] = list(ss_df.columns)

    f = WP.shape[1]

    ###############################
    # Prediction on training data #
    ###############################

    print suffix

    if suffix == 'end':

        P_train = list(ss_df.index)
        X_train = np.matrix(cl_features_df.loc[P_train])
        Y_train = np.matrix(np.identity(m))

        out_dict['X_train'] = X_train
        out_dict['Y_train'] = Y_train

        ##### Save pred for training data #####
        
        Q_mat_train = Y_train * WQ
        P_mat_train = X_train * WP

        temp = mu + (Q_mat_train * P_mat_train.T).T
        temp = temp + b_q
        train_pred_mat = (temp.T + b_p).T
        train_pred_df = pd.DataFrame(train_pred_mat, columns=ss_df.columns, index=ss_df.index)
        # convert sensitivity score to IC50
        train_pred_df *= -1

        new_out_name_train = os.path.join(out_name, 'CaDDReS_pred_end_train.csv')
        train_pred_df.to_csv(new_out_name_train)

        pd.DataFrame(P_mat_train, index=ss_df.index, columns=range(1, f+1)).to_csv(os.path.join(out_name, 'CaDDReS_P.csv'))
        pd.DataFrame(Q_mat_train, index=ss_df.columns, columns=range(1, f+1)).to_csv(os.path.join(out_name, 'CaDDReS_Q.csv'))

    ##############
    # Prediction #
    ##############

    if suffix == 'end':
        new_out_fname = os.path.join(out_name, 'CaDDReS_pred_end.csv')
        new_out_dict_fname = os.path.join(out_name, 'CaDDReS_model.pickle')
    else:
        new_out_fname = os.path.join(out_name, 'CaDDReS_pred_{:d}_ckpt.csv'.format(epoch))
        new_out_dict_fname = os.path.join(out_name, 'CaDDReS_pred_{:d}_ckpt.pickle'.format(epoch)) 


    P_test = list(ss_test_df.index)
    n_test = len(P_test)
    m_test = len(b_q)

    X_test = np.matrix(cl_features_df.loc[P_test])
    Y_test = np.matrix(np.identity(m_test))

    out_dict['X_test'] = X_test
    out_dict['Y_test'] = Y_test

    ##### save the model and other variables #####
    pickle.dump(out_dict, open(new_out_dict_fname, 'w'))



    ##### Estimate b_p_test #####
    num_seen_cl = len(set(ss_df.index).intersection(ss_test_df.index))
    if num_seen_cl == n_test:
        P_train = ss_df.index
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
        print "Estimating cell line biases ..."
        b_p_test = np.matrix(b_p) * X_test.T

    ##### Calculate prediction #####

    Q_mat_test = Y_test * WQ
    P_mat_test = X_test * WP
    
    temp = mu + (Q_mat_test * P_mat_test.T).T
    temp = temp + b_q
    pred_mat = (temp.T + b_p_test).T

    pred_df = pd.DataFrame(pred_mat, columns=ss_test_df.columns, index=ss_test_df.index)
    # convert sensitivity score to IC50
    pred_df *= -1
    pred_df.to_csv(new_out_fname)
    print 'Saved to', new_out_fname

############
# Training #
############

# Set up parameters
max_iterations = int(args.max_iterations)
split_saving = np.min([1000, max_iterations])
l_rate = float(args.l_rate)

out_dict['max_iterations'] = max_iterations
out_dict['l_rate'] = l_rate

current_err = np.inf
err_list = []
is_diverged = False

for epoch in range(1, max_iterations + 1):

    # save old results
    old_WP = WP.copy()
    old_WQ = WQ.copy()
    old_mu = mu
    old_b_p = b_p.copy()
    old_b_q = b_q.copy()

    ##### make a prediction #####

    pred = (((mu + (Y * WQ * WP.T * X.T).T) + b_q).T + b_p).T

    ##### gradient descent #####

    for u in range(n):
        err = np.nansum(np.multiply(R[u,:] - pred[u,:], WE[u,:]))/n_K
        b_p[u] += l_rate * err

    for i in range(m):
        err = np.nansum(np.multiply(R[:, i] - pred[:, i], WE[:, i])) / n_K
        b_q[i] += l_rate * err

    # Update WP
    temp = np.matrix(np.zeros(WP.shape))
    for i in range(m):
        q_i = Y[i, :] * WQ
        err_per_i = (np.multiply(R[:, i] - pred[:, i], WE[:, i])).T
        err_per_i[np.isnan(err_per_i)] = 0
        temp += (q_i.T * np.sum(np.multiply(err_per_i.T, X), axis=0)).T
    temp = temp / n_K
    WP += l_rate * temp

    # Update WQ
    temp = np.matrix(np.zeros(WQ.shape))
    for u in range(n):
        p_u = X[u, :] * WP
        err_per_u = np.multiply(R[u,:] - pred[u,:], WE[u,:])
        err_per_u[np.isnan(err_per_u)] = 0
        temp += (p_u.T * np.sum(np.multiply(err_per_u.T, Y), axis=0)).T
    temp = temp / n_K
    WQ += l_rate * temp

    new_err = np.sqrt(np.nansum(np.square(R - pred)) / n_K)
    new_ndcg = cal_exact_avg_ndcg(pred, R)
    err_list += [[epoch, new_err, new_ndcg]]

    percent_improved = (current_err-new_err)/current_err*100
    print '{:d}\t{:.3f}\t({:.3f}%)\t{:.3f}'.format(epoch, new_err, percent_improved, new_ndcg)

    if new_err < current_err:
        current_err = new_err
    else:
        print 'Stop before diverging and set values back to previous epoch'
        WP = old_WP
        WQ = old_WQ
        mu = mu
        b_p = old_b_p
        b_q = old_b_q
        save_and_predict(ss_df, ss_test_df, cl_features_df, X, WP, WQ, mu, b_p, b_q, err_list, epoch, out_name, 'end')
        break

    if epoch % split_saving == 0:
        if epoch < max_iterations:
            save_and_predict(ss_df, ss_test_df, cl_features_df, X, WP, WQ, mu, b_p, b_q, err_list, epoch, out_name, 'run')
        else:
            save_and_predict(ss_df, ss_test_df, cl_features_df, X, WP, WQ, mu, b_p, b_q, err_list, epoch, out_name, 'end')