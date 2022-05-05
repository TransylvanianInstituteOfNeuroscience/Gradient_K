import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import clust_fun as cf
import os

# Input/Output Control
path   = 'E:\\Andrei\\Clust\\'
e_path = '-groundtruth\\'
out    = 'v2\\'

# Dataset parameters
datasets  = ['jain' , 'unbl' , 'aggr' , 's4'   , 'g2_30', 'g2_50', 'synth_spike_2']
n_clust_l = [2      , 8      , 7      , 15     , 2      , 2      , 21     ]
maxiter   = 200
mindelta  = 0.01
n_runs    = 500

n_box_l   = [14     , 26     , 16     , 45     , 25     , 31     , 42     ]
sig_l     = [0.85581, 0.77081, 0.97788, 0.99968, 0.64005, 0.67587, 0.00086]
alpha_l   = [9.75935, 21.1027, 10.7597, 10.3244, 2.73753, 0.06552, 4.90508]

eps_l     = [0.07497, 0.06142, 0.04131, 0.00940, 0.03295, 0.07467, 0.00651]
min_pts_l = [3      , 3      , 6      , 2      , 7      , 2      , 5      ]

for i in range(len(datasets)):
    # Load parameters
    ds      = datasets[i]
    sig     = sig_l[i]
    n_box   = n_box_l[i]
    alpha   = alpha_l[i]
    n_clust = n_clust_l[i]
    eps     = eps_l[i]
    min_pts = min_pts_l[i]
    mindelta_gr = mindelta * n_box

    # Load data
    file     = path + ds + e_path
    sx       = np.load(file + 'sx.npy')
    sy       = np.load(file + 'sy.npy')
    color_gt = np.load(file + 'gt.npy')

    # Preproc data
    sx = sx - np.min(sx)
    sy = sy - np.min(sy)
    sx = sx / np.max(sx)
    sy = sy / np.max(sy)
    S  = np.transpose(np.stack((sx, sy)))

    # Preproc grad k
    S_bin, count, count_smooth = cf.bin_2d(sx, sy, n_box, sig)
    grad_y, grad_x = np.gradient(count_smooth)

    # Initialize output vars
    acc_gradk_o = np.zeros([n_runs])
    acc_gradk_0 = np.zeros([n_runs])
    acc_kmeans  = np.zeros([n_runs])
    acc_dbscan  = np.zeros([n_runs])

    itr_gradk_o = np.zeros([n_runs])
    itr_gradk_0 = np.zeros([n_runs])
    itr_kmeans  = np.zeros([n_runs])
    itr_dbscan  = np.zeros([n_runs])

    for run in range(n_runs):
        # Choose initial points
        ind = cf.choose_centers(S, n_clust)
        point = S_bin[ind,:]

        # Gradient-k run
        new_point_a = point
        iter_a      = 0
        delta       = 1000
        while (iter_a < maxiter) and (delta > mindelta_gr):

            a_a = new_point_a
            bclust_a, new_point_a = cf.iteration(count, grad_x, grad_y, new_point_a, alpha , count_smooth, n_clust)

            diff = np.square(a_a - new_point_a)
            delta = np.sum(np.sqrt(np.sum(diff, 1)))
            iter_a += 1

        # Gradient-k 0 imp run
        new_point_0_a = point
        iter_0_a      = 0
        delta       = 1000
        while (iter_0_a < maxiter) and (delta > mindelta_gr):

            a_0_a = new_point_0_a
            bclust_0_a, new_point_0_a = cf.iteration(count, grad_x, grad_y, new_point_0_a, 0 , count_smooth, n_clust)

            diff = np.square(a_0_a - new_point_0_a)
            delta = np.sum(np.sqrt(np.sum(diff, 1)))
            iter_0_a += 1

        # K-means clust
        new_point_k = S[ind]
        iter_k      = 0
        delta     = 1000
        while (iter_k < maxiter) and (delta > mindelta):

            a_k = new_point_k
            color_k, new_point_k = cf.k_iteration(S, new_point_k, n_clust)

            diff = np.square(a_k - new_point_k)
            delta = np.sum(np.sqrt(np.sum(diff, 1)))
            iter_k += 1

        # DBSCAN clust
        clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(S)
        color_dbs  = clustering.labels_

        # move to pointspace
        color_gradk_o = cf.move_2_pointspace(bclust_a, S_bin)
        color_gradk_0 = cf.move_2_pointspace(bclust_0_a, S_bin)

        # Compute accuracies
        acc_gradk_o[run], _ = cf.AccuracyVSGround(color_gradk_o, color_gt)
        acc_gradk_0[run], _ = cf.AccuracyVSGround(color_gradk_0, color_gt)
        acc_kmeans[run], _  = cf.AccuracyVSGround(color_k, color_gt)
        acc_dbscan[run], _  = cf.AccuracyVSGround(color_dbs, color_gt)

        # Save n_iters
        itr_gradk_o[run] = iter_a
        itr_gradk_0[run] = iter_0_a
        itr_kmeans [run] = iter_k


        # Progress display
        print('{} End run {}'.format(ds, run + 1))

    # Make otuput folder
    #os.mkdir(file + out)

    # Save outputs
    np.save(file + out + 'acc_gradk_o.npy', acc_gradk_o)
    np.save(file + out + 'acc_gradk_0.npy', acc_gradk_0)
    np.save(file + out + 'acc_kmeans.npy', acc_kmeans)
    np.save(file + out + 'acc_dbscan.npy', acc_dbscan)
    np.save(file + out + 'itr_gradk_o.npy', itr_gradk_o)
    np.save(file + out + 'itr_gradk_0.npy', itr_gradk_0)
    np.save(file + out + 'itr_kmeans.npy', itr_kmeans)
