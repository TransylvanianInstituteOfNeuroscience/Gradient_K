import numpy as np
import optuna
import os
import matplotlib.pyplot as plt
import scipy.ndimage.filters as flt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from numpy.linalg import norm
from statistics import mode
import math
import pdb

# Functions for execution of Gradient K algorithm
# Requirements        : scipy, sklearn, matplotlib, optuna

# Component functions
def bin_2d(sx, sy, n_bins, sigma):
    # computes the density space from
    # sx    : x coordinates
    # sy    : y coordinates
    # n_bins: number of boxes per dimension
    # sigma : standard deviation of smoothing kernel

    # define bins
    x_bins = np.linspace(np.min(sx) - 0.0001, np.max(sx), n_bins)
    y_bins = np.linspace(np.min(sy) - 0.0001, np.max(sy), n_bins)

    # sort points into bins
    sx_bin = np.digitize(sx, x_bins, right=True)
    sy_bin = np.digitize(sy, y_bins, right=True)

    # count number of points in each bin
    count = np.zeros([n_bins, n_bins])
    for x in range(n_bins):
        ind_x = np.argwhere(sx_bin==x)
        for y in range(n_bins):
            ind_y = np.argwhere(sy_bin==y)
            count[x,y] = np.intersect1d(ind_x, ind_y).shape[0]

    count_smooth = flt.gaussian_filter(count, sigma=sigma)
    S_bin = np.transpose(np.stack((sx_bin, sy_bin)))

    return S_bin, count, count_smooth

def choose_centers(S, n_clusters):
    # implementation of K++ initialization
    # S         : sample coordinate list
    # n_clusters: number of points to choose

    p    = np.zeros(n_clusters)
    ind  = np.random.randint(len(S))
    p[0] = ind

    for i in range(n_clusters - 1):
        # generate probability distribution from distance to last center chosen
        ind = int(p[i])
        current_point = np.reshape(S[ind,:],[1,2])
        D = np.square(cdist(current_point, S))
        D = np.squeeze(D/np.sum(D))

        # choose next point
        a = np.array(range(S.shape[0]))
        ind = np.random.choice(a=a, size=1, replace=True, p=D)
        p[i + 1] = ind

    p = p.astype(int)
    return p

def assign_box_2_clust(count, grad_x, grad_y, P, A, npoint):
    # assignment step of Gradient K algorithm
    # outputs the cluster of each box given:
    # count : number of points per box
    # grad_x: estimated gradient along the x axis
    # grad_y: estimated gradient along the y axis
    # P     : current cluster centers
    # A     : Angle importance parameter (alpha)
    # npoint: number of clusters

    nbins  = grad_x.shape[0]
    bclust = np.zeros(grad_x.shape)

    for x in range(nbins):
        for y in range(nbins):
            if count[x,y] != 0:
                scale = np.zeros([1,npoint])
                grad_vec      = [grad_x[x, y], grad_y[x, y]]

                for point in range(npoint):
                    current_point = P[point, :]
                    dist_vec      = [current_point[1] - y, current_point[0] - x]

                    cos_theta    = np.dot(grad_vec, dist_vec) / (norm(grad_vec) * norm(dist_vec))
                    scale[0,point] = np.arccos(cos_theta)

                scale = (scale/math.pi) * A + 1
                a     = [[x,y]]
                D     = cdist(a, P)
                bclust[x, y] = np.argmin(scale * D / np.max(D))

    return bclust

def compute_new_centers(bclust, density, n_clusters):
    # computes new cluster centers given:
    # bclust    : the current estimation of the cluster center for each box
    # density   : estimated density space
    # n_clusters: number of clusters

    new_point = np.zeros([n_clusters,2])

    for point in range(n_clusters):
        ind    = np.argwhere(bclust == point)
        tot_np = 0.1
        wsum   = np.array([0., 0.])

        for box in range(ind.shape[0]):
            x = ind[box, 0]
            y = ind[box, 1]
            box_count = density[x, y]

            wsum     += np.array([x, y]) * box_count
            tot_np   += box_count

        new_point[point,:] = wsum/tot_np

    return new_point

def move_2_pointspace(bclust, S_bin):
    # returns the cluster of each sample given:
    # bclust: the cluster of each box
    # S_bin : the box of each sample

    n = S_bin.shape[0]
    color = np.zeros([S_bin.shape[0],1])

    for i in range(n):
        x = S_bin[i,0]
        y = S_bin[i,1]
        color[i,0] = bclust[x, y]

    return np.squeeze(color)

def assign_point_2_clust(S, start_point):
    # Assigns samples K-MEANS clusters (for comparison)
    # S          : sample coordinate list
    # start_point: current cluster centers

    D = cdist(S, start_point)
    clust = np.argmin(D,1)
    return clust

def k_center_update(S, clust, n_clusters):
    # Updates location of K-MEANS cluster centers (for comparison)
    # S         : sample coordinate list
    # clust     : current cluster center coordinates
    # n_clusters: number of clusters

    new_point = np.zeros([n_clusters,2])

    for i in range(n_clusters):
        ind = np.squeeze(np.argwhere(clust == i))
        points_in_cluster = S[ind, :]
        new_point[i,:] = np.mean(points_in_cluster, 0)

    return new_point

def opt_start(S, ground):
    # computes real cluster center locations from ground truth:
    # S     : sample coordinate list
    # ground: ground truth cluster of each sample

    uni = np.unique(ground)

    p = []
    for val in uni:
        ind = np.squeeze(np.argwhere(ground==val))

        center = np.reshape(np.mean(S[ind,:], axis=0),[1,2])

        D = np.squeeze(cdist(center, S))
        p.append(np.argmin(D))

    p = np.array(p).astype(int)
    return p

# Big functions
def iteration(count, grad_x, grad_y, P, A, density, n_clusters):
    # executes one Gradient-K iteration
    bclust = assign_box_2_clust(count, grad_x, grad_y, P, A, n_clusters)
    point  = compute_new_centers(bclust, density, n_clusters)
    return bclust, point

def k_iteration(S, start_point, n_clusters):
    # executes one K-MEANS iteration
    clust     = assign_point_2_clust(S, start_point)
    new_point = k_center_update(S,clust,n_clusters)
    return clust, new_point

def AccuracyVSGround(guess, ground):
    # computes best case accuracy versus ground truth
    # guess : output cluster for each sample
    # ground: ground Truth

    new_color = np.copy(guess)

    # sort groundtruth clusters by density
    uni, counts = np.unique(ground, return_counts=True)
    ind         = np.argsort(counts)[::-1]
    uni         = uni[ind]

    for val in uni:

        # select all points of a gt cluster
        ind = np.squeeze(np.argwhere(ground == val))
        c_colors = guess[ind]

        # remove clustered or noisy points
        ind2 = np.squeeze(np.argwhere(c_colors >= 0))
        c_colors2 = c_colors[ind2]
        if c_colors2.size == 0:
            correct = -10
        else:
            if c_colors2.size == 1:
                c_colors2 = np.array([c_colors2])
            c_uni, count = np.unique(c_colors2, return_counts=True)
            c_ind        = np.squeeze(np.argwhere(count == np.max(count)))

            if c_ind.size == 1:
                c_ind = np.array([c_ind])

            if len(c_ind) == 1:
                correct = c_uni[c_ind]
            else:
                colorset = c_uni[c_ind]
                minclr   = np.Inf
                for clr in colorset:
                    colorind = np.squeeze(np.argwhere(guess == clr))
                    if len(colorind) < minclr:
                        correct = clr
                        minclr  = len(colorind)
            #correct  = mode(c_colors2)

        # rename points
        ind3 = np.squeeze(np.argwhere(c_colors2 == correct))
        c_colors2[:]    = -2 # original non-noise misclassified
        c_colors2[ind3] = val

        # add back to original list
        c_colors[ind2] = c_colors2
        new_color[ind] = c_colors

        # mark samples as clustered in guess
        ind = np.squeeze(np.argwhere(guess==correct))
        guess[ind] = -2

    # compute accuracy
    tot = len(guess)
    correct = np.sum(new_color == ground)
    acc     = correct / tot * 100

    return acc, new_color

class GradK_objective(object):
    # Constructor of Objective function for optimization
    def __init__(self, sx, sy, gt, maxiter_gk=200, mindelta_gk=0.01):
        self.sx      = sx
        self.sy      = sy
        self.s       = np.transpose(np.stack((sx, sy)))
        self.gt      = gt
        self.maxiter = maxiter_gk
        self.mindelt = mindelta_gk

    def __call__(self, trial):
        sx       = self.sx
        sy       = self.sy
        S        = self.s
        color_gt = self.gt
        maxiter  = self.maxiter

        numboxes   = trial.suggest_int('numboxes', 10, 100)
        sig_smooth = trial.suggest_float("sig", 0, 1)
        alpha_par  = trial.suggest_float("alpha_par", 0, 50)
        mindelta   = self.mindelt * numboxes
        numcenters = len(np.unique(color_gt))
        global maxp

        # Prepare data
        S_bin, count, count_smooth = bin_2d(sx, sy, numboxes, sig_smooth)

        # compute gradient
        grad_y, grad_x = np.gradient(count_smooth)

        # initialize datastores
        acc      = np.zeros([20])
        for run in range(20):

            # Choose initial points
            if run == 0:
                ind = opt_start(S, color_gt)
            else:
                ind = choose_centers(S, numcenters)
            point = S_bin[ind,:]

            # grad-k
            new_point = point
            iter      = 0
            delta     = 1000
            while (iter < maxiter) and (delta > mindelta):

                a = new_point
                bclust, new_point = iteration(count, grad_x, grad_y, \
                                        new_point, alpha_par , count, numcenters)

                diff = np.square(a - new_point)
                delta = np.sum(np.sqrt(np.sum(diff, 1)))
                iter += 1

            # color points as function of box
            color       = move_2_pointspace(bclust, S_bin)
            acc[run], _ = AccuracyVSGround(color, color_gt)

            if run == 0:
                if acc[run] < maxp:
                    raise optuna.TrialPruned()
                else:
                    maxp = acc[run]

        return 100 - np.mean(acc)

def clustplot(ax, sx, sy, color):
    # plots clustering results given:
    # ax    : axes on which to plot
    # sx    : x coordinates
    # sy    : y coordinates
    # color : cluster of each sample

    # misclustered points are grey
    ind = np.squeeze(np.argwhere(color == -2))
    ax.scatter(sx[ind], sy[ind], c = 'darkgray', marker='.',s=100 , alpha=0.5)

    # noise points are black
    ind = np.squeeze(np.argwhere(color == -1))
    ax.scatter(sx[ind], sy[ind], c = 'k', marker='.',s=100 , alpha=0.5)

    # rest of the points are colored
    ind = np.squeeze(np.argwhere(color >= 0))
    ax.scatter(sx[ind], sy[ind], c = color[ind], marker='.',s=100 , alpha=0.5, cmap='jet')

# Main function
def optimize_GradK_params(sx, sy, color_gt):
    # finds parameters such that output of gradient K best matches ground Truth
    # sx      : x coordinates
    # sy      : y coordinates
    # color_gt: ground truth

    # Preproc data
    sx = sx - np.min(sx)
    sy = sy - np.min(sy)
    sx = sx / np.max(sx)
    sy = sy / np.max(sy)
    S  = np.transpose(np.stack((sx, sy)))
    global maxp

    # optimize parameters
    val       = 100
    iter      = 0
    maxiter_h = 1000
    maxp      = 0
    objective = GradK_objective(sx, sy, color_gt)
    study     = optuna.create_study()

    while (iter < maxiter_h) & (val > 0):
        study.optimize(objective, n_trials=1)
        iter += 1
        val   = study.best_value

    out_para = study.best_params
    out_para['n_clust'] = len(np.unique(color_gt))
    return out_para

def Gradient_k(sx, sy, params, maxiter=200, mindelta=0.01):
    # Executes Gradient-K algorithm
    # sx      : x coordinates
    # sy      : y coordinates
    # params  : params['sig']       : smoothing parameter sigma
    #           params['numboxes']  : number of boxes per dimension
    #           params['alpha_par'] : angle importance parameter alpha
    #           params['n_clust']   : number of cluster centers
    # maxiter : maximum number of iterations
    # mindelta: minimum cluster shift (% of numboxes)

    sig      = params['sig']
    n_box    = params['numboxes']
    alpha    = params['alpha_par']
    n_clust  = params['n_clust']
    mindelta = mindelta * n_box

    # Preproc data
    sx = sx - np.min(sx)
    sy = sy - np.min(sy)
    sx = sx / np.max(sx)
    sy = sy / np.max(sy)
    S  = np.transpose(np.stack((sx, sy)))

    # Preproc grad k
    S_bin, count, count_smooth = bin_2d(sx, sy, n_box, sig)
    grad_y, grad_x = np.gradient(count_smooth)

    # Choose initial points
    ind = choose_centers(S, n_clust)
    point = S_bin[ind,:]

    # Gradient-k run
    new_point_a = point
    iter_a      = 0
    delta       = 1000
    while (iter_a < maxiter) and (delta > mindelta):

        a_a = new_point_a
        bclust_a, new_point_a = iteration(count, grad_x, grad_y, new_point_a, alpha , count_smooth, n_clust)

        diff = np.square(a_a - new_point_a)
        delta = np.sum(np.sqrt(np.sum(diff, 1)))
        iter_a += 1

    color_gradk_o = move_2_pointspace(bclust_a, S_bin)

    print('Gradient_k completed execution in {} iterations'.format(iter_a))
    return color_gradk_o
