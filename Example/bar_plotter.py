import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pdb

def prep_stats(data):
    mu = np.mean(data)
    sd = np.std(data)
    return mu, sd

def prep_stats_mdn(data):
    med = np.median(data)
    q1  = np.abs(np.percentile(data, 25) - med)
    q3  = np.percentile(data, 75) - med
    q   = np.array([q1, q3])
    #pdb.set_trace()
    return med, q


datasets = ['jain', 'unbl', 'aggr', 's4', 'g2_30', 'g2_50', 'synth_spike_2']

# Load data
base = 'E:\\Andrei\\Clust\\'
ana_type = 'median'

# initialize performance averages
perf_gradk_0_mu = np.zeros(len(datasets))
perf_gradk_o_mu = np.zeros(len(datasets))
perf_kmeans_mu  = np.zeros(len(datasets))
perf_dbscan_mu  = np.zeros(len(datasets))

# initialize performance errors
if ana_type == 'median':
    perf_gradk_0_sd = np.zeros([2, len(datasets)])
    perf_gradk_o_sd = np.zeros([2, len(datasets)])
    perf_kmeans_sd  = np.zeros([2, len(datasets)])
    perf_dbscan_sd  = np.zeros([2, len(datasets)])

if ana_type == 'mean':
    perf_gradk_0_sd = np.zeros(len(datasets))
    perf_gradk_o_sd = np.zeros(len(datasets))
    perf_kmeans_sd  = np.zeros(len(datasets))
    perf_dbscan_sd  = np.zeros(len(datasets))

# initialize iteration averages
iter_gradk_0_mu = np.zeros(len(datasets))
iter_gradk_o_mu = np.zeros(len(datasets))
iter_kmeans_mu  = np.zeros(len(datasets))

# initialize iteration errors
if ana_type == 'median':
    iter_gradk_0_sd = np.zeros([2, len(datasets)])
    iter_gradk_o_sd = np.zeros([2, len(datasets)])
    iter_kmeans_sd  = np.zeros([2, len(datasets)])

if ana_type == 'mean':
    iter_gradk_0_sd = np.zeros(len(datasets))
    iter_gradk_o_sd = np.zeros(len(datasets))
    iter_kmeans_sd  = np.zeros(len(datasets))

i = 0
for dataset in datasets:
    fpath = base + dataset + '-groundtruth\\v2'
    os.chdir(fpath)

    perf_gradk_0 = np.load('acc_gradk_0.npy')
    perf_gradk_o = np.load('acc_gradk_o.npy')
    perf_kmeans  = np.load('acc_kmeans.npy')
    perf_dbscan  = np.load('acc_dbscan.npy')

    iter_gradk_0 = np.load('itr_gradk_0.npy')
    iter_gradk_o = np.load('itr_gradk_o.npy')
    iter_kmeans  = np.load('itr_kmeans.npy')

    if ana_type == 'median':
        perf_gradk_0_mu[i], perf_gradk_0_sd[:,i] = prep_stats_mdn(perf_gradk_0)
        perf_gradk_o_mu[i], perf_gradk_o_sd[:,i] = prep_stats_mdn(perf_gradk_o)
        perf_kmeans_mu[i], perf_kmeans_sd[:,i]   = prep_stats_mdn(perf_kmeans)
        perf_dbscan_mu[i], perf_dbscan_sd[:,i]   = prep_stats_mdn(perf_dbscan)

        iter_gradk_0_mu[i], iter_gradk_0_sd[:,i] = prep_stats_mdn(iter_gradk_0)
        iter_gradk_o_mu[i], iter_gradk_o_sd[:,i] = prep_stats_mdn(iter_gradk_o)
        iter_kmeans_mu[i], iter_kmeans_sd[:,i]   = prep_stats_mdn(iter_kmeans)

    if ana_type == 'mean':
        perf_gradk_0_mu[i], perf_gradk_0_sd[i] = prep_stats(perf_gradk_0)
        perf_gradk_o_mu[i], perf_gradk_o_sd[i] = prep_stats(perf_gradk_o)
        perf_kmeans_mu[i], perf_kmeans_sd[i]   = prep_stats(perf_kmeans)
        perf_dbscan_mu[i], perf_dbscan_sd[i]   = prep_stats(perf_dbscan)

        iter_gradk_0_mu[i], iter_gradk_0_sd[i] = prep_stats(iter_gradk_0)
        iter_gradk_o_mu[i], iter_gradk_o_sd[i] = prep_stats(iter_gradk_o)
        iter_kmeans_mu[i], iter_kmeans_sd[i]   = prep_stats(iter_kmeans)

    i += 1

# all datasets
x_gradk_o = np.array([1, 6, 11, 16, 21, 26, 31])
x_gradk_0 = x_gradk_o + 1
x_kmeans  = x_gradk_o + 2
x_dbscan  = x_gradk_o + 3

w   = -1
bott = 35
datasets[-1] = 'spike'
lbl = datasets

plt.rc('font',family='Times New Roman', size=14)

f, (ax1, ax2) = plt.subplots(2, 1)

p1 = ax1.bar(x_gradk_o, perf_gradk_o_mu - bott,align='edge', bottom=bott, width=w, color='purple', yerr=perf_gradk_o_sd)
p2 = ax1.bar(x_gradk_0, perf_gradk_0_mu - bott,align='edge', bottom=bott, width=w, color='blue' , yerr=perf_gradk_0_sd, tick_label=lbl)
p3 = ax1.bar(x_kmeans, perf_kmeans_mu - bott,align='edge', bottom=bott, width=w, color='green' , yerr=perf_kmeans_sd)
p4 = ax1.bar(x_dbscan, perf_dbscan_mu - bott,align='edge', bottom=bott, width=w, color='red'   , yerr=perf_dbscan_sd)
ax1.set_ylabel('Accuracy [%]')
ax1.set_title('Accuracy', fontsize=14)

ax2.bar(x_gradk_o, iter_gradk_o_mu, width=w, color='purple', yerr=iter_gradk_0_sd)
ax2.bar(x_gradk_0, iter_gradk_0_mu, width=w, color='blue'  , yerr=iter_gradk_o_sd, tick_label=lbl)
ax2.bar(x_kmeans, iter_kmeans_mu, width=w, color='green'   , yerr=iter_kmeans_sd)
ax2.set_ylabel('Iterations')
ax2.set_title('Iterations', fontsize=14)
ax2.legend((p1[0],p2[0], p3[0], p4[0]), ('Gradient-k', 'K-means (Box space)', 'K-means', 'DBSCAN'), loc='upper left')

for i in [ax1, ax2]:
    plt.sca(i)
    plt.xticks(rotation=30)

f.set_size_inches(8.3, 8.3)
f.tight_layout()
plt.show()
