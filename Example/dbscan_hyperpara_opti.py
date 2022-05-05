import optuna
import clust_fun as cf
from sklearn.cluster import DBSCAN
import numpy as np
import pdb

# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial):

    # Variable Parameters
    eps     = trial.suggest_float("eps", 0, 0.2)
    min_pts = trial.suggest_int('min_pts', 2, 7)

    # final plot
    clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
    color_k    = clustering.labels_

    # match clusters to gt
    acc, _ = cf.AccuracyVSGround(color_k, color_gt)

    return 100 - np.mean(acc)

#parameters
path   = 'E:\\Andrei\\Clust\\'
e_path = '-groundtruth\\'
datasets    = ['jain', 'unbl', 'aggr', 's4', 'g2_30', 'g2_50', 'synth_spike_2']

all_eps = np.zeros([len(datasets)])
all_pts = np.zeros([len(datasets)])
for i in range(len(datasets)):
    ds   = datasets[i]
    file = path + ds + e_path
    sx       = np.load(file + 'sx.npy')
    sy       = np.load(file + 'sy.npy')
    color_gt = np.load(file + 'gt.npy')

    # Preproc data
    sx = sx - np.min(sx)
    sy = sy - np.min(sy)
    sx = sx / np.max(sx)
    sy = sy / np.max(sy)
    X  = np.transpose(np.stack((sx, sy)))

    # optimize parameters
    val     = 100
    iter    = 0
    maxiter = 2000
    study = optuna.create_study()

    while (iter < maxiter) & (val > 0):
        study.optimize(objective, n_trials=1)
        iter += 1
        val   = study.best_value

    all_eps[i] = np.copy(study.best_params['eps'])
    all_pts[i] = np.copy(study.best_params['min_pts'])

for i in range(len(datasets)):
    print('Best Params for {}: eps = {}; min_pts = {}'.format(datasets[i], all_eps[i], all_pts[i]))
