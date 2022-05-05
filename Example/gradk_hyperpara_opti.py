import optuna
import clust_fun as cf
import numpy as np

def objective(trial):

    # Variable Parameters
    numboxes   = trial.suggest_int('numboxes', 10, 100)
    sig_smooth = trial.suggest_float("sig", 0, 1)
    alpha_par  = trial.suggest_float("alpha_par", 0, 50)
    mindelta   = 0.01 * numboxes
    global maxp

    # Prepare data
    S_bin, count, count_smooth = cf.bin_2d(sx, sy, numboxes, sig_smooth)

    # compute gradient
    grad_y, grad_x = np.gradient(count_smooth)

    # initialize datastores
    acc      = np.zeros([20])
    for run in range(20):

        # Choose initial points
        if run == 0:
            ind = cf.opt_start(S, color_gt)
        else:
            ind = cf.choose_centers(S, numcenters)
        point = S_bin[ind,:]

        # grad-k
        new_point = point
        iter      = 0
        delta     = 1000
        while (iter < maxiter) and (delta > mindelta):

            a = new_point
            bclust, new_point = cf.iteration(count, grad_x, grad_y, \
                                    new_point, alpha_par , count, numcenters)

            diff = np.square(a - new_point)
            delta = np.sum(np.sqrt(np.sum(diff, 1)))
            iter += 1

        # color points as function of box
        color       = cf.move_2_pointspace(bclust, S_bin)
        acc[run], _ = cf.AccuracyVSGround(color, color_gt)

        if run == 0:
            if acc[run] < maxp:
                raise optuna.TrialPruned()
            else:
                maxp = acc[run]

    return 100 - np.mean(acc)

#parameters
path   = 'E:\\Andrei\\Clust\\'
e_path = '-groundtruth\\'
datasets  = ['jain' , 'unbl' , 'aggr' , 's4'   , 'g2_30', 'g2_50', 'synth_spike_2']
n_box_l   = [14     , 26     , 16     , 45     , 49     , 31     , 42     ]
sig_l     = [0.85581, 0.77081, 0.96781, 0.99968, 0.39384, 0.67587, 0.00086]
alpha_l   = [9.75935, 21.1027, 9.04926, 10.3244, 5.12984, 0.06552, 4.90508]
datasets  = ['jain' , 'aggr' ]
n_box_l   = [14     , 16     ]
sig_l     = [0.85581, 0.96781]
alpha_l   = [9.75935, 9.04926]


all_box = np.zeros([len(datasets)])
all_sig = np.zeros([len(datasets)])
all_alf = np.zeros([len(datasets)])
for i in range(len(datasets)):
    ds      = datasets[i]
    s_sig   = sig_l[i]
    s_n_box = n_box_l[i]
    s_alpha = alpha_l[i]

    file = path + ds + e_path
    sx       = np.load(file + 'sx.npy')
    sy       = np.load(file + 'sy.npy')
    color_gt = np.load(file + 'gt.npy')

    # Preproc data
    sx = sx - np.min(sx)
    sy = sy - np.min(sy)
    sx = sx / np.max(sx)
    sy = sy / np.max(sy)
    S  = np.transpose(np.stack((sx, sy)))

    # Parameters
    numcenters = len(np.unique(color_gt))
    maxiter    = 200

    # optimize parameters
    val       = 100
    iter      = 0
    maxiter_h = 1000
    maxp      = 0
    study     = optuna.create_study()

    study.enqueue_trial(
    {
        "numboxes":s_n_box,
        "sig":s_sig,
        "alpha_par":s_alpha,
    })

    while (iter < maxiter_h) & (val > 0):
        study.optimize(objective, n_trials=1)
        iter += 1
        val   = study.best_value
        print('Current dataset: {}'.format(ds))
        print('Current randmax: {}'.format(val))
        print('Current opt_max: {}'.format(100 - maxp))


    all_box[i] = study.best_params['numboxes']
    all_sig[i] = study.best_params['sig']
    all_alf[i] = study.best_params['alpha_par']


for i in range(len(datasets)):
    print('Best Params for {}: n_box = {}; sigma = {}; alpha = {}'.format(datasets[i], all_box[i], all_sig[i], all_alf[i]))
