import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from Functions import GradK_fun as cf


# Load data
file = 'Example/jain-groundtruth/'
sx       = np.load(file + 'sx.npy')
sy       = np.load(file + 'sy.npy')
color_gt = np.load(file + 'gt.npy')

params     = cf.optimize_GradK_params(sx, sy, color_gt)
color_out  = cf.Gradient_k(sx, sy, params, maxiter=200, mindelta=0.01)
acc, color = cf.AccuracyVSGround(color_out, color_gt)

# plot compare
f, ([ax1, ax2]) = plt.subplots(1, 2, sharex='all', sharey='all')
cf.clustplot(ax1, sx, sy, color_gt)
ax1.set_title('Ground Truth')
cf.clustplot(ax2, sx, sy, color)
ax2.set_title('Grad_K Clustering')
plt.show()
