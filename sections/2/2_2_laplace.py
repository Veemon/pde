import sys
import numpy as np

from tqdm import tqdm
from gif_writer import write_gif
from common_utils import *

n = 128  # simulation dimension
t = 64   # number of time steps

fps    = 60
stride = 4

n, t, fps, stride = get_args([
    ["-n", int, n],
    ["-t", int, t], 
    ["--fps", int, fps],
    ["--stride", int, stride]
], verbose=True)

def create_sample_points(ub,r=[-1.0, 1.0]):
    # assumes square matrices
    x = np.zeros([2] + list(ub.shape))

    indices   = np.array([list(range(ub.shape[0]))])
    delta     = (r[1] - r[0]) / (ub.shape[0]-1)
    x[0][:]   = r[0] + delta*indices
    x[1][:,:] = r[0] + delta*indices.T

    return x

def condition(ub,x,w=0.3,r=[-1.0, 1.0]):
    # dirichlet boundary conditions
    ub[:, 0] = 0.1 # left
    ub[:,-1] = 0.1 # right
    ub[ 0,:] = 0.1 # top
    ub[-1,:] = 0.1 # bottom

    # apply heat source condition
    m = sum(r) / 2
    p = (x[0]>m-w) & (x[0]<m+w) & (x[1]>m-w) & (x[1]<m+w)
    ub[p] = 1.0

    return ub

r      = [-1.0, 1.0]
sample = create_sample_points(np.zeros([n+2,n+2]), r=r)

# apply heat source condition
ub = np.zeros([t+1, n+2, n+2])
ub[0] = condition(ub[0], sample)

for k in tqdm(range(t)):
    ub[k+1, 1:-1, 1:-1] = ub[k, 2:,  1:-1] + ub[k, :-2, 1:-1] + \
                          ub[k, 1:-1, 2:]  + ub[k, 1:-1, :-2]
    ub[k+1, 1:-1, 1:-1] *= 0.25

    # reapply heat source condition
    ub[k+1] = condition(ub[k+1], sample)

# output
ims = color_array(ub,axis=0,stride=stride)
write_gif(ims, f"../../figures/2_2_laplace_{n}_{t}.gif", fps=fps)
