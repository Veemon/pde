import sys
import torch
import torch.nn.functional as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gif_writer import write_gif
from common_utils import *

r = [-1.0, 1.0]

n         = 1024 # simulation dimension
d         = 6    # number of grids to traverse
tolerance = 1e-4
cpu       = False


n,d,tolerance,cpu = get_args([
    ["-n", int, n],
    ["-d", int, d],
    ["--tolerance", float, tolerance],
    ["--cpu", bool, cpu]
], verbose=True)


if not torch.cuda.is_available():
    print("No capable CUDA devices, using CPU ...")
    cpu = True


def create_masks(ub,w=0.3,r=[-1.0, 1.0]):
    # assumes square matrices
    x = np.zeros([2] + list(ub.shape))

    indices   = np.array([list(range(ub.shape[0]))])
    delta     = (r[1] - r[0]) / (ub.shape[0]-1)
    x[0][:]   = r[0] + delta*indices
    x[1][:,:] = r[0] + delta*indices.T

    m = sum(r) / 2
    p = (x[0]>m-w) & (x[0]<m+w) & (x[1]>m-w) & (x[1]<m+w)

    return p

def condition(ub,p,w=0.3,r=[-1.0, 1.0]):
    # dirichlet boundary conditions
    ub[0,0,:, 0] = 0.1 # left
    ub[0,0,:,-1] = 0.1 # right
    ub[0,0, 0,:] = 0.1 # top
    ub[0,0,-1,:] = 0.1 # bottom

    # apply heat source condition
    ub[0,0,p] = 1.0

    return ub

masks = []
for i in range(d+1):
    s = n // (2**i)
    x = create_masks(np.zeros([s+2,s+2]), r=r)
    if cpu:
        masks.append(torch.BoolTensor(x))
    else:
        masks.append(torch.BoolTensor(x).cuda())
masks = masks[::-1]
print(f" - Masks   {[x.shape[1]-2 for x in masks]}")

multi_grid = []
for i in range(d):
    s = n // (2**(d-i))
    if cpu:
        x = torch.zeros([1,2,s+2,s+2])
    else:
        x = torch.zeros([1,2,s+2,s+2]).cuda()
    multi_grid.append(x)
print(f" - Grids   {[x.shape[-1]-2 for x in multi_grid]}")


w = torch.zeros(1,1,3,3)
w[0,0,1,:] = 0.25
w[0,0,:,1] = 0.25
w[0,0,1,1] = 0
print(f" - Kernel: \n{w}")

if cpu:
    w  = w
    up = torch.ones(1,1,2,2)
    ub = torch.zeros([1,2,n+2,n+2])
else:
    w  = w.cuda()
    up = torch.ones(1,1,2,2).cuda()
    ub = torch.zeros([1,2,n+2,n+2]).cuda()

# apply initial conditions
multi_grid[0] = condition(multi_grid[0], masks[0],r=r)

# apply multigridding
for i in range(d):
    k   = 0
    idx = 0
    while True:
        multi_grid[i][:,(idx+1)%2] = multi_grid[i][:,idx]
        idx = (idx + 1)%2

        # -- solve at lower res
        multi_grid[i][:,idx:idx+1] = tf.conv2d(multi_grid[i][:,idx:idx+1], w, padding=1) 
        multi_grid[i][:,idx:idx+1] = condition(multi_grid[i][:,idx:idx+1], masks[i],r=r)

        mse = ((multi_grid[i][0,0]-multi_grid[i][0,1])**2).mean()
        k += 1

        space = ' '*(len(str(d)) - len(str(i+1)))
        print(f"\r[{space}{i+1}] Iteration: {' '*(6-len(str(k)))}{k}    MSE:  {mse:.4e}{' '*6}", end="")
        if mse < tolerance:
            break
    print()

    # -- upscale
    if i < d-1:
        multi_grid[i+1][:,0:1,1:-1, 1:-1] = tf.conv_transpose2d(multi_grid[i][:,idx:idx+1,1:-1,1:-1], up, stride=2)
    else:
        ub[:, 0:1, 1:-1, 1:-1] = tf.conv_transpose2d(multi_grid[i][:,idx:idx+1,1:-1,1:-1], up, stride=2)

# solve on target grid
k   = 0
idx = 0
while True:
    ub[:,(idx+1)%2] = ub[:,idx]
    idx = (idx + 1)%2

    ub[:,idx:idx+1] = tf.conv2d(ub[:, idx:idx+1], w, padding=1) 
    ub[:,idx:idx+1] = condition(ub[:, idx:idx+1], masks[-1], r=r)

    mse = ((ub[0,0]-ub[0,1])**2).mean()
    k += 1

    s = len(str(d))
    s2 = s // 2
    space = f"{' '*(s2)}*{' '*(s-s2-1)}"
    print(f"\r[{space}] Iteration: {' '*(6-len(str(k)))}{k}    MSE:  {mse:.4e}{' '*6}", end="")
    if mse < tolerance:
        break

ub = ub.cpu().numpy()[0]

final_error = np.zeros([n+2,n+2])
final_error[1:-1,1:-1]  = (ub[idx,2:,1:-1] + ub[idx,:-2,1:-1] + ub[idx,1:-1,2:] + ub[idx,1:-1,:-2])
final_error[1:-1,1:-1] -= 4*ub[idx,1:-1,1:-1]

fig,ax = plt.subplots(1,2,figsize=(10,6))
for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

a = ax[0].imshow(ub[idx])
b = ax[1].imshow(final_error)

plt.colorbar(a, ax=[ax[0]], location='left')
plt.colorbar(b)

ax[0].set_title("$f\,(x_i,y_i)$")
ax[1].set_title("$\\nabla^2 f$")

plt.show()
