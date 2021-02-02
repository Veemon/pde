import sys
import torch
import torch.nn.functional as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gif_writer import write_gif
from common_utils import *

r = [1.0, 1.01]
g = lambda x,y: x*y*(torch.sin(x*y)**2)

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

def condition(ub,r):
    # dirichlet boundary conditions
    ub[:, 0] = 0.25 # left
    ub[:,-1] = 0.25 # right
    ub[ 0,:] = 0.25 # top
    ub[-1,:] = 0.25 # bottom

    return ub

def apply_force(x,r,g):
    indices        = torch.FloatTensor([list(range(x.shape[-1]))])
    h              = (r[1] - r[0]) / (x.shape[-1]-1)
    coords         = torch.zeros([2,*x.shape])
    coords[0][:]   = r[0] + h*indices
    coords[1][:,:] = r[0] + h*indices.T
    return 0.25 * (h**2) * g(coords[0], coords[1])

multi_grid = []
for i in range(d):
    s = n // (2**(d-i))
    x = torch.zeros([1,3,s+2,s+2])
    x[0,-1] = apply_force(x[0,-1], r, g)
    if not cpu:
        x = x.cuda()
    multi_grid.append(x)
print(f" - Grids   {[x.shape[-1]-2 for x in multi_grid]}")

w = torch.zeros(1,1,3,3)
w[0,0,1,:] = 0.25
w[0,0,:,1] = 0.25
w[0,0,1,1] = 0
print(f" - Kernel: \n{w}")

up = torch.ones(1,1,2,2)
ub = torch.zeros([1,3,n+2,n+2])
ub[0,-1] = apply_force(ub[0,-1], r, g)

if not cpu:
    w  = w.cuda()
    up = up.cuda()
    ub = ub.cuda()

# apply initial conditions
multi_grid[0][:,0:1] = condition(multi_grid[0][0,0], r)

# apply multigridding
for i in range(d):
    k   = 0
    idx = 0
    while True:
        multi_grid[i][:,(idx+1)%2] = multi_grid[i][:,idx]
        idx = (idx + 1)%2

        # -- solve at lower res
        multi_grid[i][:,idx:idx+1] = tf.conv2d(multi_grid[i][:,idx:idx+1], w, padding=1) - multi_grid[i][0,-1]
        multi_grid[i][:,idx:idx+1] = condition(multi_grid[i][0,idx], r)

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

    ub[:,idx:idx+1] = tf.conv2d(ub[:, idx:idx+1], w, padding=1) - ub[0,-1]
    ub[:,idx:idx+1] = condition(ub[0, idx], r)

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

indices        = torch.FloatTensor([list(range(n+2))])
h              = (r[1] - r[0]) / (n+2-1)
coords         = torch.zeros([2,n+2,n+2])
coords[0][:]   = r[0] + h*indices
coords[1][:,:] = r[0] + h*indices.T

fig,ax = plt.subplots(1,3,figsize=(10,6))
for i in range(3):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

a = ax[0].imshow(ub[idx])
b = ax[1].imshow(final_error)
c = ax[2].imshow(g(coords[0], coords[1]))

plt.colorbar(a, ax=[ax[0]], location='bottom')
plt.colorbar(b, ax=[ax[1]], location='bottom')
plt.colorbar(c, ax=[ax[2]], location='bottom')

ax[0].set_title("$f\,(x_i,y_i)$")
ax[1].set_title("$\\nabla^2 f$")
ax[2].set_title("$g$")

plt.show()
