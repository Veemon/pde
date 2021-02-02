import sys
import code
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gif_writer import write_gif
from common_utils import *

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
    ub[:, 0] = 0.1 # left
    ub[:,-1] = 0.1 # right
    ub[ 0,:] = 0.1 # top
    ub[-1,:] = 0.1 # bottom

    # apply heat source condition
    ub[p] = 1.0

    return ub

def smooth_laplace(u):
    # expects bounded matrices
    x = u[2:, 1:-1] + u[:-2, 1:-1] + \
        u[1:-1, 2:] + u[1:-1, :-2]
    return x * 0.25

def smooth_errors(e,r):
    # expects bounded matrices
    x = e[2:, 1:-1] + e[:-2, 1:-1] + \
        e[1:-1, 2:] + e[1:-1, :-2]
    x -= r[1:-1,1:-1]
    return x * 0.25

def compute_residuals(a):
    x  = 4*a[1:-1,1:-1]
    x -= a[2:, 1:-1] + a[:-2, 1:-1] + \
         a[1:-1, 2:] + a[1:-1, :-2]
    return x

def restrict(x):
    y = np.zeros([((i-2)//2) + 2 for i in x.shape])
    y[1:-1, 1:-1] = x[1:-1:2, 1:-1:2] + x[2::2, 1:-1:2] + \
                    x[1:-1:2, 2::2]   + x[2::2, 2::2]
    return y * 0.25

def prolong(x):
    y = np.zeros([(i-1)*2 for i in x.shape])

    # expand by factor of 2
    y[1:-1:2, 1:-1:2] = x[1:-1, 1:-1]
    y[2::2,   1:-1:2] = x[1:-1, 1:-1]
    y[1:-1:2, 2::2]   = x[1:-1, 1:-1]
    y[2::2,   2::2]   = x[1:-1, 1:-1]

    return y

def prolong_smooth(x):
    y = np.zeros([(i-1)*2 for i in x.shape])

    # expand by factor of 2
    y[1:-1:2, 1:-1:2] = x[1:-1, 1:-1]
    y[2::2,   1:-1:2] = x[1:-1, 1:-1]
    y[1:-1:2, 2::2]   = x[1:-1, 1:-1]
    y[2::2,   2::2]   = x[1:-1, 1:-1]

    # take inner averages
    y[2:-2, 2:-2] = y[3:-1, 2:-2] + y[1:-3, 2:-2] + \
                    y[2:-2, 3:-1] + y[2:-2, 1:-3]
    y *= 0.25

    # compute directional differences at the bounds
    y2 = 2 * y
    y[2:-2,  1] = y2[2:-2,  2] - y[2:-2,  3] # left  side
    y[2:-2, -2] = y2[2:-2, -3] - y[2:-2, -4] # right side
    y[1,  2:-2] = y2[ 2, 2:-2] - y[ 3, 2:-2] # top   side
    y[-2, 2:-2] = y2[-3, 2:-2] - y[-4, 2:-2] # bot   side

    # corner values are the average of their neighbors
    third = 1.0 / 3.0
    y[ 1, 1] = (y[ 1, 2] + y[ 2, 1] + y[ 2, 2]) * third #top left
    y[ 1,-2] = (y[ 1,-3] + y[ 2,-2] + y[ 2,-3]) * third #top right
    y[-2, 1] = (y[-2, 2] + y[-3, 1] + y[-3, 2]) * third #bot left
    y[-2,-2] = (y[-2,-3] + y[-3,-2] + y[-3,-3]) * third #bot right

    return y

if __name__ == "__main__":
    n = 128  # simulation dimension
    t = 64   # number of time steps
    d = 6    # number of grids to traverse
    p = 50   # number of inner-time steps 
    q = 150  # number of bottom-time steps

    fps    = 60
    stride = 4

    init        = False
    render_diff = False
    debug       = False

    n,t,d,p,q,fps,stride,init,render_diff,debug = get_args([
        ["-n", int, n],
        ["-t", int, t], 
        ["-d", int, d],
        ["-p", int, p], 
        ["-q", int, q], 
        ["--fps", int, fps],
        ["--stride", int, stride],
        ["--init", bool, init],
        ["--differences", bool, render_diff],
        ["--debug", bool, debug],
    ], verbose=True)

    print()

    # NOTE: h(n) assumes 1 pixel padding over matrix of dim n
    r = [-1.0, 1.0]

    differences = np.zeros([t,n+2,n+2])
    ub          = np.zeros([t+1, n+2, n+2])

    masks = []
    for i in range(d+1):
        s = n // (2**i)
        x = create_masks(np.zeros([s+2,s+2]), r=r)
        masks.append(x)
    print(f" - masks   {[x.shape[1]-2 for x in masks]}")

    errors = []
    for i in range(d):
        s = n // (2**(d-i))
        x = np.zeros([s+2,s+2])
        errors.append(x)
    print(f" - Errors      {[x.shape[0]-2 for x in errors]}")

    residuals = []
    for i in range(d+1):
        s = n // (2**i)
        x = np.zeros([s+2,s+2])
        residuals.append(x)
    print(f" - Residuals   {[x.shape[0]-2 for x in residuals]}")
    print()

    # solve the coarse problem as an initial guess
    if init:
        print("Applying Initialization Solve ...")
        ub[0] = condition(ub[0],masks[0],r=r)
        for _ in range(q):
            ub[0,1:-1,1:-1] = smooth_laplace(ub[0])
            ub[0] = condition(ub[0],masks[0],r=r)

        coarse = restrict(ub[0])
        for _ in range(d-1):
            coarse = restrict(coarse)

        for _ in range(q):
            coarse[1:-1,1:-1] = smooth_laplace(coarse)
            coarse = condition(coarse,masks[-1],r=r)

        for i in range(d-1):
            coarse = prolong(coarse)
            coarse = condition(coarse,masks[-i-2],r=r)

        ub[0] = prolong(coarse)


    # Apply V-Cycles
    print("Applying V-Cycle Solve ...")
    for k in tqdm(range(t)):
        # apply boundary conditions
        ub[k] = condition(ub[k],masks[0],r=r)

        # 1. smooth laplace
        ub[k+1,1:-1,1:-1] = smooth_laplace(ub[k])
        for _ in range(p-1):
            ub[k+1,1:-1,1:-1] = smooth_laplace(ub[k+1])
            ub[k+1] = condition(ub[k+1], masks[0],r=r)

        # 2. calculate residuals
        residuals[0][1:-1,1:-1] = compute_residuals(ub[k+1])

        # 3. intitial restriction
        residuals[1][...] = restrict(residuals[0])

        # 4. v-cycle - downwards
        for i in range(1,d+1):
            # init error matrix
            errors[-i][...] = 0.0

            # a. smooth errors
            iters = p if i < d else q
            for _ in range(iters):
                errors[-i][1:-1,1:-1] = smooth_errors(errors[-i], residuals[i])
            if i == d: break

            # b. calculate residuals
            residuals[i][1:-1,1:-1] = compute_residuals(errors[-i])

            # c. restrict
            residuals[i+1][...] = restrict(residuals[i])

        # 5. v-cycle - upwards
        for i in range(d-1):
            # prolong error and correct
            errors[i+1] += prolong(errors[i])

            # smooth
            for _ in range(p):
                errors[i+1][1:-1,1:-1] = smooth_errors(errors[i+1], residuals[-i-2])

        # 6. correct fine grid
        ub[k+1]        += prolong(errors[-1])
        differences[k]  = abs(ub[k+1] - ub[k])

    # reapply conditioning
    ub[-1] = condition(ub[-1],masks[0],r=r)

    # output
    if debug:
        code.interact(local=locals())
    else:
        init_str = "init_" if init else ""

        print()
        print("Rendering Convergence Array ...")
        ims = color_array(ub,axis=0,stride=stride)
        write_gif(ims, f"../../figures/2_3_{init_str}{n}_{t}_{d}.gif", fps=fps)

        if render_diff:
            print("Rendering Difference Array ...")
            ims = gray_array(differences,axis=0,stride=stride)
            write_gif(ims, f"../../figures/2_3_{init_str}differences_{n}_{t}_{d}.gif", fps=fps)

