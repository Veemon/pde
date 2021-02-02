import sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gif_writer import write_gif
from common_utils import *

common = __import__('2_3_multigrid')
create_masks      = common.create_masks
condition            = common.condition
smooth_laplace       = common.smooth_laplace
smooth_errors        = common.smooth_errors
compute_residuals    = common.compute_residuals
restrict             = common.restrict
prolong              = common.prolong
prolong_smooth       = common.prolong_smooth

if __name__ == "__main__":
    n = 128  # simulation dimension
    t = 64   # number of time steps
    d = 6    # number of grids to traverse
    p = 30   # number of inner-time steps 
    q = 50  # number of bottom-time steps

    init    = False
    _smooth = False

    n,t,d,p,q,init,_smooth = get_args([
        ["-n", int, n],
        ["-t", int, t], 
        ["-d", int, d],
        ["-p", int, p], 
        ["-q", int, q], 
        ["--init", bool, init],
        ["--smooth", bool, _smooth],
    ], verbose=True)
    if _smooth:
        prolong = prolong_smooth
    print()

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
        ub[0] = condition(ub[0],masks[0],r=r)

    fig, ax = plt.subplots(d+2,6,figsize=(7.3,9.4))
    off = -0.275

    # Apply V-Cycles
    print("Applying V-Cycle Solve ...")
    for k in tqdm(range(t)):
        # apply boundary conditions
        ub[k] = condition(ub[k],masks[0],r=r)

        ax[0][0].imshow(ub[k])
        ax[0][0].set_title("u", y=off)

        # 1. smooth laplace
        ub[k+1,1:-1,1:-1] = smooth_laplace(ub[k])
        for _ in range(p-1):
            ub[k+1,1:-1,1:-1] = smooth_laplace(ub[k+1])
            ub[k+1] = condition(ub[k+1], masks[0],r=r)

        ax[1][0].imshow(ub[k+1])
        ax[1][0].set_title("smooth u", y=off)

        # 2. calculate residuals
        residuals[0][1:-1,1:-1] = compute_residuals(ub[k+1])

        ax[2][0].imshow(residuals[0])
        ax[2][0].set_title("r", y=off)

        # 3. intitial restriction
        residuals[1][...] = restrict(residuals[0])

        ax[2][1].imshow(residuals[1])
        ax[2][1].set_title("down r", y=off)

        # 4. v-cycle - downwards
        for i in range(1,d+1):
            # init error matrix
            errors[-i][...] = 0.0

            # a. smooth errors
            iters = p if i < d else q
            for _ in range(iters):
                errors[-i][1:-1,1:-1] = smooth_errors(errors[-i], residuals[i])
            ax[2+i-1][3].imshow(errors[-i])
            ax[2+i-1][3].set_title("e", y=off)
            if i == d: break

            # b. calculate residuals
            residuals[i][1:-1,1:-1] = compute_residuals(errors[-i])
            ax[2+i][1].imshow(residuals[i])
            ax[2+i][1].set_title("r from e", y=off)

            # c. restrict
            residuals[i+1][...] = restrict(residuals[i])
            ax[2+i][2].imshow(residuals[i])
            ax[2+i][2].set_title("down r", y=off)

        # 5. v-cycle - upwards
        for i in range(d-1):
            # prolong error and correct
            errors[i+1] += prolong(errors[i])
            ax[6-i][4].imshow(errors[i+1])
            ax[6-i][4].set_title("up e", y=off)

            # smooth
            for _ in range(p):
                errors[i+1][1:-1,1:-1] = smooth_errors(errors[i+1], residuals[-i-2])
            ax[6-i][5].imshow(errors[i+1])
            ax[6-i][5].set_title("smooth e", y=off)

        # 6. correct fine grid
        residuals[0] = prolong(errors[-1])
        ax[1][4].imshow(residuals[0])
        ax[1][4].set_title("up e", y=off)

        ub[k+1] += residuals[0]
        ax[0][4].imshow(ub[k+1])
        ax[0][4].set_title("corrected u", y=off)

        # clear empty plots
        for i in range(d-1):
            ax[3+i][0].set_visible(False)
        for i in range(3):
            ax[0][1+i].set_visible(False)
            ax[1][1+i].set_visible(False)
        ax[2][2].set_visible(False)
        for i in range(2):
            ax[-1][-i-1].set_visible(False)
            ax[i][-1].set_visible(False)

        # get rid of ticks
        for j in range(d+2):
            for i in range(6):
                ax[j][i].set_xticks([])
                ax[j][i].set_yticks([])

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom= 0.05, right=1, left =0, 
                    hspace=0.25, wspace=0)
        plt.margins(0,0)

        plt.show()
        exit()

