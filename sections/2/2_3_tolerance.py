import sys
import code
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gif_writer import write_gif
from common_utils import *

common = __import__('2_3_multigrid')
create_masks      = common.create_masks
condition         = common.condition
smooth_laplace    = common.smooth_laplace
smooth_errors     = common.smooth_errors
compute_residuals = common.compute_residuals
restrict          = common.restrict
prolong           = common.prolong
prolong_smooth    = common.prolong_smooth

def downcycle(a, b, errors, residuals, p, q, d):
    for i in range(a,b):
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

def upcycle(a,b,errors,residuals,p):
    for i in range(a,b):
        # prolong error and correct
        errors[i+1] += prolong(errors[i])

        # smooth
        for _ in range(p):
            errors[i+1][1:-1,1:-1] = smooth_errors(errors[i+1], residuals[-i-2])

def v_cycle(ub, masks, residuals, errors, r, n, d, p, q):
    # apply boundary conditions
    ub[...] = condition(ub,masks[0],r=r)

    # 1. smooth laplace
    ub[1:-1,1:-1] = smooth_laplace(ub)
    for _ in range(p-1):
        ub[1:-1,1:-1] = smooth_laplace(ub)
        ub[...] = condition(ub, masks[0],r=r)

    # 2. calculate residuals
    residuals[0][1:-1,1:-1] = compute_residuals(ub)

    # 3. intitial restriction
    residuals[1][...] = restrict(residuals[0])

    # 4. v-cycle - downwards 
    downcycle(1,d+1,errors,residuals,p,q,d)

    # 5. v-cycle - upwards
    upcycle(0,d-1,errors,residuals,p)

    # 6. correct fine grid
    ub[...] += prolong(errors[-1])

    # reapply conditioning
    ub[...] = condition(ub,masks[0],r=r)

def w_cycle(ub, masks, residuals, errors, r, n, d, p, q):
    # apply boundary conditions
    ub[...] = condition(ub,masks[0],r=r)

    # 1. smooth laplace
    ub[1:-1,1:-1] = smooth_laplace(ub)
    for _ in range(p-1):
        ub[1:-1,1:-1] = smooth_laplace(ub)
        ub[...] = condition(ub, masks[0],r=r)

    # 2. calculate residuals
    residuals[0][1:-1,1:-1] = compute_residuals(ub)

    # 3. intitial restriction
    residuals[1][...] = restrict(residuals[0])

    # 4. w-cycle - downwards - full
    downcycle(1,d+1,errors,residuals,p,q,d)

    # 5. w-cycle - upwards - half
    d2 = (d-1)//2
    upcycle(0,d2,errors,residuals,p)

    # 6. w-cycle - downwards - half
    downcycle(d-d2,d+1,errors,residuals,p,q,d)

    # 7. w-cycle - upwards - full
    upcycle(0,d-1,errors,residuals,p)

    # 8. correct fine grid
    ub[...] += prolong(errors[-1])

    # reapply conditioning
    ub[...] = condition(ub,masks[0],r=r)


if __name__ == "__main__":
    n = 256  # simulation dimension
    d = 6    # number of grids to traverse
    p = 30   # number of inner-time steps 
    q = 50   # number of bottom-time steps

    init      = False
    tolerance = 1e-4
    _smooth   = False
    debug     = False
    cycle     = 'v'

    r = [-1.0, 1.0]

    n,d,p,q,init,tolerance,_smooth,debug,cycle = get_args([
        ["-n", int, n],
        ["-d", int, d],
        ["-p", int, p], 
        ["-q", int, q], 
        ["--init", bool, init],
        ["--tolerance", float, tolerance],
        ["--smooth", bool, _smooth],
        ["--debug", bool, debug],
        ["--cycle", str, cycle],
    ], verbose=True)

    cycle = cycle.lower()
    if cycle not in ['v','w']:
        print("Only V and W cycles implemented.")
        exit()

    if _smooth:
        prolong = prolong_smooth
    print()

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


    ub    = np.zeros([2, n+2, n+2])
    ub[0] = condition(ub[0],masks[0],r=r)

    # solve the coarse problem as an initial guess
    if init:
        print("Applying Initialization Solve ...")
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

    # Apply X-Cycles
    print(f"Applying {cycle.upper()}-Cycle Solve ...")
    k = 0
    i = 0
    while True:
        ub[(i+1)%2][...] = ub[i]
        i = (i+1)%2
        if cycle == 'v':
            v_cycle(ub[i], masks, residuals, errors, r,n,d,p,q)
        elif cycle == 'w':
            w_cycle(ub[i], masks, residuals, errors, r,n,d,p,q)
        mse = ((ub[1] - ub[0])**2).mean()
        k += 1
        if not debug:
            print(f"\rIteration: {' '*(6-len(str(k)))}{k}    MSE:  {mse:.4e}{' '*6}", end="")
        if mse < tolerance:
            break
    if not debug: print()

    # output smoothing
    for _ in range(q):
        ub[i, 1:-1,1:-1] = smooth_laplace(ub[i])
        ub[i] = condition(ub[i], masks[0],r=r)

    # plotting
    final_error = np.zeros([n+2,n+2])
    final_error[1:-1,1:-1]  = (ub[i, 2:, 1:-1] + ub[i, :-2, 1:-1] + ub[i, 1:-1, 2:] + ub[i, 1:-1, :-2])
    final_error[1:-1,1:-1] -= 4*ub[i, 1:-1, 1:-1]

    fig,ax = plt.subplots(1,2,figsize=(10,6))
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    a = ax[0].imshow(ub[i])
    b = ax[1].imshow(final_error)

    plt.colorbar(a, ax=[ax[0]], location='left')
    plt.colorbar(b)

    ax[0].set_title("$f\,(x_i,y_i)$")
    ax[1].set_title("$\\nabla^2 f$")

    plt.show()

