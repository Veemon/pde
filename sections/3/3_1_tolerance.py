import sys
import code
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gif_writer import write_gif
from common_utils import *

def create_forces(n, r, g):
    # assumes square matrices
    x = np.zeros([2,n,n])

    indices   = np.array([list(range(n))])
    delta     = (r[1] - r[0]) / (n-1)
    x[0][:]   = r[0] + delta*indices
    x[1][:,:] = r[0] + delta*indices.T

    return g(x[0], x[1])

def condition(u):
    # dirichlet boundary conditions
    u[:, 0] = 0.1 # left
    u[:,-1] = 0.1 # right
    u[ 0,:] = 0.1 # top
    u[-1,:] = 0.1 # bottom
    return u

def smooth_poisson(u,g,r):
    h = (r[1] - r[0]) / (g.shape[-1]-1)
    x = u[2:, 1:-1] + u[:-2, 1:-1] + \
        u[1:-1, 2:] + u[1:-1, :-2]
    x -= (h**2) * g[1:-1,1:-1]
    return x * 0.25

def smooth_errors(e,r):
    x = e[2:, 1:-1] + e[:-2, 1:-1] + \
        e[1:-1, 2:] + e[1:-1, :-2]
    x -= r[1:-1,1:-1]
    return x * 0.25

def compute_residuals(u,g,r):
    h = (r[1] - r[0]) / (g.shape[-1]-1)
    x  = (h**2) * g[1:-1,1:-1]
    x += 4*u[1:-1,1:-1]
    x -= u[2:, 1:-1] + u[:-2, 1:-1] + \
         u[1:-1, 2:] + u[1:-1, :-2]
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



def downcycle(a, b, forces, errors, residuals, p, q, d):
    for i in range(a,b):
        # init error matrix
        errors[-i][...] = 0.0

        # a. smooth errors
        iters = p if i < d else q
        for _ in range(iters):
            errors[-i][1:-1,1:-1] = smooth_errors(errors[-i], residuals[i])
        if i == d: break

        # b. calculate residuals
        residuals[i][1:-1,1:-1] = compute_residuals(errors[-i], forces[i], r)

        # c. restrict
        residuals[i+1][...] = restrict(residuals[i])

def upcycle(a, b, errors, residuals, p):
    for i in range(a,b):
        # prolong error and correct
        errors[i+1] += prolong(errors[i])

        # smooth
        for _ in range(p):
            errors[i+1][1:-1,1:-1] = smooth_errors(errors[i+1], residuals[-i-2])

def v_cycle(ub, forces, residuals, errors, r, n, d, p, q):
    # apply boundary conditions
    ub[...] = condition(ub)

    # 1. smooth laplace
    ub[1:-1,1:-1] = smooth_poisson(ub, forces[0], r)
    for _ in range(p-1):
        ub[1:-1,1:-1] = smooth_poisson(ub, forces[0], r)
        ub[...] = condition(ub)

    # 2. calculate residuals
    residuals[0][1:-1,1:-1] = compute_residuals(ub, forces[0], r)

    # 3. intitial restriction
    residuals[1][...] = restrict(residuals[0])

    # 4. v-cycle - downwards 
    downcycle(1,d+1,forces,errors,residuals,p,q,d)

    # 5. v-cycle - upwards
    upcycle(0,d-1,errors,residuals,p)

    # 6. correct fine grid
    ub[...] += prolong(errors[-1])

    # reapply conditioning
    ub[...] = condition(ub)

def w_cycle(ub, forces, residuals, errors, r, n, d, p, q):
    # apply boundary conditions
    ub[...] = condition(ub)

    # 1. smooth laplace
    ub[1:-1,1:-1] = smooth_poisson(ub, forces[0], r)
    for _ in range(p-1):
        ub[1:-1,1:-1] = smooth_poisson(ub, forces[0], r)
        ub[...] = condition(ub)

    # 2. calculate residuals
    residuals[0][1:-1,1:-1] = compute_residuals(ub, forces[0], r)

    # 3. intitial restriction
    residuals[1][...] = restrict(residuals[0])

    # 4. w-cycle - downwards - full
    downcycle(1,d+1,forces,errors,residuals,p,q,d)

    # 5. w-cycle - upwards - half
    d2 = (d-1)//2
    upcycle(0,d2,errors,residuals,p)

    # 6. w-cycle - downwards - half
    downcycle(d-d2,d+1,forces,errors,residuals,p,q,d)

    # 7. w-cycle - upwards - full
    upcycle(0,d-1,errors,residuals,p)

    # 8. correct fine grid
    ub[...] += prolong(errors[-1])

    # reapply conditioning
    ub[...] = condition(ub)


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

    g = lambda x,y: x*y*(np.sin(x*y)**2)
    r = [0.0, 4.0]

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

    forces = []
    for i in range(d+1):
        s = n // (2**i)
        x = create_forces(s+2,r,g)
        forces.append(x)
    print(f" - Forces      {[x.shape[0]-2 for x in forces]}")

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
    ub[0] = condition(ub[0])

    # Apply X-Cycles
    print(f"Applying {cycle.upper()}-Cycle Solve ...")
    k = 0
    i = 0
    while True:
        ub[(i+1)%2][...] = ub[i]
        i = (i+1)%2
        if cycle == 'v':
            v_cycle(ub[i], forces, residuals, errors, r,n,d,p,q)
        elif cycle == 'w':
            w_cycle(ub[i], forces, residuals, errors, r,n,d,p,q)
        mse = ((ub[1] - ub[0])**2).mean()
        k += 1
        if not debug:
            print(f"\rIteration: {' '*(6-len(str(k)))}{k}    MSE:  {mse:.4e}{' '*6}", end="")
        if mse < tolerance:
            break
    if not debug: print()

    # output smoothing
    for _ in range(q):
        ub[i, 1:-1,1:-1] = smooth_poisson(ub[i], forces[0], r)
        ub[i] = condition(ub[i])

    # plotting
    final_error = np.zeros([n+2,n+2])
    final_error[1:-1,1:-1]  = (ub[i, 2:, 1:-1] + ub[i, :-2, 1:-1] + ub[i, 1:-1, 2:] + ub[i, 1:-1, :-2])
    final_error[1:-1,1:-1] -= 4*ub[i, 1:-1, 1:-1]

    fig,ax = plt.subplots(1,3,figsize=(10,6))
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    a = ax[0].imshow(ub[i])
    b = ax[1].imshow(final_error)
    c = ax[2].imshow(forces[0])

    plt.colorbar(a, ax=[ax[0]], location='bottom')
    plt.colorbar(b, ax=[ax[1]], location='bottom')
    plt.colorbar(c, ax=[ax[2]], location='bottom')

    ax[0].set_title("$f\,(x_i,y_i)$")
    ax[1].set_title("$\\nabla^2 f$")
    ax[2].set_title("$g$")

    plt.show()

