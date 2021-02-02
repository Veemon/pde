import numpy as np
import matplotlib.pyplot as plt

def plot_derivative(rmin, rmax, n, f, ff):
    x  = np.linspace(rmin,rmax,n)
    y  = f(x)
    yy = ff(x)

    diff = np.zeros(n)
    h = (rmax - rmin) / n
    recip = 1.0 / (2*h)


    # central difference
    diff[1:-1] = (y[2:] - y[:-2]) * recip

    # boundary conditions
    bot_diff = y[0] - (y[1] - y[0])
    diff[0] = (y[1] - bot_diff) * recip

    top_diff = y[-1] + (y[-1] - y[-2])
    diff[-1] = (top_diff - y[-2]) * recip


    # plotting
    fig, ax = plt.subplots(1,2,figsize=(10,6))

    ax[0].set_title("raw plot")
    ax[0].plot(x,y, label="f(x)")
    ax[0].plot(x,yy, label="f'(x)")
    ax[0].plot(x,diff,'--', label="$\Delta f(x)$")
    ax[0].legend()

    ax[1].set_title("difference")
    ax[1].plot(x, abs(diff - yy))

    plt.show()


rmin = 0     # range min
rmax = 4     # range max
n    = 1000  # number of intervals

# Sinusoidal Example
f  = lambda x: np.sin(x)
ff = lambda x: np.cos(x)
plot_derivative(rmin, rmax, n, f, ff)


