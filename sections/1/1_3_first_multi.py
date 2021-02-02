import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

def array_color(phase, mag):
    a = 0.6 # saturation min
    b = 0.4 # value      min
    x = np.zeros([*phase.shape] + [3], np.uint8)
    for j in range(phase.shape[0]):
        for i in range(phase.shape[1]):
            x[j,i] = hsv_rgb(phase[j,i], a+(1.0-a)*mag[j,i], b+(1.0-b)*mag[j,i])
    return x

def hsv_rgb(h,S,V):
    while h <   0: h += 360
    while h > 360: h -= 360

    C = V * S
    X = C * (1.0 - abs(((h//60) % 2) - 1))
    M = V - C

    u = int((C+M)*255)
    v = int((X+M)*255)
    w = int((0+M)*255)

    if   0 <= h and h <=  60: return [u, v, w]
    if  60 <= h and h <= 120: return [v, u, w]
    if 120 <= h and h <= 180: return [w, u, v]
    if 180 <= h and h <= 240: return [w, v, u]
    if 240 <= h and h <= 300: return [v, w, u]
    if 300 <= h and h <= 360: return [u, w, v]

def plot_derivative(rmin, rmax, n, f, fx, fy):
    h = (rmax - rmin) / n
    recip = 1.0 / (2*h)

    X = np.linspace(rmin, rmax, n)
    Y = np.linspace(rmin, rmax, n)
    x,y = np.meshgrid(X,Y)

    z  = f(x,y)
    zx = fx(x,y)
    zy = fy(x,y)

    z_bounds = np.zeros([n+2, n+2])
    z_bounds[1:-1,1:-1] = z

    # central differences
    grad = np.zeros([n, n, 2])
    grad[:,:,0] = z_bounds[1:-1, 2:] - z_bounds[1:-1,  :-2]  # f_x
    grad[:,:,1] = z_bounds[2:, 1:-1] - z_bounds[:-2,  1:-1]  # f_y
    grad *= recip

    # plotting
    fig, ax = plt.subplots(2,3,figsize=(10,6.5))

    img1 = np.stack([zx,zy], -1)
    img2 = grad

    # -- magnitude
    img1_mag = npl.norm(img1, axis=-1)
    img2_mag = npl.norm(img2, axis=-1)
    mag_diff = abs(img2_mag - img1_mag)

    m = img1_mag.max()

    img1_mag = img1_mag / m
    img2_mag = img2_mag / m
    img2_mag[img2_mag > 1.0] = 1.0
    mag_diff = mag_diff / mag_diff.max()

    ax[0][0].set_title("magnitude $\\nabla f$")
    ax[0][0].imshow(img1_mag)

    ax[0][1].set_title("magnitude $\\Delta f$")
    ax[0][1].imshow(img2_mag)

    ax[0][2].set_title("difference")
    ax[0][2].imshow(mag_diff)

    # -- phase
    img1_phase = np.degrees(np.arctan2(img1[:,:,1], img1[:,:,0]))
    img2_phase = np.degrees(np.arctan2(img2[:,:,1], img2[:,:,0]))
    phase_diff = abs(img2_phase - img1_phase)

    img1_phase = array_color(img1_phase, img1_mag)
    img2_phase = array_color(img2_phase, img2_mag)
    phase_diff = phase_diff / phase_diff.max()

    ax[1][0].set_title("phase $\\nabla f$")
    ax[1][0].imshow(img1_phase)

    ax[1][1].set_title("phase $\\Delta f$")
    ax[1][1].imshow(img2_phase)

    ax[1][2].set_title("difference")
    ax[1][2].imshow(phase_diff)

    plt.show()


rmin = 0     # range min
rmax = 4     # range max
n    = 32   # number of intervals

# Sinusoidal Example
f  = lambda x,y: np.sin(x*y)
fx = lambda x,y: y*np.cos(x*y)
fy = lambda x,y: x*np.cos(x*y)
plot_derivative(rmin, rmax, n, f, fx, fy)


