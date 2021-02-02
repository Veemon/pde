import sys
import numpy as np

def get_args(info,verbose=False):
    caught_first_warning = False

    # info - tag, type, value
    idx      = None
    setter   = False
    set_type = None
    for arg in sys.argv[1:]:
        handled = False
        if setter:
            setter        = False
            info[idx][-1] = set_type(arg)
            continue

        for i, (_tag, _type, _) in enumerate(info):
            if arg == _tag:
                if _type is bool:
                    info[i][-1] = True
                    handled  = True
                    break
                else:
                    setter   = True
                    handled  = True
                    set_type = _type
                    idx      = i
                    break

        if not handled:
            if not caught_first_warning:
                print()
                caught_first_warning = True
            print(f"[Warning] Unkown argument '{arg}'")

    if caught_first_warning:
        print()

    if verbose:
        s = 0
        for tag, _, _ in info:
            if len(tag) > s:
                s = len(tag)

        for _tag, _, _val in info:
            space = ' ' * (s-len(_tag)+4)
            print(f"{_tag}{space}--  {_val}")

    return [x[-1] for x in info]

def gray_array(x, axis=-1, stride=1):
    # normalize
    xmin = x.min()
    xmax = x.max()
    x = (x - xmin) / ((xmax-xmin)+1e-8)
    x = np.swapaxes(x,axis,0)

    ims = []
    for i in range(0,x.shape[0],stride):
        im = np.dstack([x[i] * 255]*3)
        im = np.rint(im-1e-8)
        ims.append(im)
    return ims


def color_array(x, axis=-1, stride=1):
    low  = np.array([ 25,  15,  30], dtype=np.float32)
    mid  = np.array([255,  70,  40], dtype=np.float32)
    high = np.array([255, 250,  80], dtype=np.float32)

    # normalize
    xmin = x.min()
    xmax = x.max()
    x = (x - xmin) / ((xmax-xmin)+1e-8)
    x = np.swapaxes(x,axis,0)
    
    # the current gif table supports 256 colors.
    # if 3 std. devs from the mean encapsulates 99.7% of the data, well ..
    nc = 256
    bandwidth = (2.0 * (x.mean() + 3.5*x.std())) / nc
    for j in range(nc):
        mm = (bandwidth*j < x)
        if j < nc-1:
            mm &= (x <= bandwidth*(j+1))
        x[mm] = bandwidth*j

    ims = []
    for i in range(0,x.shape[0],stride):
        im  = np.dstack([x[i]]*3)

        m   = (im > 0.5).astype(np.float32)

        im  = (im-0.5)*2.0*m + im*2.0*(1.0-m)
        inv = 1.0 - im
        im  = (high*im + mid*inv)*m + (mid*im + low*inv)*(1.0-m)
        im  = np.ceil(im)

        ims.append(im)

    return ims

def optimize_gif_params(n,t):
    tol = 0.2
    eps = 1e-8
    f   = lambda s: n / (s*t)
    lr  = lambda f: 2*int(f > 100.0 - eps)-1

    r = [1, n]

    # solve the continous problem
    i = 1
    s = (r[0] + r[1])/2
    while True:
        p = s
        i += 1
        s += lr(f(s)) * (r[1] - r[0]) / (2**i)
        if abs(p-s) < tol: break

    # solve the discrete problem
    ss = int(s)
    for d in [ss-1, ss, ss+1]:
        if d == 0: continue
        z = int(f(d))
        if 1 <= z and z <= 100:
            return d,z

    return -1,-1
