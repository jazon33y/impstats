import operator
import warnings
import numpy as np
from math import trunc
from scipy import interpolate as interp
import matplotlib.pyplot as plt
from impstats import interval
from impstats import pbox


def impplot(toplot):    
    if not isinstance(toplot, pbox.Pbox):
        toplot = pbox.Pbox(toplot)
        
    ps = np.array(range(toplot.n)) / toplot.n
    
    plt.subplot(1, 1, 1)
    plt.plot([toplot.lo[0], toplot.hi[0], *toplot.hi], [0, 0, *ps], '-')
    plt.plot([*toplot.lo, toplot.lo[toplot.n - 1], toplot.hi[toplot.n - 1]], [*ps, 1, 1], '-')
    plt.show()


def autoselect(x, y, op="+"):
    # # dependency tracking not yep implemented
    # if indep(x, y):
    #     x._conv(y, op=op)
    # elif perfect(x, y):
    #     x._perfectconv(y, op=op)
    # elif (opposite(x,y)):
    #     x._oppositeconv(y, op=op)
    # else:
    #     return x._frechetconv(y, op=op)
    #
    return x._frechetconv(y, op=op)

def makepbox(*args):
    if len(args) == 1:
        return [pbox.Pbox(args[0])]

    return [pbox.Pbox(arg) if not isinstance(arg, pbox.Pbox) else arg for arg in args]

def imp_interval(*args):
    mmm = args[0]
    for each in args:
        mmm = interval.Interval(lo=np.max([left(mmm), left(each)]), hi=np.min([right(mmm), right(each)]))
    
    # should check whether they now cross
    if mmm.hi < mmm.lo: 
        warnings.warn('Imposition is empty')
        mmm = None
        
    return mmm

def imp_pbox(*args):
    his = [arg.hi for arg in args]
    los = [arg.lo for arg in args]
    mean_los = [arg.mean_lo for arg in args]
    mean_his = [arg.mean_hi for arg in args]
    var_los = [arg.var_lo for arg in args]
    var_his = [arg.var_hi for arg in args]
    
    hi = np.minimum(*his)
    lo = np.maximum(*los)
    mean_lo = np.max(mean_los) #myvectormax
    mean_hi = np.min(mean_his) #myvectormin
    var_lo = np.max(var_los) #myvectormax
    var_hi = np.min(var_his) #myvectormin

    if np.any(hi < lo):
        raise Exception('Imposition does not exist')
    else:
        pbox_parms = {
            "hi": hi,
            "lo": lo,
            "mean_lo": mean_lo,
            "mean_hi": mean_hi,
            "var_lo": var_lo,
            "var_hi": var_hi
        }

        return pbox.Pbox(**pbox_parms)

def imp(*args):
    int_imp = [isinstance(arg, interval.Interval) for arg in args]
    pbox_imp = [isinstance(arg, pbox.Pbox) for arg in args]

    if np.all(np.array(int_imp)):
        return imp_interval(*args)
    elif np.all(np.array(pbox_imp)):
        return imp_pbox(*args)
    else:
        raise Exception("not all Intervals or all Pbox objects were given")

def get_op(op):
    ops = {
        '+' : operator.add,
        '-' : operator.sub,
        '*' : operator.mul,
        '/' : operator.truediv,
        '%' : operator.mod,
        '^' : operator.xor
    }

    return ops[op]

def is_iterable(obj):
    try:
        iter(obj) # check if iterable
        return True
    except:
        return False

def ii(pbox_steps=200):
    return np.array(range(pbox_steps)) / pbox_steps

def jj(pbox_steps=200):
    return np.array(range(1, pbox_steps + 1)) / pbox_steps

def iii(pbox_steps=200, bot=0.001):
    steps = np.array(range(pbox_steps)) / pbox_steps
    steps[0] = bot

    return steps

def jjj(pbox_steps=200, top=0.999):
    steps = np.array(range(1, pbox_steps + 1)) / pbox_steps
    steps[-1] = top

    return steps

def env_int(*args):
    lo = min([min(i) if is_iterable(i) else i for i in args])
    hi = max([max(i) if is_iterable(i) else i for i in args])
    return interval.Interval(lo, hi)


def env_pbox(*args):
    args = makepbox(*args)
    lo = np.amin([arg.lo for arg in args], 0)
    hi = np.amax([arg.hi for arg in args], 0)
    mean_lo = np.min([arg.mean_lo for arg in args])
    mean_hi = np.max([arg.mean_lo for arg in args])
    var_lo = np.min([arg.var_lo for arg in args])
    var_hi = np.max([arg.var_hi for arg in args])
    shape = [arg.shape for arg in args]

    if len(set(shape)) > 1:
        shape = ''
    else:
        shape = shape[0] # there can be only one
    
    pbox_parms = {
        'lo': lo,
        'hi': hi,
        'mean_lo': mean_lo,
        'mean_hi': mean_hi,
        'var_lo': var_lo,
        'var_hi': var_hi,
        'shape': shape
    }

    return pbox.Pbox(**pbox_parms)

def env(*args):
    int_imp = [isinstance(arg, interval.Interval) for arg in args]
    pbox_imp = [isinstance(arg, pbox.Pbox) for arg in args]

    if np.all(np.array(int_imp)):
        return env_int(*args)
    elif np.all(np.array(pbox_imp)):
        return env_pbox(*args)
    else:
        raise Exception("not all Intervals or all Pbox objects were given")

def left(imp):
    if isinstance(imp, interval.Interval) or isinstance(imp, pbox.Pbox):
        return imp.left()
    elif is_iterable(imp):
        return min(imp)
    else:
        return imp

def right(imp):
    if isinstance(imp, interval.Interval) or isinstance(imp, pbox.Pbox):
        return imp.right()
    elif is_iterable(imp):
        return max(imp)
    else:
        return imp

def left_list(implist, verbose=False):
    if not is_iterable(implist):
        return np.array(implist)

    return np.array([left(imp) for imp in implist])

def right_list(implist, verbose=False):
    if not is_iterable(implist):
        return np.array(implist)
    
    return np.array([right(imp) for imp in implist])

def qleftquantiles(pp, x, p): # if first p is not zero, the left tail will be -Inf
    return [max(left_list(x)[right_list(p) <= P]) for P in pp]

def qrightquantiles(pp, x, p):  # if last p is not one, the right tail will be Inf 
    return [min(right_list(x)[P <= left_list(p)]) for P in pp]

def quantiles(x, p, pbox_steps=200):
    lo = qleftquantiles(ii(pbox_steps=pbox_steps), x, p)
    hi = qrightquantiles(jj(pbox_steps=pbox_steps), x, p)
    return pbox.Pbox(lo=lo, hi=hi)  # quantiles are in x and the associated cumulative probabilities are in p

def interp_step(u, pbox_steps=200): 
    u = np.sort(u)

    seq = np.linspace(start=0, stop=len(u) - 0.00001, num=pbox_steps, endpoint=True)
    seq = np.array([trunc(seq_val) for seq_val in seq])
    return u[seq]

def interp_cubicspline(vals, pbox_steps=200):
    vals = np.sort(vals) # sort
    vals_steps = np.array(range(len(vals))) + 1
    vals_steps = vals_steps / len(vals_steps)

    steps = np.array(range(pbox_steps)) + 1
    steps = steps / len(steps)

    interped = interp.CubicSpline(vals_steps, vals)
    return interped(steps)

def interp_left(u, pbox_steps=200): 
    p = np.array(range(len(u))) / (len(u) - 1)
    pp, x = ii(pbox_steps=pbox_steps), u
    return qleftquantiles(pp, x, p)

def interp_right(d, pbox_steps=200): 
    p = np.array(range(len(d))) / (len(d) - 1)
    pp, x = jj(pbox_steps=pbox_steps), d
    return qrightquantiles(pp, x, p)

def interp_outer(x, left, pbox_steps=200):
    if (left) :
        return interp_left(x, pbox_steps=pbox_steps)
    else: 
        return interp_right(x, pbox_steps=pbox_steps)

def interp_linear(V, pbox_steps=200):
    m = len(V) - 1

    if m == 0: return np.repeat(V, pbox_steps)
    if pbox_steps == 1: return np.array([min(V), max(V)])

    d = 1 / m
    n = round(d * pbox_steps * 20)

    if n == 0:
        c = V
    else: 
        c = []
        for i in range(m):
            v = V[i]
            w = V[i + 1]
            c.extend(np.linspace(start=v, stop=w, num=n))
    
    u = [c[round((len(c) - 1) * (k + 0) / (pbox_steps - 1))] for k in range(pbox_steps)]
    
    return np.array(u)
  
def interpolate(u, interpolation='linear', left=True, pbox_steps=200): 
    if interpolation == 'outer':
        return interp_outer(u, left, pbox_steps=pbox_steps)
    elif interpolation == 'spline':
        return interp_cubicspline(u, pbox_steps=pbox_steps)
    elif interpolation == 'step':
        return interp_step(u, pbox_steps=pbox_steps)
    else:
        return interp_linear(u, pbox_steps=pbox_steps)

def sideVariance(w, mu=None):
    if not isinstance(w, np.ndarray): w = np.array(w)
    if mu is None: mu = np.mean(w)
    return max(0, np.mean((w - mu) ** 2))

def dwMean(p_box):
    return interval.Interval(np.mean(p_box.hi), np.mean(p_box.lo))

def dwVariance(pbox):
    if np.any(np.isinf(pbox.lo)) or np.any(np.isinf(pbox.hi)):
        return interval.Interval(0, np.inf)

    if np.all(pbox.hi[0] == pbox.hi) and np.all(pbox.lo[0] == pbox.lo):
        return interval.Interval(0, (pbox.hi[0] - pbox.lo[0]) ** (2 / 4))

    vr = sideVariance(pbox.lo, np.mean(pbox.lo))
    w = np.copy(pbox.lo)
    n = len(pbox.lo)

    for i in reversed(range(n)):
        w[i] = pbox.hi[i]
        v = sideVariance(w, np.mean(w))

        if np.isnan(vr) or np.isnan(v):
            vr = np.inf
        elif vr < v:
            vr = v
    
    if pbox.lo[n - 1] <= pbox.hi[0]:
        vl = 0.0
    else:
        w = np.copy(pbox.hi)
        vl = sideVariance(w, np.mean(w))
        
        for i in reversed(range(n)):
            w[i] = pbox.lo[i]
            here = w[i]

            if 1 < i:
                for j in reversed(range(i-1)):
                    if w[i] < w[j]:
                        w[j] = here

            v = sideVariance(w, np.mean(w))

            if np.isnan(vl) or np.isnan(v):
                vl = 0
            elif v < vl:
                vl = v
    
    return interval.Interval(vl, vr)

def straddles(x):
    return (left(x) <= 0) and (0 <= right(x)) # includes zero

def straddlingzero(x):
    return (left(x) < 0) and (0 < right(x)) # neglects zero as an endpoint
