import warnings
import numpy as np
import scipy.stats as sps
import scipy.special as spsp
import interval
import pbox
import utils


##-- todo
# - broken
# poissonbinomial
# hypergeometric0
# hypergeometric



# N
# normal0
# normal
# gaussian
# benford
# bernoulli
# beta
# betabinomial
# binomial 
# cantor
# cauchy
# chi
# chisquared
# dagum
# delta
# dirac
# discreteuniform 
# erlang
# exponential
# exponentialpower
# extremevalue
# F
# frechet
# gamma 
# gamma1
# gamma2
# gammaexponential
# generalizedextremevalue
# geometric
# gumbel
# hyperbolicsecant
# hypergeometric0
# hypergeometric
# hypoexponential
# inversegamma
# kumaraswamy
# laplace
# logistic 
# lognormal0
# lognormal
# logtriangular
# loguniform
# lomax
# muth
# negativebinomial 
# pareto
# pascal
# poisson
# poissonbinomial
# powerfunction
# quadratic
# quadratic1
# rayleigh
# reciprocal 
# rectangular
# shiftedloglogistic
# skellam
# skewnormal
# stable
# student
# trapezoidal
# Trapezoidal
# triangular
# uquadratic
# uquadratic1
# uniform
# voigt
# wakeby
# weibull
# wilcoxon

# # -- Pbox constructor utility functions; very slow...
# # --  I suspect all env* functions represent execution 
# # --  bottlenecks.

def closest(r, s):
    '''
    returns the value in the interval r that is closest to the scalar s
    '''
    if s < utils.left(r): 
        return utils.left(r)
    elif right(r) < s:
        return utils.right(r)
    else:
        return(s)

def nothing(val):
    return np.abs(val) < 1e-100

def lambertW(x):
    if not isinstance(x, np.array):
        x = np.array(x)
        
    prec = 1e-12
    
    w = np.where(500 < x, np.log(x - 4.0) - (1.0 - 1.0 / np.log(x)) * np.log(np.log(x)), np.nan)
    lx1 = np.log(x + 1.0)
    
    w = np.where((-1 / np.exp(1) < x) and (x <= 500), 0.665 * (1 + 0.0195 * lx1) * lx1 + 0.04, w)
    
    for i in range(100):
        wew = w * np.exp(w)
        wpew = (w + 1) * np.exp(w)
        w = w - (wew - x) / (wpew - (w + 2) * (wew - x) / (2 * w + 2))
        
    return w


def pyrange(array):
    return np.array([np.min(array), np.max(array)])

def intparm(*args):
    for arg in args:
        if utils.left(arg) < utils.right(arg):
            return True

    return False


def env1_2(dname, i, **kwargs):
    a = utils.env(
        dname(utils.left(i), **kwargs),
        dname(utils.right(i), **kwargs)
    )
    
    return a

def env2_4(dname, i, j, **kwargs):
    a = utils.env(
        dname(utils.left(i), utils.left(j), **kwargs),
        dname(utils.right(i), utils.left(j), **kwargs),
        dname(utils.left(i), utils.right(j), **kwargs),
        dname(utils.right(i), utils.right(j), **kwargs)
    )
    
    return a

def env2_2(dname, i, j, **kwargs):
    if utils.right(i) < utils.left(j):
        a = utils.env(
            dname(utils.left(i), utils.left(j), **kwargs),
            dname(utils.left(i), utils.right(j), **kwargs),
            dname(utils.right(i), utils.left(j), **kwargs),
            dname(utils.right(i), utils.right(j), **kwargs)
        )
        
    elif (utils.left(i) < utils.left(j)) and (utils.right(i) < utils.right(j)):
        a = utils.env(
            dname(utils.left(i), utils.left(j), **kwargs),
            utils.left(j),
            dname(utils.left(i), utils.right(j), **kwargs),
            dname(utils.right(i), utils.right(j), **kwargs)
        )
        
    elif (utils.left(j) <= utils.left(i)) and (utils.right(i) <= utils.right(j)):
        a = utils.env(
            utils.left(i),
            dname(utils.left(i), utils.right(j), **kwargs),
            dname(utils.right(i), utils.right(j), **kwargs)
        )
        
    elif (utils.left(i) <= utils.left(j)) and (utils.right(j) <= utils.right(i)):
        a = utils.env(
            dname(utils.left(i), utils.left(j), **kwargs),
            dname(utils.left(i), utils.right(j), **kwargs),
            utils.right(j)
        )
        
    elif utils.right(j) >= utils.left(i):
        a = utils.env(
            utils.left(i),
            utils.right(j),
            dname(utils.left(i), utils.right(j), **kwargs)
        ) 
        
    else: 
        raise Exception('Minimum must be smaller than maximum')
        
    if dname == Sloguniform:
        a.shape = 'loguniform'
    else:
        a.shape = 'uniform'
    
    return a
 
def env3_8(dname, i, j, k, **kwargs):
    a = pbox.Pbox(-np.inf, np.inf)
    
    for I in np.unique(pyrange(i)):
        for J in np.unique(pyrange(j)):
            for K in np.unique(pyrange(k)):
                a = utils.env(a, dname(I, J, K, **kwargs))
                
    return a

def env4_16(dname, i, j, k, l, **kwargs):
    a = pbox.Pbox(-np.inf, np.inf)

    for I in np.unique(pyrange(i)):
        for J in np.unique(pyrange(j)):
            for K in np.unique(pyrange(k)):
                for L in np.unique(pyrange(l)):
                    a = utils.env(a, dname(I, J, K, L, **kwargs))
                    
    return a

def env5_32(dname, i, j, k, l, m, **kwargs):
    a = pbox.Pbox(-np.inf, np.inf)
    
    for I in np.unique(pyrange(i)):
        for J in np.unique(pyrange(j)):
            for K in np.unique(pyrange(k)):
                for L in np.unique(pyrange(l)):
                    for M in np.unique(pyrange(m)):
                        a = utils.env(a, dname(I, J, K, L, M, **kwargs))
                        
    return a    


# # - distribution function definitions

def qarcsin(p):
	return (1 / 2) - np.cos(p * np.pi) / 2

def arcsin(pbox_steps=200):
    pbox_parms = {
        'lo': qarcsin(utils.ii(pbox_steps=pbox_steps)), 
        'hi': qarcsin(utils.jj(pbox_steps=pbox_steps)), 
        'shape': 'arc sine',
        'mean_lo': 1 / 2, 
        'mean_hi': 1 / 2
    }
    
    return pbox.Pbox(**pbox_parms)

def qarctan(p, alpha, phi): # untested
	return phi + np.tan(-np.arctan(alpha * phi) + p * np.arctan(alpha * phi) + p * np.pi / 2) / alpha

def bernoulli(p, pbox_steps=200): # untested
	lo = np.array([0 if i < 1 - utils.left(p) else 1 for i in utils.ii(pbox_steps=pbox_steps)])
	hi = np.array([0 if i < 1 - utils.right(p) else 1 for i in utils.jj(pbox_steps=pbox_steps)])

	pbox_parms = {
		'lo': lo, 
		'hi': hi, 
		'shape': 'Bernoulli', 
		'mean_lo': utils.left(p), 
		'mean_hi': utils.right(p), 
		'var_lo': 0.25 - (utils.right(p) - 0.5) ** 2, 
		'var_hi': 0.25 - (utils.left(p) - 0.5) ** 2
	}

	return pbox.Pbox(**pbox_parms)


def beta(shape1, shape2, pbox_steps=200):    
    if intparm(shape1, shape2): 
        return env2_4(beta, shape1, shape2, pbox_steps=pbox_steps)
    
    if (shape1 == 0) and (shape2 == 0):
        return pbox.Pbox(0, 1, shape='beta')
    
    if shape1 == 0:
        return pbox.Pbox(0, shape='beta')
    
    if shape2 == 0:
        return pbox.Pbox(1, shape='beta')
    
    shape1 = np.float64(shape1) # np.float to avoid div by 0 err
    
    m = shape1 / (shape1 + shape2)
    v = shape1 * shape2 / ((shape1 + shape2) ** 2 * (shape1 + shape2 + 1))
    
    pbox_parms = {
        'lo': sps.beta.ppf(utils.ii(pbox_steps=pbox_steps), shape1, shape2),
        'hi': sps.beta.ppf(utils.jj(pbox_steps=pbox_steps), shape1, shape2),
        'shape': 'beta',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

Be = beta # beta alias...


def betabinomial(a, b, n, pbox_steps=200):    
    if intparm(a, b):
        return env3_8(betabinomial, a, b, n, pbox_steps=pbox_steps)
    
    a = np.float64(a) # np.float to avoid div by 0 err
    
    k = np.array(range(n + 1))
    p = np.cumsum(spsp.binom(n, k) * spsp.beta(k + a, n - k + b) / spsp.beta(a, b))
    m = n * a / (a + b)
    v = n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1))
    
    rep_what = np.array(range(n + 1), dtype='int64').reshape(n + 1, 1)
    rep_howmanytimes_c = np.diff(np.array(np.ceil(np.array([0, *p]) * pbox_steps), dtype='int64'))
    rep_howmanytimes_f = np.diff(np.array(np.floor(np.array([0, *p]) * pbox_steps), dtype='int64'))
    
    lo = np.repeat(rep_what, rep_howmanytimes_c)[range(pbox_steps)]
    hi = np.array([*np.repeat(rep_what, rep_howmanytimes_f), n])[range(pbox_steps)]
    
    pbox_parms = {
        'lo': lo,
        'hi': hi,
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v,
        'shape': 'beta-binomial'
    }
    
    return pbox.Pbox(**pbox_parms)

BB = betabinomial # betabinomial alias...


def binomial(size=None, prob=None, mean=None, std=None, pbox_steps=200):    
    if (size is None) and (mean is not None) and (std is not None):
        size = mean / (1 - std ** 2 / mean)
        
    if (prob is None) and (mean is not None) and (std is not None):
        prob = 1 - std ** 2 / mean
        
    if intparm(size, prob):
        return env2_4(binomial, size, prob, pbox_steps=pbox_steps)
    
    std = np.float64(std) # np.float to avoid div by 0 err
    
    m = size * prob
    v = size * prob * (1 - prob)
    
    lo = sps.binom.ppf(utils.ii(pbox_steps=pbox_steps), size, prob)
    hi = sps.binom.ppf(utils.jj(pbox_steps=pbox_steps), size, prob)
    
    pbox_parms = {
        'lo': lo,
        'hi': hi, 
        'shape': 'binomial',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

B = binomial # binomial alias...

def bradford(location, scale, shape, pbox_steps=200):    
    if intparm(location, scale, shape):
        return env3_8(bradford, location, scale, shape, pbox_steps=pbox_steps)
    
    shape =  np.float64(shape) # np.float to avoid div by 0 err

    k = np.log(shape + 1)
    m = (shape * (scale - location) + k * (location * (shape + 1) - scale)) / (shape * k)
    v = ((scale - location) ** 2 * (shape * (k - 2) + 2 * k)) / (2 * shape * k ** 2)
    lo = sps.bradford.ppf(utils.iii(pbox_steps=pbox_steps), shape, location, scale)
    hi = sps.bradford.ppf(utils.jjj(pbox_steps=pbox_steps), shape, location, scale)
    
    pbox_parms = {
        'lo': lo,
        'hi': hi,
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

def qburr(p, c, k):
    return np.exp(np.log(np.exp(-np.log(1 - p) / k) - 1) / c)

def burr(c, k, pbox_steps=200):
    if intparm(c, k):
        return env2_4(burr, c, k, pbox_steps=pbox_steps)
    
    k = np.float64(k) # np.float to avoid div by 0 err
    m  = k * spsp.beta(k - 1 / c, 1 + 1 / c)
    
    pbox_parms = {
        'lo': qburr(utils.ii(pbox_steps=pbox_steps), c, k),
        'hi': qburr(utils.jjj(pbox_steps=pbox_steps), c, k),
        'shape': 'Burr',
        'mean_lo': m,
        'mean_hi': m
    }
    
    return pbox.Pbox(**pbox_parms)

# def cantor(*args):
#     fnp1 = lambda x: (fn[k %% 2 == 0] / 3), (fn[k %% 2 == 0] / 3) + (2 / 3)
# # cantor <- function(...){ 
# #   fnp1 <- function(fn) {fa <- fn[k %% 2 == 0] / 3; fnext <- c(fa, fa + 2/3); fnext }; 
# #   k <- 1:(Pbox$steps+2); 
# #   fn <- k / (Pbox$steps+2); 
# #   for (kk in 1:20) fn <- fnp1(fn); 
# #   i <- 1:Pbox$steps; 
# #   j <- 2:(Pbox$steps+1); 
# #   pbox(u=fn[i], d=fn[j], shape='Cantor', ml=0.5, mh=0.5, vl=1/8, vh=1/8, ...) 
# #   }

def cauchy(location, scale, pbox_steps=200):
    if intparm(location, scale):
        return env2_4(cauchy, location, scale, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': sps.cauchy.ppf(utils.iii(pbox_steps=pbox_steps), location, scale),
        'hi': sps.cauchy.ppf(utils.jjj(pbox_steps=pbox_steps), location, scale),
        'shape': 'Cauchy',
        'mean_lo': -np.inf,
        'mean_hi': np.inf,
        'var_lo': 0,
        'var_hi': np.inf
    }
    
    return pbox.Pbox(**pbox_parms)


def chi(df, pbox_steps=200):
    if intparm(df):
        return env1_2(chi, df, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': sps.chi.ppf(utils.ii(pbox_steps=pbox_steps), df),
        'hi': sps.chi.ppf(utils.jjj(pbox_steps=pbox_steps), df),
        'shape': 'chi',
        'mean_lo': df,
        'mean_hi': df,
        'var_lo': 2 * df,
        'var_hi': 2 * df
    }
    
    return pbox.Pbox(**pbox_parms)

def chisquared(df, pbox_steps=200):
    if intparm(df):
        return env1_2(chisquared, df, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': sps.chi2.ppf(utils.ii(pbox_steps=pbox_steps), df),
        'hi': sps.chi2.ppf(utils.jjj(pbox_steps=pbox_steps), df),
        'shape': 'chi-squared',
        'mean_lo': df,
        'mean_hi': df,
        'var_lo': 2 * df,
        'var_hi': 2 * df
    }
    
    return pbox.Pbox(**pbox_parms)

def qdagum(p, q, a, b):
    return b * np.exp(np.log(np.exp(np.log(p) / (-q)) - 1) / (-a))

def dagum(q, a, b, pbox_steps=200):
    if intparm(q, a, b):
        return env3_8(dagum, q, a, b, pbox_steps=pbox_steps)
    
    mean_lo, var_lo = 0, 0
    mean_hi, var_hi = np.inf, np.inf
    
    if 1 < a:
        mean_lo = (-b / a) * (spsp.gamma(-1 / a) * spsp.gamma(1 / a + q)) / spsp.gamma(q)
        mean_hi = (-b / a) * (spsp.gamma(-1 / a) * spsp.gamma(1 / a + q)) / spsp.gamma(q)
        
    if 2 < a:
        var_lo = (-(b ** 2) / (a ** 2)) * (2 * a * spsp.gamma(-2 / a) * spsp.gamma(2 / a + q) / spsp.gamma(q) + (spsp.gamma(-1 / a) * spsp.gamma(1 / a + q) / spsp.gamma(q)) ** 2)
        var_hi = (-(b ** 2) / (a ** 2)) * (2 * a * spsp.gamma(-2 / a) * spsp.gamma(2 / a + q) / spsp.gamma(q) + (spsp.gamma(-1 / a) * spsp.gamma(1 / a + q) / spsp.gamma(q)) ** 2)
        
    pbox_parms = {
        'lo': qdagum(utils.ii(pbox_steps=pbox_steps), q, a, b),
        'hi': qdagum(utils.jjj(pbox_steps=pbox_steps), q, a, b),
        'shape': 'Dagum',
        'mean_lo': mean_lo,
        'mean_hi': mean_hi,
        'var_lo': var_lo,
        'var_hi': var_hi
    }
    
    return pbox.Pbox(**pbox_parms)


def delta(x):
    return pbox.Pbox(utils.left(x), utils.right(x), shape='delta')

dirac = delta # delta alias...

# def discreteuniform(max=None, mean=None, pbox_steps=200):
#     if (mas is None) and (mean is not None):
#         max = 2 * mean
#     if intparm(max):
#         return env1_2(discreteuniform, max, pbox_steps=pbox_steps)
#
#
#
# # discreteuniform <- function(max=NULL, mean=NULL, name='', ...)  {
# #   if (is.null(max) & !is.null(mean)) max <- 2 * mean
# #   if (intparm(max)) return(env1.2(discreteuniform,max,name=name, ...))
# #   pbox(Min(max,int(uniform(0, max+1))), shape='discrete uniform', ml=max/2, mh=max/2, vl=max*(max+2)/12, vh=max*(max+2)/12)  
# #   }


def exponential(mean, pbox_steps=200):
    if intparm(mean):
        return env1_2(exponential, mean, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': sps.expon.ppf(utils.ii(pbox_steps=pbox_steps), scale = mean),
        'hi': sps.expon.ppf(utils.jjj(pbox_steps=pbox_steps), scale = mean),
        'shape': 'exponential',
        'mean_lo': mean,
        'mean_hi': mean,
        'var_lo': mean ** 2,
        'var_hi': mean ** 2
    }
    
    return pbox.Pbox(**pbox_parms)

def qexponentialpower(p, lmbda, kappa):
    return np.exp(-(np.log(lmbda) - np.log(np.log(1 - np.log(1 - p)))) / kappa)

def exponentialpower(lmbda, kappa, pbox_steps=200):
    if intparm(lmbda, kappa):
        return env2_4(exponentialpower, lmbda, kappa, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': qexponentialpower(utils.ii(pbox_steps=pbox_steps), lmbda, kappa), 
        'hi': qexponentialpower(utils.jjj(pbox_steps=pbox_steps), lmbda, kappa), 
        'shape': 'exponential-power'
    }
    
    return pbox.Pbox(**pbox_parms)

def fishersnedecor(df1, df2, pbox_steps=200):
    if intparm(df1, df2):
        return env2_4(fishersnedecor, df1, df2, pbox_steps=pbox_steps)
    
    df2 = np.float64(df2) # np.float to avoid div by 0 err
    
    m = df2 / (df2 - 2)
    v = 2 * df2 ** 2 * (df1 + df2 - 2) / (df1 * (df2 - 2) ** 2 * (df2 - 4))    

    pbox_parms = {
        'lo': sps.f.ppf(utils.ii(pbox_steps=pbox_steps), df1, df2),
        'hi': sps.f.ppf(utils.jjj(pbox_steps=pbox_steps), df1, df2),
        'shape': 'F',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

f = fishersnedecor # fishersnedecor alias...
F = fishersnedecor # fishersnedecor alias...
    
def qfrechet(p, b, c):
    return b * np.exp((-1 / c) * np.log(np.log(1 / p)))
    
def frechet(b, c, pbox_steps=200):
    if intparm(b, c):
        return env2_4(frechet, b, c, pbox_steps=pbox_steps)
    
    mean_lo = -np.inf
    mean_hi = np.inf
    
    if 1 < c:
        mean_lo = b * spsp.gamma(np.float64(1) - 1 / c) # np.float to avoid div by 0 err
        mean_hi = b * spsp.gamma(np.float64(1) - 1 / c) # np.float to avoid div by 0 err
        
    var_lo = 0
    var_hi = np.inf
    
    if 2 < c:
        e = spsp.gamma(np.float64(1) - 1 / c) # np.float to avoid div by 0 err
        var_lo = b * b * (spsp.gamme(np.float64(1) - 2 / c) - e * e) # np.float to avoid div by 0 err
        var_hi = b * b * (spsp.gamme(np.float64(1) - 2 / c) - e * e) # np.float to avoid div by 0 err
    
    pbox_parms = {
        'lo': qfrechet(utils.ii(pbox_steps=pbox_steps), b, c),
        'hi': qfrechet(utils.jjj(pbox_steps=pbox_steps), b, c),
        'shape': 'Frechet',
        'mean_lo': mean_lo,
        'mean_hi': mean_hi,
        'var_lo': var_lo,
        'var_hi': var_hi
    }
    
    return pbox.Pbox(**pbox_parms)
    
def gamma(scale, shape, pbox_steps=200):
    if intparm(scale, shape):
        return env2_4(gamma, scale, shape, pbox_steps=pbox_steps)
    
    m = scale * shape
    v = scale ** 2 * shape
    
    pbox_parms = {
        'lo': sps.gamma.ppf(utils.ii(pbox_steps=pbox_steps), shape, scale=scale),
        'hi': sps.gamma.ppf(utils.jjj(pbox_steps=pbox_steps), shape, scale=scale),
        'shape': 'gamma',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

erlang = gamma # gamma alias...

def gamma1(mu, sigma, pbox_steps=200):
    # np.float to avoid div by 0 err
    return gamma(scale=np.float64(sigma) ** 2 / mu, shape=(np.float64(mu) / sigma) ** 2, pbox_steps=pbox_steps)

def gamma2(shape, rate=1, pbox_steps=200): # scale = 1 / rate,
    scale = np.float64(1) / rate # np.float to avoid div by 0 err
    return gamma(scale, shape, pbox_steps=pbox_steps)

def inversegamma(a, b, pbox_steps=200):
    if intparm(a, b):
        return env2_4(inversegamma, a, b, pbox_steps=pbox_steps)
    
    b = np.float64(b) # np.float to avoid div by 0 err
    m = b / (a - 1)
    v = b ** 2 / ((a - 1) ** 2 * (a - 2))
    
    pbox_parms = {
        'lo': np.flip(1 / sps.gamma.ppf(utils.jj(pbox_steps=pbox_steps), a, scale=1 / b)),
        'hi': np.flip(1 / sps.gamma.ppf(utils.iii(pbox_steps=pbox_steps), a, scale=1 / b)),
        'shape': 'inverse gamma',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

def qgammaexponential(p=None, scale=1, shape=None):
    rate = 1 / scalex
    return rate * ((1 - p) ** (-1 / shape) - 1)

def gammaexponential(scale=1, shape=None, pbox_steps=200):
    rate = 1 / scale
    scale = 1 / rate
    
    if intparm(scale, shape):
        return env2_4(gammaexponential, scale, shape, pbox_steps=pbox_steps)
    
    mean_lo = -np.inf
    mean_hi = np.inf
    var_lo = 0
    var_hi = np.inf
    
    pbox_parms = {
        'lo': qgammaexponential(utils.ii(pbox_steps=pbox_steps), scale, shape),
        'hi': qgammaexponential(utils.jjj(pbox_steps=pbox_steps), scale, shape),
        'shape': 'gamma-exponential',
        'mean_lo': mean_lo,
        'mean_hi': mean_hi,
        'var_lo': var_lo,
        'var_hi': var_hi
    }
    
    return pbox.Pbox(**pbox_parms)

def geometric(prob=None, mean=None, pbox_steps=200):
    if (prob is None) and (mean is not None):
        prob = 1 / (1 + mean)
        
    if intparm(prob):
        return env1_2(geometric, prob, pbox_steps=pbox_steps)
    
    m = (1 - prob) / prob
    v = (1 - prob) / prob ** 2
    
    pbox_parms = {
        'lo': sps.geom.ppf(utils.ii(pbox_steps=pbox_steps), p=prob),
        'hi': sps.geom.ppf(utils.jjj(pbox_steps=pbox_steps), p=prob),
        'shape': 'geometric',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

pascal = geometric # geometric alias...

def qgumbel(p, a, b):
    return a - b * np.log(np.log(1.0 / p))

def qgev(p, a, b, c):
    if c == 0:
        return qgumbel(p, a, b)
    else: 
        return a + b * (np.exp(-c * np.log(-log(p))) - 1) / c

def generalizedextremevalue(a=0, b=1, c=0, pbox_steps=200):
    if intparm(a, b, c):
        return env3_8(generalizedextremevalue, a, b, c, pbox_steps=pbox_steps)
    
    if np.all(c == 0):
        return gumbel(a, b, pbox_steps=pbox_steps)
    
    g1 = spsp.gamma(1 - c)
    g2 = spsp.gamma(1 - 2 * c)
    
    if c == 0:
        m = a + b * 0.57721566490153
    elif 1 <= c:
        m = np.inf
    elif c < 1:
        m = a + (g1 - 1) * (b / c)
        
    if c == 0:
        v = (b * np.pi) ** 2 / 6
    elif 0.5 <= c:
        v = np.inf
    elif c < 1:
        v = (b / c) ** 2 * (g2 - g1 ** 2)
        
    if 0 < c:
        lo = qgev(utils.ii(pbox_steps=pbox_steps), a, b, c)
        hi = qgev(utils.jjj(pbox_steps=pbox_steps), a, b, c)
    
    if c < 0:
        lo = qgev(utils.iii(pbox_steps=pbox_steps), a, b, c)
        hi = qgev(utils.jj(pbox_steps=pbox_steps), a, b, c)
        
    pbox_parms = {
        'lo': lo,
        'hi': hi,
        'shape': 'generalized extreme value',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)
        

GEV = generalizedextremevalue # generalizedextremevalue alias...
gev = generalizedextremevalue # generalizedextremevalue alias...

def gumbel(a=None, b=None, mean=None, std=None, var=None, pbox_steps=200):
    em = 0.577215665
    
    if (std is None) and (var is not None):
        std = np.sqrt(var)
        
    if (a is None) and (b is None):
        a = mean - std * em * np.sqrt(6) / np.pi
        b = std * np.sqrt(6) / np.pi
        
    if intparm(a, b):
        return env2_4(gumbel, a, b, pbox_steps=pbox_steps)
    
    if mean is None:
        mean = a + b * em
    
    if var is None:
        var = ((b * np.pi) ** 2) / 6
        
    pbox_parms = {
        'lo': qgumbel(utils.iii(pbox_steps=pbox_steps), a, b),
        'hi': qgumbel(utils.jjj(pbox_steps=pbox_steps), a, b),
        'shape': 'Gumbel',
        'mean_lo': mean,
        'mean_hi': mean,
        'var_lo': var,
        'var_hi': var
    }
    
    return pbox.Pbox(**pbox_parms)
    

extremevalue = gumbel # gumbel alias...

def hyperbolicsecant(pbox_steps=200, *args):
    pbox_parms = {
        'lo': (2 / np.pi) * np.log(np.tan(utils.iii(pbox_steps=pbox_steps) * np.pi / 2)),
        'hi': (2 / np.pi) * np.log(np.tan(utils.jjj(pbox_steps=pbox_steps) * np.pi / 2)),
        'shape': 'hyperbolic secant',
        'mean_lo': 0,
        'mean_hi': 0,
        'var_lo': 1,
        'var_hi': 1
    }
    
    return pbox.Pbox(**pbox_parms)


Hyperbolicsecant = hyperbolicsecant # hyperbolicsecant alias...


# hypergeometric0 and hypergeometric do not yet match R results     
# it looks like the ppf parameter specification
# is not the same as qhpyer in R
def hypergeometric0(m, n, k, pbox_steps=200):
    if intparm(m, n, k):
        return env3_8(hypergeometric0)
    
    N = m + n
    mean = k / (1 + n / m)
    v = (k * m / N) * (1 - m / N) * (N - k) / (N - 1)
    
    print(mean)
    print(v)
    print(sps.hypergeom.ppf(utils.iii(pbox_steps=pbox_steps), m, n, k))
    
    pbox_parms = {
        'lo': sps.hypergeom.ppf(utils.iii(pbox_steps=pbox_steps), m, n, k),
        'hi': sps.hypergeom.ppf(utils.jjj(pbox_steps=pbox_steps), m, n, k),
        'shape': 'hypergeometric',
        'mean_lo': mean,
        'mean_hi': mean,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)
    

def hypergeometric(whites=None, blacks=None, draws=None, balls=None, pbox_steps=200): 
    if (blacks is None) and (balls is not None) and (whites is not None):
        blacks = balls - whites
        
    if (whites is None) and (balls is not None) and (blacks is not None):
        whites = balls - blacks
        
    if whites + blacks < draws:
        raise Exception('too many draws to specify the hypergeometric distribution')
        
    if (blacks is not None) and (whites is not None) and (draws is not None):
        return hypergeometric0(whites, blacks, draws, pbox_steps=pbox_steps)
    else:
        raise Exception('not enough information to specify the hypergeometric distribution')


def qhypoexponential(p, lmbda):
    return -np.log(1 - p) / lmbda
        
def hypoexponential(lmbda, pbox_steps=200):
    if intparm(lmbda):
        return env1_2(hypoexponential, lmbda, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': qhypoexponential(utils.ii(pbox_steps=pbox_steps),lmbda),
        'hi': qhypoexponential(utils.jjj(pbox_steps=pbox_steps), lmbda),
        'shape': 'hypoexponential'
    }
    
    return pbox.Pbox(**pbox_parms)


def qkumaraswamy(p, a, b): 
    return np.exp((1 / a) * np.log(1 - np.exp((1 / b) * np.log(1 - p))))

def kumaraswamy(a, b, pbox_steps=200):
    if intparm(a, b):
        return env2_4(kumaraswamy, a, b, pbox_steps=pbox_steps)
    
    m = (b * spsp.gamma(1 + 1 / a) * spsp.gamma(b)) / spsp.gamma(1 + b + 1 / a)
    v = (b * spsp.gamma(1 + 2 / a) * spsp.gamma(b)) / spsp.gamma(1 + b + 2 / a)
    v = v - m ** 2
    
    pbox_parms = {
        'lo': qkumaraswamy(utils.ii(pbox_steps=pbox_steps), a, b),
        'hi': qkumaraswamy(utils.jj(pbox_steps=pbox_steps), a, b),
        'shape': 'Kumaraswamy',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)


def qlaplace(p, a, b):
    return np.array([a + b * np.log(2.0 * pp) if pp <= 0.5 else a - b * np.log(2.0 * (1.0 - pp)) for pp in p])


def laplace(a, b, pbox_steps=200):
    if intparm(a, b):
        return env2_4(laplace, a, b, pbox_steps=pbox_steps)
    
    m = a
    v = 2 * b * b
    
    pbox_parms = {
        'lo': qlaplace(utils.iii(pbox_steps=pbox_steps), a, b),
        'hi': qlaplace(utils.jjj(pbox_steps=pbox_steps), a, b),
        'shape': 'Laplace',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)


def logistic(location, scale, pbox_steps=200):
    if intparm(location, scale):
        return env2_4(logistic, location, scale, pbox_steps=pbox_steps)
    
    m = location
    v = (np.pi * scale) ** 2 / 3
    
    pbox_parms = {
        'lo': sps.logistic.ppf(utils.iii(pbox_steps=pbox_steps), location, scale),
        'hi': sps.logistic.ppf(utils.jjj(pbox_steps=pbox_steps), location, scale),
        'shape': 'logistic',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

def qloglogistic(p, lmbda, kappa):
    return np.exp(-(np.log(lmbda) - np.log(-p / (p - 1)) / kappa))

def loglogistic(lmbda, kappa, pbox_steps=200):
    if intparm(lmbda, kappa):
        return env2_4(loglogistic, lmbda, kappa, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': qloglogistic(utils.ii(pbox_steps=pbox_steps),lmbda, kappa),
        'hi': qloglogistic(utils.jjj(pbox_steps=pbox_steps),lmbda, kappa),
        'shape': 'loglogistic'
    }
    
    return pbox.Pbox(**pbox_parms)

def lognormal0(meanlog, stdlog, pbox_steps=200):
    if intparm(meanlog, stdlog):
        return env2_4(lognormal10, meanlog, stdlog, pbox_steps=pbox_steps)
    
    m = np.exp(meanlog + 0.5 * stdlog * stdlog)
    v = np.exp(2.0 * meanlog + stdlog * stdlog) * (np.exp(stdlog * stdlog) - 1.0)
    
    pbox_parms = {
        'lo': sps.lognorm.ppf(utils.iii(pbox_steps=pbox_steps), stdlog, scale=np.exp(meanlog)),
        'hi': sps.lognorm.ppf(utils.jjj(pbox_steps=pbox_steps), stdlog, scale=np.exp(meanlog)),
        'shape': 'lognormal',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

def lognormal(mean=None, std=None, meanlog=None, stdlog=None, median=None, cv=None, pbox_steps=200):
    if (meanlog is None) and (median is not None):
        meanlog = np.log(median)
        
    if (stdlog is None) and (cv is not None):
        stdlog = np.sqrt(np.log(cv ** 2 + 1))
    
    if (meanlog is None) and (mean is not None) and (std is not None):
        meanlog = np.log(mean ** 2 / np.sqrt(mean ** 2 + std ** 2))
    
    if (stdlog is None) and (mean is not None) and (std is not None):
        stdlog = np.sqrt(np.log((mean **2 + std ** 2)/ mean **2))
    
    if (meanlog is not None) and (stdlog is not None):
        return lognormal0(meanlog, stdlog, pbox_steps=pbox_steps)
    else:
        raise Exception('not enough information to specify the lognormal distribution')

L = lognormal # lognormal alias...

def qlogtriangular(p, i_min, i_mode, i_max):
    a = np.log(i_min)
    b = np.log(i_mode) # could this really be correct?
    c = np.log(i_max)
    
    return np.exp(qtriangular(p, a, b, c))

def logtriangular(min=None, mode=None, max=None, minlog=None, midlog=None, maxlog=None, pbox_steps=200):
    if (min is None) and (minlog is not None):
        min = np.exp(minlog)
        
    if (max is None) and (maxlog is not None):
        max = np.exp(maxlog)
    
    def logt_innerfunc0(a, b): 
        if nothing(a - b): 
            return -2 * ((a + b) / 2) ** 2 
        else: 
            return (a ** 2 - b ** 2) / np.log(b / a)
        
    def logt_innerfunc(a, b): 
        if nothing(a - b): 
            return -2 * ((a + b) / 2) ** 2 
        else: 
            return (a ** 2 - b ** 2) / np.log(b / a)

    if intparm(min, mode, max):
        return env3_8(logtriangular, min, mode, mode, pbox_steps=pbox_steps)
    
    m = (2 / np.log(max / min)) * (logt_innerfunc0(max, mode) - logt_innerfunc0(mode, min))
    v = (logt_innerfunc(min, mode) - logt_innerfunc(mode, max)) / (2 * np.log(max / min)) - (m * m)
    
    pbox_parms = {
        'lo': qlogtriangular(utils.ii(pbox_steps=pbox_steps), min, mode, max),
        'hi': qlogtriangular(utils.jj(pbox_steps=pbox_steps), min, mode, max),
        'shape': 'logtriangular',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

def qloguniform(p, one, two):
    m = np.log(one)
    return np.exp((np.log(two) - m) * p + m)

def loguniform(min=None, max=None, minlog=None, maxlog=None, mean=None, std=None, pbox_steps=200):
    if (min is None) and (minlog is not None):
        return
    if (max is None) and (maxlog is not None):
        return
    if (max is None) and (mean is not None) and (std is not None) and (min is not None):
        return
    
    if intparm(min, max):
        return env2_4(loguniform, min, mode, pbox_steps=pbox_steps)
    
    mean = (max - min) / (np.log(max) - np.log(min))
    z = np.log(max / min)
    v = (mean - (max - min) / z) ** 2 + (max ** 2 - min ** 2) / (2 * z) - ((max - min) / z) ** 2
    
    pbox_parms = {
        'lo': qloguniform(utils.ii(pbox_steps=pbox_steps), min, max),
        'hi': qloguniform(utils.jj(pbox_steps=pbox_steps), min, max),
        'shape': 'loguniform',
        'mean_lo': mean,
        'mean_hi': mean,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)


def qlomax(p, lmbda, kappa):
    return ((1 - p) ** (-1 / kappa) - 1) / lmbda


def lomax(lmbda, kappa, pbox_steps=200):
    if intparm(lmbda, kappa):
        return env2_4(lomax, lmbda, kappa, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': qlomax(utils.ii(pbox_steps=pbox_steps), lmbda, kappa),
        'hi': qlomax(utils.jjj(pbox_steps=pbox_steps), lmbda, kappa),
        'shape': 'Lomax'
    }
    
    return pbox.Pbox(**pbox_parms)


def qmuth(p, a):
    return -(lambertW(-np.exp(np.log(1 - p) - 1 / a) / a) * a + 1 - np.log(1 - p) * a) / (a ** 2)

def muth(kappa, pbox_steps=200):
    if intparm(kappa):
        return env1_2(muth, kappa, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': qmuth(utils.ii(pbox_steps=pbox_steps), kappa),
        'hi': qmuth(utils.jjj(pbox_steps=pbox_steps), kappa),
        'shape': 'Muth'
    }
    
    return pbox.Pbox(**pbox_parms)

def negativebinomial(size, prob, pbox_steps=200):
    if intparm(size, prob):
        return env2_4(negativebinomial, size, prob, pbox_steps=pbox_steps)
    
    m = size * (1 / prob - 1)
    v = size * (1 - prob) / (prob ** 2)
    
    pbox_parms = {
        'lo': sps.nbinom.ppf(utils.ii(pbox_steps=pbox_steps), size, prob),
        'hi': sps.nbinom.ppf(utils.jjj(pbox_steps=pbox_steps), size, prob),
        'shape': 'negative binomial', 
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)


def normal0(mean, std, pbox_steps=200):
    if intparm(mean, std):
        return env2_4(normal0, mean, std, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': sps.norm.ppf(utils.iii(pbox_steps=pbox_steps), mean, std),
        'hi': sps.norm.ppf(utils.jjj(pbox_steps=pbox_steps), mean, std),
        'shape': 'normal',
        'mean_lo': mean,
        'mean_hi': mean,
        'var_lo': std ** 2,
        'var_hi': std ** 2
    }
    
    return pbox.Pbox(**pbox_parms)

def normal(mean=None, std=None, median=None, mode=None, cv=None, iqr=None, var=None, pbox_steps=200):
    if (mean is None) and (median is not None):
        mean = median
        
    if (mean is None) and (mode is not None):
        mean = mode
        
    if (std is None) and (cv is not None) and (mean is not None):
        std = mean * cv
        
    if (mean is not None) and (std is not None):
        return normal0(mean, std, pbox_steps=pbox_steps)
    else:
        raise Exception('not enough information to specify the normal distribution')

N = normal # normal alias...
Normal = normal # normal alias...
gaussian = normal # normal alias...

def skewnormal(location, scale, skew, pbox_steps=200):
    if intparm(location, scale):
        return env2_4(knewnormal, location, scale, pbox_steps=pbox_steps)
    
    d = skew / np.sqrt(1 + skew ** 2)
    m = location + scale * d * np.sqrt(2 / np.pi)
    v = scale ** 2 * (1 - 2 * d ** 2 / np.pi)
    
    pbox_parms = {
        'lo': sps.skewnorm.ppf(utils.iii(pbox_steps=pbox_steps), skew, location, scale),
        'hi': sps.skewnorm.ppf(utils.jjj(pbox_steps=pbox_steps), skew, location, scale),
        'shape': 'skew-normal',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

SN = skewnormal # skewnormal alias...

def qpareto(p, mode, c):
    return mode * np.exp((-1.0 / c) * np.log(1.0 - p))

def pareto(mode, c, pbox_steps=200):
    if intparm(mode, c):
        return env2_4(pareto, mode, c, pbox_steps=pbox_steps)
    
    mean_lo = -np.inf
    mean_hi = np.inf
    var_lo = 0
    var_hi = np.inf
    
    if 2 < c:
        var_lo = mode * mode * c / ((c - 1) * (c - 1) * (c - 2))
        var_hi = mode * mode * c / ((c - 1) * (c - 1) * (c - 2))
    
    pbox_parms = {
        'lo': qpareto(utils.iii(pbox_steps=pbox_steps), mode, c),
        'hi': qpareto(utils.jjj(pbox_steps=pbox_steps), mode, c),
        'shape': 'Pareto',
        'mean_lo': mean_lo,
        'mean_hi': mean_hi,
        'var_lo': var_lo,
        'var_hi': var_hi
    }
    
    return pbox.Pbox(**pbox_parms)

def qpowerfunction(p, b, c):
    return b * np.exp((1.0 / c) * np.log(p))

def powerfunction(b, c, pbox_steps=200):
    if intparm(b, c):
        return env2_4(powerfunction, b, c, pbox_steps=pbox_steps)
    
    m = b * c / (c + 1)
    v = b * b * c / ((c + 2) * (c + 1) * (c + 1))
    
    pbox_parms = {
        'lo': qpowerfunction(utils.iii(pbox_steps=pbox_steps), b, c),
        'hi': qpowerfunction(utils.jjj(pbox_steps=pbox_steps), b, c),
        'shape': 'power function',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)


def poisson(lmbda, pbox_steps=200):
    if intparm(lmbda):
        return env1_2(poisson, lmbda, pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': sps.poisson.ppf(utils.ii(pbox_steps=pbox_steps), lmbda),
        'hi': sps.poisson.ppf(utils.jjj(pbox_steps=pbox_steps), lmbda),
        'shape': 'Poisson',
        'mean_lo': lmbda,
        'mean_hi': lmbda,
        'var_lo': lmbda,
        'var_hi': lmbda
    }
    
    return pbox.Pbox(**pbox_parms)


def poissonbinomial(p, pbox_steps=200): #doesnt work
    if isinstance(p, interval.Interval):
        return env(poissonbinomial(utils.left(p)), poissonbinomial(utils.right(p)))
    
    try:
        n = len(p)
    except:
        n = 1
        
    m = np.sum(p)
    v = np.sum(p * (1 - p))
    Prk = np.repeat(0, n + 1)
    Prk[0] = np.prod(1 - p)
    
    T = lambda i: np.sum((1 / (1 / p - 1)) ** i)    
    Ti = np.array([T(i + 1) for i in range(n)])
    
    for k in range(n):
        i = range(k)
        Prk[k + 1] = (1 / k) * np.sum((-1) ** (i - 1) * Prk[k - i + 1] *  Ti[i])
        
    C = np.cumsum(Prk)
    i = utils.ii(pbox_steps=pbox_steps)
    j = utils.jj(pbox_steps=pbox_steps)
    
    lo = np.repeat(0, pbox_steps)
    hi = np.repeat(0, pbox_steps)
    
    for k in range(n):
        lo[C[k] <= i] = k
        
    for k in range(n):
        hi[C[k] <= j] = k 
    
#     lo = np.array(lo)
#     hi = np.array(hi)
    
    pbox_parms = {
        'lo': lo,
        'hi': hi,
        'shape': 'Poisson-binomial',
        'mean_lo': utils.left(m),
        'mean_hi': utils.right(m),
        'var_lo': left(v),
        'var_hi': right(v)
    }
    
    return pbox.Pbox(**pbox_parms)



# poissonbinomial <- function(p, name='') {  # p is an array of (possibly different) probability values, which can be scalars or intervals
#   # the distribution of the sum of independent Bernoulli trials, with possibly different probabilities of success p
#   # support = 0:(length(p)), mean = sum(p), variance = sum(p(1-p))
#   # poissonbinomial(p[1],p[2],p[3],...,p[n]) ~ binomial(n, p*) iff p[i]=p*, i=1,...n
#   # http://en.wikipedia.org/wiki/Poisson_binomial_distribution, 8 May 2012
#   # Wang, Y. H. (1993). "On the number of successes in independent trials". Statistica Sinica 3 (2): 295-312.
#   if (class(p) != 'numeric') return(env(poissonbinomial(left(p)), poissonbinomial(right(p))))
#   n = length(p)
#   m = sum(p)
#   v = sum(p*(1-p))
#   Prk = rep(0,n+1)
#   Prk[0+1] = prod(1-p)
#   T <- function(i) sum((1/(1/p-1))^i)    #sum((p[j]/(1-p[j]))^i)
#   Ti = rep(0,n)
#   for (i in 1:n) Ti[i] = T(i)
#   for (k in 1:n) {
#     i = 1:k
#     Prk[k +1] = (1/k) * sum((-1)^(i-1) * Prk[k-i +1] * Ti[i])
#     }
#   C = cumsum(Prk)
#   u = d = rep(0,Pbox$steps)
#   i = ii()
#   j = jj()
#   for (k in 1:n) u[C[k] <= i] = k
#   for (k in 1:n) d[C[k] <= j] = k
#   pbox(u, d, shape='Poisson-binomial', name=name, ml=left(m), mh=right(m), vl=left(v), vh=right(v))
#   }


def unitqquad(P):
    # June 2009 Draft NASA Report "Measurement Uncertainty Analysis: Principles and Methods". Appendix B, page 151 
    d = 4 * P - 2
    h = d ** 2 / 4 + -1
    i = np.sqrt(d ** 2 / 4 - h)
    j = np.where(i < 0, (-1 * -i) ** (1 / 3), i ** (1 / 3))
    k = np.arccos(-(d / (2 * i)))
    m = np.cos(k / 3)
    n = 1.73205080756888 * np.sin(k / 3)
    return -j * (m - n)

def qquadratic(p, m, a):
    return unitqquad(p) * (a - m) + m

def quadratic(mean, max, pbox_steps=200):
    if mean == max:
        return pbox.Pbox(mean)
    
    if max < mean:
        warnings.warn('max is less than mean...')
        
    if mean >= max:
        raise Exception('mean is greater than or equal to max')
        
    if intparm(mean, max):
        return env2_4(quadratic, mean, max, pbox_steps=pbox_steps)
    
    m = mean
    v = (max - mean) ** 2 / 5
    
    pbox_parms = {
        'lo': qquadratic(utils.ii(pbox_steps=pbox_steps), mean, max),
        'hi': qquadratic(utils.jj(pbox_steps=pbox_steps), mean, max),
        'shape': 'quadratic',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

def quadratic1(mean=0, std=1, pbox_steps=200):
    return quadratic(mean, np.sqrt(5) * std + mean)

def ququadratic(p,a,b):
    # http://en.wikipedia.org/wiki/U-quadratic_distribution
    cuberoot = lambda z: np.where(0 < z, z ** (1 / 3), -(np.abs(z) ** (1 / 3)))
    alpha = 12 / (b - a) ** 3
    beta = (a + b) / 2
    return beta + cuberoot(3 * p / alpha - (beta - a) ** 3)

def uquadratic(a=0, b=1, pbox_steps=200):
    if a == b:
        return Pbox(a)

    if b < a:
        warnings.warn('max is less than mean...')
        
    if a >= b:
        raise Exception('mean is greater than or equal to max')    
        
    if intparm(a, b):
        return env2_4(uquadratic, a, b, pbox_steps=pbox_steps)
    
    m = (a + b) / 2
    v = 3 * (b - a) ** 2 / 20
    
    pbox_parms = {
        'lo': ququadratic(utils.ii(pbox_steps=pbox_steps), a, b),
        'hi': ququadratic(utils.jj(pbox_steps=pbox_steps), a, b),
        'shape': 'uquadratic',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)


def uquadratic1(mean=0, std=1, pbox_steps=200):
    return uquadratic(mean - std * np.sqrt(5 / 3), mean + std + np.sqrt(5 / 3), pbox_steps=pbox_steps)

def qrayleigh(p, b):
    return np.sqrt(-2.0 * b * b * np.log(1.0 - p))

def rayleigh(b, pbox_steps=200):
    if intparm(b):
        return env1_2(rayleigh, b, pbox_steps=pbox_steps)
    
    m = b * np.sqrt(np.pi / 2)
    v = (2 - np.pi / 2) * b * b
    
    pbox_parms = {
        'lo': qrayleigh(utils.ii(pbox_steps=pbox_steps), b),
        'hi': qrayleigh(utils.jjj(pbox_steps=pbox_steps), b),
        'shape': 'Rayleigh',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

def qreciprocal(p, a=10.0): 
    return np.exp(p * np.log(a))

def reciprocal(a=10.0, pbox_steps=200):
    if intparm(a):
        return env1_2(reciprocal, a, pbox_steps=pbox_steps)
    
    m = (a - 1) / np.log(a)
    
    pbox_parms = {
        'lo': qreciprocal(utils.ii(pbox_steps=pbox_steps), a),
        'hi': qreciprocal(utils.jj(pbox_steps=pbox_steps), a),
        'shape': 'reciprocal',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': 0,
        'var_hi': np.inf
    }
    
    return pbox.Pbox(**pbox_parms)

def qshiftedloglogistic(p, a, b, c):
    return a + b * (np.exp(c * np.log(1 / (1 / p - 1))) - 1) / c

def shiftedloglogistic(a=0, b=1, c=0, pbox_steps=200):
    if intparm(a, b, c):
        return env3_8(shiftedloglogistic, a, b, c, pbox_steps=pbox_steps)
    
    if np.all(c == 0):
        return logistic(a, b, pbox_steps=pbox_steps)
    
    csc = lambda x: 1 / np.sin(x)
    
    m = a + b * (np.pi * c * csc(np.pi * c)) / c
    v = b ** 2 * (2 * np.pi * c * csc(2 * np.pi * c) - (np.pi * c * csc(np.pi * c)) ** 2) / (c ** 2)
    
    if 0 < c:
        lo = qshiftedloglogistic(utils.ii(pbox_steps=pbox_steps), a, b, c)
        hi = qshiftedloglogistic(utils.jjj(pbox_steps=pbox_steps), a, b, c)
    if c < 0:
        lo = qshiftedloglogistic(utils.iii(pbox_steps=pbox_steps), a, b, c)
        hi = qshiftedloglogistic(utils.jj(pbox_steps=pbox_steps), a, b, c)
        
    pbox_parms = {
        'lo': lo,
        'hi': hi,
        'shape': 'shifted loglogistic',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)
    
def qtrapezoidal(p, a, b, c, d):
    if nothing(d - a):
        return np.repeat(a, len(p))
    
    if (nothing(c-b)):
        return qtriangular(p, a, b, d)
        
    h = 2 / (c + d - b - a)
    p1 = h * (b - a) / 2
    p2 = p1 + h * (c - b)
    r = np.where(p <= p2, (p - p1) / h + b, d - np.sqrt(2 * (1 - p) * (d - c) / h))
    r[p <= p1] = a + np.sqrt(2 * p[p <= p1] * (b - a) / h)   
    
    return r
  
def trapezoidal(a, b, c, d, pbox_steps=200):
    if intparm(a, b, c, d):
        return env4_16(trapezoidal, a, b, c, d, pbox_steps=pbox_steps)
    
    ab = a + b
    cd = c + d
    
    if nothing(cd - ab):
        h = 1
    else:
        h = 1.0 / (3.0 * (cd - ab))
    
    m = h * (c * cd + d * d - (a * ab + b * b))
    
    if nothing(d - a):
        m = a
        v = 0.0
    else:
        v = 0.5 * h * (cd * (c * c + d * d) -  ab * (a * a + b * b)) - m * m
        
    pbox_parms = {
        'lo': qtrapezoidal(utils.ii(pbox_steps=pbox_steps), a, b, c, d),
        'hi': qtrapezoidal(utils.jj(pbox_steps=pbox_steps), a, b, c, d),
        'shape': 'trapezoidal',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)



def Strapezoidalmean(a, b, c, d):
    ab = a + b
    cd = c + d
    if nothing(cd - ab):
        h = 1
    else:
        h = 1.0 / (3.0 * (cd - ab))
        
    return h * (c * cd + d * d - (a * ab + b * b))

def Strapezoidalvar(a, b, c, d):
    ab = a + b
    cd = c + d
    
    if nothing(cd - ab):
        h = 1
    else:
        h = 1.0 / (3.0 * (cd - ab))
        
    m = h * (c * cd + d * d - (a * ab + b * b))
    
    if nothing(d - a):
        m = a
        v = 0.0
    else:
        v = 0.5 * h * (cd * (c * c + d * d) - ab * (a * a + b * b)) - m * m
        
    return v

def Trapezoidal(min, lmode, rmode, max):
    a = utils.left(min)
    b = utils.right(min)
    c = utils.left(lmode)
    d = utils.right(lmode)
    e = utils.left(rmode)
    f = utils.right(rmode)
    g = utils.left(max)
    h = utils.right(max)
    
    if c < a: c = a
    if e < c: e = c
    if g < e: g = e
    if h < f: f = h
    if f < d: d = f
    if d < b: b = d

    ml = Strapezoidalmean(a, c, e, g)
    mh = Strapezoidalmean(b, d, f, h)
    
    if g <= b: 
        vl = 0 
    else: 
        vl = Strapezoidalvar(b, closest(lmode, (b + g) / 2), closest(rmode, (b + g) / 2), g)
    
    vh = Strapezoidalvar(a, closest(interval.Interval(c, d), a), closest(interval.Interval(e, f), h), h)
    
    pbox_parms = {
        'lo': qtrapezoidal(utils.ii(pbox_steps=pbox_steps), a, c, e, g),
        'hi': qtrapezoidal(utils.jj(pbox_steps=pbox_steps), b, d, f, h),
        'shape': 'trapezoidal',
        'mean_lo': ml,
        'mean_hi': mh,
        'var_lo': vl,
        'var_hi': vh
    }
    
    return pbox.Pbox(**pbox_parms)


# ## I cannot resolve which of the preceding two implementations is better
# #t = trapezoidal(1,2,4,8)
# #T = Trapezoidal(1,2,4,8)
# #
# #t = trapezoidal(I(1,2), I(2,3), I(5,7), I(9,12))
# #T = trapezoidal(I(1,2), I(2,3), I(5,7), I(9,12))
# #
# #t = trapezoidal(I(1,3), I(2,6), I(5,7), I(9,12))
# #T = trapezoidal(I(1,3), I(2,6), I(5,7), I(9,12))

def qtriangular(p, min, mid, max):
    pm = (mid - min) / (max - min)
#     ifelse(p<=pm, min + sqrt(p * (max - min) * (mid - min)), max - sqrt((1.0 - p) * (max - min) * (max - mid)))    
    ps = []
    
    for pp in p:
        if pp <= pm:
            ps.append(min + np.sqrt(pp * (max - min) * (mid - min)))
        else:
            ps.append(max - np.sqrt((1.0 - pp) * (max - min) * (max - mid)))
            
    return np.array(ps)

# # Not sure which implementation of triangular below is better

def triangular(min, mode, max, pbox_steps=200):
    if intparm(min, mode, max):
        return env3_8(triangular, min, mode, max, pbox_steps=pbox_steps)
    
    m = (min + mode + max) / 3
    v = (min * min + mode * mode + max * max - min * mode - min * max - mode * max) / 18
    
    pbox_parms = {
        'lo': qtriangular(utils.ii(pbox_steps=pbox_steps), min, mode, max),
        'hi': qtriangular(utils.jj(pbox_steps=pbox_steps), min, mode, max),
        'shape': 'triangular',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)


def T(min, mode, max, pbox_steps=200):
    a = utils.left(min)
    b = utils.right(min)
    c = utils.left(mode)
    d = utils.right(mode)
    e = utils.left(max)
    f = utils.right(max)
    
    if c < a: c = a   # implicit constraints
    if e < c: e = c
    if f < d: d = f
    if d < b: b = d    

    m = closest(interval.Interval(c, d), (b + e) / 2)
    
    if e <= b:
        vl = 0 
    else:
        vl = (b ** 2 + m ** 2 + e ** 2 - (b * m + b * e + m * e)) / 18
        
    vh = (a ** 2 + f ** 2 + np.max(c ** 2 - (c * (a + f) + a * f), d ** 2 - (d * (a + f) + a * f))) / 18
    
    pbox_parms = {
        'lo': qtriangular(utils.ii(pbox_steps=pbox_steps), a, c, e),
        'hi': qtriangular(utils.jj(pbox_steps=pbox_steps), b, d, f),
        'shape': 'triangular',
        'mean_lo': ml,
        'mean_hi': mh,
        'var_lo': vl,
        'var_hi': vh
    }
    
    return pbox.Pbox(**pbox_parms)


def uniform(min, max, pbox_steps=200):
    if intparm(min, max): 
        return env2_4(uniform, min, max)
    
    min, max = np.float64(min), np.float64(max) # np.float to avoid div by 0 err
    
    i_mm = interval.I(min, max)
    
    min = i_mm.lo
    max = i_mm.hi
    
    m = (min + max) / 2
    v = (min - max) ** 2 / 12
    
    if min == max:
        lo = np.array([min for i in range(pbox_steps)])
        hi = np.copy(lo)
    else:
        scale = i_mm.hi - i_mm.lo
        lo = sps.uniform.ppf(utils.ii(pbox_steps=pbox_steps), min, scale)
        hi = sps.uniform.ppf(utils.jj(pbox_steps=pbox_steps), min, scale)
    
    pbox_parms = {
        'lo': lo,
        'hi': hi,
        'shape': 'uniform',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }

    return pbox.Pbox(**pbox_parms)

U = uniform # uniform alias...
rectangular = uniform # uniform alias...

def skellam(m1, m2, pbox_steps=200):
    if intparm(m1, m2):
        return env2_4(skellam, m1, m2, pbox_steps=pbox_steps)
    
    return pbox.Pbox(Spoisson(m1)._conv(Spoisson(m2), '-'), shape='Skellam')

# stable <- function(a,b,c,d, name='',...){
#   library('stabledist')
#   if (intparm(a,b,c,d)) return(env4.16(stable,a,b,c,d,name=name,...))
#   squeal <- function(test, msg) if (test) stop(paste('Improper parameters for stable:',msg))
#   squeal(a <= 0, 'a<=0')
#   squeal(2 < a, '2<a')
#   squeal(b < -1, 'b<-1')
#   squeal(1 < b, '1<b')
#   squeal(c <= 0, 'c<=0')
#   pbox(u=qstable(iii(),a,b,c,d), d=qstable(jjj(),a,b,c,d), shape='stable', name=name) #, ml=m, mh=m, vl=0, vh=Inf,...)
#   }


def student(df):
    if intparm(df):
        return env3_8(student, df, pbox_steps=pbox_steps)
    
    m = 0
    v = 1 / (1 - 2 / df)
    
    pbox_parms = {
        'lo': sps.t.ppf(utils.iii(pbox_steps=pbox_steps), df),
        'hi': sps.t.ppf(utils.jjj(pbox_steps=pbox_steps), df),
        'shape': 'Student\'s t',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)
    
t = student # student alias...

def voigt(sigma, gamma, pbox_steps=200):
    if intparm(sigma, gamma):
        return env2_4(voigt, sigma, gamma)
    
    p = normal(0, sigma, pbox_steps=pbox_steps)._conv(cauchy(0, gamma, pbox_steps=pbox_steps), '+')
    
    p = pbox.Pbox(p, shape='Voigt')
    p.mean_lo = -np.inf
    p.mean_hi = np.inf
    p.var_lo = 0
    p.var_hi = np.inf
    
    return p



def qwakeby(p, e, a, b, c, d):
    # x = (a/b) * (1 - (1-p)^b) - (c/d) * (1 - (1-p)^(-d)) + e
    # http://dspace.mit.edu/bitstream/handle/1721.1/31278/MIT-EL-77-033WP-04146753.pdf?sequence=1
    # Based on the Fortran code QUAWAK written by J.R.M. Hosking
    # http://sgi62.ncep.noaa.gov:8080/PQPF/pqf/lmoments
    ufl = -170.0 # ufl should be the smallest value that doesn't cause underflow
    
    y1 = -np.log(1 - p)
    z = -np.log(1 - p)
    temp = -b * z
    
    if (b != 0):
        y1 = np.where(temp < ufl, 1 / b, (1 - np.exp(temp)) / b)
    
    y2 = z
    
    if d != 0:
        y2 = (1 - np.exp(d * y2)) / (-d)
    
    return a * y1 + c * y2 + e


def wakeby(e, a, b, c, d, pbox_steps=200):
    if intparm(e, a, b, c, d):
        return env5_32(wakeby, e, a, b, c, d, pbox_steps=pbox_steps)
    
    badmsg = 'Improper parameters for wakeby:'
    
    if (b + d) < 0:
        raise Exception(f'{badmsg} b + d < 0')
    if ((b + d) <= 0) and ((b == 0) and (c == 0) and (d == 0)):
        raise Exception('b + d <= 0 when b, c, d are nonzero')
    if 0 < c:
        if d < 0:
            raise Exception(f'{badmsg} d < 0 when 0 < c')
    if c < 0:
        raise Exception(f'{badmsg} c < 0')
    if (a + c) < 0:
        raise Exception(f'{badmsg} a + c < 0')
        
    m = e + a / (b + 1) + c / (1 - d)
    
    if (d < 0) or (c == 0):
        j = utils.jj(pbox_steps=pbox_steps)
    else:
        j = utils.jjj(pbox_steps=pbox_steps)
    
    pbox_parms = {
        'lo': qwakeby(utils.ii(pbox_steps=pbox_steps), e, a, b, c, d),
        'hi': qwakeby(j, e, a, b, c, d),
        'shape': 'Wakeby',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': 0,
        'var_hi': np.inf
    }

    return pbox.Pbox(**pbox_parms)

def wilcoxon(m, n, pbox_steps=200):
    if intparm(m ,n):
        return env2_4(wilcoxon, m, n, pbox_steps=pbox_steps)
    
    mn = m * n / 2
    vr = m * n * (m + n + 1) / 12
    
    pbox_parms = {
        'lo': None,
        'hi': None,
        'shape': 'Wilcoxon',
        'mean_lo': mn,
        'mean_hi': mn,
        'var_lo': vr,
        'var_hi': vr
    }
    
    return pbox.Pbox(**pbox_parms)

# wilcoxon <- function(m, n, name='',...){ # distribution of the Wilcoxon rank sum statistic (see R's qwilcox)
#   if (intparm(m,n)) return(env2.4(wilcoxin,m,n,name=name,...))
#   mn = m * n / 2
#   vr = m * n * (m + n + 1) / 12
#   pbox(u=qwilcox(ii(), m, n), d=qwilcox(jj(), m, n), shape='Wilcoxon', name=name, ml=mn, mh=mn, vl=vr, vh=vr,...)
#   }


def weibull(scale, shape, pbox_steps):
    if intparm(scale, shape):
        return env2_4(weibull, scale, shape, pbox_steps=pbox_steps)
    
    m = scale * np.gamma(1 + 1 / shape)
    v = scale ** 2 * (np.gamma(1 + 2 / shape) - (np.gamma(1 + 1 / shape)) ** 2)
    
    pbox_parms = {
        'lo': None,
        'hi': None,
        'shape': 'Weibull',
        'mean_lo': m,
        'mean_hi': m,
        'var_lo': v,
        'var_hi': v
    }
    
    return pbox.Pbox(**pbox_parms)

# weibull <- function(scale, shape, name='',...){  # argument order disagrees with R's qweibull function, but agrees with Wikipedia (in November 2013)
#   if (intparm(scale,shape)) return(env2.4(weibull,scale,shape,name=name,...))
#   m <- scale * gammafunc(1+1/shape)
#   v <- scale^2*(gammafunc(1+2/shape)-(gammafunc(1+1/shape))^2)
#   pbox(u=qweibull(ii(), shape, scale), d=qweibull(jjj(), shape, scale), shape='Weibull', name=name, ml=m, mh=m, vl=v, vh=v,...)
#   }
