import numpy as np
from impstats import interval
from impstats import utils
from impstats import rowebounds
from impstats import vkmean


class Pbox(object):
    """
    Bounded probability distrituion object
    """

    def __init__(self, lo=None, hi=None, steps=200, shape=None, mean_lo=None, mean_hi=None, var_lo=None, var_hi=None, interpolation='linear'):
        if isinstance(lo, Pbox):
            self.lo = lo.lo
            self.hi = lo.hi
            self.steps = lo.steps
            self.n = lo.steps
            self.shape = lo.shape
            self.mean_lo = lo.mean_lo
            self.mean_hi = lo.mean_hi
            self.var_lo = lo.var_lo
            self.var_hi = lo.var_hi
            
        else:
            if lo is None and hi is None:
                lo = -np.inf
                hi = np.inf
                
            if (lo is not None) and (hi is None):
                hi = lo

            if isinstance(lo, interval.Interval) or (not hasattr(lo, '__len__')):
                lo = np.array([utils.left(lo)])

            if isinstance(hi, interval.Interval) or (not hasattr(hi, '__len__')):
                hi = np.array([utils.right(hi)])

            if len(lo) != steps:
                lo = utils.interpolate(lo, interpolation=interpolation, left=True, pbox_steps=steps)

            if len(hi) != steps:
                hi = utils.interpolate(hi, interpolation=interpolation, left=False, pbox_steps=steps)

            self.lo = lo
            self.hi = hi
            self.steps = steps
            self.n = self.steps
            self.shape = shape
            self.mean_lo = -np.inf
            self.mean_hi = np.inf
            self.var_lo = 0
            self.var_hi = np.inf

        self._computemoments()
        if shape is not None: self.shape = shape
        if mean_lo is not None: self.mean_lo = np.max([mean_lo, self.mean_lo])
        if mean_hi is not None: self.mean_hi = np.min([mean_hi, self.mean_hi])
        if var_lo is not None: self.var_lo = np.max([var_lo, self.var_lo])
        if var_hi is not None: self.var_hi = np.min([var_hi, self.var_hi])
        self._checkmoments()
        
    def __len__(self):
            return 1
        
    def __repr__(self):
        if self.mean_lo == self.mean_hi:
            mean_text = f'{round(self.mean_lo, 4)}'
        else:
            mean_text = f'[{round(self.mean_lo, 4)}, {round(self.mean_hi, 4)}]'

        if self.var_lo == self.var_hi:
            var_text = f'{round(self.var_lo, 4)}'
        else:
            var_text = f'[{round(self.var_lo, 4)}, {round(self.var_hi, 4)}]'
        
        range_text = f'[{round(np.min([self.lo, self.hi]), 4), round(np.max([self.lo, self.hi]), 4)}'

        if self.shape is None:
            shape_text = ' '
        else:
            shape_text = f' {self.shape}' # space to start; see below lacking space

        return f'Pbox: ~{shape_text}(range={range_text}, mean={mean_text}, var={var_text})'
    
    def __neg__(self):
        if self.shape in ['uniform','normal','cauchy','triangular','skew-normal']:
            s = self.shape
        else:
            s = ''
        
        pbox_parms = {
            "hi": -np.flip(self.lo), 
            "lo": -np.flip(self.hi), 
            "shape": s, 
            "mean_lo": -self.mean_hi, 
            "mean_hi": -self.mean_lo, 
            "var_lo": self.var_lo, 
            "var_hi": self.var_hi
        }

        return Pbox(**pbox_parms)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float):
            return self._shift(other)

        elif isinstance(other, Pbox):
            return self._frechetconv(other, "+")

        elif isinstance(other, interval.Interval):
            return self._frechetconv(Pbox(other), "+")

        elif isinstance(other, np.ndarray):
            return other + self

        else:
            raise ValueError(f"unsupported operand type(s) for +: 'Pbox' and '{type(other).__name__}'")

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float):
            return self._multnumeric(other)

        elif isinstance(other, interval.Interval):
            return self._frechetconv(Pbox(other), "*")

        elif isinstance(other, Pbox):
            return self._frechetconv(other, "*")

        elif isinstance(other, np.ndarray):
            return other * self

        else:
            raise ValueError(f"unsupported operand type(s) for *: 'Pbox' and '{type(other).__name__}'")

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float) or isinstance(other, np.ndarray):
            return self._multnumeric(1 / other)
        
        elif isinstance(other, interval.Interval):
            return self / Pbox(other)

        elif isinstance(other, Pbox):
            return self * other.reciprocal()

        else:
            raise ValueError(f"unsupported operand type(s) for /: 'Pbox' and '{type(other).__name__}'")

    def __iadd__(self, other):
        return self + other

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    
    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float) or isinstance(other, np.ndarray) or isinstance(other, type(self)):
            return other * self.reciprocal()
        else:
            raise ValueError(f"unsupported operand type(s) for /: '{type(other).__name__}' and 'Interval'")          

    def _multnumeric(self, m):
        multipliable = self.shape in ['uniform', 'normal', 'cauchy', 'triangular', 'skew-normal']
        scalable = self.shape in ['exponential','lognormal']
        if multipliable or (scalable and (0 <= self.lo[0])):
            s = self.shape
        else:
            s = ''
        
        if m < 0:
            return -(self._multnumeric(abs(m)))
        else:
            pbox_parms = {
                "hi": m * self.hi, 
                "lo": m * self.lo, 
                "shape": s, 
                "mean_lo": m * self.mean_lo, 
                "mean_hi": m * self.mean_hi, 
                "var_lo": (m ** 2) * self.var_lo, 
                "var_hi": (m ** 2) * self.var_hi
            }

            return Pbox(**pbox_parms)

    def _shift(self, other):
        if self.shape in ['uniform','normal','cauchy','triangular','skew-normal']: 
            s = self.shape
        else: 
            s = ''
        pbox_parms = {
            "hi": other + self.hi, 
            "lo": other + self.lo, 
            "shape": s, 
            "mean_lo": self.mean_lo + other, 
            "mean_hi": self.mean_hi + other, 
            "var_lo": self.var_lo, 
            "var_hi": self.var_hi
        }

        return Pbox(**pbox_parms)

    def _computemoments(self):    # should we compute mean if it is a Cauchy, var if it's a t distribution?
        self.mean_lo = np.max([self.mean_lo, np.mean(self.lo)])
        self.mean_hi = np.min([self.mean_hi, np.mean(self.hi)])

        if not (np.any(self.lo <= -np.inf) or np.any(np.inf <= self.hi)):
            V, JJ = 0, 0
            j = np.array(range(self.n))

            for J in np.array(range(self.n)) - 1:
                ud = [*self.lo[j < J], *self.hi[J <= j]]
                v = utils.sideVariance(ud)

                if V < v:
                    JJ = J
                    V = v

            self.var_hi = V

    def _checkmoments(self):
        a = interval.Interval(self.mean_lo, self.mean_hi) #mean(x)
        b = utils.dwMean(self)

        self.mean_lo = np.max([utils.left(a), utils.left(b)])
        self.mean_hi = np.min([utils.right(a), utils.right(b)])

        if self.mean_hi < self.mean_lo:
            # use the observed mean
            self.mean_lo = utils.left(b)
            self.mean_hi = utils.right(b)

        a = interval.Interval(self.var_lo, self.var_hi) #var(x)
        b = utils.dwVariance(self)

        self.var_lo = np.max([utils.left(a), utils.left(b)])
        self.var_hi = np.min([utils.right(a), utils.right(b)])
        
        if self.var_hi < self.var_lo:
            # use the observed variance
            self.var_lo = utils.left(b)
            self.var_hi = utils.right(b)

    def _conv(self, other, op, pbox_steps=200):
        if op == '-':
            return self._conv(-other, '+')
        if op == '/':
            return self._conv(other.reciprocal(), '*')
        
        # x = makepbox(x)
        # y = makepbox(y)
        m = self.n
        p = other.n
        n = np.min([pbox_steps, m * p])
        L = m * p / n
        k = np.array(range(n))

        c = (utils.get_op(op)([[i] for i in self.hi], other.hi)).flatten()
        c.sort()
        Zd = c[(k * L + L - 1).astype(int)]

        c = (utils.get_op(op)([[i] for i in self.lo], other.lo)).flatten()
        c.sort()
        Zu = c[(k * L).astype(int)]

        # mean
        ml = -np.inf
        mh = np.inf
        
        if op in ['+', '-', '*']:
            ml = utils.get_op(op)(self.mean_lo, other.mean_lo)
            mh = utils.get_op(op)(self.mean_hi, other.mean_hi)
    
        vl = 0
        vh = np.inf
        
        if op in ['+', '-']:
            vl = self.var_lo + other.var_lo
            vh = self.var_hi + other.var_hi
        
        pbox_parms = {
            "hi": Zd, 
            "lo": Zu, 
            "mean_lo": ml, 
            "mean_hi": mh, 
            "var_lo": vl, 
            "var_hi": vh
        }

        return Pbox(**pbox_parms)

    def _perfectconv(self, other, op): # prolly doesn't work for ^ in interesting cases
        if op in ['-', '/']:
            cu = utils.get_op(op)(self.lo, other.hi)
            cd = utils.get_op(op)(self.hi, other.lo)
        else:
            cu = utils.get_op(op)(self.lo, other.lo)
            cd = utils.get_op(op)(self.hi, other.hi)
        
        scu = np.sort(cu)
        scd = np.sort(cd)

        return Pbox(hi=scd, lo=scu)

    def _oppositeconv(self, other, op):# prolly doesn't work for ^ in interesting cases
        if op in ['-', '/']:
            cu = utils.get_op(op)(self.lo, np.flip(other.hi))
            cd = utils.get_op(op)(self.hi, np.flip(other.lo))
        else:
            cu = utils.get_op(op)(self.lo, np.flip(other.lo))
            cd = utils.get_op(op)(self.hi, np.flip(other.hi))

        cu.sort()
        cd.sort()

        return Pbox(hi=cd, lo=cu)

    def _frechetconv(self, other, op, pbox_steps=200):
        if op == "-":
            return self._frechetconv(-other, '+')
        if op == '/':
            return self._frechetconv(other.reciprocal(), '*')
        if op == '*':
            if utils.straddlingzero(self) or utils.straddlingzero(other):
                return utils.imp(self._balchprod(other), self._naivefrechetconv(other, '*'))
        
        # x = makepbox(x)
        # y = makepbox(y)
        zu = np.zeros(pbox_steps) # []
        zd = np.zeros(pbox_steps) # []
        
        for i in range(pbox_steps):
            j = np.array(range(i, pbox_steps))
            k = np.flip(j)
            zd[i] = np.min(utils.get_op(op)(self.hi[j], other.hi[k]))

            j = np.array(range(0, i + 1))
            k = np.flip(j)
            zu[i] = np.max(utils.get_op(op)(self.lo[j], other.lo[k]))
        
        ml = -np.inf
        mh = np.inf

        if op in ['+', '-']:
            ml = utils.get_op(op)(self.mean_lo, other.mean_lo)
            mh = utils.get_op(op)(self.mean_hi, other.mean_hi)

        vl = 0
        vh = np.inf

        pbox_parms = {
            "hi": zd, # np.array(zd),
            "lo": zu, # np.array(zu),
            "mean_lo": ml,
            "mean_hi": mh,
            "var_lo": vl,
            "var_hi": vh
        }

        return Pbox(**pbox_parms)

    def _naivefrechetconv(self, other, op='*'):
        if op == '+':
            return self._frechetconv(other, '+')
        if op == '-':
            return self._frechetconv(-other, '+')
        if op == '/':
            return self._naivefrechetconv(other.reciprocal(), '*')
            
        # x = makepbox(x)
        # y = makepbox(y)
        n = len(self.hi)

        c = (utils.get_op(op)([[i] for i in self.hi], other.hi)).flatten()
        c.sort()
        Zd = c[np.array(range(n * n - n, n * n))]

        c = (utils.get_op(op)([[i] for i in self.lo], other.lo)).flatten()
        c.sort()
        Zu = c[np.array(range(n))]
        
        # mean
        m = interval.Interval(self.mean_lo, self.mean_hi) * interval.Interval(other.mean_lo, other.mean_hi)
        a = np.sqrt(interval.Interval(self.var_lo, self.var_hi) * interval.Interval(other.var_lo, other.var_hi))
        
        ml = m - a
        mh = m + a
        
        VK = vkmean.VKmeanproduct(self, other)
        m = utils.imp(interval.Interval(ml, mh), VK)
        
        # variance
        vl = 0
        vh = np.inf
        
        pbox_parms = {
            "hi": Zd, 
            "lo": Zu, 
            "mean_lo": utils.left(m), 
            "mean_hi": utils.right(m), 
            "var_lo": vl, 
            "var_hi": vh
        }

        return Pbox(**pbox_parms)

    def _balchprod(self, other):
        if utils.straddles(self) and utils.straddles(other):
            x0 = utils.left(self)
            y0 = utils.left(other)
            xx0 = self - x0
            yy0 = other - y0
            a = xx0._frechetconv(yy0, '*') 
            b = (y0 * xx0)._frechetconv(x0 * yy0, '+')

            return a._frechetconv(b, '+') + x0 * y0
        
        if straddles(self):
            x0 = utils.left(self)
            xx0 = self - x0
            a = xx0._frechetconv(other, '*') 
            b = x0 * other

            return a._frechetconv(b, '+')
        
        if straddles(other):
            y0 = utils.left(other)
            yy0 = other - y0
            a = self._frechetconv(yy0, '*')
            b = self * y0

            return a._frechetconv(b, '+')
        
        return self._frechetconv(other, '*')

    def left(self):
        return self.lo[0]

    def right(self):
        return self.hi[-1]    

    def reciprocal(self):
        if self.shape in ['Cauchy', '{min, max, median}', '{min, max, percentile}', '{min, max}']:
            sh = self.shape
        elif self.shape == 'Pareto':
            sh = 'power function'
        elif self.shape == 'power function':
            sh = 'Pareto'
        else:
            sh = ''
        t = lambda x: 1 / x

        if utils.left(self) <= 0 and utils.right(self) >= 0:
            return np.nan
        elif utils.left(self) > 0:
            mymean = rowebounds.transformedMean(self, t, False, True)
            myvar = rowebounds.transformedVariance(self, t, False, True, mymean)
        else:
            mymean = rowebounds.transformedMean(self, t, False, False)
            myvar = rowebounds.transformedVariance(self, t, False, False, mymean)

        pbox_parms = {
            "hi": 1 / np.flip(self.lo), 
            "lo": 1 / np.flip(self.hi), 
            "shape": sh, 
            "mean_lo": utils.left(mymean), 
            "mean_hi": utils.right(mymean), 
            "var_lo": utils.left(myvar), 
            "var_hi": utils.right(myvar)
        }

        return Pbox(**pbox_parms)
    
    def plot(self):
        return utils.impplot(self)
