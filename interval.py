import numpy as np
import warnings
import utils
import pbox

class Interval(object):
    """
    Interval object
    """

    def __init__(self, lo=None, hi=None):
        # kill complex nums
        if isinstance(lo, np.complex):
            lo = 0

        if isinstance(hi, np.complex):
            hi = 0

        if lo is None and hi is None: # construct vaccuous interval
            lo = -np.inf
            hi = np.inf
        elif lo is None or hi is None:
            lo = hi = [i for i in [lo, hi] if i is not None][0]

        if hasattr(lo, '__iter__'): # if iterable, find min
            lo = min(lo)

        if hasattr(hi, '__iter__'): # if iterable, find max
            hi = max(hi)

        lo_hi = [lo, hi]
        l_lo = min(lo_hi)
        h_hi = max(lo_hi)

        self.lo = l_lo
        self.hi = h_hi

    def __iter__(self):
        for bound in [self.lo, self.hi]:
            yield bound

    def __len__(self):
        if self.lo == self.hi:
            return 1
        else:
            return 2

    def conjugate(self): # complex conjugate
        return self # Intervals do not yet support complex numbers

    def reciprocal(self):
        lo = 1 / self.hi
        hi = 1 / self.lo
        return Interval(lo, hi)

    def sqrt(self):
        return self ** 0.5

    def power(self, other):
        return self ** other

    def square(self):
        return np.power(self, 2)

    def midpoint(self):
        return (self.hi - self.lo) / 2

    def truncate(self, min_val, max_val):
        lo = min(max_val, max(min_val, self.lo))
        hi = min(max_val, max(min_val, self.hi))
        return Interval(lo, hi)

    #trig xforms
    def cos(self):
        return Interval(np.cos(self.lo), np.cos(self.hi))

    def sin(self):
        return Interval(np.sin(self.lo), np.sin(self.hi))

    def tan(self):
        return Interval(np.tan(self.lo), np.tan(self.hi))

    def arccos(self):
        return Interval(np.arccos(self.lo), np.arccos(self.hi))

    def arcsin(self):
        return Interval(np.arcsin(self.lo), np.arcsin(self.hi))

    def arctan(self):
        return Interval(np.arctan(self.lo), np.arctan(self.hi))

    #rep
    def __repr__(self):
        return f'Interval({self.lo}, {self.hi})'

    #unary funcs
    def __neg__(self):
        return Interval(-self.lo, -self.hi)

    def __pos__(self):
        return Interval(+self.lo, +self.hi)

    def __abs__(self):
        return Interval(abs(self.lo), abs(self.hi))

    def __contains__(self, other): # x in y
        if isinstance(other, type(self)):
            return (self.lo <= other.lo) and (other.hi <= self.hi)
        else:
            return self.hi >= other >= self.lo

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            other = Interval(other, other)

        if self.hi < other.lo:
            return True
        elif other.hi < self.lo:
            return False
        else:
            return False

    def __le__(self, other):
        if not isinstance(other, type(self)):
            other = Interval(other, other)

        if self.hi <= other.lo:
            return True
        elif other.hi <= self.lo:
            return False
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.lo == other.lo) and (self.hi == other.hi)
        elif other in self:
            return False
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, type(self)):
            return not ((self.lo == other.lo) and (self.hi == other.hi))
        elif other in self:
            return True
        else:
            return True

    def __gt__(self, other):
        if not isinstance(other, type(self)):
            other = Interval(other, other)

        if self.lo > other.hi:
            return True
        elif other.lo > self.hi:
            return False
        else:
            return False

    def __ge__(self, other):
        if not isinstance(other, type(self)):
            other = Interval(other, other)

        if self.lo >= other.hi:
            return True
        elif other.lo >= self.hi:
            return False
        else:
            return False

    #arithmatic methods
    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float):
            lo = self.lo + other
            hi = self.hi + other
            return Interval(lo, hi)

        elif isinstance(other, type(self)):
            lo = self.lo + other.lo
            hi = self.hi + other.hi
            return Interval(lo, hi)

        elif isinstance(other, np.ndarray) or isinstance(other, pbox.Pbox):
            return other + self

        else:
            raise ValueError(f"unsupported operand type(s) for +: 'Interval' and '{type(other).__name__}'")

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float) or isinstance(other, np.ndarray):
            return self * (1 / other)

        elif isinstance(other, type(self)):
            return self * other.reciprocal()

        else:
            raise ValueError(f"unsupported operand type(s) for /: 'Interval' and '{type(other).__name__}'")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float):
            if other < 0:
                new_int = abs(other) * self
                return -Interval(new_int.lo, new_int.hi)
            else:
                return Interval((+other) * self.lo, other * self.hi)

        elif isinstance(other, type(self)):
            lo = min(self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
            hi = max(self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
            return Interval(lo, hi)

        elif isinstance(other, np.ndarray):
            return other * self

        else:
            raise ValueError(f"unsupported operand type(s) for *: 'Interval' and '{type(other).__name__}'")

    def __pow__(self, other, modulo=None):
        if other == 0:
            ans = 1
        elif other == 1:
            ans = self
        elif (other % 2) == 0:
            if self.lo >= 0:
                ans = Interval(lo=self.lo ** other, hi=self.hi ** other)
            if (self.lo < 0) and (self.hi > 0):
                ans = Interval(lo=0, hi=max(self.lo ** other, self.hi ** other))
            if (self.hi <= 0):
                ans = Interval(lo=self.hi ** other, hi=self.lo ** other)
        else:
            ans = Interval(lo=self.lo ** other, hi=self.hi ** other)

        return ans

    def __iadd__(self, other):
        return self + other

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __rpow__(self, other):
        np.power(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64) or isinstance(other, float) or isinstance(other, np.ndarray) or isinstance(other, type(self)):
            return other * self.reciprocal()
        else:
            raise ValueError(f"unsupported operand type(s) for /: '{type(other).__name__}' and 'Interval'")

    def left(self):
        return self.lo

    def right(self):
        return self.hi
    
    def plot(self):
        return utils.impplot(self)


I = Interval # Interval class alias
