import numpy as np
import interval
import utils


def VKmeanlo(p, q, k1, k2, k3, k4): # double p, double q, double k1, double k2, double k3, double k4)
    return np.max([p + q - 1, 0.0]) * k1 + np.min([p, 1 - q]) * k2 + np.min([1 - p, q]) * k3 + np.max([1 - p - q, 0.0]) * k4

def VKmeanup(p, q, k1, k2, k3, k4): # double p, double q, double k1, double k2, double k3, double k4)
  return np.min([p, q]) * k1 + np.max([p - q, 0.0]) * k2 + np.max([q - p, 0.0]) * k3 + np.min([1 - p, 1 - q]) * k4

def touching(a, b): # const Interval& a, const Interval& b )
    if np.all(utils.left(a) == utils.right(a)) and np.all(utils.left(b) == utils.right(b)):
        return utils.left(a) == utils.left(b)
    if utils.straddles(a - b):
        return True
    
    return False

def VKmeanlower(pa, pb, k1, k2, k3, k4): # Interval pa, Interval pb, double k1, double k2, double k3, double k4)
    # double p1, p2, ec1;
    lpa = utils.left(pa)
    rpa = utils.right(pa)
    lpb = utils.left(pb)
    rpb = utils.right(pb)
    touch = touching(pa + pb, 1.0)
    p1 = lpa
    p2 = lpb
    ec1 = VKmeanlo(p1, p2, k1, k2, k3, k4)
    p1 = lpa
    p2 = rpb
    ec1 = np.min([ec1, VKmeanlo(p1, p2, k1, k2, k3, k4)])
    p1 = rpa
    p2 = lpb
    ec1 = np.min([ec1, VKmeanlo(p1, p2, k1, k2, k3, k4)])
    p1 = rpa
    p2 = rpb
    ec1 = np.min([ec1, VKmeanlo(p1, p2, k1, k2, k3, k4)])
    p1 = np.max([lpa, 1 - rpb])
    p2 = 1 - p1

    if touch:
        ec1 = np.min([ec1, VKmeanlo(p1, p2, k1, k2, k3, k4)])

    p1 = np.min([rpa, 1 - lpb])
    p2 = 1 - p1

    if touch:
        ec1 = np.min([ec1, VKmeanlo(p1, p2, k1, k2, k3, k4)])
    
    return ec1

def VKmeanupper(pa, pb, k1, k2, k3, k4): # Interval pa, Interval pb, double k1, double k2, double k3, double k4)
    #double p1, p2, ec2;
    lpa = utils.left(pa)
    rpa = utils.right(pa)
    lpb = utils.left(pb)
    rpb = utils.right(pb)
    touch = touching(pa, pb)
    p1 = lpa
    p2 = lpb
    ec2 = VKmeanup(p1, p2, k1, k2, k3, k4)
    p1 = lpa
    p2 = rpb
    ec2 = np.max([ec2, VKmeanup(p1, p2, k1, k2, k3, k4)])
    p1 = rpa
    p2 = lpb
    ec2 = np.max([ec2, VKmeanup(p1, p2, k1, k2, k3, k4)])
    p1 = rpa
    p2 = rpb
    ec2 = np.max([ec2, VKmeanup(p1, p2, k1, k2, k3, k4)])
    p1 = np.max([lpa, 1 - rpb])
    p2 = 1 - p1

    if (touch): ec2 = np.max([ec2, VKmeanup(p1, p2, k1, k2, k3, k4)])

    p1 = np.min([rpa, 1 - lpb])
    p2 = 1-p1

    if (touch): ec2 = np.max([ec2, VKmeanup(p1, p2, k1, k2, k3, k4)])

    return ec2

def VKmeanproduct(a, b):
    # Interval ea,eb,ec,pa,pb;
    # double la,ra,lb,rb,k1,k2,k3,k4,ec1,ec2;
    ec = interval.Interval(-np.inf, np.inf)
    ea = interval.Interval(a.mean_lo, a.mean_hi)
    eb = interval.Interval(b.mean_lo, b.mean_hi)
    la = utils.left(a)
    ra = utils.right(a)
    lb = utils.left(b)
    rb = utils.right(b)
    k1 = ra * rb
    k2 = ra * lb
    k3 = la * rb
    k4 = la * lb
    pa = interval.Interval((utils.left(ea) - la) / (ra - la), (utils.right(ea) - la) / (ra - la))
    pb = interval.Interval((utils.left(eb) - lb) / (rb - lb), (utils.right(eb) - lb) / (rb - lb))
    ec = utils.env_int(VKmeanlower(pa, pb, k1, k2, k3, k4), VKmeanupper(pa, pb, k1, k2, k3, k4))
    
    return ec
