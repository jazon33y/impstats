import numpy as np
from scipy import optimize
import interval


def transformedMean(B, t, posderiv, pos2ndderiv):
    if pos2ndderiv is None: # np.isnan(pos2ndderiv):
        return transformedMeanNull2ndDeriv(B, t, posderiv)
    elif pos2ndderiv:
        return transformedMeanPos2ndDeriv(B, t, posderiv)
    else:
        newT = lambda x: (-1) * t(x)
        tempResult = transformedMeanPos2ndDeriv(B, newT, not posderiv)

        if tempResult is None: # np.isnan(tempResult):
            transformedMean_lower = (-1) * (tempResult.hi)
            transformedMean_upper = (-1) * (tempResult.lo)
            return interval.Interval(transformedMean_lower, transformedMean_upper)
        else:
            return None


def transformedMeanNull2ndDeriv(B, t, posderiv):
    transformedMean_lower = None
    transformedMean_upper = None
    transformedSum = 0     

    for i in range(B.n):
        transformedSum += t(B.lo[i])

    if posderiv:
        transformedMean_lower = transformedSum / (B.n)
    else:
        transformedMean_upper = transformedSum / (B.n)

    transformedSum = 0

    for i in range(B.n):
        transformedSum += t(B.hi[i])

    if posderiv:
        transformedMean_upper = transformedSum / (B.n)
    else:
        transformedMean_lower = transformedSum / (B.n)

    return interval.Interval(transformedMean_lower, transformedMean_upper)


def transformedMeanPos2ndDeriv(B, t, posderiv):
    mu_lower = B.mean_lo
    mu_upper = B.mean_hi    
    ssum = 0
    transformedSum = 0     

    for i in range(B.n):
        ssum += B.lo[i]
        transformedSum += t(B.lo[i])

    computedMu_lower = ssum / (B.n)

    if posderiv:
        transformedMean_lower = transformedSum / (B.n)
    else:
        transformedMean_upper = transformedSum / (B.n)
	
    ssum = 0
    transformedSum = 0

    for i in range(B.n):
        ssum += B.hi[i]
        transformedSum += t(B.hi[i])
        
    computedMu_upper = ssum / (B.n)

    if posderiv:
        transformedMean_upper = transformedSum / (B.n)    
    else:
        transformedMean_lower = transformedSum / (B.n)

    if computedMu_lower > mu_upper or computedMu_upper < mu_lower:
        return None
    elif computedMu_lower >= mu_lower and computedMu_upper <= mu_upper:
        return interval.Interval(transformedMean_lower, transformedMean_upper)
    else:           	
        if posderiv:
            if computedMu_lower < mu_lower:
                transformedMean_lower = None
            if computedMu_upper > mu_upper:
                transformedMean_upper = None

        else:
            if computedMu_lower < mu_lower:
                transformedMean_upper = None
            if computedMu_upper > mu_upper:
                transformedMean_lower = None

    if transformedMean_lower is None: # np.isnan(transformedMean_lower):
        if posderiv:
            mu =  mu_lower
        else:
            mu =  mu_upper

        endingPoints = getBoundEndingPoints(B)
  
        ssum = 0
        transformedSum = 0 

        for i in range(B.n):
            ssum += B.lo[i]
            transformedSum += t(B.lo[i])

        num_middle = 0
        computedMu = ssum / (B.n)

        if computedMu >= mu_lower and computedMu <= mu_upper:
            transformedMean_lower = transformedSum / (B.n)
		
        for i in range(2 * (B.n)):
            if endingPoints[1, i] < 0:
                ssum -= endingPoints[0, i]
                transformedSum -= t(endingPoints[0, i])
                num_middle +=  1
            else:
                ssum =+ endingPoints[0, i]
                transformedSum += t(endingPoints[0, i])
                num_middle -=  1

            if num_middle == 0:
                computedMu = ssum / (B.n) 
                if computedMu >= mu_lower and computedMu <= mu_upper:
                    currentTransformedMean = transformedSum / (B.nn)
                    if transformedMean_lower is None: # np.isnan(transformedMean_lower):
                        transformedMean_lower = currentTransformedMean
                    else:
                        transformedMean_lower = min(transformedMean_lower, currentTransformedMean)

            else:
                value_middle = (mu * (B.n) - ssum ) / num_middle
         
                if value_middle >= endingPoints[0, i] and  value_middle <= endingPoints[0, i + 1]:
                    currentTransformedMean = (transformedSum + t(value_middle) * num_middle) / (B.n)
                    if transformedMean_lower is None: # np.isnan(transformedMean_lower):
                        transformedMean_lower = currentTransformedMean
                    else:
                        transformedMean_lower = min(transformedMean_lower, currentTransformedMean)

    if transformedMean_upper is None: # np.isnan(transformedMean_upper):
        if posderiv:
            mu =  mu_upper
        else:
            mu =  mu_lower

        ssum = mu * (B.n)	 
        sum_lower = 0
        transformedSum_lower = 0 

        for i in range(B.n):
            sum_lower += B.hi[i]
            transformedSum_lower = transformedSum_lower + t(B.hi[i])
    		
        for i in range(B.n):
            sum_upper = sum_lower
            transformedSum_upper = transformedSum_lower
            sum_lower += (B.lo[i] - B.hi[i])
            transformedSum_lower += (t(B.lo[i]) - t(B.hi[i]))

            if ssum >= sum_lower and ssum <= sum_upper:
                a = (sum_upper - ssum) / (B.hi[i] - B.lo[i])
                currentTransformedMean = (transformedSum_upper - a * (t(B.hi[i]) - t(B.lo[i]))) / (B.n)

                if transformedMean_upper is None:#np.isnan(transformedMean_upper):
                    transformedMean_upper = currentTransformedMean
                else:
                    transformedMean_upper = max(transformedMean_upper, currentTransformedMean)

    return interval.Interval(transformedMean_lower, transformedMean_upper)


def transformedVariance(B, t, posderiv, pos2ndderiv, transformedMean):
    if pos2ndderiv is None: #np.isnan(pos2ndderiv):
        return transformedVarianceByIntervalComp(B, t, posderiv, transformedMean)
    elif pos2ndderiv:
        return transformedVariancePos2ndDeriv(B, t, posderiv, transformedMean)
    else:
        newT = lambda x: (-1) * t(x)
        newTransformedMean = interval.Interval((-1) * (transformedMean.hi), (-1) * (transformedMean.lo))
        return transformedVariancePos2ndDeriv(B, newT, not posderiv, newTransformedMean)


def transformedVariancePos2ndDeriv(B, t, posderiv, transformedMean):
    result1 = transformedVarianceByIntervalComp(B, t, posderiv, transformedMean)
    if result1 is None:#np.isnan(result1):
        return None
    else:
        result2 = transformedVarianceByRowesPos2ndDeriv(B, t, posderiv, transformedMean)
        result_lower = np.max([result1.lo, result2.lo])
        result_upper = np.min([result1.hi, result2.hi])
        
        return interval.Interval(result_lower, result_upper)


def transformedVarianceByRowesPos2ndDeriv(B, t, posderiv, transformedMean):
    m = B.lo[0]
    M = B.hi[B.n - 1]
    mu_lower = B.mean_lo
    mu_upper = B.mean_hi
    v_lower = B.var_lo
    v_upper = B.var_hi
    
    transformedMean_lower = transformedMean.lo
    transformedMean_upper = transformedMean.hi
    
    eta_lower = optimize.brentq(lambda x: t(x) - transformedMean_lower, a=m, b=M, xtol = 0.0001)
    eta_upper = optimize.brentq(lambda x: t(x) - transformedMean_upper, a=m, b=M, xtol = 0.0001)
    
    if posderiv:
        transformedVariance_lower = (transformedMean_lower - t(m)) ** 2 / (eta_lower - m) ** 2 * (v_lower + getIntervalDistance(eta_lower, eta_lower, mu_lower, mu_upper).lo ** 2)
        transformedVariance_upper = (transformedMean_upper - t(M)) ** 2 / (eta_upper - M) ** 2 * (v_upper + getIntervalDistance(eta_upper, eta_upper, mu_lower, mu_upper).hi ** 2)
    else:
        transformedVariance_lower = (transformedMean_upper - t(M)) ** 2 / (eta_upper - M) ** 2 * (v_upper + getIntervalDistance(eta_upper, eta_upper, mu_lower, mu_upper).lo ** 2)
        transformedVariance_upper = (transformedMean_lower - t(m)) ** 2 / (eta_lower - m) ** 2 * (v_lower + getIntervalDistance(eta_lower, eta_lower, mu_lower, mu_upper).hi ** 2)

    return interval.Interval(transformedVariance_lower, transformedVariance_upper)


def transformedVarianceByIntervalComp(B, t, posderiv, transformedMean):
    transformedMean_lower = transformedMean.lo
    transformedMean_upper = transformedMean.hi
    transformedSum = 0
    
    for j in range(B.n):
        transformedSum += t(B.lo[j])

    currentTransformedMean = transformedSum / (B.n)
    
    if posderiv:
        if currentTransformedMean > transformedMean_upper:
            return None
    else:
        if currentTransformedMean < transformedMean_lower:
            return None
  	
    transformedSum = 0
    
    for j in range(B.n):
        transformedSum += t(B.hi[j])
    
    currentTransformedMean = transformedSum / (B.n)
    
    if posderiv:
        if currentTransformedMean < transformedMean_lower:
            return None
    else:
        if currentTransformedMean > transformedMean_upper:
            return None

    transformedVariance_lower = transVarLowerByIntervalCompPos2ndDeriv(B, t, posderiv, transformedMean)
    transformedVariance_upper = transVarUpperByIntervalCompPos2ndDeriv(B, t, posderiv, transformedMean)

    if transformedVariance_lower is None or transformedVariance_upper is None:#np.isnan(transformedVariance_lower) or np.isnan(transformedVariance_upper):
        return None
    else:
        return interval.Interval(transformedVariance_lower, transformedVariance_upper)


def transVarUpperByIntervalCompPos2ndDeriv(B, t, posderiv, transformedMean):
    transformedMean_lower = transformedMean.lo
    transformedMean_upper = transformedMean.hi
    transformedVariance_upper = None
    transformedSum = 0
    transformedSquareSum = 0
    
    for j in range(B.n):
        transformedSum += t(B.hi[j])
        transformedSquareSum += t(B.hi[j]) ** 2
    
    currentTransformedMean = transformedSum / (B.n)
    
    if currentTransformedMean >= transformedMean_lower and currentTransformedMean <= transformedMean_upper:
        currentTransformedVariance = transformedSquareSum / (B.n) - currentTransformedMean ** 2
        transformedVariance_upper = currentTransformedVariance
    
    for i in range(B.n):
        transformedSum += t(B.lo[i]) - t(B.hi[i])
        transformedSquareSum += (t(B.lo[i])) ** 2 - (t(B.hi[i])) ** 2
        currentTransformedMean = transformedSum / (B.n)
        
        if currentTransformedMean >= transformedMean_lower and currentTransformedMean <= transformedMean_upper:
            currentTransformedVariance = transformedSquareSum / (B.n) - currentTransformedMean ** 2
            
            if transformedVariance_upper is None: #np.isnan(transformedVariance_upper):
                transformedVariance_upper = currentTransformedVariance
            else:
                transformedVariance_upper = np.max([transformedVariance_upper, currentTransformedVariance])

        else:
            value_middle = (B.n) * transformedMean_lower - (transformedSum - t(B.lo[i]))
            currentTransformedMean = (transformedSum - t(B.lo[i]) +  value_middle) / (B.n)

            if (value_middle >= t(B.hi[i]) and value_middle <= t(B.lo[i]) and posderiv) or \
                (value_middle >= t(B.lo[i]) and value_middle <= t(B.hi[i]) and not posderiv):
                
                currentTransformedVariance = (transformedSquareSum - t(B.lo[i]) ** 2 + value_middle ** 2) / (B.n) - currentTransformedMean ** 2
                
                if transformedVariance_upper is None: # np.isnan(transformedVariance_upper):
                    transformedVariance_upper = currentTransformedVariance
                else:
                    transformedVariance_upper = np.max([transformedVariance_upper, currentTransformedVariance])
            
            value_middle = (B.n) * transformedMean_upper - (transformedSum - t(B.lo[i]))
            currentTransformedMean = (transformedSum - t(B.lo[i]) +  value_middle) / (B.n)
            
            if (value_middle >= t(B.hi[i]) and value_middle <= t(B.lo[i]) and posderiv) or \
                (value_middle >= t(B.lo[i]) and value_middle <= t(B.hi[i]) and not posderiv):
                
                currentTransformedVariance = (transformedSquareSum - t(B.lo[i]) ** 2 + value_middle ** 2) / (B.n) - currentTransformedMean ** 2
                
                if transformedVariance_upper is None: # np.isnan(transformedVariance_upper):
                    transformedVariance_upper = currentTransformedVariance
                else:
                    transformedVariance_upper = np.max([transformedVariance_upper, currentTransformedVariance])

    return(transformedVariance_upper)


def transVarLowerByIntervalCompPos2ndDeriv(B, t, posderiv, transformedMean):
    transformedMean_lower = transformedMean.lo
    transformedMean_upper = transformedMean.hi
    transformedVariance_lower = None
    endingPoints = getTransformedBoundEndingPoints(B, t, posderiv)
    transformedSum = 0
    transformedSquareSum = 0
    
    if posderiv:
        for j in range(B.n):
            transformedSum += t(B.lo[j])
            transformedSquareSum += t(B.lo[j]) ** 2
    else:
        for j in range(B.n):
            transformedSum += t(B.hi[j])
            transformedSquareSum += t(B.hi[j]) ** 2
    
    currentTransformedMean = transformedSum / (B.n)
    
    if currentTransformedMean >= transformedMean_lower and currentTransformedMean <= transformedMean_upper:
        currentTransformedVariance = transformedSquareSum / (B.n) - currentTransformedMean ** 2
        transformedVariance_lower = currentTransformedVariance
    
    num_middle = 0

    for i in range(2 * (B.n) -1):
        if (endingPoints[1, i] < 0):
            transformedSum += endingPoints[0, i]
            transformedSquareSum += endingPoints[0, i]  ** 2
            num_middle += 1
        else:
            transformedSum += endingPoints[0, i]
            transformedSquareSum += endingPoints[0, i] ** 2
            num_middle -= 1

        currentTransformedMean_lower = (transformedSum + num_middle * endingPoints[0, i]) / (B.n)
        currentTransformedMean_upper = (transformedSum + num_middle * endingPoints[0, i + 1]) / (B.n)
        
        if currentTransformedMean_upper >= transformedMean_lower and currentTransformedMean_lower <= transformedMean_upper:
            currentTransformedMean_lower = np.max([currentTransformedMean_lower, transformedMean_lower])
            currentTransformedMean_upper = np.min([currentTransformedMean_upper, transformedMean_upper])
            cuurentTransformedMean = transformedSum / (B.n - num_middle) # never used? typo? ...
            
            if currentTransformedMean >= currentTransformedMean_lower and currentTransformedMean <= currentTransformedMean_upper:
                value_middle = currentTransformedMean
                currentTransformedVariance = (transformedSquareSum + num_middle * value_middle ** 2) / (B.n) - currentTransformedMean ** 2
            
                if transformedVariance_lower is None: # np.isnan(transformedVariance_lower):
                    transformedVariance_lower = currentTransformedVariance
                else:
                    transformedVariance_lower = np.min([transformedVariance_lower, currentTransformedVariance])

            else:
                value_middle = (currentTransformedMean_lower * (B.n) - transformedSum) / num_middle
                currentTransformedVariance = (transformedSquareSum + num_middle * value_middle ** 2) / (B.n) - currentTransformedMean_lower ** 2
                value_middle = (currentTransformedMean_upper * (B.n) - transformedSum) / num_middle
                currentTransformedVariance = np.min([currentTransformedVariance, (transformedSquareSum + num_middle * value_middle ** 2) / (B.n) - currentTransformedMean_upper ** 2])

                if transformedVariance_lower is None: # np.isnan(transformedVariance_lower):
                    transformedVariance_lower = currentTransformedVariance
                else:
                    transformedVariance_lower = np.min([transformedVariance_lower, currentTransformedVariance])
    
    transformedSum += endingPoints.flatten(order='F')[2 * (B.n)]
    transformedSquareSum += (endingPoints.flatten(order='F')[2 * (B.n)]) ** 2
    currentTransformedMean = transformedSum / (B.n)
    
    if currentTransformedMean >= transformedMean_lower and currentTransformedMean <= transformedMean_upper:
        currentTransformedVariance = transformedSquareSum / (B.n) - currentTransformedMean ** 2

        if transformedVariance_lower is None: # np.isnan(transformedVariance_lower):
            transformedVariance_lower = currentTransformedVariance
        else:
            transformedVariance_lower = np.min([transformedVariance_lower, currentTransformedVariance])
    
    return transformedVariance_lower


def getBoundEndingPoints(B):
    endingPoints = []
    i_u = 0
    i_d = 0

    for i in range(2 * (B.n)):
        if i_u >= B.n:
            endingPoints.append(B.hi[i_d])
            endingPoints.append(1)
            i_d += 1

        elif B.lo[i_u] < B.hi[i_d]:
            endingPoints.append(B.lo[i_u])
            endingPoints.append(-1)
            i_u += 1
        else:
            endingPoints.append(B.hi[i_d])
            endingPoints.append(1)
            i_d += 1
            
    endingPoints = np.array(endingPoints)
    
    return endingPoints.reshape(2, 2 * (B.n), order="F")


def getTransformedBoundEndingPoints(B, t, posderiv):
    endingPoints = []
    i_u = 0
    i_d = 0

    for i in range(2 * (B.n)):
        if i_u >= B.n:
            endingPoints.append(t(B.hi[i_d]))
            endingPoints.append(1)
            i_d += 1

        elif B.lo[i_u] < B.hi[i_d]:
            endingPoints.append(t(B.lo[i_u]))
            endingPoints.append(-1)
            i_u += 1
        else:
            endingPoints.append(t(B.hi[i_d]))
            endingPoints.append(1)
            i_d += 1
            
    endingPoints = np.array(endingPoints)
    endingPoints = endingPoints.reshape(2, 2 * (B.n), order="F")
    
    if not posderiv:
        newEndingPoints = []
        for i in range(2 * (B.n)):
            newEndingPoints.append(endingPoints[0, ((2 * (B.n)) - (1 + i))])
            newEndingPoints.append(-1 * endingPoints[1, ((2 * (B.n)) - (1 + i))])

        newEndingPoints = np.array(newEndingPoints)
        newEndingPoints = newEndingPoints.reshape(2, 2 * (B.n), order="F")

        return newEndingPoints

    else:
        return endingPoints


def compareInterval(lower1, upper1, lower2, upper2):
    result = 0
    if upper1 <= lower2:
        result = -1
    elif lower1 >= upper2:
        result = 1
    else:
        result = 0

    return result

def getIntervalDistance(lower1, upper1, lower2, upper2):
    dis_lower = 0
    dis_upper = 0
    
    if upper1 <= lower2:
        dis_lower = lower2 - upper1
        dis_upper = upper2 - lower1
    elif lower1 >= upper2:
        dis_lower = lower1 - upper2
        dis_upper = upper1 - lower2
    else:
        dis_lower = 0
        dis_upper = np.max([np.abs(lower1 - upper2), np.abs(lower2 - upper1)])
    
    return interval.Interval(dis_lower, dis_upper)


def testPd(f, B):
    n = 100
    lower_x = left(B)
    upper_x =  right(B)
    x = lower_x
    
    for i in range(n):
        middle_x = lower_x + (upper_x - lower_x) * i / 100
        x.append(middle_x)
    x = np.array(x)

    pd = f(x) - f(x + 0.01) < 0
 
    if (np.all(pd)):
        return True
    elif (np.all(not pd)):
        return False
    else:
        return None

def testP2d(f, B):
    n = 100
    lower_x = left(B)
    upper_x =  right(B)
    x = lower_x
    
    for i in range(n):
        middle_x = lower_x + (upper_x - lower_x) * i / 100
        x.append(middle_x)
    x = np.array(x)

    pd = (f(x) + f(x + 0.02) > 2 * f(x + 0.01))
 
    if (np.all(pd)):
        return True
    elif (np.all(not pd)):
        return False
    else:
        return None

