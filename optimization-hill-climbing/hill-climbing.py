import math
from random import uniform, random

def rosenbrock(x):
    return float(100 * (x[1] - x[0]**2)**2 + (1-x[0])**2)

def rastrigin(x):
    return float(20+(x[0]**2-10*math.cos(2*math.pi*x[0]))+(x[1]**2-10*math.cos(2*math.pi*x[1])))

def styblinsky_tang(x):
    return float(0.5*((x[0]**4-16*x[0]**2+5*x[0])+(x[1]**4- 16*x[1]**2+5*x[1])))

def getDx(xLow, xHigh, fitness):
    return ((xHigh[0] - xLow[0])*(1-fitness), (xHigh[1] - xLow[1])*(1-fitness))

def ranIncrement(Dx):
    return (-0.5*Dx[0]+random()*Dx[0], -0.5*Dx[1]+random()*Dx[1])

def sumTuplas(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])


def hc_min(func, xLow, xHigh, n):
    x = (uniform(xLow[0], xHigh[0]), uniform(xLow[1], xHigh[1]))
    y = func(x)
    Dx = getDx(xLow, xHigh, 0)
    dx = ranIncrement(Dx)

    for _ in range(n):
        tempX = sumTuplas(x, dx)
        tempY = func(tempX)
        if(tempY < y):
            x = tempX
            y = tempY
            Dx = getDx(xLow, xHigh, 1/(1+y))
        else:
            dx = ranIncrement(Dx)

    return x, y

def run_tests():
    test_cases = {
        "Rosenbrock": rosenbrock,
        "Rastrigin": rastrigin,
        "Styblinsky-Tang": styblinsky_tang
    }
    
    iterations = [100, 10_000, 1_000_000]
    xlow = (-5, -5)
    xhigh = (5, 5)
    
    for name, func in test_cases.items():
        print(f"\nFuncion {name}:")
        for n in iterations:
            xmin, fmin = hc_min(func, xlow, xhigh, n)
            print(f"  n = {n}: x* = {xmin}, f(x*) = {fmin}")



# Test functions
def rosenbrock(x):
    return float(100 * (x[1] - x[0]**2)**2 + (1-x[0])**2)

def rastrigin(x):
    return float(20+(x[0]**2-10*math.cos(2*math.pi*x[0]))+(x[1]**2-10*math.cos(2*math.pi*x[1])))

def styblinskyTang(x):
    return float(0.5*((x[0]**4-16*x[0]**2+5*x[0])+(x[1]**4- 16*x[1]**2+5*x[1])))


#######     Hill Climbing       ######
def getDx(xLow, xHigh, fitness):
    return ((xHigh[0] - xLow[0])*(1-fitness), (xHigh[1] - xLow[1])*(1-fitness))

def ranIncrement(Dx):
    return (-0.5*Dx[0]+random()*Dx[0], -0.5*Dx[1]+random()*Dx[1])

def sumTuplas(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])


def hc_min(func, xLow, xHigh, n):
    x = (uniform(xLow[0], xHigh[0]), uniform(xLow[1], xHigh[1]))
    y = func(x)
    Dx = getDx(xLow, xHigh, 0)
    dx = ranIncrement(Dx)

    for _ in range(n):
        tempX = sumTuplas(x, dx)
        tempY = func(tempX)
        if(tempY < y):
            x = tempX
            y = tempY
            Dx = getDx(xLow, xHigh, 1/(1+y))
        else:
            dx = ranIncrement(Dx)

    return x, y

if __name__ == '__main__':
    run_tests()