import random
import math

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1-x[0])**2

def rastrigin(x):
    return 20+(x[0]**2-10*math.cos(2*math.pi*x[0]))+(x[1]**2-10*math.cos(2*math.pi*x[1]))

def styblinsky_tang(x):
    return 0.5*((x[0]**4-16*x[0]**2+5*x[0])+(x[1]**4-16*x[1]**2+5*x[1]))

def random_min(func, xlow: tuple, xhigh: tuple, n: int):
    best_x = None
    best_f = float('inf')
    
    decimals = 7
    
    for _ in range(n):
        x1 = random.uniform(xlow[0], xhigh[0])
        x2 = random.uniform(xlow[1], xhigh[1])
        x = (x1, x2)
        f = func(x)
        if f < best_f:
            best_f = f
            best_x = x
    
    return best_x, best_f

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
            xmin, fmin = random_min(func, xlow, xhigh, n)
            print(f"  n = {n}: x* = {xmin}, f(x*) = {fmin}")

if __name__ == '__main__':
    run_tests()
