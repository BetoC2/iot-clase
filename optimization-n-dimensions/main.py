from random import uniform
from math import cos, exp, pi, sqrt
from time import time

def randx(xl, xh, dim):
    return [uniform(xl[i], xh[i]) for i in range(dim)]
  
def random_min(func, xl, xh, dim, n):
  xm = ()
  fm = float('inf')
  for i in range(n):
      x = randx(xl, xh, dim)
      f = func(x, dim)
      if f < fm:
          fm = f
          xm = x
  return xm

def hc_min(func, xl, xh, dim, n):
  xbest = randx(xl, xh, dim)
  ybest = func(xbest, dim)
  DX = [xh[i] - xl[i] for i in range(dim)]
  dx = [-DX[i] / 2 + uniform(0, DX[i]) for i in range(dim)]
  x = [0 for _ in range(dim)]
  for i in range(n):
      for i in range(dim): x[i] = xbest[i] + dx[i]
      y = func(x, dim)
      if y < ybest:
          ybest = y
          for i in range(dim): xbest[i] = x[i]
          fit = 1 / (1 + ybest)
          for i in range(dim): DX[i] = (xh[i] - xl[i]) * (1 - fit)
      else:
          for i in range(dim): dx[i] = -DX[i] / 2 + uniform(0, DX[i])
  return xbest

from random import uniform

def pso_min(func, xl, xh, dim, n):
    w = 0.001
    c1 = 0.1
    c2 = 0.1
    m = 100

    def add_vectors(v1, v2):
        return [v1[i] + v2[i] for i in range(len(v1))]

    def sub_vectors(v1, v2):
        return [v1[i] - v2[i] for i in range(len(v1))]

    def mul_vector_scalar(v, s):
        return [v[i] * s for i in range(len(v))]

    class Particle:
        # Propiedades estáticas para almacenar la mejor posición global
        global_best_p = None
        global_best_y = float("inf")
        W = w
        C1 = c1
        C2 = c2

        def __init__(self):
            self.position = [uniform(xl[i], xh[i]) for i in range(dim)]
            self.velocity = [uniform(-1, 1) for _ in range(dim)]
            self.local_best_p = self.position
            self.local_best_y = func(self.position, dim)
            # Actualizamos la mejor posición global al crear la partícula
            if Particle.global_best_p is None or self.local_best_y < Particle.global_best_y:
                Particle.global_best_p = self.local_best_p
                Particle.global_best_y = self.local_best_y

        def update_particle(self, func):
            r1 = uniform(0, 1)
            r2 = uniform(0, 1)
            # Cálculo de la nueva velocidad
            inertia = mul_vector_scalar(self.velocity, Particle.W)
            parte1 = mul_vector_scalar(sub_vectors(self.local_best_p, self.position), Particle.C1 * r1)
            parte2 = mul_vector_scalar(sub_vectors(Particle.global_best_p, self.position), Particle.C2 * r2)
            new_velocity = add_vectors(inertia, add_vectors(parte1, parte2))
            self.velocity = new_velocity

            # Actualizar la posición
            self.position = add_vectors(self.position, self.velocity)
            y = func(self.position, dim)

            # Actualizar la mejor posición local
            if y < self.local_best_y:
                self.local_best_y = y
                self.local_best_p = self.position

            # Actualizar la mejor posición global
            if y < Particle.global_best_y:
                Particle.global_best_y = y
                Particle.global_best_p = self.position

            return y

    Particle.global_best_p = None
    Particle.global_best_y = float("inf")

    particles = [Particle() for _ in range(m)]

    for _ in range(n):
        for p in particles:
            p.update_particle(func)

    return Particle.global_best_p


def rosenbrock(x, dim):
    fx = 0
    for i in range(dim - 1):
        fx += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
    return fx

def rastrigin(x, dim):
    fx = 0
    for i in range(dim):
        fx += x[i]**2 - 10 * cos(2 * pi * x[i])
    return 10 * dim + fx

def styblinski(x, dim):
    fx = 0
    for i in range(dim):
        fx += x[i]**4 - 16 * x[i]**2 + 5 * x[i]
    return fx / 2

def ackley(x, dim):
    parte1 = 0
    parte2 = 0
    for i in range(dim):
        parte1 += x[i]**2
        parte2 += cos(2 * pi * x[i])
    return -20 * exp(-0.2 * sqrt(parte1 / dim)) - exp(parte2 / dim) + 20 + exp(1)

# Bloque principal
if __name__ == '__main__':
    func = ackley
    n = 100_000
    dim = 5
    xl = [-5 for _ in range(dim)]
    xh = [5 for _ in range(dim)]

    # Búsqueda aleatoria
    start = time()
    minx = random_min(func, xl, xh, dim, n)
    end = time()
    print("Búsqueda aleatoria:")
    print(f"x = {minx}")
    print(f"f(x) = {func(minx, dim):.9f}")
    print(f"{(end - start):.2f} segundos\n")

    # Hill Climbing
    start = time()
    minx = hc_min(func, xl, xh, dim, n)
    end = time()
    print("Hill Climbing:")
    print(f"x = {minx}")
    print(f"f(x) = {func(minx, dim):.9f}")
    print(f"{(end - start):.2f} segundos\n")

    # PSO (se utiliza n//100 generaciones para reducir el tiempo de cómputo)
    start = time()
    minx = pso_min(func, xl, xh, dim, n // 100)
    end = time()
    print("PSO:")
    print(f"x = {minx}")
    print(f"f(x) = {func(minx, dim):.9f}")
    print(f"{(end - start):.2f} segundos")

""" Salida de consola:
Búsqueda aleatoria:
x = [-0.11397225253515764, -0.21592809910827526, -0.3523292680693091, -0.09190026304002075, -0.26477031622819336]
f(x) = 2.364596084
0.26 segundos

Hill Climbing:
x = [0.06646754599709892, -0.06642316375692081, -0.9325066202729904, -0.1675075052881705, -0.19948756344876006]
f(x) = 2.352020990
0.31 segundos

PSO:
x = [-0.0016321512865632574, 0.005422291292310927, -0.0012201736483560851, -0.003371310521216239, 0.002462458624581836]
f(x) = 0.013315850
0.67 segundos
"""