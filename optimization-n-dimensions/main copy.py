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

    # Inicializar partículas como diccionarios
    particles = []
    global_best_p = None
    global_best_y = float("inf")

    for _ in range(m):
        pos = [uniform(xl[i], xh[i]) for i in range(dim)]
        vel = [uniform(-1, 1) for _ in range(dim)]
        local_best_p = pos[:]        # copia de la posición inicial
        local_best_y = func(pos, dim)
        # Actualizar el global si corresponde
        if local_best_y < global_best_y:
            global_best_y = local_best_y
            global_best_p = pos[:]
        particles.append({
            "position": pos,
            "velocity": vel,
            "local_best_p": local_best_p,
            "local_best_y": local_best_y
        })

    # Iterar n generaciones actualizando cada partícula
    for _ in range(n):
        for particle in particles:
            r1 = uniform(0, 1)
            r2 = uniform(0, 1)
            inertia = mul_vector_scalar(particle["velocity"], w)
            part1 = mul_vector_scalar(sub_vectors(particle["local_best_p"], particle["position"]), c1 * r1)
            part2 = mul_vector_scalar(sub_vectors(global_best_p, particle["position"]), c2 * r2)
            new_velocity = add_vectors(inertia, add_vectors(part1, part2))
            particle["velocity"] = new_velocity

            new_position = add_vectors(particle["position"], new_velocity)
            particle["position"] = new_position
            current_value = func(new_position, dim)

            # Actualizar la mejor posición local de la partícula
            if current_value < particle["local_best_y"]:
                particle["local_best_y"] = current_value
                particle["local_best_p"] = new_position[:]

            # Actualizar la mejor posición global
            if current_value < global_best_y:
                global_best_y = current_value
                global_best_p = new_position[:]

    return global_best_p, global_best_y

def ackley(x, dim):
    parte1 = 0
    parte2 = 0
    for i in range(dim):
        parte1 += x[i]**2
        parte2 += cos(2 * pi * x[i])
    return -20 * exp(-0.2 * sqrt(parte1 / dim)) - exp(parte2 / dim) + 20 + exp(1)

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