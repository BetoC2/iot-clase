from random import uniform
from math import cos
from math import pi


def randx(xlow: tuple, xhigh: tuple):
  return uniform(xlow[0], xhigh[0]), uniform(xlow[1], xhigh[1])

def increment(Dx: tuple):
  return (-0.5*Dx[0] + uniform(0, 1) * Dx[0], -0.5*Dx[1] + uniform(0, 1) * Dx[1])

def sumTuples(a: tuple, b: tuple):
  return a[0] + b[0], a[1] + b[1]

def getDX(xlow: tuple, xhigh: tuple, fitness: float):
  return ((xhigh[0] - xlow[0])*(1-fitness), (xhigh[1] - xlow[1])*(1-fitness))

def hc_min(func, xlow: tuple, xhigh: tuple, n: int):
  # 1. Crear una solución aleatoria en la zona factible [x0, x1]
  # 2. Calcula y = funcion(x)
  # 3. Calcular el incremento máximo DX
  # 4. Calcular el incremento aleatorio dx delimitado por DX
  # 5. Por cada iteración:
  #    a. Crear una solución x = mejor_solucion + incremento
  #    b. Calcular y = función(x)
  #    c. Si y es mejor que el actual mejor:
  #         i.  Actualizar x, y mejores
  #        ii. Hacer más fino el máximo incremento DX en términos de la aptitud
  #    d. Si no:
  #         i. Calcular nuevo incremento aleatorio dx
  # 6. Devolver mejor x
  x = randx(xlow, xhigh)
  y = func(x)
  Dx = (xhigh[0] - xlow[0], xhigh[1] - xlow[1])
  dx = increment(Dx)

  best_x = x
  best_y = y

  for _ in range(n):
    temp_x = sumTuples(best_x, dx)
    temp_y = func(x)
    if temp_y < best_y:
      best_x = temp_x
      best_y = temp_y
      Dx = getDX(xlow, xhigh, best_y)
    else:
      Dx = increment(Dx)

  return best_x, best_y


def random_min(func, xlow, xhigh, n):
  xm = ()
  fm = float('inf')
  for i in range(n):
    x = randx(xlow, xhigh)
    f = func(x)
    if f < fm:
      fm = f
      xm = x
  return xm


# 𝑓(𝑥) = 100(𝑥2 − 𝑥1^2)^2 + (1 − 𝑥1)^2
def rosenbrock(x):
  return 100 * (x[1] - x[0]**2)** 2 + (1 - x[0])**2


def rastrigin(x):
  return 20 + x[0] ** 2 - 10 * cos(2 * pi * x[0]) + x[1] ** 2 - 10 * cos(2 * pi * x[1])


def styblinski(x):
  return (x[0] ** 4 - 16 * x[0] ** 2 + 5 * x[0] + x[1] ** 4 - 16 * x[1] ** 2 + 5 * x[1]) / 2


if __name__ == '__main__':
  """
  func = rosenbrock
  xlow = (-5, -5)
  xhigh = (5, 5)
  xmin = random_min(func, xlow, xhigh, 10000000)
  print("xmin =", xmin)
  f = rosenbrock(xmin)
  print(f"f(xmin) = {f:.8f}")
  
  xmin = hc_min(func, xlow, xhigh, 1000000)
  print("xmin =", xmin)
  f = rosenbrock(xmin)
  print("f(xmin) =", f)
  """

  xlow = (-5, -5)
  xhigh = (5, 5)
  
  x, y = hc_min(rosenbrock, xlow, xhigh, 1_000_000)
  print(x, y)