from random import uniform
import random
from math import cos, pi


def sum_vectors(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def mul_vector_escalar(v, e):
    return [v[i] * e for i in range(len(v))]

def sub_vectors(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]


class Particle:
    global_best_p = None
    global_best_y = float("inf")  # Inicializamos con infinito para encontrar mínimos
    W = 0.001
    C1 = 0.1
    C2 = 0.1

    def __init__(self, xlow, xhigh, func):
        self.position = [uniform(xlow[i], xhigh[i]) for i in range(len(xlow))]
        self.velocity = [random.uniform(-1, 1) for _ in range(len(xlow))]
        self.local_best_p = self.position[:]
        self.local_best_y = func(self.position)

        # Actualizamos la mejor posición global
        if Particle.global_best_p is None or self.local_best_y < Particle.global_best_y:
            Particle.global_best_p = self.local_best_p[:]
            Particle.global_best_y = self.local_best_y

    def __repr__(self):
        return f"x: {self.position}, v: {self.velocity}, b: {self.local_best_p}"
    
    def update_particle(self, func):
        r1 = uniform(0, 1)
        r2 = uniform(0, 1)

        # Cálculo de la nueva velocidad
        parte1 = mul_vector_escalar(self.velocity, Particle.W)
        parte2 = mul_vector_escalar(sub_vectors(self.local_best_p, self.position), Particle.C1 * r1)
        parte3 = mul_vector_escalar(sub_vectors(Particle.global_best_p, self.position), Particle.C2 * r2)
        new_velocity = sum_vectors(parte1, sum_vectors(parte2, parte3))

        # Actualizar la posición y evaluar la función objetivo
        self.position = sum_vectors(self.position, new_velocity)
        y = func(self.position)

        # Actualizar la mejor posición local
        if y < self.local_best_y:
            self.local_best_p = self.position[:]
            self.local_best_y = y

        # Actualizar la mejor posición global
        if y < Particle.global_best_y:
            Particle.global_best_p = self.position[:]
            Particle.global_best_y = y

        self.velocity = new_velocity


def pso_min(func, xlow, xhigh, n, m):
    # Reiniciar la mejor posición global antes de iniciar una nueva ejecución
    Particle.global_best_p = None
    Particle.global_best_y = float("inf")

    # Inicializar partículas
    particles = [Particle(xlow, xhigh, func) for _ in range(m)]

    for _ in range(n):
        for p in particles:
            p.update_particle(func)

    return Particle.global_best_p, Particle.global_best_y


# Funciones de prueba
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rastrigin(x):
    return 20 + x[0] ** 2 - 10 * cos(2 * pi * x[0]) + x[1] ** 2 - 10 * cos(2 * pi * x[1])

def styblinski(x):
    return (x[0] ** 4 - 16 * x[0] ** 2 + 5 * x[0] + x[1] ** 4 - 16 * x[1] ** 2 + 5 * x[1]) / 2


if __name__ == '__main__':
    test_cases = {
        "Rosenbrock": rosenbrock,
        "Rastrigin": rastrigin,
        "Styblinsky-Tang": styblinski
    }

    xlow = (-5, -5)
    xhigh = (5, 5)

    for name, func in test_cases.items():
        print(f"\nFunción {name}:")
        for n in [1000, 10_000]:
          xmin, fmin = pso_min(func, xlow, xhigh, n, n // 10)
          print(f"n = {n}, m = {n // 10} x* = {xmin}, f(x*) = {fmin}")