from math import pow
from random import randint

class Individual:
    def __init__(self, xl, xh):
        self.genotype = randint(0, 10**13 - 1)
        self.phenotype = [0, 0]
        self.toPhenotype(xl, xh)
        self.fitness = 0
        self.expectedValue = 0

    def toPhenotype(self, xl, xh):
        self.phenotype[0] = xl[0] + (self.genotype // 10**6) * (xh[0] - xl[0]) / 10**7-1
        self.phenotype[1] = xl[1] + (self.genotype % 10**6) * (xh[1]- xl[0]) / 10**7-1

    def __repr__(self):
        return f'{self.genotype}, {self.phenotype[0]}, {self.phenotype[1]}\n'

    

def styblinski(x, dim=2):
  fx = 0
  for i in range(dim):
      fx += pow(x[i], 4) - 16 * pow(x[i], 2) + 5 * x[i]
  return fx / 2


def fitness(x, fxl, fxh):
    return 0


def updateFitness():
    pass


def createPop(M, xl, xh):
    return [Individual(xl, xh) for _ in range (M)]


def selection():
    pass


def crossover():
    pass


def mutation():
    pass



def runGA(N, M, xl, xh, fxl, fxh):
    pop = createPop(M, xl, xh)
    print(pop)
    #for i in range(N):
    #    updateFitness()
    #    selection()
    #    crossover()
    #    mutation()
    #updateFitness()


if __name__ == '__main__':
    runGA(10, 10, [-5, -5], [5, 5], -80, 250)
