from random import uniform, shuffle, random, randint
from time import time
from math import exp

def random_net(n):
    return [[int(uniform(1, 2 * n)) for _ in range(n)] for _ in range(n)]

net1 = [[0, 4, 2, 1], 
        [3, 0, 6, 5], 
        [2, 3, 0, 2], 
        [3, 8, 2, 0]]   # min_distance = 9

net2 = [[0, 4, 2, 1, 3, 5, 5, 6, 8, 6], 
        [3, 0, 6, 5, 7, 2, 2, 5, 8, 2], 
        [2, 3, 0, 2, 1, 4, 6, 8, 5, 4],
        [3, 8, 2, 0, 4, 5, 7, 9, 6, 5], 
        [6, 1, 3, 8, 0, 2, 4, 7, 9, 1], 
        [1, 2, 3, 4, 5, 0, 5, 9, 7, 8],
        [1, 2, 3, 4, 5, 6, 0, 4, 8, 3], 
        [6, 5, 4, 3, 2, 1, 7, 0, 5, 3], 
        [6, 3, 8, 7, 9, 1, 2, 5, 0, 5],
        [7, 3, 8, 9, 1, 2, 6, 5, 4, 0]]  # min_distance = 19

net3 = random_net(10)  # min_distance = ??
net = net2
N = len(net)

def tsp_exhaustive(start):
    from itertools import permutations
    min_weight = float("inf")
    for perm in permutations(range(N)):
        if perm[0] != start:
            continue
        weight = sum(net[perm[i]][perm[i+1]] for i in range(N-1)) + net[perm[-1]][perm[0]]
        min_weight = min(min_weight, weight)
    return min_weight

def get_nearest(st, visited):
    min_distance = float('inf')
    nearest = -1
    for i in range(N):
        if not visited[i] and net[st][i] < min_distance:
            min_distance = net[st][i]
            nearest = i
    return nearest, min_distance

def tsp_nearest_neighbor(start):
    visited = [False] * N
    visited[start] = True
    solution = [start]
    distance = 0
    for _ in range(N - 1):
        nearest, d = get_nearest(solution[-1], visited)
        visited[nearest] = True
        solution.append(nearest)
        distance += d
    return distance + net[solution[-1]][start]

def eval_solution(solution):
    return sum(net[solution[i]][solution[i+1]] for i in range(N))

def tsp_random(start, n):
    node_list = [i for i in range(N) if i != start]
    min_weight = float('inf')
    for _ in range(n):
        shuffle(node_list)
        random_solution = [start] + node_list + [start]
        min_weight = min(min_weight, eval_solution(random_solution))
    return min_weight

def swap(solution, i, j):
    solution[i], solution[j] = solution[j], solution[i]
    return solution

def inverse(solution, i, j):
    solution[i:j+1] = reversed(solution[i:j+1])
    return solution

def insert(solution, i, j):
    solution.insert(j, solution.pop(i))
    return solution

def recocido(start, n):
    T = 100
    alpha = 0.99
    x1 = list(range(N))
    shuffle(x1)
    x1 = [start] + [x for x in x1 if x != start] + [start]
    perturbations = [swap, inverse, insert]
    fx = eval_solution(x1)

    for _ in range(n):
        i, j = sorted([randint(1, N-1), randint(1, N-1)])
        while i == j:
            j = randint(1, N-1)
        x2 = perturbations[randint(0, 2)](x1[:], i, j)
        fy = eval_solution(x2)
        if fy < fx or random() < exp((fx - fy) / T):
            x1, fx = x2, fy
        T *= alpha
    return fx

if __name__ == '__main__':

    net = net1
    N = len(net)
    print("\nNet1, 4 nodos, min = 9")
    start = time()
    print(f"- Algoritmo aleatorio = {tsp_random(0, 1000)}, time: {time()-start}")
    start = time()
    print(f"- Vecino más cercano  = {tsp_nearest_neighbor(0)}, time: {time()-start}")
    start = time()
    print(f"- Búsqueda exhaustiva = {tsp_exhaustive(0)}, time: {time()-start}")
    start = time()
    print(f"- Recocido = {recocido(0, 1000)}, time: {time()-start}")

    net = net2
    N = len(net)
    print("\nNet2, 10 nodos, min = 19")
    start = time()
    print(f"- Algoritmo aleatorio = {tsp_random(0, 1000)}, time: {time()-start}")
    start = time()
    print(f"- Vecino más cercano  = {tsp_nearest_neighbor(0)}, time: {time()-start}")
    start = time()
    print(f"- Búsqueda exhaustiva = {tsp_exhaustive(0)}, time: {time()-start}")
    start = time()
    print(f"- Recocido = {recocido(0, 1000)}, time: {time()-start}")
    

    net = net3
    N = len(net)
    print("\nNet3, 11 nodos, min = ???")
    start = time()
    print(f"- Algoritmo aleatorio = {tsp_random(0, 1000)}, time: {time()-start}")
    start = time()
    print(f"- Vecino más cercano  = {tsp_nearest_neighbor(0)}, time: {time()-start}")
    start = time()
    print(f"- Búsqueda exhaustiva = {tsp_exhaustive(0)}, time: {time()-start}")
    start = time()
    print(f"- Recocido = {recocido(0, 1000)}, time: {time()-start}")
    print()