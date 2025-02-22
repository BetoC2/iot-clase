# import random
from random import uniform, sample


def random_net(n):
    return [[int(uniform(1, 2 * n)) for _ in range(n)] for _ in range(n)]


net1 = [
    [0, 4, 2, 1], 
    [3, 0, 6, 5], 
    [2, 3, 0, 2], 
    [3, 8, 2, 0]
    ]  # min_distance = 9
net2 = [
    [0, 4, 2, 1, 3, 5, 5, 6, 8, 6],
    [3, 0, 6, 5, 7, 2, 2, 5, 8, 2],
    [2, 3, 0, 2, 1, 4, 6, 8, 5, 4],
    [3, 8, 2, 0, 4, 5, 7, 9, 6, 5],
    [6, 1, 3, 8, 0, 2, 4, 7, 9, 1],
    [1, 2, 3, 4, 5, 0, 5, 9, 7, 8],
    [1, 2, 3, 4, 5, 6, 0, 4, 8, 3],
    [6, 5, 4, 3, 2, 1, 7, 0, 5, 3],
    [6, 3, 8, 7, 9, 1, 2, 5, 0, 5],
    [7, 3, 8, 9, 1, 2, 6, 5, 4, 0],
]  # min_distance = 19
net3 = random_net(11)  # min_distance = ??
net = net2
n = len(net)



#### Algoritmo exhaustivo
def process(st: int, k: int, visited: list[int], solution: list[int]) -> int:
    if k == n:
        return net[solution[k-1]][st]
    
    min_distance = float('inf')
    best_path = []
    
    for i in range(n):
        if visited[i]:
            continue
        
        visited[i] = True
        solution[k] = i
        
        current_distance = net[solution[k-1]][i] + process(st, k+1, visited, solution)
        
        if current_distance < min_distance:
            min_distance = current_distance
            best_path = solution[k:].copy()
        
        visited[i] = False
    
    if best_path:
        solution[k:] = best_path[:]
    
    return min_distance

def tsp_exhaustive(st: int) -> int:
    # Si st=2, n=6:
    # Cómo generamos listas por "comprension"?
    # visited = [False, False, True, False, False, False]
    # Solution = [2, -1 -1, -1, -1, 2]
    # LLAMAR A PROCESS PASANDO TODOS ESTOS DATOS, con k = 1, d = 0
    visited = [i == st for i in range(n)]
    solution = [st if i == 0 or i == n else -1 for i in range(n + 1)]
    return process(st, 1, visited, solution)
    



#### Algoritmo vecino más cercano
def get_nearest(st: int, visited: list[int]) -> tuple[int, int]:
    min_distance = float('inf')
    nearest = -1
    for i in range(n):
        if visited[i]:
            continue

        if net[st][i] < min_distance:
            min_distance = net[st][i]
            nearest = i
    
    return nearest, min_distance

def tsp_nearest_neighbor(start: int) -> int:
    visited = [i == start for i in range(n)]
    solution = [start if i == 0 or i == n else -1 for i in range(n + 1)]
    distance = 0
    
    for i in range(n-1):
        nearest, d = get_nearest(solution[i], visited)
        visited[nearest] = True
        solution[i + 1] = nearest
        distance += d
    
    return distance + net[solution[n-1]][start]
        




#### Algoritmo aleatorio
def eval(start, solution):
    return sum(net[solution[i]][solution[i+1]] for i in range(n))


def tsp_random(start: int, iterations: int):
    return min(eval(start, [start] + sample([i for i in range(n) if i != start], n - 1) + [start]) for _ in range(iterations))


if __name__ == "__main__":
    minD = tsp_random(0, 10000)
    print(minD)
    minD = tsp_exhaustive(0)
    print(minD)
    minD = tsp_nearest_neighbor(0)
    print(minD)
