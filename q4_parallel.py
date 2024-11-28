import random
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

terminals = [f"x{i}" for i in range(16)]

def AND(x, y):
    return x and y

def OR(x, y):
    return x or y

def NOT(x):
    return not x

def IF(cond, x, y):
    return x if cond else y

def SUM(*args):
    return sum(args)

def GT(x, y):
    return x > y

def LT(x, y):
    return x < y

functions = {
    "AND": AND,
    "OR": OR,
    "NOT": NOT,
    "IF": IF,
    "SUM": SUM,
    "GT": GT,
    "LT": LT,
}

class Node:
    def __init__(self, value, children=None):
        self.value = value
        if children:
            self.children = children
        else:
            self.children = []

    def evaluate(self, inputs):
        # If node is a terminal, return its value
        if self.value in terminals:
            return inputs[self.value]
        # If node is a function, evaluate children recursively
        elif self.value in functions:
            func = functions[self.value]
            args = [child.evaluate(inputs) for child in self.children]
            return func(*args)
        return self.value

    def __str__(self):
        if not self.children:
            return str(self.value)
        return f"{self.value}({', '.join(str(child) for child in self.children)})"

def generate_random_tree(max_depth):
    if max_depth == 0 or (max_depth > 1 and random.random() > 0.5):
        return Node(random.choice(terminals))
    func = random.choice(list(functions.keys()))
    num_args = len(functions[func].__code__.co_varnames)
    return Node(func, [generate_random_tree(max_depth - 1) for _ in range(num_args)])

def fitness(args):
    program, test_cases = args
    correct = 0
    for inputs, expected in test_cases:
        try:
            if program.evaluate(inputs) == expected:
                correct += 1
        except Exception:
            pass
    return correct / len(test_cases)

def crossover(parent1, parent2):
    if random.random() > 0.5:
        return parent1, parent2
    if not parent1.children or not parent2.children:
        return parent1, parent2
    cross_point1 = random.choice(parent1.children)
    cross_point2 = random.choice(parent2.children)
    child1 = Node(parent1.value, parent1.children.copy())
    child2 = Node(parent2.value, parent2.children.copy())
    child1.children.remove(cross_point1)
    child1.children.append(cross_point2)
    child2.children.remove(cross_point2)
    child2.children.append(cross_point1)
    return child1, child2

def mutate(node, max_depth):
    if random.random() < 0.1:
        return generate_random_tree(max_depth)
    if not node.children:
        return node
    return Node(node.value, [mutate(child, max_depth - 1) for child in node.children])

def generate_test_cases():
    test_cases = []
    for i in range(2**16):
        binary = bin(i)[2:].zfill(16)
        inputs = {f"x{j}": int(binary[j]) for j in range(16)}
        bit_sum = sum(inputs[f"x{k}"] for k in range(16))
        expected = 1 if 7 <= bit_sum <= 9 else 0
        test_cases.append((inputs, expected))
    return test_cases

def genetic_programming(pop_size, generations, max_depth, subset_ratio):
    population = [generate_random_tree(max_depth) for _ in range(pop_size)]
    test_cases = generate_test_cases()
    subset_size = int(subset_ratio * len(test_cases))
    fitness_progress = []

    with Pool() as pool:
        for generation in range(generations):
            if subset_size == len(test_cases):
                subset_test_cases = test_cases
            else:
                subset_test_cases = random.sample(test_cases, subset_size)

            # Prepare arguments for parallel fitness evaluation
            fitness_args = [(individual, subset_test_cases) for individual in population]

            # Evaluate fitness in parallel
            fitness_values = pool.map(fitness, fitness_args)
            population_fitness = list(zip(population, fitness_values))
            population_fitness.sort(key=lambda pf: -pf[1])
            population = [pf[0] for pf in population_fitness]

            # Record best fitness for this generation
            best_fitness = fitness((population[0], test_cases))
            fitness_progress.append(best_fitness)

            if best_fitness == 1.0:
                fitness_progress.extend([1.0] * (generations - generation - 1))
                break

            # Create a new population
            new_population = population[:int(0.2 * pop_size)]
            while len(new_population) < pop_size:
                if random.random() < 0.7:
                    p1, p2 = random.sample(population[:int(0.5 * pop_size)], 2)
                    child1, child2 = crossover(p1, p2)
                    new_population.append(mutate(child1, max_depth))
                    if len(new_population) < pop_size:
                        new_population.append(mutate(child2, max_depth))
                else:
                    new_population.append(mutate(random.choice(population), max_depth))
            population = new_population

        best_program = population[0]
        best_fitness = fitness((best_program, test_cases))

    return best_program, best_fitness, fitness_progress

def plot_fitness_progress(fitness_progress):
    plt.plot(fitness_progress)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Progression Over Generations")
    plt.show()

if __name__ == "__main__":
    start = time.time()
    best_program, best_fitness, fitness_progress = genetic_programming(pop_size=100, generations=300, max_depth=20, subset_ratio=0.1)

    end = time.time()
    print(f"Best fitness: {int(65536 * best_fitness)}/65536")
    print("Best program:", best_program)
    print(f"Time taken: {round(end - start, 2)}s")
    plot_fitness_progress(fitness_progress)
