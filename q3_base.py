import random
import matplotlib.pyplot as plt
import time

terminals = ["a0", "a1", "a2", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
functions = {
    "AND": lambda x, y: x and y,
    "OR": lambda x, y: x or y,
    "NOT": lambda x: not x,
    "IF": lambda cond, x, y: x if cond else y,
}

class Node:
    def __init__(self, value, children=None):
        self.value = value
        if children:
            self.children = children
        else:
            self.children = []

    def evaluate(self, inputs):
        # If node is a terminal, return
        if self.value in terminals:
            return inputs[self.value]
        # If node is a function, evaluate children recursively
        elif self.value in functions:
            func = functions[self.value]
            args = [child.evaluate(inputs) for child in self.children]
            return func(*args)
        return self.value

    # Print the tree in a readable format
    def __str__(self):
        if not self.children:
            return str(self.value)
        return f"{self.value}({', '.join(str(child) for child in self.children)})"

def generate_random_tree(max_depth):
    # Base case or random chance to create terminal node
    if max_depth == 0 or (max_depth > 1 and random.random() > 0.5):
        return Node(random.choice(terminals))
    
    # Randomly select a function and recursively generate children
    func = random.choice(list(functions.keys()))
    num_args = len(functions[func].__code__.co_varnames)
    return Node(func, [generate_random_tree(max_depth - 1) for _ in range(num_args)])

def fitness(program, test_cases):
    correct = 0
    for inputs, expected in test_cases:
        if program.evaluate(inputs) == expected:
            correct += 1
    return correct / len(test_cases)

def crossover(parent1, parent2):
    # 50% chance to return parents as is
    if random.random() > 0.5:
        return parent1, parent2
    
    # Crossover not possible for terminal nodes
    if not parent1.children or not parent2.children:
        return parent1, parent2
    
    # Single point crossover
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
    # 10% chance to generate a new random tree
    if random.random() < 0.1:
        return generate_random_tree(max_depth)
    
    # Cannot mutate terminal nodes
    if not node.children:
        return node
    
    # Mutate children recursively
    return Node(node.value, [mutate(child, max_depth - 1) for child in node.children])

def generate_test_cases():
    test_cases = []
    for a0 in [0, 1]:
        for a1 in [0, 1]:
            for a2 in [0, 1]:
                for d_values in [[0, 0, 0, 0, 0, 0, 0, 1], 
                                 [0, 0, 0, 0, 0, 0, 1, 0], 
                                 [0, 0, 0, 0, 0, 1, 0, 0], 
                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0]]:
                    inputs = {"a0": a0, "a1": a1, "a2": a2}
                    for i, d in enumerate(d_values):
                        inputs[f"d{i}"] = d
                    
                    address = a2 * 4 + a1 * 2 + a0
                    expected = inputs[f"d{address}"]
                    
                    test_cases.append((inputs, expected))
    return test_cases

# Updated to only test a subset of the test cases
def genetic_programming(pop_size, generations, max_depth, subset_ratio):
    population = [generate_random_tree(max_depth) for _ in range(pop_size)]
    test_cases = generate_test_cases()
    subset_size = int(subset_ratio * len(test_cases))
    fitness_progress = []

    for generation in range(generations):
        # Select a random subset of test cases for this generation
        if subset_size == len(test_cases):
            subset_test_cases = test_cases
        else:
            subset_test_cases = random.sample(test_cases, subset_size)

        # Sort population by fitness evaluated on the subset
        population = sorted(population, key=lambda p: -fitness(p, subset_test_cases))

        # Evaluate the best individual's fitness on the full test set
        best_fitness = fitness(population[0], test_cases)
        fitness_progress.append(best_fitness)

        # If a perfect solution is found, stop early
        if best_fitness == 1.0:
            fitness_progress.extend([1.0] * (generations - generation - 1))
            break

        # Create a new population, keeping the top 20%
        new_population = population[:int(0.2 * pop_size)]
        while len(new_population) < pop_size:
            # 70% chance to perform crossover
            if random.random() < 0.7:
                p1, p2 = random.sample(population[:int(0.5 * pop_size)], 2)
                child1, child2 = crossover(p1, p2)
                new_population.append(mutate(child1, max_depth))
                if len(new_population) < pop_size:
                    new_population.append(mutate(child2, max_depth))
            # 30% chance to perform mutation
            else:
                new_population.append(mutate(random.choice(population), max_depth))

        population = new_population

    # Get the final best program and its fitness
    population = sorted(population, key=lambda p: -fitness(p, test_cases))
    best_program = population[0]
    best_fitness = fitness(best_program, test_cases)

    return best_program, best_fitness, fitness_progress

def plot_fitness_progress(fitness_progress):
    plt.plot(fitness_progress)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Progression Over Generations")
    plt.show()

if __name__ == "__main__":
    start = time.time()
    best_program, best_fitness, fitness_progress = genetic_programming(
        pop_size=300, generations=100, max_depth=30, subset_ratio=0.5
    )
    end = time.time()
    print(f"Best fitness: , {int(2048* best_fitness)}/2048")
    print("Best program:", best_program)
    print(f"Time taken: {round(end - start, 2)}s")
    plot_fitness_progress(fitness_progress)