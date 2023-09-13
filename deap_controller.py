import sys, os
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from evoman.controller import Controller
import numpy as np

# Define DEAP fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

class player_controller(Controller):
  def __init__(self, n_hidden):
    self.n_hidden = n_hidden
  
# Define evaluation function
def evaluate_individual(individual):
  player = player_controller(n_hidden=n_hidden_neurons)
  player.set(individual, n_inputs=10)
  
  total_fitness = 0
  
  for en in range(1, 9):
      env.update_parameter('enemies', [en])
      fitness, *_ = env.play(player.control)
      total_fitness += fitness
  
  return (total_fitness,)

# Define DEAP Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1, 1)  # Assuming controller weights are in range [-1, 1]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=16)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Assuming want to use blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selBest)

experiment_name = 'deap_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 0

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2,
                  visuals=True)

# tests saved demo solutions for each enemy
for en in range(1, 9):
    env.update_parameter('enemies',[en])
    sol = np.loadtxt('solutions_demo/demo_'+str(en)+'.txt')
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
    env.play(sol)

# Create and evolve the population
population = toolbox.population(n=50)
generations = 10

for gen in range(generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)  # Adjust crossover and mutation probabilities
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Retrieve best individual and its performance
best_individual = tools.selBest(population, k=1)[0]
best_fitness = best_individual.fitness.values[0]



