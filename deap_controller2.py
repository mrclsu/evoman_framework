import os, random, array
from deap import base, benchmarks, creator, tools, algorithms
from evoman.environment import Environment
from evoman.controller import Controller
import numpy as np

IND_SIZE = 10
MIN_VALUE = 10
MAX_VALUE = 10
MIN_STRATEGY = 0.2
MAX_STRATEGY = 0.7

class player_controller(Controller):
  def __init__(self, n_hidden):
    self.n_hidden = n_hidden

experiment_name = 'deap_specialist_demo'
if not os.path.exists(experiment_name):
  os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden=n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2,
                  visuals=True)

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

def simulation(env,x):
  fit, _, _, _ = env.play(pcont=x)
  return fit

def evaluate(individual):
  return simulation(env, individual),

# Define DEAP fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Define DEAP Toolbox
def generateES(icls, scls, size, imin, imax, smin, smax):
  ind = icls(random.uniform(imin, imax) for _ in range(size))
  ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
  return ind

def checkStrategy(minstrategy):
  def decorator(func):
      def wrappper(*args, **kargs):
          children = func(*args, **kargs)
          for child in children:
              for i, s in enumerate(child.strategy):
                  if s < minstrategy:
                      child.strategy[i] = minstrategy
          return children
      return wrappper
  return decorator

toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
  IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", benchmarks.sphere)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

def main():
  MU, LAMBDA = 100, 100
  pop = toolbox.population(n=MU)
  hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)

  pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
      cxpb=0.6, mutpb=0.3, ngen=500, stats=stats, halloffame=hof, verbose=True)

  return pop, logbook, hof

if __name__ == "__main__":
  main()