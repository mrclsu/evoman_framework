#######################################################################################
# EvoMan EA			                              		              #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm            #
#                                                                                     #
# Author: Austin Dickerson       			                              #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

import numpy as np
import random

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="fastest",
				  enemymode="static",
				  level=2,
				  visuals=False)
print(env.get_num_sensors())
num_vars = n_hidden_neurons*(env.get_num_sensors()+1) + (n_hidden_neurons+1)*5
mutagenic_temperature = 0.3
threshold = 0.2
replacement = 0.5
pop_size = 10
generations = 200

def initialize_population(pop_size, num_vars):
    pop = []
    for i in range(pop_size):
        member = np.zeros(num_vars)
        for j in range(num_vars):
            member[j] = random.uniform(-1,1)
        pop.append(member)
    return pop
    
def simulate(env,unit):
    fit, _, _, _ = env.play(pcont=unit)
    return fit

def test_population(pop):
    scores = []
    for unit in pop:
        scores.append(simulate(env,unit))
    return scores, max(scores)

def normalize(scores):
    lo = min(scores)
    hi = max(scores)
    scores -= lo
    scores /= (hi-lo)
    return scores

def combine_genes(p1, p2):
    first_key = np.random.randint(2, len(p1))
    second_key = np.ones(len(p1))-first_key
    zygote1 = first_key*p1
    zygote2 = second_key*p2
    child = zygote1 + zygote2
    return child

def select_parents(pop, low, high):
    first = random.randint(low,high-1)
    second = random.randint(low,high-1)
    while first == second:
        second = random.randint(low,high-1)
    p1 = pop[first]
    p2 = pop[second]
    return p1, p2

def mutate_offspring(mutagenic_temperature, child):
    for i in range(len(child)):
        if random.random() < mutagenic_temperature:
            child[i] += random.uniform(-0.1,0.1)
            if child[i] < -1:
                child[i] = -1
            if child[1] > 1:
                child[i] = 1
    return child

def reproduce(pop, fit_scores):
    sorted_pairs = sorted(zip(pop, fit_scores), key=lambda x: x[1])
    pop, fit_scores = zip(*sorted_pairs)
    pop = list(pop)
    fit_scores = list(fit_scores)
    replaced = int(len(pop)*replacement)
    for i in range(replaced):
        p1, p2 = select_parents(pop, replaced, len(pop))
        child = combine_genes(p1, p2)
        pop[i] = mutate_offspring(mutagenic_temperature, child)
    return pop


performance = []  
pop = initialize_population(pop_size, d_hi, d_lo, num_vars)
for i in range(generations):
    fit_scores, maxi = test_population(pop)
    performance.append(maxi)
    pop = reproduce(pop, fit_scores)
print(performance)


