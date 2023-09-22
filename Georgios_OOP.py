###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Georgios Xanthopoulos        			                              #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10



# initializes simulation in individual evolution mode, for single static enemy.

env = Environment(experiment_name=experiment_name,
                  enemies=[5],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=True)

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

def simulation(env,x):
     f,p,e,t = env.play(pcont=x)
     if e==0:
         print(f"Success ! Enemy killed hp bar {p},Fitness {f}, Time {t} ") 
     return f



n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 



class Individual:
    def __init__(self,mutation_step_size,age=0,n_values=265,given_genome=None):
        """
        Initialize an individual with its genome and fitness.

        Parameters:
        - n_values: The number of values in the genome.
        - env: The environment where the individual's fitness is evaluated.
        - mutation_step_size: The size of mutation step.
        - age: Age of the individual (default is 0).
        - given_genome: If given, initializes individual with the given genome.
        """
        if given_genome is None:
            self.genome = np.random.uniform(-1, 1, n_values)
        else:
            self.genome = given_genome
        self.fitness = self.evaluate()
        self.mutation_step_size=mutation_step_size
        self.age=age
    def evaluate(self):
        return env.play(self.genome)[0]
    def gaussian_mutation(self,mutation_rate,mutation_step_size=1, lower_bound=-1 ,upper_bound=1):
        # create a mask to determine which genes should undergo mutation based on the mutation_rate
        genes_2_mutate=np.random.uniform(0,1,len(self.genome))
        #mutate by adding white noise
        self.genome[genes_2_mutate<mutation_rate]+=np.random.normal(0,mutation_step_size,genes_2_mutate[genes_2_mutate<mutation_rate].shape)
        #Make sure values lie in [-1,1]
        self.genome=np.clip(self.genome,lower_bound,upper_bound)
    def uniform_mutation(self,mutation_rate,lower_bound,upper_bound):
        #create a mask to determine which genes should undergo mutation based on the mutation_rate
        genes_2_mutate=np.random.uniform(0,1,len(self.genome))
        #mutate by adding a completely new value from a uniform distribution in the range [lower_bound,upper_bound]
        self.genome[genes_2_mutate<mutation_rate]=np.random.uniform(lower_bound,upper_bound,genes_2_mutate[genes_2_mutate<mutation_rate].shape)
    def blend_crossover1(self, other,alpha=None,stochastic=True):
        if stochastic==True:
            alpha=np.random.uniform(0,1)
            self.genome=self.genome*alpha+(1-alpha)*other.genome
        elif stochastic==False:
            self.genome=self.genome*alpha+(1-alpha)*other.genome
    def self_adaptive_mutation(self,tau=1):
        #log normal operator
        self.mutation_step_size=np.exp(tau*np.random.normal(0,1))
        #Boundary condition
        if self.mutation_step_size<0.001:
            self.mutation_step_size=0.001
        self.gaussian_mutation(0.1,self.mutation_step_size)
    #------------Recombination operators-----------------
    def single_arithmetic_crossover(self,other,alpha=0.5 ,stochastic=True):
        mutation_index = np.random.choice(range(len(self.genome)))
        if stochastic==True:
            alpha=np.random.uniform(0,1)
        # For child1 based on parent1
        child1_genome = np.copy(self.genome)  
        child1_genome[mutation_index] = alpha * self.genome[mutation_index] + (1 - alpha) * other.genome[mutation_index]
        # For child2 based on parent2
        child2_genome=np.copy(other.genome)  # Start with all genes from parent2
        child2_genome[mutation_index] = alpha * other.genome[mutation_index] + (1 - alpha) * self.genome[mutation_index]

        child1 = Individual(len(child1_genome),  given_genome=child1_genome)
        child2 = Individual(len(child2_genome),  given_genome=child2_genome)

        return child1, child2
    def blend_crossover(self, other, alpha=0.5, stochastic=True):
        if stochastic:
            alpha = np.random.uniform(0, 1)

        parent1 = self.genome
        parent2 = other.genome
        child1_genome = np.zeros(len(parent1))
        child2_genome = np.zeros(len(parent2))

        for i in range(len(parent1)):
            side = min(parent1[i], parent2[i])
            distance = abs(parent1[i] - parent2[i])
            new_alleles = np.random.uniform(side - distance*alpha, side + distance*alpha, 2)

            # Ensuring values are within the desired range.
            child1_genome[i] = np.clip(new_alleles[0], -1, 1)
            child2_genome[i] = np.clip(new_alleles[1], -1, 1)

        child1 = Individual(len(child1_genome), env, given_genome=child1_genome)
        child2 = Individual(len(child2_genome), env, given_genome=child2_genome)

        return child1, child2

class Population:
    def __init__(self, size, distance, mutation_step_size, dom_l=-1, dom_u=1):
        # self.size=size
        # self.individuals=[Individual(10,env,mutation_step_size) for _ in range(self.size)]
        # self.env=env
        # self.mutation_step_size=mutation_step_size
        # self.population_fitness=[individual.fitness for individual in self.individuals]
        # self.population_worst_fitness=np.min(self.population_fitness)
        # self.distance = distance
        self.size = size
        self.individuals = [Individual(mutation_step_size) for _ in range(self.size)]
        self.mutation_step_size = mutation_step_size
        self.population_fitness = [individual.fitness for individual in self.individuals]
        self.population_worst_fitness = np.min(self.population_fitness)
        self.population_best_fitness = np.max(self.population_fitness)
        self.distance = distance
        self.dom_l = dom_l
        self.dom_u = dom_u

    def age_population(self):
            for individual in self.individuals:
                individual.age += 1
    #------------Scale-Population-Fitness-----------------
    def windowing_scaling(self):
        # Scale the fitnesses by shifting them by beta worst fintess in the last n gens
        beta=self.population_worst_fitness
        self.population_fitness=[fitness-beta for fitness in self.population_fitness]

    def sigma_scalling(self,c=2):

        self.population_fitness=list(map(lambda x: x-(self.population_avg_fitness-c*self.population_std_fitness),self.population_fitness))
        self.population_fitness=np.clip(self.population_fitness,0,None)
    def normalize_fitness(self):
        #normalize fitness to be in the range [0,1]
        self.population_fitness=list(map(lambda x: (x-self.population_worst_fitness)/(self.population_best_fitness-self.population_worst_fitness),self.population_fitness)) 
        # make sure fitness is not negative and not NaN
        self.population_fitness=np.clip(self.population_fitness,0,None)

    #------------Selection operators-----------------
    def population_statistics(self):
        self.population_avg_fitness=np.mean(self.population_fitness)
        self.population_best_fitness=np.max(self.population_fitness)
        self.population_best_individual=self.individuals[np.argmax(self.population_fitness)]
        self.population_worst_fitness=np.min(self.population_fitness)
        self.population_worst_individual=self.individuals[np.argmin(self.population_fitness)]
        self.population_std_fitness=np.std(self.population_fitness)

    def roullete_selection(self,selection_size):
        fitnesses=np.array(self.population_fitness)
        print(f'Fitnesses {fitnesses} and number of individuals {len(self.individuals)}')
        probabilities=fitnesses/np.sum(fitnesses)
        #select the individuals
        selected_individuals=np.random.choice(self.individuals,size=selection_size,p=probabilities)
        return selected_individuals
    

    def rank_selection(self,selection_size):
        #sort the individuals based on their fitness
        # Get the indices of the sorted individuals
        sorted_indices = np.argsort(self.population_fitness)
        # create empty array to store the ranks
        ranks=np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(self.size)
        probabilities = ranks / (self.size - 1)


        #change fitness to rank from the best having fitness pop-1 to the worst 0
        for i in range(len(self.individuals)):
            self.individuals[i].fitness=i
        #select the individuals


    def tournament_selection(self,k=2,deterministic=True,replace=False):
        # Select k members at random from the population
        winners=[]
        for _ in range(k):
            if deterministic: 
                #select k individuals
                selected_individuals=np.random.choice(self.individuals,size=k,replace=replace)
                #get the best individual based on fitness
                best_individual=selected_individuals[np.argmax([individual.fitness for individual in selected_individuals])]
                winners.append(best_individual)
            elif not deterministic:
                #select k individuals
                selected_individuals=np.random.choice(self.individuals,size=k,replace=replace)
                #decide the winner randomly
                best_individual=selected_individuals[np.random.choice(range(k))]
                winners.append(best_individual)

#-------------Survivor-selection operators-------------------------------------------###
    def mu_lambda_strategy(self, offspring_list):
            # 1. Evaluate offspring fitness
            offspring_fitness = [offspring.evaluate() for offspring in offspring_list]
            
            # 2. Select the top μ individuals from λ offspring
            sorted_indices = np.argsort(offspring_fitness)[::-1]  # Sort in descending order
            self.individuals = [offspring_list[i] for i in sorted_indices[:self.size]]

            # 3. Update the population's fitness list for the new generation
            self.population_fitness = [offspring_fitness[i] for i in sorted_indices[:self.size]]
    def lambda_plus_mu_strategy(self, offspring_list):
            # Combine both parents and offspring
            combined_population = self.individuals + offspring_list
                
                # Evaluate fitness of the combined list
            combined_fitness = [individual.evaluate(self.env) for individual in combined_population]
                
                # Select the top μ individuals from the combined list
            sorted_indices = np.argsort(combined_fitness)[::-1]  # Sort in descending order
            self.individuals = [combined_population[i] for i in sorted_indices[:self.size]]

                # Update the population's fitness list for the new generation
            self.population_fitness = [combined_fitness[i] for i in sorted_indices[:self.size]]
    
    def doomsday(self, pro=0.5):
        npop = len(self.individuals)
        n_vars = len(self.individuals[0].genome)

        # Identify the worst and best individuals based on fitness
        order = np.argsort(self.population_fitness)
        worst_inds = order[:int(npop/4)]
        best_ind = self.individuals[order[-1]]

        for o in worst_inds:
            individual = self.individuals[o]
            for j in range(n_vars):
                if np.random.uniform(0, 1) <= pro:
                    individual.genome[j] = np.random.uniform(self.dom_l, self.dom_u)
                else:
                    individual.genome[j] = best_ind.genome[j]
            self.population_fitness[o] = individual.evaluate(self.env)        
    #----Preserving diversity operators-----------------
    @staticmethod
    def manhatan_distance(self,individual1,individual2):
        return np.sum(np.abs(individual1.genome-individual2.genome))
    @staticmethod
    def euclidean_distance(self,individual1,individual2):
        return np.sqrt(np.sum(np.square(individual1.genome-individual2.genome)))
    @staticmethod
    def hamming_distance(self,individual1,individual2):
        return np.sum(individual1.genome!=individual2.genome)
    

    def sharing_function(self, distance, sigma_sh=1.0, alpha=1.0):
        if distance < sigma_sh:
            return 1 - (distance/sigma_sh)**alpha
        return 0

    # def apply_fitness_sharing(self, sigma_sh=1.0):
    #     for i, individual in enumerate(self.individuals):
    #         sum_sh = sum(self.sharing_function(distance(individual, other), sigma_sh) for other in self.individuals if other != individual)
    #         individual.fitness /= (1 + sum_sh)
    # def deterministic_crowding_replace(self, parent1, parent2, child1, child2):
    #     if distance(child1, parent1) + distance(child2, parent2) < distance(child1, parent2) + distance(child2, parent1):
    #         if child1.fitness > parent1.fitness:
    #             self.replace_individual(parent1, child1)
    #         if child2.fitness > parent2.fitness:
    #             self.replace_individual(parent2, child2)
    #     else:
    #         if child1.fitness > parent2.fitness:
    #             self.replace_individual(parent2, child1)
    #         if child2.fitness > parent1.fitness:
    #             self.replace_individual(parent1, child2)
            
    def replace_individual(self, old_individual, new_individual):
        idx = self.individuals.index(old_individual)
        self.individuals[idx] = new_individual
        self.population_fitness[idx] = new_individual.fitness



class Island(Population):




    def __init__(self, size, mutation_step_size, evolution_scheme, **kwargs):
        super().__init__(size, env, mutation_step_size, **kwargs)
        self.evolution_scheme = evolution_scheme 
        self.epoch=epoch()
    def epoch(self):
        pass
    def island_evolution():
            pass
    def migration():    
            pass
    


#================ Evolutionary Algorithm ====================#
POPULATION_SIZE = 4
GENERATIONS = 1
MUTATION_RATE = 0.1
MUTATION_STEP_SIZE = 0.1
SELECTION_SIZE = 2
OFFSPRING_SIZE = 4



pop = Population(POPULATION_SIZE, distance=Population.euclidean_distance, mutation_step_size=MUTATION_STEP_SIZE)
print(f'Population initialization check')

# Run evolution
#save statistics
avg_fitness = []
best_fitness = []
best_individual = []
standard_deviation = []
for gen in range(GENERATIONS):
    pop.normalize_fitness()
    # print(f'Population Fitness After normalization {pop.population_fitness}')
    print(f'Normalize check')

    # Selection of parents
    parents = pop.roullete_selection(SELECTION_SIZE)
    print(f'Selection check')

    # Generate offspring through crossover
    offspring = []
    for i in range(0, OFFSPRING_SIZE,2):# 2 offspring per iteration so stepsize should be doubled
        parent1, parent2 = np.random.choice(parents, 2, replace=False)
        child1, child2 = parent1.single_arithmetic_crossover(parent2)
        offspring.extend([child1, child2])
        print(f'Crossover check | number of offsprings {len(offspring)}')

    # Mutate offspring
    for child in offspring:
        child.gaussian_mutation(MUTATION_RATE, MUTATION_STEP_SIZE)
    
    # Replace old population with the new offspring (this is the (μ,λ) strategy)
    pop.mu_lambda_strategy(offspring)
    print(f'Mu Lambda check')
    print(f'=================Population size is {len(pop.individuals)}')
    # Calculate and print statistics about the current generation
    pop.population_statistics()
    print(f"Generation: {gen}, Avg Fitness: {pop.population_avg_fitness}, Best Fitness: {pop.population_best_fitness}")
    avg_fitness.append(pop.population_avg_fitness)
    best_fitness.append(pop.population_best_fitness)
    best_individual.append(pop.population_best_individual)
    standard_deviation.append(pop.population_std_fitness)
    print(f'Statistics check')

# Plot statistics in one figure

import matplotlib.pyplot as plt

plt.figure()
plt.plot(avg_fitness, label="Average Fitness")
plt.plot(best_fitness, label="Best Fitness")
plt.plot(standard_deviation, label="Standard Deviation")
plt.legend()
plt.show()


# Run a simulation with the best individual
best_individual = pop.population_best_individual
best_fitness = pop.population_best_fitness
print(f"Best individual: {best_individual.genome}")
print(f"Best fitness: {best_fitness}")
print(f"Simulation of best individual: {best_individual.evaluate()}")

env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
		  		  speed="normal",
				  enemymode="static",
				  level=2,visuals=True)

env.play(pcont=best_individual.genome)

