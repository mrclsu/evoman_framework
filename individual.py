###############################################################################
## Individual module                                                         ##
###############################################################################

# imports framework
import sys
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import sys
sys.path.append(r"C:\Users\ordix\OneDrive\Desktop\Evoman Project\evoman")
from environment import Environment


n_hidden_neurons = 10
run_mode = 'train' # train or test

class Individual:
    def __init__(self,mutation_step_size,lp,age=0,n_values=265,provided_genome=None):
        """
        Initialize an individual with its genome and fitness.

        Parameters:
        - n_values: The number of values in the genome.
        - env: The environment where the individual's fitness is evaluated.
        - mutation_step_size: The size of mutation step.
        - age: Age of the individual (default is 0).
        - given_genome: If given, initializes individual with the given genome.
        """
        if provided_genome is None:
            self.genome = np.random.uniform(-1, 1, n_values)
        else:
            self.genome = provided_genome
        self.fitness=self.evaluate()    
        self.fitness_history = []
        self.lp=lp
        self.mutation_step_size=mutation_step_size
        self.age=age
    def evaluate(self):
        
        self.fitness=env.play(self.genome)[0]
        self.fitness_history.append(self.fitness)
        return self.fitness
        return env.play(self.genome)[0]
    def lifepoints(self):
        return env.play(self.genome)[1]
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
    
    #------------Mutations Self Adaptive stepsize------
    def self_adaptive_stepsize(self):
        self.mutation_step_size
    
    # def self_adaptive_mutation(self,type,tau=1):
    #     if type=="uncorrelated_np":
    #         #log normal operator
    #         self.mutation_step_size=np.exp(tau*np.random.normal(0,1))
    #         #Boundary condition
    #         if self.mutation_step_size<0.001:
    #             self.mutation_step_size=0.001
    #         self.gaussian_mutation(0.1,self.mutation_step_size)
    #     elif type=="correlated_p":
    #         self.correlated_self_adaptive_mutation(tau,t)
    #         self.mutation_step_size*=np.exp(tau_prime*np.random.normal(0,1)+tau*np.random.normal(0,1))
    #     elif type=="correlated":

    
    
    
    #------------Recombination operators-----------------
    def single_arithmetic_crossover(self,other,alpha=0.5 ,stochastic=True):
        mutation_index = np.random.choice(range(len(self.genome)))
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
    def convex_crossover(self,other,alpha=0.5,stochastic=True):
        if stochastic==True:
            alpha=np.random.uniform(0,1)
        child1_genome=alpha*self.genome+(1-alpha)*other.genome
        child2_genome=alpha*other.genome+(1-alpha)*self.genome
        child1 = Individual(len(child1_genome), given_genome=child1_genome)
        child2 = Individual(len(child2_genome), given_genome=child2_genome)

        return child1,child2




    def blend_crossover(self, other, alpha=0.5, stochastic=True):
        if stochastic:
            alpha = np.random.uniform(0, 1)

        parent1 = self.genome
        parent2 = other.genome
        child1_genome = np.zeros(len(parent1))
        child2_genome = np.zeros(len(parent2))
        side = min(parent1[i], parent2[i])

        for i in range(len(parent1)):
            distance = abs(parent1[i] - parent2[i])
            new_alleles = np.random.uniform(side - distance*alpha, side + distance*alpha, 2)

            # Ensuring values are within the desired range.
            child1_genome[i] = np.clip(new_alleles[0], -1, 1)
            child2_genome[i] = np.clip(new_alleles[1], -1, 1)

        child1 = Individual(len(child1_genome), env, given_genome=child1_genome)
        child2 = Individual(len(child2_genome), env, given_genome=child2_genome)

        return child1, child2