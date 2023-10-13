
import numpy as np
from individual import Individual

class Population:
    def __init__(self, size, mutation_step_size=1,distance=None, dom_l=-1, dom_u=1):
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
        
        # self.distance = distance
        # Historical Statistics
        self.avg_fitness_history = []
        self.best_fitness_history = []
        self.best_individual_history = []
        self.standard_deviation_history = []

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
    #------------Population-Statistics-----------------
    def update_statistics(self):
    # Calculate the statistics
        self.population_avg_fitness = np.mean(self.population_fitness)
        self.population_best_fitness = np.max(self.population_fitness)
        self.population_best_individual = self.individuals[np.argmax(self.population_fitness)]
        self.population_worst_fitness = np.min(self.population_fitness)
        self.population_worst_individual = self.individuals[np.argmin(self.population_fitness)]
        self.population_std_fitness = np.std(self.population_fitness)

        # Append the statistics to their respective lists
        self.avg_fitness_history.append(self.population_avg_fitness)
        self.best_fitness_history.append(self.population_best_fitness)
        self.best_individual_history.append(self.population_best_individual)
        self.standard_deviation_history.append(self.population_std_fitness)


    #------------Selection operators-----------------

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
    def elitism_selection(self,n_elites):
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        elites = sorted_individuals[:n_elites]


#-------------Survivor-selection operators-------------------------------------------###
    def mu_lambda_strategy(self, offspring_list):
        # 1. Evaluate offspring fitness
        offspring_fitness = [offspring.evaluate() for offspring in offspring_list]
        
        # 2. Select the top μ individuals from λ offspring
        sorted_indices = np.argsort(offspring_fitness)[::-1]  # Sort in descending order
        new_population = [offspring_list[i] for i in sorted_indices[:self.size]]

        # 3. Replace the current population with the new population
        self.individuals = new_population
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
    def age_population(self):
        for individual in self.individuals:
            individual.age += 1

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
#=================Evolution=======================

    def evolve_population(self):
                # 1. Selection
                selected_parents = self.roullete_selection(selection_size=10)  # For example
                
                # 2. Crossover to produce offspring (using the code from the previous answer)
                offspring = []
                for i in range(0, len(selected_parents), 2):
                    parent1 = selected_parents[i]
                    parent2 = selected_parents[i+1]
                    
                    if np.random.uniform(0, 1) < 0.5:
                        child1, child2 = parent1.single_arithmetic_crossover(parent2)
                    else:
                        child_genome = parent1.convex_crossover(parent2)
                        child = Individual(len(child_genome), given_genome=child_genome)
                        offspring.append(child)
                        continue

                    offspring.extend([child1, child2])
                for child in offspring:
                     child.gaussian_mutation(0.1,0.1)
                # 3. Survivor selection
                self.mu_lambda_strategy(offspring)
                self.age_population()
                self.update_statistics()
                self.normalize_fitness()
                
                    