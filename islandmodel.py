from population import Population
import random


class IslandModel:
    
    def __init__(self, num_islands, island_size, migration_rate, envs, mutation_step_size, **kwargs):
        self.islands = [Population(island_size, mutation_step_size) for _ in range(num_islands)]
        self.epoch_count = 0
        self.migration_rate = migration_rate

    def increment_epoch(self):
        self.epoch_count += 1

    def migrate(self):
        # Iterate over each island for migration
        for i in range(len(self.islands)):
            source_island = self.islands[i]
            
            # If it's the last island, set target to the first island, else set to next island
            target_island = self.islands[0] if i == len(self.islands) - 1 else self.islands[i + 1]
            
            # Choose a migrant from the source island
            migrant = random.choice(source_island.individuals)
            
            # Decide placement in target island: Here, we replace a random individual
            replace_idx = random.randint(0, len(target_island.individuals) - 1)
            target_island.individuals[replace_idx] = migrant

    def evolve_island(self, num_generations):
        for generation in range(num_generations):
            for island in self.islands:
                island.evolve_population()

