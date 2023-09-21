import neat
from evoman.controller import Controller
from evoman.environment import Environment
import numpy as np
import os
import csv

experiment_name = 'neat-controller'
visuals = False
env = Environment(
            experiment_name=experiment_name,
            enemies=[4],
            playermode="ai",
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=visuals,
            sound='off',
)

def sigmoid_activation(x):
    return 1./(1.+np.exp(-x))

class player_controller(Controller):
    def __init__(self, net):
        self.net = net  # NEAT network
    
    def control(self, inputs, _):
        # Normalizes the input using min-max scaling
        #inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        print(inputs)

        output = self.net.activate(inputs)

        return [1 if o > 0.5 else 0 for o in output]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        player = player_controller(net)

        env.player_controller = player
        genome.fitness = env.play()[0]


def run(config, working_dir, run_num = 0):
        # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # checkpointer = neat.Checkpointer(20)
    # checkpointer.filename_prefix = os.path.join(working_dir, f'{run_num}-neat-checkpoint-')
    # p.add_reporter(checkpointer)
    
    # todo change back to 100
    winner = p.run(eval_genomes, 1)
    stats.save()
    # Save stats to a csv with the run number
    with open(os.path.join(working_dir, f'stats-{run_num}.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['generation', 'max_fitness', 'mean_fitness', 'stdev_fitness'])

        max_fitness = [c.fitness for c in stats.most_fit_genomes]
        mean_fitness = stats.get_fitness_mean()
        stdev_fitness = stats.get_fitness_stdev()

        for g in zip(range(0, len(max_fitness)), max_fitness, mean_fitness, stdev_fitness):
            writer.writerow([g[0], g[1], g[2], g[3]])
    # winner.save(f'winner-{run_num}')
    print(winner)


def main():
    # Create folder for evomen output
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run(config = config, working_dir = experiment_name, run_num = 0)

if __name__ == '__main__':
    main()