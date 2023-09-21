import neat
from evoman.controller import Controller
from evoman.environment import Environment
import numpy as np
import os
import csv
import pickle

experiment_name = 'neat-controller'
visuals = False
env = None

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


def run(config, working_dir, enemy, run_num = 0):
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
    winner = p.run(eval_genomes, 100)
    # stats.save()
    # Save stats to a csv with the run number
    with open(os.path.join(working_dir, f'stats-{enemy}-{run_num}.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['generation', 'max_fitness', 'mean_fitness', 'stdev_fitness'])

        max_fitness = [c.fitness for c in stats.most_fit_genomes]
        mean_fitness = stats.get_fitness_mean()
        stdev_fitness = stats.get_fitness_stdev()

        for g in zip(range(0, len(max_fitness)), max_fitness, mean_fitness, stdev_fitness):
            writer.writerow([g[0], g[1], g[2], g[3]])

    print(winner)

    # Save winner to a pickle file
    with open(os.path.join(working_dir, f'winner-{enemy}-{run_num}.pkl'), 'wb') as output:
        pickle.dump(winner, output, 1)


def main():
    # Create folder for evomen output
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    global env 
    env = Environment(
            experiment_name=experiment_name,
            playermode="ai",
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=visuals,
            sound='off',
    )

    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    for enemy in [1, 4, 6]:
        env.enemies = [enemy]
        for run in range(0, 1): # todo change back to 10
            run(config = config, working_dir = experiment_name, enemy = enemy, run_num = 0)

def run_winner():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    global env 
    env = Environment(
            experiment_name=experiment_name,
            enemies=[4],
            playermode="ai",
            enemymode="static",
            level=2,
            speed="normal",
            visuals=True,
            sound='off',
    )

    with open(os.path.join(experiment_name, 'winner-0.pkl'), 'rb') as input_file:
        genome = pickle.load(input_file)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        player = player_controller(net)

        env.player_controller = player
        print(env.play()[0])

if __name__ == '__main__':
    # main()
    run_winner()