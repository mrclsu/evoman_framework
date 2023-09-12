import neat
from evoman.controller import Controller
from evoman.environment import Environment
import numpy as np
import os

experiment_name = 'neat-controller'
visuals = True

def sigmoid_activation(x):
    return 1./(1.+np.exp(-x))

class player_controller(Controller):
    def __init__(self, net):
        self.net = net  # NEAT network
    
    def control(self, inputs, _):
        # Normalizes the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        print(inputs)

        output = self.net.activate(inputs)

        return [1 if o > 0.5 else 0 for o in output]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        player = player_controller(net)

        env = Environment(
            experiment_name=experiment_name,
            enemies=[1],
            playermode="ai",
            player_controller=player,
            enemymode="static",
            level=1,
            speed="fastest",
            fullscreen=False,
            visuals=visuals,
        ) 

        genome.fitness = env.play()[0]


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

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)
    print(winner)
    os.chdir(experiment_name)
    stats.save()

if __name__ == '__main__':
    main()