
import sys
sys.path.append(r'C:\Users\ordix\OneDrive\Desktop\Evoman Project')
from IslandModel import IslandModel
from Population import Population
from Individual import Individual
from evoman.environment import Environment


enemies = [1, 2, 3, 4, 5, 6]  # assuming these are the enemy levels
envs = [Environment(experiment_name=experiment_name,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    speed="normal",
                    enemymode="static",
                    level=enemy,
                    visuals=True) for enemy in enemies]

if __name__ == "__main__":
    NUM_ISLANDS = 4
    ISLAND_SIZE = 100
    MIGRATION_INTERVAL = 25  # Adjust this to match your desired migration interval
    ENEMY_LEVELS = [1, 2, 3, 4, 5, 6]
    NUM_EPOCHS = 3
    NUM_GENERATIONS = 4

