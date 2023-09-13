import sys, os
import numpy as np
import pandas as pd
from evoman.environment import Environment
from demo_controller import player_controller

def simulate(env,unit):
    fit, _, _, _ = env.play(pcont=unit)
    return fit

experiment_name = 'controller_specialist_demo'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

bests2 = pd.read_csv("Reseed_Boss_1.csv", delimiter=",")
performance = []

for j in range(4):

    scores = []

    env = Environment(experiment_name=experiment_name,
				  playermode="ai",
                  enemies=[1],
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)
    
    for i in range(5):    

        score = simulate(env,np.array(bests2.iloc[j]))
        scores.append(score)
    performance.append(np.mean(np.array(scores)))

for i in range(len(performance)):

    print(f"Mutation_Degree {(i+1)*0.1} average score {performance[i]}")
