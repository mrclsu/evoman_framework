from deap import base, creator, tools, algorithms
from evoman.environment import Environment
import random
import numpy as np
import time
import os
from demo_controller import player_controller
import csv
import pickle
import argparse

experiment_name = 'deap_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Parameters
n_hidden_neurons = 10
mu_size = 100  # Number of initial individuals
lambda_size = 100  # Number of children to produce at each generation
crossover_prob = 0.7
mutation_prob = 0.2
gen_count = 2500
dom_u = 1
dom_l = -1


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[3, 5, 6, 8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  multiplemode="yes",
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(np.array(x))
    # print(f)
    return p - e, e

def simulation2(env, x):
    f, p, e, t = env.play(np.array(x))
    # print(f)
    return f, p - e, e

def evaluate(individual):
    return simulation(env, individual)

creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, dom_l, dom_u)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("mean", np.mean)
stats.register("max", np.max)
stats.register("std", np.std)

def train(pop = toolbox.population(n=mu_size)): 
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    with open(f'{experiment_name}/hof1_500.pkl', 'rb') as f:
        hof_old = pickle.load(f)


    env.enemies = [2, 3, 4, 5, 6, 8]
    hof = tools.HallOfFame(100)
    _, logbook1 = algorithms.eaMuCommaLambda(list(hof_old), toolbox, mu_size, lambda_size, crossover_prob, mutation_prob, gen_count,
                                stats=stats, halloffame=hof, verbose=True)

    # env.enemies = [2, 3, 4]
    # hof2 = tools.HallOfFame(100)
    # _, logbook2 = algorithms.eaMuCommaLambda(list(hof), toolbox, mu_size, lambda_size, crossover_prob, mutation_prob, gen_count,
    #                             stats=stats, halloffame=hof2, verbose=True)


    # env.enemies = [2, 3, 4, 5, 6, 8]
    # hof10 = tools.HallOfFame(10)
    # _, logbook2 = algorithms.eaMuCommaLambda(list(hof), toolbox, mu_size, lambda_size, crossover_prob, mutation_prob, gen_count,
    #                             stats=stats, halloffame=hof10, verbose=True)

    with open(f'{experiment_name}/hof1.pkl', 'wb') as f:
        pickle.dump(hof, f)
    
    # with open(f'{experiment_name}/hof2.pkl', 'wb') as f:
    #     pickle.dump(hof2, f)

    # with open(f'{experiment_name}/hof10.pkl', 'wb') as f:
    #     pickle.dump(hof10, f)

    with open(f"{experiment_name}/statistics_run1.csv", "w", newline="") as csvfile:
        fieldnames = ["gen", "mean", "max", "std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in logbook1:
            writer.writerow({
                'gen': entry['gen'],
                'mean': entry['mean'],
                'max': entry['max'],
                'std': entry['std']
            })

    # with open(f"{experiment_name}/statistics_run2.csv", "w", newline="") as csvfile:
    #     fieldnames = ["gen", "mean", "max", "std"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
    #     writer.writeheader()
    #     for entry in logbook2:
    #         writer.writerow({
    #             'gen': entry['gen'],
    #             'mean': entry['mean'],
    #             'max': entry['max'],
    #             'std': entry['std']
    #         })

    # with open(f"{experiment_name}/statistics_run3.csv", "w", newline="") as csvfile:
    #     fieldnames = ["gen", "mean", "max", "std"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
    #     writer.writeheader()
    #     for entry in logbook2:
    #         writer.writerow({
    #             'gen': entry['gen'],
    #             'mean': entry['mean'],
    #             'max': entry['max'],
    #             'std': entry['std']
    #         })

def run_pop():
    # Load pickled population
    with open(f'{experiment_name}/hof1.pkl', 'rb') as f:
        hof = pickle.load(f)

        # print(hof)

        global env
        env = Environment(experiment_name=experiment_name,
                    enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    multiplemode="yes",
                    speed="fastest",
                    visuals=False)
        

        res = map(lambda p: (toolbox.evaluate(p), p), hof)
        fit = map(lambda r: r[0], res)
        for f in fit:
            print(f)

        
    np.savetxt('winner.txt', hof[np.argmax(fit)])
    winner = hof[np.argmax(fit)]

    # winner = [-30.09642510576107, 27.86585526984371, -1.830557761346726, 69.70582177456663, 3.361997748162752, -29.954992229461766, 14.377500107784565, 3.636944970854176, -27.455578298534892, 14.814160194222621, -11.914590266273342, 57.08671870740764, -9.205832256172629, 0.2625055768399509, 51.281753234298634, 3.2055206504465743, 5.0441031547712, -31.163002860032837, 59.404965659585415, -1.1835042681679273, -33.67643238239931, -18.61454920538473, 10.534973600840406, 1.3749917188084781, -18.6375717834189, 51.33050579623604, 210.60446277250406, -7.54977180086016, -107.72541907990293, -28.915853961842, -152.2846046458004, -64.8281264799534, 77.52533120346803, 59.27341185776926, -59.77238101803789, -146.71695412325718, 13.719368764304784, -1.8337658020327958, -33.939533437820465, -11.81744927506879, 24.118373172823645, 16.407315821610826, 30.198227473402895, -460.1949309012352, -14.567593692262637, 10.513933697269723, 75.46309148910643, -13.364665769312186, -111.20870061377587, 21.633278019100235, -10.59526992511412, 66.6033708544714, -11.159616049768204, 22.248676588815897, -39.57261750858755, -27.534872149334145, -28.067549228975803, 29.503481599560786, -38.294985809156984, 11.809559513419169, 22.71041159029405, -14.674388570383162, 43.457801190084986, -28.74833792246362, -33.072714927114106, -5.554356378629278, 892.2881279855576, 16.488992809974434, -33.466182648279116, -27.215388535407627, -38.21791808058269, -4.013391531097522, 26.566396630470884, -140.44210690468867, 23.297553659406354, 36.46481594455112, -46.305927574545365, -114.38544727970474, -480.37806695819665, -8.052355316333873, 2.550104780351896, -68.45889835583161, -71.2698366125874, -179.73393055259515, -16.486347969473776, -7.667193197126988, -8.871332151873137, -235.04983057971523, -46.245105775829664, -30.028978999881723, -14.1204907447883, 36.2137860958038, -31.405902148900793, -36.36980517555686, 1.5144919087561008, 3.4236938441650886, -12.972638647152923, 32.259107551895646, -41.77346376106016, -6.203382445489895, 37.632373785466214, 12.520578946850971, -69.35604269072421, -465.993843179181, -36.294225152467135, 20.01812791985656, 35.94808593043357, -27.772516210245847, 123.29001295614583, -2.547248616195641, -15.78172516267632, -63.21685650148555, -100.52747490741207, -13.695781900928987, 9.236578147173967, 23.876553198583984, 138.52609737373834, 16.061328366529118, -220.29320318386198, -17.98547106435923, -43.687953885606404, 20.231858125339027, -46.816144028181796, -32.58654493245373, -44.75450464076556, -6.3904742378693165, -8.91487057131051, 0.032581652168153244, 38.304606732250264, -21.592490205089653, -38.62592416546724, -144.74426696968297, -14.340783657643573, -270.279380092466, -23.892580921460983, 36.122425182037475, 15.51117335517064, -15.474424015141572, 71.27374993059553, -4.483072000759457, -24.845835311181283, -58.76030355661322, 10.416977199259716, -1.3079493115756535, -12.347592602201074, -68.12885289921192, -40.12592459620268, 27.663687144660617, -12.370184112372156, 4.885174791596834, 20.882419552715113, -49.95788794842892, -64.08483466024715, 39.022842031442934, -8.891588113267009, 15.273931063107003, 3.387576820317653, 24.938119004091647, -46.87894282821106, 3.237479534445435, 81.7837297584559, -25.627017042137922, -50.51493275070911, -209.3657482694733, -6.525654903241087, -19.54718456815404, 34.58556222365544, 2.2530244021975196, 74.63796481004803, -1.9581534272782743, 44.69345139362273, -39.771059908794655, 46.08917392301679, -195.86711114703388, 27.213624401405504, -48.47555898633788, -37.32415935909978, -24.741175657339554, -100.19680378434732, 19.77860013840936, -76.8224633560419, -29.91715856296319, 18.481528176930134, 124.50915078802689, 19.916143130339243, -12.017716820100159, 49.21562719687999, -28.79921190881518, 6.394551176920556, -3.399925161737524, -24.728814488879674, -35.315728489742725, -47.2475558515837, -304.5884190597865, 8.908547954908256, 33.70791049053926, 65.75819167974244, -27.20712718579309, 155.699844212021, 11.166826740846217, 48.52444423209748, 17.544159353911763, 31.639839901330426, -118.41276729909282, 10.425164425269955, -48.97160888021036, 40.920462109761445, -17.40214474063063, 43.92708973128557, 15.05726468989266, 1.6124322991075875, 8.425120100629055, 71.11164448422332, 31.875632831373895, 1.1490920596550593, 117.37716030288773, -120.39391792510374, -113.21901654641825, 368.93106460210765, -52.189034922446766, -12.5176282137101, 14.082911843520414, -71.99234300952892, -197.68341869380288, 28.979756983162275, 91.40009982980061, 40.780859090116174, 17.604523921280137, 99.84480539117858, 37.34501706320479, 1.8563302592865871, -82.19059250310457, 69.95542307455624, 76.68644421676636, 79.34426226841174, 17.200510948216355, -78.60683513771426, 58.68304428988866, -73.60712950084981, -20.549338162407413, -57.59221726665069, -28.797857132864696, -24.61612780026739, -29.2412000301023, 22.697718996790428, -14.916225920068136, -3.6254678638799893, 83.15136543937732, 220.61208158721553, 142.3961081573848, 42.340550769325006, -107.39024454422956, 121.62257439800061, -46.84502799899285, -67.07522312384606, 42.629540101398135, -59.49776569778237, -34.14602706645002, -46.70253719562453, -33.5525618113146, -21.17544591354448, -77.02243262499685, 32.367145651847565, 81.03385013617549, -27.57663268744468]

        # Run individual with highest fitness
    env.visuals = True
    env.speed = 'normal'
    env.multiplemode = 'no'

    for i in range(1, 9):
        env.enemies = [i]
        fitness = simulation2(env, winner)
        print(f'Enemy: {i}, Fitness: {fitness[0]}, Gain: {fitness[1]}')
    fitness = simulation2(env, hof[np.argmax(fit)]) 
    # print(hof[np.argmax(fit)])
    # print(f'Fitness: {fitness}')


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DEAP generalist algorithm')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--run', action='store_true', help='Run population')

    args = parser.parse_args()

    if args.train:
        train()
    elif args.run:
        run_pop()

if __name__ == "__main__":
    main()