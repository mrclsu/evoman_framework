import pandas as pd
import matplotlib.pyplot as plt
import os

folder_path = "ga_stats"
enemy_numbers = [1, 4, 6]
types = ["Means", "Peaks", "St_Dev"]
runs = range(10)
num_runs = 10

def plot_ga_stats():
    # Create a dictionary to store data for each enemy and type
    data = {enemy: {t: [] for t in types} for enemy in enemy_numbers}

    # Read the CSV files and store the data
    for enemy in enemy_numbers:
        for t in types:
            for run in runs:
                file_name = f"V22.1_Test_{enemy}_{run}_{t}.csv"
                file_path = os.path.join(folder_path, file_name)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, header=None)
                    series_data = df.iloc[:, 0]  # Convert the single column to a Series
                    data[enemy][t].append(series_data)

    # Plot the data
    for enemy in enemy_numbers:
        plt.figure(figsize=(10, 6))
        
        # Calculate the average and std for Means and Peaks
        means_avg = pd.concat(data[enemy]["Means"], axis=1).mean(axis=1)
        means_std = pd.concat(data[enemy]["Means"], axis=1).std(axis=1)
        
        peaks_avg = pd.concat(data[enemy]["Peaks"], axis=1).mean(axis=1)
        peaks_std = pd.concat(data[enemy]["Peaks"], axis=1).std(axis=1)

        # Plot Means
        plt.plot(means_avg, label="Average Fitness")
        plt.fill_between(means_avg.index, means_avg - means_std, means_avg + means_std, alpha=0.2)

        # Plot Peaks
        plt.plot(peaks_avg, label="Max Fitness")
        plt.fill_between(peaks_avg.index, peaks_avg - peaks_std, peaks_avg + peaks_std, alpha=0.2)

        plt.title(f"GA | Enemy {enemy}")
        plt.xlabel("Generations")
        plt.ylabel("Fitness Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"plots/ga_{enemy}_plot.png", dpi=300)
        plt.show()

# def box_plot():
#     num_tests = 5
#     results = {enemy: [] for enemy in enemies}

#     # Run the best solution for each enemy, for each of the 10 runs, 5 times each
#     for enemy in enemies:
#         for run in range(num_runs):
#             gains = []
#             for test in range(num_tests):
#                 gain = run_solution(enemy, "best_solution") 
#                 gains.append(gain)
#             mean_gain = sum(gains) / num_tests
#             results[enemy].append(mean_gain)

#     # Plot the results using box plots
#     plt.figure(figsize=(12, 6))
#     data_to_plot = [results[enemy] for enemy in enemies]
#     plt.boxplot(data_to_plot)
#     plt.xticks(range(1, len(enemies) + 1), [f"Enemy {enemy}" for enemy in enemies])
#     plt.ylabel("Individual Gain")
#     plt.title("Comparison of Algorithms by Enemy")
#     plt.grid(True, axis='y')
#     plt.tight_layout()
#     plt.show()

plot_ga_stats()
#box_plot()