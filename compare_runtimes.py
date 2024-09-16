import pandas as pd
import matplotlib.pyplot as plt

def compare_runtimes():
    # Read the data from the CSV files
    ga_runtimes = pd.read_csv('ga_stats/V22.1_Test_Runtimes.csv')
    neat_runtimes = pd.read_csv('neat-controller/stats/run-times.csv')

    # Convert times from ns to minutes
    ga_runtimes['time'] = ga_runtimes['time'] / 60000000000
    neat_runtimes['time'] = neat_runtimes['time'] / 60000000000

    # List of enemies
    enemies = [1, 4, 6]

    # Plot the runtimes for each enemy
    for enemy in enemies:
        plt.figure(figsize=(10, 6))

        # Filter data for the current enemy
        ga_data = ga_runtimes[ga_runtimes['enemy'] == enemy]
        neat_data = neat_runtimes[neat_runtimes['enemy'] == enemy]

        # Plot the data
        plt.plot(ga_data['run'], ga_data['time'], label='GA Runtimes', marker='o')
        plt.plot(neat_data['run'], neat_data['time'], label='NEAT Runtimes', marker='x')

        # Set plot labels and title
        plt.xlabel('Run Number')  # Corrected label here
        plt.ylabel('Runtime (in minutes)')
        plt.title(f'Runtime Comparison for Enemy {enemy}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"plots/runtimes_{enemy}.png", dpi=300)
        # Show the plot
        plt.show()

# Call the function
compare_runtimes()
