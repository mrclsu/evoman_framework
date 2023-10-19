import pandas as pd

with open('deap_specialist/350_gens/no_time_winners.csv', 'r') as f:
    # Read the string data into a pandas DataFrame
    df = pd.read_csv(f)

    # Sort the DataFrame based on the 'fitness' column in descending order
    df_sorted = df.sort_values(by='fitness', ascending=False)

    # Extract the top 10 rows
    top_10 = df_sorted.head(10)
    
    # Save the top 10 rows to a CSV file
    top_10.to_csv('no_time_top_10.csv', index=False)