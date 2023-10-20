import pandas as pd
import os

# Initialize empty lists to store dataframes for each group
bonus_time_dfs = []
no_time_dfs = []
def_dfs = []

# Loop through each file in the current directory
for file in os.listdir():
    if file.startswith('bonus_time_'):
        bonus_time_dfs.append(pd.read_csv(file))
    elif file.startswith('no_time_'):
        no_time_dfs.append(pd.read_csv(file))
    elif file.startswith('def_'):
        def_dfs.append(pd.read_csv(file))

# Function to process the list of dataframes and generate the final dataframe
def process_dataframes(dfs):
    # Concatenate all dataframes in the list
    df_concat = pd.concat(dfs, axis=1)
    
    # Calculate the average of 'mean_fitness' and 'max_fitness'
    mean_avg = df_concat['mean_fitness'].mean(axis=1)
    max_avg = df_concat['max_fitness'].mean(axis=1)
    
    # Calculate the standard deviation for 'mean_fitness' and 'max_fitness'
    std_mean_fitness = df_concat['mean_fitness'].std(axis=1)
    std_max_fitness = df_concat['max_fitness'].std(axis=1)
    
    # Construct the final dataframe
    final_df = pd.DataFrame({
        'mean_fitness': mean_avg,
        'std_mean_fitness': std_mean_fitness,
        'max_fitness': max_avg,
        'std_max_fitness': std_max_fitness
    })
    
    return final_df

# Process data and save to CSV
final_bonus_time = process_dataframes(bonus_time_dfs)
final_bonus_time.to_csv('final_bonus_time.csv', index=False)

final_no_time = process_dataframes(no_time_dfs)
final_no_time.to_csv('final_no_time.csv', index=False)

final_def = process_dataframes(def_dfs)
final_def.to_csv('final_def.csv', index=False)
