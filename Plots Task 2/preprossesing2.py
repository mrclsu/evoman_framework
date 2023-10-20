import os
import pandas as pd

# Define a function to group CSVs and calculate the required statistics
def group_and_process_csvs(prefix):
    # Get all CSV filenames in the current directory with the given prefix
    filenames = [f for f in os.listdir() if f.startswith(prefix) and f.endswith('.csv')]
    
    # List to store DataFrames
    dataframes = []
    
    # Read each CSV and append to dataframes list
    for filename in filenames:
        df = pd.read_csv(filename)
        dataframes.append(df)
    
    # Concatenate all the DataFrames
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    # Calculate average and std for mean_fitness, max_fitness, and time
    grouped = combined_df.groupby(combined_df.index)
    final_df = pd.DataFrame({
        'mean_fitness': grouped['mean_fitness'].mean(),
        'std_mean_fitness': grouped['mean_fitness'].std(),
        'max_fitness': grouped['max_fitness'].mean(),
        'std_max_fitness': grouped['max_fitness'].std(),
        'mean_time': grouped['time'].mean(),
        'std_time': grouped['time'].std()
    })
    
    return final_df

# Group and process CSVs based on prefixes
final_bonus_time = group_and_process_csvs('bonus_time_')
final_no_time = group_and_process_csvs('no_time_')
final_def = group_and_process_csvs('def_')

# Save the final processed DataFrames to new CSVs
final_bonus_time.to_csv('final_bonus_time_2.csv')
final_no_time.to_csv('final_no_time_2.csv')
final_def.to_csv('final_def_2.csv')
