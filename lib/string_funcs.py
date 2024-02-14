import os
import pandas as pd
import numpy as np

def read_until_termination(directory, file_extension, search_string, termination_char):
    try:
        for file_name in os.listdir(directory):
            if file_name.endswith(file_extension):
                file_path = os.path.join(directory, file_name)
                with open(file_path, 'r') as file:
                    found = False
                    result = []
                    for line in file:
                        if found:
                            if termination_char in line:
                                return result
                            
                            result.append(line)

                            # found = False  # Reset the found flag after processing the next line
                        if search_string in line:
                            found = True
        print(f"No files with extension {file_extension} found in {directory}.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def process_string(input_string):
    input_string = input_string.replace('\n', '')

    # Split the string on spaces
    string_array = input_string.split()
    
    result = [float(number) for number in string_array]
    return result

def string_arr_2_float_arr(string_arr):
    # Purpose: Converts an array of strings that contain numbers into a nested list of floats 
    
    # Init array to hold floats
    float_arr = [None] * len(string_arr)

    # Convert each string and store the result as a nested list
    for i, string in enumerate(string_arr):
        float_arr[i] = process_string(string)

    return float_arr

def float_arr_2_df(float_arr):
    # Purpose: Convert float array to df
    # Assumes no column headers
    return pd.DataFrame(float_arr)

def closest_row_indices(df, column_to_compare, target_values, tol = 1e-2):
    # Purpose: Find the closest values in a df column to the target values
    closest_rows_indices = []
    for value in target_values:
        #  Find the difference between target and df column values
        diffence_arr = np.abs(df[column_to_compare].values[:, np.newaxis] - value)
        
        # Get the minimum difference
        min_value = np.min(diffence_arr)

        # Get the row corresponding to min value
        min_row = np.argmin(diffence_arr)
        if min_value > tol:
            print(f"The difference between target: {value} and the closest value {df[column_to_compare].iloc[min_row]} greater than tolerance ")
        else:
            closest_rows_indices.append(min_row)

    return closest_rows_indices