import pandas as pd
from lib.string_funcs import read_until_termination, string_arr_2_float_arr

def get_num_elements(directory_path):
    search_string = "$$STARTCOUNTERS"
    termination_char = "$"
    file_extension = ".GOM"

    # Read the lines from the .GOM file between search string and termination char
    file_lines = read_until_termination(directory_path, file_extension, search_string, termination_char)
    
    # Convert the string lines to floats
    float_arr = string_arr_2_float_arr(file_lines)

    num_elements = float_arr[0][0] 
    
    return num_elements

def get_num_MPs_per_element(directory_path):
    # Purpose: Gets the number of MPs per element
    # TODO: Assumes that all elements have the same number of MPs
    search_string = "$$START_NUMBER_OF_MATERIAL_POINTS"
    termination_char = "$"
    file_extension = ".GOM"

    # Read the lines from the .GOM file between search string and termination char
    file_lines = read_until_termination(directory_path, file_extension, search_string, termination_char)
    
    # Convert the string lines to floats
    float_arr = string_arr_2_float_arr(file_lines)

    num_MPs_per_element = float_arr[0][0] 

    return num_MPs_per_element

def get_num_solid_MPs(directory_path):
    # Purpose: Get the total number of MPs in the model
    search_string = "$$START_NUMBER_OF_MATERIAL_POINTS"
    termination_char = "$"
    file_extension = ".GOM"

    # Read the lines from the .GOM file between search string and termination char
    file_lines = read_until_termination(directory_path, file_extension, search_string, termination_char)
    
    # Convert the string lines to floats
    float_arr = string_arr_2_float_arr(file_lines)

    df = pd.DataFrame(float_arr)

    return df[0].sum()
