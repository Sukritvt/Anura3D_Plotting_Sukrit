import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

PI = np.pi

def calc_hydraulic_conductivity(intrinsic_permeability, unit_wt_water, dynamic_viscosity):
    # Purpose: Calc hydraulic conductivity [m/s]: https://en.wikipedia.org/wiki/Hydraulic_conductivity
    # intrinsic_permeability [m^2]
    # unit_wt_water [N/m^3]: Unit weight of water
    # dynamic_viscosity [Pa-s]: dynamic viscosity of water
    return intrinsic_permeability *  unit_wt_water/dynamic_viscosity

def calc_coeff_consolidation(hydraulic_conductivity, coeff_volume_change, unit_wt_water):
    # Purpose: Calculate the coefficient of Consolidation, C_v
    # hydraulic_conductivity [m/s] of the soil
    # Coefficient of volume change (m_v) (Inverse of the constrained modulus)
    return hydraulic_conductivity/(unit_wt_water * coeff_volume_change)

def calc_terzhagi_pore_pressure(applied_load, z, H, time_factor, num_iterations):
    # Calculate u_exceess at a given spatial and temporal location
    # Copied from Lambe and Whitman Pg. 409 who got it from Taylor (1948)
    # qu: Applied load (also initial excess pore pressure)
    # z: Elevation
    # H: Inital model height
    # time_factor (T_v)
    temp = 0
    for i in range(num_iterations):
        M = PI * (2 * i + 1)/2 
        # temp+= 1/(2 * i + 1) * np.sin((2 * i + 1) * PI * z/(2 * H)) * np.exp( -((2 * i + 1)**2 * PI**2)/4 * time_factor)
        temp+= 1/M * np.sin(M * (H-z)/(H)) * np.exp( -M**2 * time_factor)
    
    return 2 * applied_load * temp

def calc_normalized_pore_pressure(pore_pressure, applied_load):
    return pore_pressure/applied_load

def calc_time_factor(coeff_consolidation, t, height):
    # Purpose: Calculate the time factor (T_v)
    # coeff_consolidation (C_v)
    # t (time)
    return coeff_consolidation * t/(height **2)

def calc_constrained_modulus(E, nu):
    # Calc the constrained modulus (M, Eoed)
    # Formula from: https://en.wikipedia.org/wiki/Elastic_modulus
    # Also known as p-wave or oedometric modulus

    return E * (1-nu)/((1+nu) * (1-2*nu))

def calc_coeff_volume_change(E, nu):
    # Calc the coefficient of volume change (m_v) 
    # it's the inverse of the constrained modulus
    
    constrained_modulus = calc_constrained_modulus(E, nu)
    return 1/constrained_modulus

def calc_degree_consolidation(time_factor, tolerance):
    # Purpose: Calc the degree of consolidation (U) [-] U = 1-  \frac{u_excess}{u_0}
    # time_factor: non-dimensional time factor Tv
    # num_iterations: Number of summation iterations
    temp = 0.0
    relative_difference = tolerance + 1
    i = 0
    while relative_difference > tolerance:
        # Store the last iteration
        init_temp =  temp 

        # Calc the iteration
        m = (2 * i + 1)
        temp += 1/m**2 * np.exp(-(m * PI/2)**2 * time_factor)

        # Calc relative difference between iterations
        relative_difference  = np.abs(init_temp - temp)

        i+=1

    # Calc degree  of consolidation 
    return 1- 8/PI**2 * temp

def time_factor_difference(guess_time_factor, degree_consolidation, consolidation_tolerance):
    degree_consolidation_calc = calc_degree_consolidation(guess_time_factor,  consolidation_tolerance)
    
    # Find the difference between the guess and the desired value
    return degree_consolidation - degree_consolidation_calc

def find_time_factor(degree_consolidation, consolidation_tolerance):
    # Given a degree of consolidation find the corresponding Tv value
    # As consolidation has sharp gradients  in the solution fsolve struggles for degrees of consolidation less than 30%
    # For values less than 30% a brute force search is done
    # There's probably a way to get fsolve to work but brute force works for now

    # Degrees of consolidation below this number don't work well with fsolve
    U_fsolve_crit  = 30e-2

    # Check if the degree of consolidation
    if degree_consolidation  >= U_fsolve_crit:
        guess = 0.1
        # print("Using fsovle")
        Tv = fsolve(time_factor_difference, x0 = guess, args = (degree_consolidation, consolidation_tolerance), xtol = 1e-15)

    elif 3e-2 < degree_consolidation  < U_fsolve_crit:
        # Need to brute force the solution
        array_size = 100

        # Tv corresponding to a  degree of consolidation of 35%
        Tv_35 = 0.0962
        Tv_array = np.linspace(0, Tv_35, array_size)
        U = np.zeros(len(Tv_array))
        j= 0
        while j < 10:
        # Get the U values for the assumed Tv
            for i,t in enumerate(Tv_array):
                U[i] = calc_degree_consolidation(t,  1e-10)
            # Find the closest value to the desired  U
            # min_difference = np.min(np.abs(U-degree_consolidation))
            # print(min_difference)
            
            # Get the arg corresponding to  the min value
            min_arg  = np.argmin(np.abs(U-degree_consolidation))
            
            # Store the closest  Tv value
            Tv = Tv_array[min_arg]

            # reset the Tv array
            if min_arg == 0:
                Tv_array = np.linspace(Tv_array[min_arg], Tv_array[min_arg+1], array_size)
            elif min_arg == array_size-1:
                Tv_array = np.linspace(Tv_array[min_arg-1], Tv_array[min_arg], array_size)
            else:
                Tv_array = np.linspace(Tv_array[min_arg-1], Tv_array[min_arg+1], array_size)
            
            j+=1
    else:
        Tv = 0
    return Tv

def calc_Terzaghi_settlement(coeff_volume_change, applied_load, init_soil_height, time_factor, tolerance):
    #Settlement(t) = m_{v} * applied_load * H * U
    # where U: Degree of consolidation

    temp = 0.0
    relative_difference = tolerance+1
    i = 0

    while relative_difference > tolerance:
        # Store the last iteration
        init_temp =  temp 

        # Calc the iteration
        m = (2 * i + 1)
        temp += 1/m**2 * np.exp(-(m * PI/2)**2 * time_factor)

        # Calc relative difference between iterations
        relative_difference  = np.abs(init_temp - temp)

        i+=1

    # Calc degree  of consolidation 
    return coeff_volume_change * applied_load * init_soil_height * (1- 8/PI**2 * temp)

def plot_settlement(model_data, coeff_volume_change, applied_load, init_soil_height, ax, plot_theoretical = False):
    # Plot the theoretical and MP settlements
    #TODO: Add capacility to only plot selected Mp settlements
    # Really I should make a function for plotting the theoretical settlement and one for the MPs

    if plot_theoretical:
        # Get the time factor data
        theo_time_factor = model_data[0]["Time_Factor"]

        theo_settlement = np.zeros(len(theo_time_factor))

        # Calc the theoretical settlement 
        for i, Tv in enumerate(theo_time_factor):
            theo_settlement[i] = calc_Terzaghi_settlement(coeff_volume_change, applied_load, init_soil_height, Tv, tolerance = 1e-10)

        # Plot the settlement for all the material points in the model_data
        ax.plot(theo_time_factor, theo_settlement,  label = "Theoretical")

    for mp_data in model_data:
        # Select the MP time factor data    
        time_factor = mp_data["Time_Factor"]

        # Calc the mp settlement
        mp_settlement = -1 * (mp_data["Y"] - mp_data["Y"][0])

        # Plot the settlement of that MP
        ax.plot(time_factor, mp_settlement)