#TODO: Add the capability to read the model parameters from the GOM file and compare the results to the analytical solution automatically

# Read in the model parameters from the GOM file
# water density
# $$DENSITY_LIQUID 
# Permeability
# young_modulus
# $$YOUNG_MODULUS
# poisson ratio
# $$POISSON_RATIO
# gravity - only needed for the unit weight of water


# Read in the .Par files for each directory

# GOM file reads
## Number of elements

## Number of MPs

# need to select MP at the top of the model so that the model height can be known (right?)



settlement  = np.zeros(len(results_list[0].DATA[0]["Time_Factor"]))

for i, tv in enumerate(results_list[0].DATA[0]["Time_Factor"]):
    # Calc the settlement
    settlement[i] = calc_Terzaghi_settlement(coeff_volume_change, applied_load, init_soil_height, tv, tolerance = 1e-10) 

plt.plot(results_list[0].DATA[0]["Time_Factor"], settlement)

MP_id = 0
plt.plot(results_list[0].DATA[MP_id]["Time_Factor"], -1* (results_list[0].DATA[MP_id]["Y"] - results_list[0].DATA[MP_id]["Y"][0]))
plt.gca().invert_yaxis()

plt.ylabel("Settlement [m]")
plt.xlabel("Time Factor  ($T_{v}$)")
plt.show()