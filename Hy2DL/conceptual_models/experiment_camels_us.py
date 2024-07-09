# ## Code to calibrate a process-based hydrological model

# **Description**
# 
# The following notebook contains the code to calibrate a process-based rainfall-runoff model. The calibration is done 
# using the library [Spotpy](https://doi.org/10.1371/journal.pone.0145180)[1]. When multiple basins are being 
# calibrated, the code runs the calibration in parallel (one basin per core) to speed up the computation.
# 
# **Authors:**
# - Eduardo Acuna Espinoza (eduardo.espinoza@kit.edu)
# 
# **References:**
# [1]: "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made Python 
# Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"

#Import necessary packages
import sys
import os
import pandas as pd
import numpy as np
import random
import spotpy
import time

os.chdir(sys.path[0])
sys.path.append('..')

#from spotpy.objectivefunctions import rmse as obj_func # objective function used during optimization
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as obj_func
from hydrological_models import HBV as hydrological_model
from optimization_methods import dream as optimization_method

from modelcalibration_camelsus import ModelCalibrationCamelsUS as model_calibration
from modelcalibration_camelsus import check_basins_camelsus as check_basins

from modelcalibration import calibrate_basins
from evaluation_metrics import nse_loss

# -----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__': 
    # Initialize information
    path_entities= "../../data/basin_id/basins_camels_us_531.txt"
    path_data = "../../data/CAMELS_US"
    forcing = "daymet"
    path_additional_features = "../../data/CAMELS_US/pet_hargreaves.pickle"
    input_variables = ["prcp(mm/day)", "pet(mm/day)", "tmax(C)", "tmin(C)"]
    target_variables = ['QObs(mm/d)']
    #training_period = ["1980-10-01","1997-12-31"]
    #testing_period = ["1997-01-01","2008-12-31"]
    training_period = "../../data/CAMELS_US/train_split_file_new.p"
    testing_period = "../../data/CAMELS_US/test_split_file_new.p"
    batch_size = 75
    warmup_period = 365
    random_seed = 42

    # Save results
    path_output = '../results/HBV_extrapolation/'

    # use when one select the best parameters, depends on the loss function one wants. 
    maximize = True # True for gaussian_likelihood, False of rmse

    # set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Read information
    basins_id= np.loadtxt(path_entities, dtype='str').tolist()
    selected_basins_id = check_basins(path_data=path_data,basins_id=basins_id, training_period= training_period, 
                                      testing_period= testing_period)
    
    # Check if the path where one will store the results exists. In case it does not, it creates such path.
    if not os.path.exists(path_output):
        # Create the folder
        os.makedirs(path_output)
        print(f"Folder '{path_output}' created successfully.")
    else:
        print(f"Folder '{path_output}' already exists.")
    
    
    # Process the basins in batches (avoid memory issues)
    dfs = []
    batches = [selected_basins_id[i:i + batch_size] for i in range(0, len(selected_basins_id), batch_size)]

    start_time = time.time()
    for basin_batch in batches:
        training_object = {}
        testing_object = {}

        for i, basin in enumerate(basin_batch):
            training_object[basin] = model_calibration(model = hydrological_model(), 
                                                       path_data = path_data,
                                                       forcing=forcing,
                                                       basin_id = basin,
                                                       input_variables = input_variables , 
                                                       target_variables = target_variables,
                                                       time_period = training_period, 
                                                       obj_func = obj_func, 
                                                       warmup_period = warmup_period,
                                                       path_additional_features=path_additional_features)

            testing_object[basin] = model_calibration(model = hydrological_model(), 
                                                      path_data = path_data,
                                                      forcing=forcing,
                                                      basin_id = basin,
                                                      input_variables = input_variables , 
                                                      target_variables = target_variables,
                                                      time_period = testing_period, 
                                                      obj_func = obj_func, 
                                                      warmup_period = warmup_period,
                                                      path_additional_features=path_additional_features)


        # Run the calibration of the different basins in parallel ---------------------------------------------------------
        optimizer = optimization_method(random_state =random_seed)
        calibrate_basins(training_object=training_object, 
                         optimization_method=optimizer, 
                         basins=basin_batch, 
                         path_output=path_output,
                         random_seed=random_seed)
        

        # Process and summarize the results ------------------------------------------------------------------------------
        hyd_model = hydrological_model()
        optimizer = optimization_method()
        df_calibration = pd.DataFrame(index=range(len(basin_batch)), 
                                      columns=["basin_id", "NSE_training"] + list(hyd_model.parameter_ranges) + ["NSE_testing"])
        
        for i, basin in enumerate(basin_batch):
            # extract calibrated parameters
            file_name = path_output + hyd_model.name + '_' + optimizer.name + '_' + str(basin)
            results = spotpy.analyser.load_csv_results(file_name)
            calibrated_param = spotpy.analyser.get_best_parameterset(results, maximize=maximize)[0]

            # Training period ------------------------------------------
            q_sim = training_object[basin].simulation(calibrated_param)
            q_obs = training_object[basin].evaluation()

            # Calculate loss
            evaluation = q_obs[warmup_period:][training_object[basin].data_split[warmup_period:]]
            simulation = q_sim[warmup_period:][training_object[basin].data_split[warmup_period:]]
            mask_nans = ~np.isnan(evaluation)
            NSE_training = nse_loss(evaluation=evaluation[mask_nans].flatten(),
                                    simulation=simulation[mask_nans].flatten())

            # Testing period ------------------------------------------
            q_sim = testing_object[basin].simulation(calibrated_param)
            q_obs = testing_object[basin].evaluation()

            # Calculate loss
            evaluation = q_obs[warmup_period:][testing_object[basin].data_split[warmup_period:]]
            simulation = q_sim[warmup_period:][testing_object[basin].data_split[warmup_period:]]
            mask_nans = ~np.isnan(evaluation)
            NSE_testing = nse_loss(evaluation=evaluation[mask_nans].flatten(),
                                    simulation=simulation[mask_nans].flatten())
            
            # Save the result of the basin
            df_calibration.loc[i] = [basin, NSE_training] + list(calibrated_param) + [NSE_testing]


        #Dataframe of the batch
        dfs.append(df_calibration)

    # Save the results
    combined_df = pd.concat(dfs)
    combined_df.to_csv(path_output+hyd_model.name+"_"+optimizer.name+"_summary.csv", index=False)
    # Calculate and print the calibration time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total calibration time: {int(elapsed_time)} seconds")