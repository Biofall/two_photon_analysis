# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import os
import numpy as np
print('test')
# %%
# Opto intensity sweep w/ flash experiments (with MoCo!) 2/8/22

# Multiple ROIs
# Fly 1
astar1_fly1_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "2", "proximal_multiple"]]
astar1_fly1_alt_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "4", "proximal_multiple"]]
astar1_fly1_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "5", "proximal_multiple"]]
astar1_fly1_pre_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "2", "medial_1_multiple"]]
astar1_fly1_alt_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "4", "medial_1_multiple"]]
astar1_fly1_post_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "5", "medial_1_multiple"]]
astar1_fly1_pre_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "2", "medial_2_multiple"]]
astar1_fly1_alt_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "4", "medial_2_multiple"]]
astar1_fly1_post_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "5", "medial_2_multiple"]]
astar1_fly1_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "2", "distal_multiple"]]
astar1_fly1_alt_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "4", "distal_multiple"]]
astar1_fly1_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "5", "distal_multiple"]]
# Fly 2
astar1_fly2_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "8", "proximal_multiple"]]
astar1_fly2_alt_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "9", "proximal_multiple"]]
astar1_fly2_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "10", "proximal_multiple"]]
astar1_fly2_pre_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "8", "medial_1_multiple"]]
astar1_fly2_alt_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "9", "medial_1_multiple"]]
astar1_fly2_post_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "10", "medial_1_multiple"]]
astar1_fly2_pre_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "8", "medial_2_multiple"]]
astar1_fly2_alt_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "9", "medial_2_multiple"]]
astar1_fly2_post_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "10", "medial_2_multiple"]]
astar1_fly2_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "8", "distal_multiple"]]
astar1_fly2_alt_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "9", "distal_multiple"]]
astar1_fly2_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220526.common_moco", "2022-05-26", "10", "distal_multiple"]]
# Fly 3
astar1_fly3_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "1", "proximal_multiple"]]
astar1_fly3_alt_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "2", "proximal_multiple"]]
astar1_fly3_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "3", "proximal_multiple"]]
astar1_fly3_pre_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "1", "medial_1_multiple"]]
astar1_fly3_alt_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "2", "medial_1_multiple"]]
astar1_fly3_post_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "3", "medial_1_multiple"]]
astar1_fly3_pre_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "1", "medial_2_multiple"]]
astar1_fly3_alt_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "2", "medial_2_multiple"]]
astar1_fly3_post_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "3", "medial_2_multiple"]]
astar1_fly3_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "1", "distal_multiple"]]
astar1_fly3_alt_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "2", "distal_multiple"]]
astar1_fly3_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "3", "distal_multiple"]]
# Fly 4
astar1_fly4_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "proximal_multiple"]]
astar1_fly4_alt_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "proximal_multiple"]]
astar1_fly4_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "proximal_multiple"]]
astar1_fly4_pre_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "medial_1_multiple"]]
astar1_fly4_alt_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "medial_1_multiple"]]
astar1_fly4_post_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "medial_1_multiple"]]
astar1_fly4_pre_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "medial_2_multiple"]]
astar1_fly4_alt_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "medial_2_multiple"]]
astar1_fly4_post_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "medial_2_multiple"]]
astar1_fly4_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "distal_multiple"]]
astar1_fly4_alt_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "distal_multiple"]]
astar1_fly4_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "distal_multiple"]]
# Fly 5
astar1_fly5_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "14", "proximal_multiple"]]
astar1_fly5_alt_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "16", "proximal_multiple"]]
astar1_fly5_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "17", "proximal_multiple"]]
astar1_fly5_pre_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "14", "medial_1_multiple"]]
astar1_fly5_alt_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "16", "medial_1_multiple"]]
astar1_fly5_post_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "17", "medial_1_multiple"]]
astar1_fly5_pre_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "14", "medial_2_multiple"]]
astar1_fly5_alt_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "16", "medial_2_multiple"]]
astar1_fly5_post_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "17", "medial_2_multiple"]]
astar1_fly5_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "14", "distal_multiple"]]
astar1_fly5_alt_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "16", "distal_multiple"]]
astar1_fly5_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "17", "distal_multiple"]]


#Concatenate those suckers



# mi1_prox_all = np.concatenate(
#                        (mi1_fly1_prox, mi1_fly2_prox,), 
#                         axis = 0,
#                       )
# mi1_medi_all = np.concatenate(
#                        (mi1_fly1_medi, mi1_fly2_medi,), 
#                         axis = 0,
#                       )
# mi1_dist_all = np.concatenate(
#                        (mi1_fly1_dist, mi1_fly2_dist,), 
#                         axis = 0,
#                       )
# mi1_all_multiple = np.concatenate(
#                                   (mi1_fly1_prox, mi1_fly2_prox, mi1_fly1_medi, mi1_fly2_medi, mi1_fly1_dist, mi1_fly2_dist,),
#                                    axis = 0,
#                                  )

#%% 
# Which one to plot
layer = astar1_fly3_pre_prox 

file_path = os.path.join(layer[0][0], layer[0][1] + ".hdf5")
ID = imaging_data.ImagingDataObject(file_path, layer[0][2], quiet=True)
roi_data = ID.getRoiResponses(layer[0][3])

# %% Pull the trial's data based on the parameter key

# For both
parameter_keys = ('current_spatial_period', 'current_temporal_frequency')

unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(np.expand_dims(roi_data.get('epoch_response')[0][:][:], axis=0), parameter_key=parameter_keys)
# calc the sem + / -
sem_plus = mean_response + sem_response
sem_minus = mean_response - sem_response

# Just current_spatial_period
parameter_key_spatial = 'current_spatial_period'
unique_parameter_values_spatial, mean_response_spatial, sem_response_spatial, trial_response_by_stimulus_spatial = ID.getTrialAverages(np.expand_dims(roi_data.get('epoch_response')[0][:][:], axis=0), parameter_key=parameter_key_spatial)
# calc the sem + / -
sem_plus_spatial = mean_response_spatial + sem_response_spatial
sem_minus_spatial = mean_response_spatial - sem_response_spatial

# Just current_temporal_temporal
parameter_key_temporal = 'current_temporal_temporal'
unique_parameter_value_temporal, mean_response_temporal, sem_response_temporal, trial_response_by_stimulus_temporal = ID.getTrialAverages(np.expand_dims(roi_data.get('epoch_response')[0][:][:], axis=0), parameter_key=parameter_key_temporal)
# calc the sem + / -
sem_plus_temporal = mean_response_temporal + sem_response_temporal
sem_minus_temporal = mean_response_temporal - sem_response_temporal
# %%
