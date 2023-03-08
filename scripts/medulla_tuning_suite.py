# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import os
from pathlib import Path
import numpy as np
from two_photon_analysis import medulla_analysis as ma

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
# astar1_fly4_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "proximal_multiple"]]
# astar1_fly4_alt_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "proximal_multiple"]]
# astar1_fly4_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "proximal_multiple"]]
# astar1_fly4_pre_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "medial_1_multiple"]]
# astar1_fly4_alt_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "medial_1_multiple"]]
# astar1_fly4_post_medi1 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "medial_1_multiple"]]
# astar1_fly4_pre_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "medial_2_multiple"]]
# astar1_fly4_alt_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "medial_2_multiple"]]
# astar1_fly4_post_medi2 = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "medial_2_multiple"]]
# astar1_fly4_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "9", "distal_multiple"]]
# astar1_fly4_alt_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "10", "distal_multiple"]]
# astar1_fly4_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20220527.common_moco", "2022-05-27", "11", "distal_multiple"]]
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

astar1_alt_prox_1_2 = np.concatenate((astar1_fly1_alt_prox, astar1_fly2_alt_prox,), axis=0, )
astar1_alt_prox_3_5 = np.concatenate((astar1_fly5_alt_prox,), axis=0, )

astar1_alt_prox_all = np.concatenate(
                                      (astar1_fly1_alt_prox, astar1_fly2_alt_prox, astar1_fly3_alt_prox, astar1_fly5_alt_prox,), axis=0
                                    )

astar1_alt_medi1_all = np.concatenate(
                                      (astar1_fly1_alt_medi1, astar1_fly2_alt_medi1, astar1_fly3_alt_medi1, astar1_fly5_alt_medi1,), axis=0
                                    )

astar1_alt_medi2_all = np.concatenate(
                                      (astar1_fly1_alt_medi2, astar1_fly2_alt_medi2, astar1_fly3_alt_medi2, astar1_fly5_alt_medi2,), axis=0
                                    )

astar1_alt_dist_all = np.concatenate(
                                      (astar1_fly1_alt_dist, astar1_fly2_alt_dist, astar1_fly3_alt_dist, astar1_fly5_alt_dist,), axis=0
                                    )

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

# %% Pull the trial's data based on the parameter key
def get_spatiotemporal_responses(trials):
  # if opto stim
  parameter_keys = ('current_spatial_period', 'current_temporal_frequency')

  unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(trials, parameter_key=parameter_keys)
  # calc the sem + / -
  sem_plus = mean_response + sem_response
  sem_minus = mean_response - sem_response

  # Just current_spatial_period
  parameter_key_spatial = 'current_spatial_period'
  unique_parameter_values_spatial, mean_response_spatial, sem_response_spatial, trial_response_by_stimulus_spatial = ID.getTrialAverages(trials, parameter_key=parameter_key_spatial)
  # calc the sem + / -
  sem_plus_spatial = mean_response_spatial + sem_response_spatial
  sem_minus_spatial = mean_response_spatial - sem_response_spatial

  # Just current_temporal_temporal
  parameter_key_temporal = 'current_temporal_temporal'
  unique_parameter_values_temporal, mean_response_temporal, sem_response_temporal, trial_response_by_stimulus_temporal = ID.getTrialAverages(trials, parameter_key=parameter_key_temporal)
  # calc the sem + / -
  sem_plus_temporal = mean_response_temporal + sem_response_temporal
  sem_minus_temporal = mean_response_temporal - sem_response_temporal

  return unique_parameter_values, mean_response, sem_response, sem_plus, sem_minus, unique_parameter_values_spatial, mean_response_spatial, sem_response_spatial, sem_plus_spatial, sem_minus_spatial, unique_parameter_values_temporal, mean_response_temporal, sem_response_temporal, sem_plus_temporal, sem_minus_temporal

save_directory = "/Volumes/ABK2TBData/lab_repo/analysis/outputs/medulla_tuning_suite/" #+ experiment_file_name + "/"
Path(save_directory).mkdir(exist_ok=True)

#%% 
# Which one to plot
layer = astar1_alt_prox_3_5
background_subtraction = False
alt_pre_time = .7
savefig = True

print('\n\n\n')
print('======================================================================================')
print(f'FLY = {str(layer)}')
print(f'Background Subtraction = {str(background_subtraction)}')
print(f'Alt Pre Time = {str(alt_pre_time)}')
print('======================================================================================')

print('\n\n\n')
#%%  Loop through experiments/layers here:
# instantiate the big bois
flies_nopto_mean_response = []
flies_yopto_mean_response = []
flies_nopto_sem_response = []
flies_yopto_sem_response = []

for fly_ind in range(len(layer)):
  print(f'Loop starts!!! fly_ind = {fly_ind}')
  file_path = os.path.join(layer[fly_ind][0], layer[fly_ind][1] + ".hdf5")
  ID = imaging_data.ImagingDataObject(file_path, layer[fly_ind][2], quiet=True)
  roi_data = ID.getRoiResponses(layer[fly_ind][3], background_subtraction=background_subtraction, background_roi_name='bg_distal')
  # Testing opto vs no opto
  # first, get roi_data
  # getAltEpochResponseMatrix b/c opto comes on during typical pre-time
  test_time_vector, test_epoch_response = ma.getAltEpochResponseMatrix(ID, np.vstack(roi_data['roi_response']), alt_pre_time=alt_pre_time)
  # second, filter by opto
  yopto_query = {'opto_stim': True}
  nopto_query = {'opto_stim': False}
  yes_opto_trials = shared_analysis.filterTrials(test_epoch_response, ID, query=yopto_query)
  no_opto_trials = shared_analysis.filterTrials(test_epoch_response, ID, query=nopto_query)
  # run the function
  yopto_unique_parameter_values, yopto_mean_response, yopto_sem_response, yopto_sem_plus, yopto_sem_minus, yopto_unique_parameter_values_spatial, yopto_mean_response_spatial, yopto_sem_response_spatial, yopto_sem_plus_spatial, yopto_sem_minus_spatial, yopto_unique_parameter_values_temporal, yopto_mean_response_temporal, yopto_sem_response_temporal, yopto_sem_plus_temporal, yopto_sem_minus_temporal = get_spatiotemporal_responses(yes_opto_trials)
  nopto_unique_parameter_values, nopto_mean_response, nopto_sem_response, nopto_sem_plus, nopto_sem_minus, nopto_unique_parameter_values_spatial, nopto_mean_response_spatial, nopto_sem_response_spatial, nopto_sem_plus_spatial, nopto_sem_minus_spatial, nopto_unique_parameter_values_temporal, nopto_mean_response_temporal, nopto_sem_response_temporal, nopto_sem_plus_temporal, nopto_sem_minus_temporal = get_spatiotemporal_responses(no_opto_trials)

  print(f'nopto_mean_response shape = {nopto_mean_response.shape}')
  # add fly_ind into flies big boi
  if fly_ind == 0:
    print('made it here....')
    flies_nopto_mean_response = nopto_mean_response
    flies_yopto_mean_response = yopto_mean_response
    flies_nopto_sem_response = nopto_sem_response
    flies_yopto_sem_response = yopto_sem_response
    print(f'flies_nopto_mean_response shape = {flies_nopto_mean_response.shape}')

  else:
    flies_nopto_mean_response = np.append(flies_nopto_mean_response, nopto_mean_response, axis=0)
    flies_yopto_mean_response = np.append(flies_yopto_mean_response, yopto_mean_response, axis=0)
    flies_nopto_sem_response = np.append(flies_nopto_sem_response, nopto_sem_response, axis=0)
    flies_yopto_sem_response = np.append(flies_yopto_sem_response, yopto_sem_response, axis=0)
    print(f'flies_nopto_mean_response shape = {flies_nopto_mean_response.shape}')



#%%

flies_nopto_mean_response3 = flies_nopto_mean_response 
flies_yopto_mean_response3 = flies_yopto_mean_response
flies_nopto_sem_response3 = flies_nopto_sem_response
flies_yopto_sem_response3 = flies_yopto_sem_response



#%%
file_path = os.path.join(layer[0][0], layer[0][1] + ".hdf5")
ID = imaging_data.ImagingDataObject(file_path, layer[0][2], quiet=True)
roi_data = ID.getRoiResponses(layer[0][3], background_subtraction=background_subtraction, background_roi_name='bg_distal')
# Testing opto vs no opto
# first, get roi_data
#epoch_response = roi_data.get('epoch_response') 
# getAltEpochResponseMatrix b/c opto comes on during typical pre-time
test_time_vector, test_epoch_response = ma.getAltEpochResponseMatrix(ID, np.vstack(roi_data['roi_response']), alt_pre_time=alt_pre_time)

# second, filter by opto
yopto_query = {'opto_stim': True}
nopto_query = {'opto_stim': False}
yes_opto_trials = shared_analysis.filterTrials(test_epoch_response, ID, query=yopto_query)
no_opto_trials = shared_analysis.filterTrials(test_epoch_response, ID, query=nopto_query)

# run the function
yopto_unique_parameter_values, yopto_mean_response, yopto_sem_response, yopto_sem_plus, yopto_sem_minus, yopto_unique_parameter_values_spatial, yopto_mean_response_spatial, yopto_sem_response_spatial, yopto_sem_plus_spatial, yopto_sem_minus_spatial, yopto_unique_parameter_values_temporal, yopto_mean_response_temporal, yopto_sem_response_temporal, yopto_sem_plus_temporal, yopto_sem_minus_temporal = get_spatiotemporal_responses(yes_opto_trials)
nopto_unique_parameter_values, nopto_mean_response, nopto_sem_response, nopto_sem_plus, nopto_sem_minus, nopto_unique_parameter_values_spatial, nopto_mean_response_spatial, nopto_sem_response_spatial, nopto_sem_plus_spatial, nopto_sem_minus_spatial, nopto_unique_parameter_values_temporal, nopto_mean_response_temporal, nopto_sem_response_temporal, nopto_sem_plus_temporal, nopto_sem_minus_temporal = get_spatiotemporal_responses(no_opto_trials)

# plotting the raw traces
# Plotting nopto on top of opto
# cmap = plt.get_cmap('cool') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
# colors = [cmap(i) for i in np.linspace(0.1, 1.0, len(yopto_unique_parameter_values))]

#%% Plotting the whole trace averaging across ROIs
fh, ax = plt.subplots(len(nopto_unique_parameter_values), 1, figsize=(16, 8*len(nopto_unique_parameter_values)))
for up_ind, up in enumerate(nopto_unique_parameter_values): # up = unique parameter
  ax[up_ind].plot(roi_data['time_vector'], yopto_mean_response[:, up_ind, :].mean(axis=0), color='red', alpha=0.9, label='opto'+str(up))
  ax[up_ind].plot(roi_data['time_vector'], nopto_mean_response[:, up_ind, :].mean(axis=0), color='black', alpha=0.9, label='no opto'+str(up))

  ax[up_ind].fill_between(roi_data['time_vector'], yopto_sem_plus[:, up_ind, :].mean(axis=0), 
                  yopto_sem_minus[:, up_ind, :].mean(axis=0),
                  color='red', alpha=0.1)
  ax[up_ind].fill_between(roi_data['time_vector'], nopto_sem_plus[:, up_ind, :].mean(axis=0), 
              nopto_sem_minus[:, up_ind, :].mean(axis=0),
              color='black', alpha=0.1)
  # Legend, Grid, Axis
  ax[up_ind].legend(loc="upper right", fontsize=15)
  ax[up_ind].grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
  #x_locator = FixedLocator(list(range(-1, 20)))
  #ax.xaxis.set_major_locator(x_locator)
  ax[up_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
  ax[up_ind].grid(axis="y", color="k", alpha=.1, linewidth=.5)
  ax[up_ind].set_xlabel('Time in Seconds')
  ax[up_ind].set_ylabel('DF/F')
  ax[up_ind].set_title(f'spatio-temporal: {up}')
fh.suptitle(f'Traces (ROI avg) for {layer[0][1]}, Series {layer[0][2]} | ROI={layer[0][3]} | AltPreTime={alt_pre_time} | BgSub={background_subtraction}', fontsize=13)

if savefig == True:
    fh.savefig(
    save_directory
    + "AvgTraces."
    + str(layer[0][1])
    + ".Series"
    + str(layer[0][2])
    + ".ROI"
    + str(layer[0][3])
    + ".AltPreTime"
    + str(alt_pre_time)
    + ".BgSub"
    + str(background_subtraction)
    + ".pdf",
    dpi=300,
    )

# Plotting the whole trace, each ROI
fh, ax = plt.subplots(len(nopto_unique_parameter_values), 1, figsize=(16, 8*len(nopto_unique_parameter_values)))
for up_ind, up in enumerate(nopto_unique_parameter_values): # up = unique parameter
  for roi_ind in range(len(nopto_mean_response)):
    ax[up_ind].plot(roi_data['time_vector'], yopto_mean_response[roi_ind, up_ind, :], color='red', alpha=0.9, label='opto'+str(up))
    ax[up_ind].plot(roi_data['time_vector'], nopto_mean_response[roi_ind, up_ind, :], color='black', alpha=0.9, label='no opto'+str(up))

    ax[up_ind].fill_between(roi_data['time_vector'], yopto_sem_plus[roi_ind, up_ind, :], 
                    yopto_sem_minus[roi_ind, up_ind, :],
                    color='red', alpha=0.1)
    ax[up_ind].fill_between(roi_data['time_vector'], nopto_sem_plus[roi_ind, up_ind, :], 
                nopto_sem_minus[roi_ind, up_ind, :],
                color='black', alpha=0.1)
    # Legend, Grid, Axis
    #ax[up_ind].legend(loc="upper right", fontsize=15)
    ax[up_ind].grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
    #x_locator = FixedLocator(list(range(-1, 20)))
    #ax.xaxis.set_major_locator(x_locator)
    ax[up_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
    ax[up_ind].grid(axis="y", color="k", alpha=.1, linewidth=.5)
    ax[up_ind].set_xlabel('Time in Seconds')
    ax[up_ind].set_ylabel('DF/F')
    ax[up_ind].set_title(f'spatio-temporal: {up}')
fh.suptitle(f'Traces x ROI for {layer[0][1]}, Series {layer[0][2]} | ROI={layer[0][3]} | AltPreTime={alt_pre_time} | BgSub={background_subtraction}', fontsize=13)

if savefig == True:
    fh.savefig(
    save_directory
    + "TracePerROI."
    + str(layer[0][1])
    + ".Series"
    + str(layer[0][2])
    + ".ROI"
    + str(layer[0][3])
    + ".AltPreTime"
    + str(alt_pre_time)
    + ".BgSub"
    + str(background_subtraction)
    + ".pdf",
    dpi=300,
    )


# Collect metrics for mean, max, min for inside stim presentation window
vis_start = ID.getRunParameters('pre_time')
vis_length = ID.getRunParameters('stim_time')
getting_going_time = 0.5
early_cutoff_time = 0.1
window_length = vis_length - getting_going_time - early_cutoff_time
window_time = vis_start + getting_going_time
window_frames = int(np.ceil(window_length / ID.getResponseTiming().get('sample_period')))
nopto_windows = np.zeros((len(nopto_unique_parameter_values), window_frames))
nopto_windows_sem = np.zeros((len(nopto_unique_parameter_values), window_frames))
yopto_windows = np.zeros((len(nopto_unique_parameter_values), window_frames))
yopto_windows_sem = np.zeros((len(nopto_unique_parameter_values), window_frames))

fh, ax = plt.subplots(len(nopto_unique_parameter_values), 1, figsize=(18, 4*len(nopto_unique_parameter_values)))

# Collect windowed responses and plot them
cmap = plt.get_cmap('viridis') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0)]
for up_ind, up in enumerate(nopto_unique_parameter_values):
  start_index = np.where(roi_data.get('time_vector') > window_time)[0][0]
  nopto_windows[up_ind, :] = nopto_mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
  nopto_windows_sem[up_ind, :] = nopto_sem_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
  yopto_windows[up_ind, :] = yopto_mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
  yopto_windows_sem[up_ind, :] = yopto_sem_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)

  # Plot: Each Window for a given LED Intensity
  ax[up_ind].plot(nopto_windows[up_ind, :], label='no opto')
  ax[up_ind].plot(yopto_windows[up_ind, :], label='yes opto')
  ax[up_ind].set_title('up={}'.format(up))
  ax[up_ind].legend()
fh.suptitle(f'Window Traces for {layer[0][1]}, Series {layer[0][2]} | ROI={layer[0][3]} | AltPreTime={alt_pre_time} | BgSub={background_subtraction}', fontsize=13)

if savefig == True:
    fh.savefig(
    save_directory
    + "Window_Traces."
    + str(layer[0][1])
    + ".Series"
    + str(layer[0][2])
    + ".ROI"
    + str(layer[0][3])
    + ".AltPreTime"
    + str(alt_pre_time)
    + ".BgSub"
    + str(background_subtraction)
    + ".pdf",
    dpi=300,
    )

# find and then plot Mean, Max, Min Opto Vs No Opto for each layer
# nopto_windows.shape = unique_parameters x window_frames
# response_max_nopto = max for each unique_parameters
response_max_nopto = np.max(nopto_windows, axis=-1)
response_min_nopto = np.min(nopto_windows, axis=-1)
response_mean_nopto = np.mean(nopto_windows, axis=-1)
response_sem_mean_nopto = np.mean(nopto_windows_sem, axis=-1)
response_max_yopto = np.max(yopto_windows, axis=-1)
response_min_yopto = np.min(yopto_windows, axis=-1)
response_mean_yopto = np.mean(yopto_windows, axis=-1)
response_sem_mean_yopto = np.mean(yopto_windows_sem, axis=-1)
# mean - max to approximate peak-->trough distance
response_PtT_nopto = response_max_nopto - response_min_nopto
response_PtT_yopto = response_max_yopto - response_min_yopto
# now find the indecies of max and min to later pull the sem for those values
max_indecies_nopto = np.argmax(nopto_windows, axis=-1)
max_indecies_yopto = np.argmax(yopto_windows, axis=-1)
min_indecies_nopto = np.argmin(nopto_windows, axis=-1)
min_indecies_yopto = np.argmin(yopto_windows, axis=-1)

# pull the sem for the max and min values
#initialize:
sem_max_nopto = np.empty(len(nopto_windows))
sem_max_yopto = np.empty(len(yopto_windows))
sem_min_nopto = np.empty(len(nopto_windows))
sem_min_yopto = np.empty(len(yopto_windows))
for i in range(len(nopto_windows)):
  sem_max_nopto[i] = nopto_windows_sem[i][max_indecies_nopto[i]]
  sem_max_yopto[i] = yopto_windows_sem[i][max_indecies_yopto[i]]
  sem_min_nopto[i] = nopto_windows_sem[i][min_indecies_nopto[i]]
  sem_min_yopto[i] = yopto_windows_sem[i][min_indecies_yopto[i]]
# Calc the mean of max, min st error
sem_PtT_nopto = abs(sem_max_nopto-sem_min_nopto)/2
sem_PtT_yopto = abs(sem_max_yopto-sem_min_yopto)/2

# plotting those metrics
cmap = plt.get_cmap('tab20c') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(nopto_unique_parameter_values))]

fh, ax = plt.subplots(4, 1, figsize=(8, 24))
for up_ind, up in enumerate(nopto_unique_parameter_values):
  ax[0].scatter(response_max_nopto[up_ind], response_max_yopto[up_ind], color=colors[up_ind], label = up)
  ax[0].errorbar(response_max_nopto[up_ind], response_max_yopto[up_ind], xerr=sem_max_nopto[up_ind], yerr=sem_max_yopto[up_ind], elinewidth=4, alpha=0.2)
  ax[0].set_title('Maximum Response - No Opto v Opto')
  ax[0].set_xlabel('No Opto')
  ax[0].set_ylabel('Opto')

  ax[1].scatter(response_min_nopto[up_ind], response_min_yopto[up_ind], color=colors[up_ind])
  ax[1].errorbar(response_min_nopto[up_ind], response_min_yopto[up_ind], xerr=sem_min_nopto[up_ind], yerr=sem_min_yopto[up_ind], elinewidth=4, alpha=0.2)
  ax[1].set_title('Minimum Response - No Opto v Opto')
  ax[1].set_xlabel('No Opto')
  ax[1].set_ylabel('Opto')

  ax[2].scatter(response_mean_nopto[up_ind], response_mean_yopto[up_ind], color=colors[up_ind])
  ax[2].errorbar(response_mean_nopto[up_ind], response_mean_yopto[up_ind], xerr=response_sem_mean_nopto[up_ind], yerr=response_sem_mean_yopto[up_ind], elinewidth=4, alpha=0.2)
  ax[2].set_title('Mean Response - No Opto v Opto')
  ax[2].set_xlabel('No Opto')
  ax[2].set_ylabel('Opto')

  ax[3].scatter(response_PtT_nopto[up_ind], response_PtT_yopto[up_ind], color=colors[up_ind])
  ax[3].errorbar(response_PtT_nopto[up_ind], response_PtT_yopto[up_ind], xerr=sem_PtT_nopto[up_ind], yerr=sem_PtT_yopto[up_ind], elinewidth=4, alpha=0.2)
  ax[3].set_title('Max-Min Response - No Opto v Opto')
  ax[3].set_xlabel('No Opto')
  ax[3].set_ylabel('Opto')



# Finding unity lines by selecting the largest and smallest points on the plot
unity_lower_max = min(np.min(response_max_nopto), np.min(response_max_yopto))*0.9
unity_upper_max = max(np.max(response_max_nopto), np.max(response_max_yopto))*1.1
unity_lower_min = min(np.min(response_min_nopto), np.min(response_min_yopto))*0.9
unity_upper_min = max(np.max(response_min_nopto), np.max(response_min_yopto))*1.1
unity_lower_mean = min(np.min(response_mean_nopto), np.min(response_mean_yopto))*0.9
unity_upper_mean = max(np.max(response_mean_nopto), np.max(response_mean_yopto))*1.1
unity_lower_PtT = min(np.min(response_PtT_nopto), np.min(response_PtT_yopto))*0.9
unity_upper_PtT = max(np.max(response_PtT_nopto), np.max(response_PtT_yopto))*1.1
ax[0].plot([unity_lower_max, unity_upper_max], [unity_lower_max, unity_upper_max], 'k--', label='unity', alpha=0.5, linewidth=3)
ax[1].plot([unity_lower_min, unity_upper_min], [unity_lower_min, unity_upper_min], 'k--', alpha=0.5, linewidth=3)
ax[2].plot([unity_lower_mean, unity_upper_mean], [unity_lower_mean, unity_upper_mean], 'k--', alpha=0.5, linewidth=3)
ax[3].plot([unity_lower_PtT, unity_upper_PtT], [unity_lower_PtT, unity_upper_PtT], 'k--', alpha=0.5, linewidth=3)

fh.legend(loc='lower right')
fh.suptitle(f'Metrics for {layer[0][1]}, Series {layer[0][2]} | ROI={layer[0][3]} | AltPreTime={alt_pre_time} | BgSub={background_subtraction}', fontsize=13)

if savefig == True:
    fh.savefig(
    save_directory
    + "Metrics."
    + str(layer[0][1])
    + ".Series"
    + str(layer[0][2])
    + ".ROI"
    + str(layer[0][3])
    + ".AltPreTime"
    + str(alt_pre_time)
    + ".BgSub"
    + str(background_subtraction)
    + ".pdf",
    dpi=300,
    )




#=============================================================================================================================================
#=============================================================================================================================================
#=============================================================================================================================================
#=============================================================================================================================================
#=============================================================================================================================================
# %% 
#=============================================================================================================================================
#=============================================================================================================================================
#=============================================================================================================================================
#=============================================================================================================================================
# Collect across experiments function



def getMetricsFromExperiment(layer, alt_pre_time = 1, background_subtraction = False, background_roi_name = 'bg'):
  file_path = os.path.join(layer[0][0], layer[0][1] + ".hdf5")
  ID = imaging_data.ImagingDataObject(file_path, layer[0][2], quiet=True)
  roi_data = ID.getRoiResponses(layer[0][3], background_subtraction=background_subtraction, background_roi_name='bg_distal')
  # Testing opto vs no opto
  # first, get roi_data
  #epoch_response = roi_data.get('epoch_response') 
  # getAltEpochResponseMatrix b/c opto comes on during typical pre-time
  time_vector, epoch_response = ma.getAltEpochResponseMatrix(ID, np.vstack(roi_data['roi_response']), alt_pre_time=alt_pre_time)

  # second, filter by opto
  yopto_query = {'opto_stim': True}
  nopto_query = {'opto_stim': False}
  yes_opto_trials = shared_analysis.filterTrials(epoch_response, ID, query=yopto_query)
  no_opto_trials = shared_analysis.filterTrials(epoch_response, ID, query=nopto_query)

  # run the function
  yopto_unique_parameter_values, yopto_mean_response, yopto_sem_response, yopto_sem_plus, yopto_sem_minus, yopto_unique_parameter_values_spatial, yopto_mean_response_spatial, yopto_sem_response_spatial, yopto_sem_plus_spatial, yopto_sem_minus_spatial, yopto_unique_parameter_values_temporal, yopto_mean_response_temporal, yopto_sem_response_temporal, yopto_sem_plus_temporal, yopto_sem_minus_temporal = get_spatiotemporal_responses(yes_opto_trials)
  nopto_unique_parameter_values, nopto_mean_response, nopto_sem_response, nopto_sem_plus, nopto_sem_minus, nopto_unique_parameter_values_spatial, nopto_mean_response_spatial, nopto_sem_response_spatial, nopto_sem_plus_spatial, nopto_sem_minus_spatial, nopto_unique_parameter_values_temporal, nopto_mean_response_temporal, nopto_sem_response_temporal, nopto_sem_plus_temporal, nopto_sem_minus_temporal = get_spatiotemporal_responses(no_opto_trials)

  # Collect metrics for mean, max, min for inside stim presentation window
  vis_start = ID.getRunParameters('pre_time')
  vis_length = ID.getRunParameters('stim_time')
  getting_going_time = 0.5
  early_cutoff_time = 0.1
  window_length = vis_length - getting_going_time - early_cutoff_time
  window_time = vis_start + getting_going_time
  window_frames = int(np.ceil(window_length / ID.getResponseTiming().get('sample_period')))
  nopto_windows = np.zeros((len(nopto_unique_parameter_values), window_frames))
  nopto_windows_sem = np.zeros((len(nopto_unique_parameter_values), window_frames))
  yopto_windows = np.zeros((len(nopto_unique_parameter_values), window_frames))
  yopto_windows_sem = np.zeros((len(nopto_unique_parameter_values), window_frames))

  for up_ind, up in enumerate(nopto_unique_parameter_values):
    start_index = np.where(roi_data.get('time_vector') > window_time)[0][0]
    nopto_windows[up_ind, :] = nopto_mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
    nopto_windows_sem[up_ind, :] = nopto_sem_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
    yopto_windows[up_ind, :] = yopto_mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
    yopto_windows_sem[up_ind, :] = yopto_sem_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)

  # find and then plot Mean, Max, Min Opto Vs No Opto for each layer
  # nopto_windows.shape = unique_parameters x window_frames
  # response_max_nopto = max for each unique_parameters
  response_max_nopto = np.max(nopto_windows, axis=-1)
  response_min_nopto = np.min(nopto_windows, axis=-1)
  response_mean_nopto = np.mean(nopto_windows, axis=-1)
  response_sem_mean_nopto = np.mean(nopto_windows_sem, axis=-1)
  response_max_yopto = np.max(yopto_windows, axis=-1)
  response_min_yopto = np.min(yopto_windows, axis=-1)
  response_mean_yopto = np.mean(yopto_windows, axis=-1)
  response_sem_mean_yopto = np.mean(yopto_windows_sem, axis=-1)
  # mean - max to approximate peak-->trough distance
  response_PtT_nopto = response_max_nopto - response_min_nopto
  response_PtT_yopto = response_max_yopto - response_min_yopto
  # now find the indecies of max and min to later pull the sem for those values
  max_indecies_nopto = np.argmax(nopto_windows, axis=-1)
  max_indecies_yopto = np.argmax(yopto_windows, axis=-1)
  min_indecies_nopto = np.argmin(nopto_windows, axis=-1)
  min_indecies_yopto = np.argmin(yopto_windows, axis=-1)

  # pull the sem for the max and min values
  #initialize:
  sem_max_nopto = np.empty(len(nopto_windows))
  sem_max_yopto = np.empty(len(yopto_windows))
  sem_min_nopto = np.empty(len(nopto_windows))
  sem_min_yopto = np.empty(len(yopto_windows))
  for i in range(len(nopto_windows)):
    sem_max_nopto[i] = nopto_windows_sem[i][max_indecies_nopto[i]]
    sem_max_yopto[i] = yopto_windows_sem[i][max_indecies_yopto[i]]
    sem_min_nopto[i] = nopto_windows_sem[i][min_indecies_nopto[i]]
    sem_min_yopto[i] = yopto_windows_sem[i][min_indecies_yopto[i]]
  # Calc the mean of max, min st error
  sem_PtT_nopto = abs(sem_max_nopto-sem_min_nopto)/2
  sem_PtT_yopto = abs(sem_max_yopto-sem_min_yopto)/2

  return response_max_nopto, response_min_nopto, response_mean_nopto, response_sem_mean_nopto, response_PtT_nopto, sem_PtT_nopto,\
         response_max_yopto, response_min_yopto, response_mean_yopto, response_sem_mean_yopto, response_PtT_yopto, sem_PtT_yopto

# Plotting function
def plotMTSMetrics(unique_parameter_values, response_max_nopto, response_min_nopto, response_mean_nopto, response_PtT_nopto,\
                   response_max_yopto, response_min_yopto, response_mean_yopto, response_PtT_yopto, layer_name, save_fig=True
                  ):
  # plotting those metrics
  cmap = plt.get_cmap('tab20c') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
  colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(nopto_unique_parameter_values))]

  fh, ax = plt.subplots(4, 1, figsize=(8, 24))
  for up_ind, up in enumerate(nopto_unique_parameter_values):
    ax[0].scatter(response_max_nopto[up_ind], response_max_yopto[up_ind], color=colors[up_ind], label = up)
    #ax[0].errorbar(response_max_nopto[up_ind], response_max_yopto[up_ind], xerr=sem_max_nopto[up_ind], yerr=sem_max_yopto[up_ind], elinewidth=4, alpha=0.2)
    ax[0].set_title('Maximum Response - No Opto v Opto')
    ax[0].set_xlabel('No Opto')
    ax[0].set_ylabel('Opto')

    ax[1].scatter(response_min_nopto[up_ind], response_min_yopto[up_ind], color=colors[up_ind])
    #ax[1].errorbar(response_min_nopto[up_ind], response_min_yopto[up_ind], xerr=sem_min_nopto[up_ind], yerr=sem_min_yopto[up_ind], elinewidth=4, alpha=0.2)
    ax[1].set_title('Minimum Response - No Opto v Opto')
    ax[1].set_xlabel('No Opto')
    ax[1].set_ylabel('Opto')

    ax[2].scatter(response_mean_nopto[up_ind], response_mean_yopto[up_ind], color=colors[up_ind])
    #ax[2].errorbar(response_mean_nopto[up_ind], response_mean_yopto[up_ind], xerr=response_sem_mean_nopto[up_ind], yerr=response_sem_mean_yopto[up_ind], elinewidth=4, alpha=0.2)
    ax[2].set_title('Mean Response - No Opto v Opto')
    ax[2].set_xlabel('No Opto')
    ax[2].set_ylabel('Opto')

    ax[3].scatter(response_PtT_nopto[up_ind], response_PtT_yopto[up_ind], color=colors[up_ind])
    #ax[3].errorbar(response_PtT_nopto[up_ind], response_PtT_yopto[up_ind], xerr=sem_PtT_nopto[up_ind], yerr=sem_PtT_yopto[up_ind], elinewidth=4, alpha=0.2)
    ax[3].set_title('Max-Min Response - No Opto v Opto')
    ax[3].set_xlabel('No Opto')
    ax[3].set_ylabel('Opto')

  # Finding unity lines by selecting the largest and smallest points on the plot
  unity_lower_max = min(np.min(response_max_nopto), np.min(response_max_yopto))*0.9
  unity_upper_max = max(np.max(response_max_nopto), np.max(response_max_yopto))*1.1
  unity_lower_min = min(np.min(response_min_nopto), np.min(response_min_yopto))*0.9
  unity_upper_min = max(np.max(response_min_nopto), np.max(response_min_yopto))*1.1
  unity_lower_mean = min(np.min(response_mean_nopto), np.min(response_mean_yopto))*0.9
  unity_upper_mean = max(np.max(response_mean_nopto), np.max(response_mean_yopto))*1.1
  unity_lower_PtT = min(np.min(response_PtT_nopto), np.min(response_PtT_yopto))*0.9
  unity_upper_PtT = max(np.max(response_PtT_nopto), np.max(response_PtT_yopto))*1.1
  ax[0].plot([unity_lower_max, unity_upper_max], [unity_lower_max, unity_upper_max], 'k--', label='unity', alpha=0.5, linewidth=3)
  ax[1].plot([unity_lower_min, unity_upper_min], [unity_lower_min, unity_upper_min], 'k--', alpha=0.5, linewidth=3)
  ax[2].plot([unity_lower_mean, unity_upper_mean], [unity_lower_mean, unity_upper_mean], 'k--', alpha=0.5, linewidth=3)
  ax[3].plot([unity_lower_PtT, unity_upper_PtT], [unity_lower_PtT, unity_upper_PtT], 'k--', alpha=0.5, linewidth=3)

  fh.legend(loc='lower right')
  fh.suptitle(f'Metrics for {layer_name}, AltPreTime={alt_pre_time} | BgSub={background_subtraction}', fontsize=13)

  if save_fig == True:
      fh.savefig(
      save_directory
      + "Metrics."
      + str(layer_name)
      + ".AltPreTime"
      + str(alt_pre_time)
      + ".BgSub"
      + str(background_subtraction)
      + ".pdf",
      dpi=300,
      )
# %% single run
response_max_nopto, response_min_nopto, response_mean_nopto, response_sem_mean_nopto, response_PtT_nopto, sem_PtT_nopto,\
response_max_yopto, response_min_yopto, response_mean_yopto, response_sem_mean_yopto, response_PtT_yopto, sem_PtT_yopto \
= getMetricsFromExperiment(astar1_fly5_post_dist, alt_pre_time = 0.7)

# %% multi_run
layer = astar1_alt_dist_all
outer_response_max_nopto = []; outer_response_min_nopto = []; outer_response_mean_nopto = []; 
outer_response_sem_mean_nopto = []; outer_response_PtT_nopto = []; outer_sem_PtT_nopto = []
outer_response_max_yopto = []; outer_response_min_yopto = []; outer_response_mean_yopto = []
outer_response_sem_mean_yopto = []; outer_response_PtT_yopto = []; outer_sem_PtT_yopto = []

for layer_ind in range(len(layer)):
  response_max_nopto, response_min_nopto, response_mean_nopto, response_sem_mean_nopto, response_PtT_nopto, sem_PtT_nopto,\
  response_max_yopto, response_min_yopto, response_mean_yopto, response_sem_mean_yopto, response_PtT_yopto, sem_PtT_yopto = getMetricsFromExperiment(layer, alt_pre_time = 0.7)
  outer_response_max_nopto.append(response_max_nopto)
  outer_response_min_nopto.append(response_min_nopto)
  outer_response_mean_nopto.append(response_mean_nopto)
  outer_response_sem_mean_nopto.append(response_sem_mean_nopto)
  outer_response_PtT_nopto.append(response_PtT_nopto)
  outer_sem_PtT_nopto.append(sem_PtT_nopto)
  outer_response_max_yopto.append(response_max_yopto)
  outer_response_min_yopto.append(response_min_yopto)
  outer_response_mean_yopto.append(response_mean_yopto)
  outer_response_sem_mean_yopto.append(response_sem_mean_yopto)
  outer_response_PtT_yopto.append(response_PtT_yopto)
  outer_sem_PtT_yopto.append(sem_PtT_yopto)

# average those suckers
mean_response_max_nopto = np.mean(outer_response_max_nopto, axis = 0)
mean_response_min_nopto = np.mean(outer_response_min_nopto, axis = 0)
mean_response_mean_nopto = np.mean(outer_response_mean_nopto, axis = 0)
mean_response_sem_mean_nopto = np.mean(outer_response_sem_mean_nopto, axis = 0)
mean_response_PtT_nopto = np.mean(outer_response_PtT_nopto, axis = 0)
mean_sem_PtT_nopto = np.mean(outer_sem_PtT_nopto, axis = 0)

mean_response_max_yopto = np.mean(outer_response_max_yopto, axis = 0)
mean_response_min_yopto = np.mean(outer_response_min_yopto, axis = 0)
mean_response_mean_yopto = np.mean(outer_response_mean_yopto, axis = 0)
mean_response_sem_mean_yopto = np.mean(outer_response_sem_mean_yopto, axis = 0)
mean_response_PtT_yopto = np.mean(outer_response_PtT_yopto, axis = 0)
mean_sem_PtT_yopto = np.mean(outer_sem_PtT_yopto, axis = 0)

# %%
plotMTSMetrics(nopto_unique_parameter_values, mean_response_max_nopto, mean_response_min_nopto, mean_response_mean_nopto, mean_response_PtT_nopto,
               mean_response_max_yopto, mean_response_min_yopto, mean_response_mean_yopto, mean_response_PtT_yopto,
               layer_name = 'distal',
               save_fig = True
              )
# %%
