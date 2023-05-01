# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import os
from pathlib import Path
import numpy as np
from two_photon_analysis import medulla_analysis as ma
from scipy import interpolate
from matplotlib.pyplot import cm
import itertools


# Loading and concatenating all the data
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
# Something is wrong with Fly 4
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


# Concatenate those suckers

astar1_alt_prox_1_2 = np.concatenate((astar1_fly1_alt_prox, astar1_fly2_alt_prox,), axis=0, )
astar1_alt_prox_3_5 = np.concatenate((astar1_fly5_alt_prox,), axis=0, )

astar1_alt_prox_all = np.concatenate(
                                      #(astar1_fly1_alt_prox, astar1_fly2_alt_prox, astar1_fly3_alt_prox, astar1_fly4_alt_prox, astar1_fly5_alt_prox,), axis=0
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

# Set directories and function to pull trial data
# Important directory establishing
save_directory = "/Volumes/ABK2TBData/lab_repo/analysis/outputs/medulla_tuning_suite/" #+ experiment_file_name + "/"
Path(save_directory).mkdir(exist_ok=True)

#  Interpolation function
def interpolate_to_common_trial_length(ID, original_values):
  # Get run parameters (in s)
  pre_time = ID.getRunParameters("pre_time")
  stim_time = ID.getRunParameters("stim_time")
  tail_time = ID.getRunParameters("tail_time")
  interp_len = 44 # hardcoded. also could be: int(trial_len * 10)

  interp_value = np.zeros((original_values.shape[0], original_values.shape[1], interp_len))
  # len of trial in frames
  frame_count = len(original_values[0, 0, :])
  old_time_vector = np.linspace(0, frame_count-1, num=frame_count)
  new_time_vector = np.linspace(0, frame_count-1, num=interp_len)

  # Loop through ROI and unique parameter values
  for roi_ind in range(len(original_values)):
    for up in range(original_values.shape[1]):
      # Interpolate the sample period into time vector
      # interpolate yopto_mean_response[roi_data, up, :] into interp_len
      interp_value[roi_ind, up, :] = np.interp(new_time_vector, old_time_vector, original_values[roi_ind, up, :])

  return interp_value, new_time_vector



#%% Pulling Traces separated by opto and unique parameter values
layer = astar1_alt_prox_all  #astar1_alt_prox_all, astar1_fly5_alt_prox
background_subtraction = False
alt_pre_time = 0 # this now is backwards from the vis stim time
dff = False
display_fix = True
layer_name = 'astar1_alt_prox_all'

print('\n\n\n')
print('======================================================================================')
print(f'FLY = {str(layer)}')
print('======================================================================================')
print('\n\n\n')
#Loop through experiments/layers here:
# instantiate the big bois
flies_nopto_mean_response = []
flies_yopto_mean_response = []
flies_nopto_sem_response = []
flies_yopto_sem_response = []

spatial_periods = [10, 20, 40, 80]
temporal_frequencies = [0.5, 1, 2, 4]
optos = [0, 1]

for fly_ind in range(len(layer)):
  file_path = os.path.join(layer[fly_ind][0], layer[fly_ind][1] + ".hdf5")
  if display_fix == True:
    cfg_dict = {'timing_channel_ind': 1}
    ID = imaging_data.ImagingDataObject(file_path, layer[fly_ind][2], quiet=True, cfg_dict = cfg_dict)
  else:  
    ID = imaging_data.ImagingDataObject(file_path, layer[fly_ind][2], quiet=True)

  
  roi_data = ID.getRoiResponses(layer[fly_ind][3], background_subtraction=background_subtraction, background_roi_name='bg_distal')

  # first, get roi_data
  # getAltEpochResponseMatrix b/c opto comes on during typical pre-time
  time_vector, epoch_response = ma.getAltEpochResponseMatrix(ID, np.vstack(roi_data['roi_response']), dff=dff, alt_pre_time=alt_pre_time)
  
  parameter_keys = ('opto_stim', 'current_spatial_period', 'current_temporal_frequency')
  unique_parameter_values, mean_response, sem_response, _ = ID.getTrialAverages(epoch_response,
                                                                                parameter_key=parameter_keys)

  interp_mean, interp_time = interpolate_to_common_trial_length(ID, mean_response)
  interp_sem, _ = interpolate_to_common_trial_length(ID, sem_response)

  unique_parameter_values = np.array(unique_parameter_values)

  reordered_mean = np.zeros((interp_mean.shape[0], len(optos), len(spatial_periods), len(temporal_frequencies), interp_mean.shape[-1]))
  reordered_mean[:] = np.nan
  reordered_sem = np.zeros((interp_sem.shape[0], len(optos), len(spatial_periods), len(temporal_frequencies), interp_sem.shape[-1]))
  reordered_sem[:] = np.nan

  # Put mean responses into reordered_mean based on opto, spatial, and temporal
  for opto_ind, opto in enumerate(optos):
    for spatial_ind, spatial in enumerate(spatial_periods):
      for temporal_ind, temporal in enumerate(temporal_frequencies):
        tmp_ind = np.intersect1d(np.where(opto == unique_parameter_values[:, 0]),
                                  np.where(spatial == unique_parameter_values[:, 1]))
        pull_ind = np.intersect1d(np.where(temporal == unique_parameter_values[:, 2]),
                                  tmp_ind)
        if len(pull_ind) == 0:
          pass # skip
        elif len(pull_ind) == 1:
          reordered_mean[:, opto_ind, spatial_ind, temporal_ind, :] = interp_mean[:, pull_ind[0], :]
          reordered_sem[:, opto_ind, spatial_ind, temporal_ind, :] = interp_sem[:, pull_ind[0], :]
        else:
          print('This should never happen')

  if np.any(np.isnan(reordered_mean)):
    print('Nans found - missing param combo')

  # add fly_ind into flies big boi
  if fly_ind == 0:
    flies_nopto_mean_response = reordered_mean[:, 0, :, :, :]
    flies_yopto_mean_response = reordered_mean[:, 1, :, :, :]
    flies_nopto_sem_response = reordered_sem[:, 0, :, :, :]
    flies_yopto_sem_response = reordered_sem[:, 1, :, :, :]


  else:
    flies_nopto_mean_response = np.append(flies_nopto_mean_response, reordered_mean[:, 0, :, :, :], axis=0)
    flies_yopto_mean_response = np.append(flies_yopto_mean_response, reordered_mean[:, 1, :, :, :], axis=0)
    flies_nopto_sem_response = np.append(flies_nopto_sem_response, reordered_sem[:, 0, :, :, :], axis=0)
    flies_yopto_sem_response = np.append(flies_yopto_sem_response, reordered_sem[:, 1, :, :, :], axis=0)

# calculate the sem plus and minus
flies_nopto_sem_plus = flies_nopto_mean_response + flies_nopto_sem_response
flies_nopto_sem_minus = flies_nopto_mean_response - flies_nopto_sem_response
flies_yopto_sem_plus = flies_yopto_mean_response + flies_yopto_sem_response
flies_yopto_sem_minus = flies_yopto_mean_response - flies_yopto_sem_response


# Now, working in the average space. Must recalculate sem here because the mean is across ROIs
# calculate mean and sem across ROIs
across_roi_mean_nopto = np.nanmean(flies_nopto_mean_response, axis=0)
across_roi_mean_yopto = np.nanmean(flies_yopto_mean_response, axis=0)
# calculate the standard error of the mean across ROIs
across_roi_sem_nopto = np.nanstd(flies_nopto_mean_response, axis=0) / np.sqrt(flies_nopto_mean_response.shape[0])
across_roi_sem_yopto = np.nanstd(flies_yopto_mean_response, axis=0) / np.sqrt(flies_yopto_mean_response.shape[0])
# calculate the mean plus sem and mean minus sem
across_roi_mean_plus_sem_nopto = across_roi_mean_nopto + across_roi_sem_nopto
across_roi_mean_minus_sem_nopto = across_roi_mean_nopto - across_roi_sem_nopto
across_roi_mean_plus_sem_yopto = across_roi_mean_yopto + across_roi_sem_yopto
across_roi_mean_minus_sem_yopto = across_roi_mean_yopto - across_roi_sem_yopto

# Find unique parameter values with opto column removed and np.unique to remove duplications
optoless_unique_parameter_values = np.unique(np.delete((unique_parameter_values), 0, axis=1), axis=0)



# %%  MTS.1 Plot the whole trace averaging across ROIs
savefig = True
darkmode = True

if darkmode == True:
  # Set the plots to a dark grid background
  with plt.style.context('dark_background'):
    # define the plot color
    c = [193/255, 70/255, 255/255]

    fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(32, 24))
    for sp_ind, spatial in enumerate(spatial_periods):
      for tf_ind, temporal in enumerate(temporal_frequencies):

        ax[sp_ind, tf_ind].plot(interp_time, across_roi_mean_nopto[sp_ind, tf_ind, :], color='w', alpha=0.9, label='no opto')
        ax[sp_ind, tf_ind].plot(interp_time, across_roi_mean_yopto[sp_ind, tf_ind, :], color=c, alpha=0.9, label='opto')
        ax[sp_ind, tf_ind].fill_between(interp_time, across_roi_mean_plus_sem_nopto[sp_ind, tf_ind, :], across_roi_mean_minus_sem_nopto[sp_ind, tf_ind, :],
                                        color='w', alpha=0.15) 
        ax[sp_ind, tf_ind].fill_between(interp_time, across_roi_mean_plus_sem_yopto[sp_ind, tf_ind, :], across_roi_mean_minus_sem_yopto[sp_ind, tf_ind, :],
                                          color=c, alpha=0.3)
        
        # Legend, Grid, Axis
        #ax[sp_ind, tf_ind].legend()
        ax[sp_ind, tf_ind].grid(axis="x", color="w", alpha=.1, linewidth=1, linestyle=":")
        #x_locator = FixedLocator(list(range(-1, 20)))
        #ax.xaxis.set_major_locator(x_locator)
        ax[sp_ind, tf_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
        ax[sp_ind, tf_ind].grid(axis="y", color="w", alpha=.1, linewidth=.5)
        ax[sp_ind, tf_ind].set_xlabel('Frames')
        #ax[sp_ind, tf_ind].set_ylabel('Response')
        ax[sp_ind, tf_ind].set_title(f'spatal: {spatial} | temporal: {temporal}')
    # fig suptitle
    fh.suptitle(f"Average Traces Across ROIs for {layer_name}", fontsize=16)
    fh.text(0.09, 0.5, 'Response', ha='center', va='center', rotation='vertical', fontsize=14)

    handles, labels = ax[0,0].get_legend_handles_labels()
    fh.legend(handles, labels, loc=2)

else:
  fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(8*len(spatial_periods), 8*len(temporal_frequencies)))
  for sp_ind, spatial in enumerate(spatial_periods):
    for tf_ind, temporal in enumerate(temporal_frequencies):

      ax[sp_ind, tf_ind].plot(interp_time, across_roi_mean_nopto[sp_ind, tf_ind, :], color='black', alpha=0.9, label='no opto')
      ax[sp_ind, tf_ind].plot(interp_time, across_roi_mean_yopto[sp_ind, tf_ind, :], color='red', alpha=0.9, label='opto')
      ax[sp_ind, tf_ind].fill_between(interp_time, across_roi_mean_plus_sem_nopto[sp_ind, tf_ind, :], across_roi_mean_minus_sem_nopto[sp_ind, tf_ind, :],
                                      color='black', alpha=0.1) 
      ax[sp_ind, tf_ind].fill_between(interp_time, across_roi_mean_plus_sem_yopto[sp_ind, tf_ind, :], across_roi_mean_minus_sem_yopto[sp_ind, tf_ind, :],
                                        color='red', alpha=0.1)
      
      # Legend, Grid, Axis
      ax[sp_ind, tf_ind].legend()
      ax[sp_ind, tf_ind].grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
      #x_locator = FixedLocator(list(range(-1, 20)))
      #ax.xaxis.set_major_locator(x_locator)
      ax[sp_ind, tf_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
      ax[sp_ind, tf_ind].grid(axis="y", color="k", alpha=.1, linewidth=.5)
      ax[sp_ind, tf_ind].set_xlabel('Time in Seconds')
      ax[sp_ind, tf_ind].set_ylabel('DF/F')
      ax[sp_ind, tf_ind].set_title(f'spatal: {spatial} | temporal: {temporal}')
  # fig suptitle
  fh.suptitle(f"Average Traces Across ROIs for {layer_name}", fontsize=20)


if savefig == True:
    fh.savefig(
    save_directory
    + "AvgTraces."
    + layer_name
    + ".DFF: "
    + str(dff)
    + "AltPreTime: "
    + str(alt_pre_time)
    + "Darkmode="
    + str(darkmode)
    + ".pdf",
    dpi=300, bbox_inches='tight', transparent=True,
    )

# %% For each unique parameter value, plot every ROI
error_bars = False

fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(8*len(spatial_periods), 8*len(temporal_frequencies)))
for sp_ind, spatial in enumerate(spatial_periods):
  for tf_ind, temporal in enumerate(temporal_frequencies):
    # Have to reset these inside the loop, otherwise the colors don't reset
    reds = iter(cm.autumn(np.linspace(0, 1, len(flies_yopto_mean_response))))
    blues = iter(cm.winter(np.linspace(0, 1, len(flies_yopto_mean_response))))


    for roi_ind in range(len(flies_yopto_mean_response)):

      # Loop through shades of red for opto
      red_color = next(reds)
      ax[sp_ind, tf_ind].plot(interp_time, flies_yopto_mean_response[roi_ind, sp_ind, tf_ind, :], color=red_color, alpha=0.6, label='ROI: '+str(roi_ind)+' | opto')
      # Loop through shades of blue for no opto
      blue_color = next(blues)
      ax[sp_ind, tf_ind].plot(interp_time, flies_nopto_mean_response[roi_ind, sp_ind, tf_ind, :], color=blue_color, alpha=0.6, label='ROI: '+str(roi_ind)+' | no opto')

      if error_bars == True:
        ax[sp_ind, tf_ind].fill_between(interp_time, flies_yopto_sem_plus[roi_ind, sp_ind, tf_ind, :],
                    flies_yopto_sem_minus[roi_ind, sp_ind, tf_ind, :],
                    color=red_color, alpha=0.1)
        ax[sp_ind, tf_ind].fill_between(interp_time, flies_nopto_sem_plus[roi_ind, sp_ind, tf_ind, :],
                    flies_nopto_sem_minus[roi_ind, sp_ind, tf_ind, :],
                    color=blue_color, alpha=0.1)
        
      # Legend, Grid, Axis
      ax[sp_ind, tf_ind].legend()
      ax[sp_ind, tf_ind].grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
      ax[sp_ind, tf_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
      ax[sp_ind, tf_ind].grid(axis="y", color="k", alpha=.1, linewidth=.5)
      ax[sp_ind, tf_ind].set_xlabel('Time in Seconds')
      ax[sp_ind, tf_ind].set_ylabel('Response')
      ax[sp_ind, tf_ind].set_title(f'spatal: {spatial} | temporal: {temporal}')
# title for entire figure. Each unique parameter value, plot every ROI
fh.suptitle('Traces for each unique parameter value, every ROI')
if savefig == True:
    fh.savefig(
    save_directory
    + "TracePerROI."
    + "DFF: "
    + str(dff)
    + ".AltPreTime: "
    + str(alt_pre_time)
    + ".pdf",
    dpi=300,
    )

# %% For each ROI, plot every unique parameter value
error_bars = False
# Create a figure with as many subplots as there are ROIs
fh, ax = plt.subplots(len(flies_yopto_mean_response), 1, figsize=(16, 8*len(flies_yopto_mean_response)))

for roi_ind in range(len(flies_yopto_mean_response)):
  # Have to reset these inside the loop, otherwise the colors don't reset
  colors1 = iter(cm.winter(np.linspace(0.2, 1, len(optoless_unique_parameter_values))))
  colors2 = iter(cm.autumn(np.linspace(0.2, 1, len(optoless_unique_parameter_values))))

  marker1 = itertools.cycle((',', 'v', 's', 'P', 'x', '+', '.', 'X', 'o', '*', '4', '>', 'p', '1', '<', '|', '_', 'H', '8')) 
  marker2 = itertools.cycle((',', 'v', 's', 'P', 'x', '+', '.', 'X', 'o', '*', '4', '>', 'p', '1', '<', '|', '_', 'H', '8')) 

  # loop through the spatial periods and then temporal frequencies
  for sp_ind, spatial in enumerate(spatial_periods):
    for tf_ind, temporal in enumerate(temporal_frequencies):

      # Loop through shades of red for opto
      color1_color = next(colors1)
      ax[roi_ind].plot(interp_time, flies_yopto_mean_response[roi_ind, sp_ind, tf_ind, :], marker=next(marker1), color=color1_color, alpha=0.8, label='spatio-temporal: '+str(up)+' | opto')
      # Loop through shades of blue for no opto
      color2_color = next(colors2)
      ax[roi_ind].plot(interp_time, flies_nopto_mean_response[roi_ind, sp_ind, tf_ind, :], marker=next(marker2), color=color2_color, alpha=0.8, label='spatio-temporal: '+str(up)+' | no opto')

      if error_bars == True:
        ax[roi_ind].fill_between(interp_time, flies_yopto_sem_plus[roi_ind, sp_ind, tf_ind, :],
                    flies_yopto_sem_minus[roi_ind, sp_ind, tf_ind, :],
                    color=color1_color, alpha=0.1)
        ax[roi_ind].fill_between(interp_time, flies_nopto_sem_plus[roi_ind, sp_ind, tf_ind, :],
                    flies_nopto_sem_minus[roi_ind, sp_ind, tf_ind, :],
                    color=color2_color, alpha=0.1)
        
      # Legend, Grid, Axis for each subplot
      ax[roi_ind].legend()
      ax[roi_ind].grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
      ax[roi_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
      ax[roi_ind].grid(axis="y", color="k", alpha=.1, linewidth=.5)
      ax[roi_ind].set_xlabel('Time in Seconds')
      ax[roi_ind].set_ylabel('Response')
      ax[roi_ind].set_title(f'ROI: {roi_ind}')
fh.suptitle(f'For each ROI, show all the unique parameter values')

# %% For each ROI, plot the mean of all parameter values
error_bars = True

fh, ax = plt.subplots(len(flies_yopto_mean_response), 1, figsize=(16, 8*len(flies_yopto_mean_response)))
for roi_ind in range(len(flies_yopto_mean_response)):

  ax[roi_ind].plot(new_time_vector, np.mean(flies_yopto_mean_response[roi_ind, :, :], axis=0), color='red', alpha=0.9, label='opto')
  ax[roi_ind].plot(new_time_vector, np.mean(flies_nopto_mean_response[roi_ind, :, :], axis=0), color='black', alpha=0.9, label='no opto')

  if error_bars == True:
    ax[roi_ind].fill_between(new_time_vector, np.mean(flies_yopto_sem_plus[roi_ind, :, :], axis=0), 
                    np.mean(flies_yopto_sem_minus[roi_ind, :, :], axis=0),
                    color='red', alpha=0.1)
    ax[roi_ind].fill_between(new_time_vector, np.mean(flies_nopto_sem_plus[roi_ind, :, :], axis=0), 
                np.mean(flies_nopto_sem_minus[roi_ind, :, :], axis=0),
                color='black', alpha=0.1)
  # Legend, Grid, Axis
  ax[roi_ind].legend()
  ax[roi_ind].grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")

  ax[roi_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
  ax[roi_ind].grid(axis="y", color="k", alpha=.1, linewidth=.5)
  ax[roi_ind].set_xlabel('Interpolated Frames')
  ax[roi_ind].set_title(f'ROI: {roi_ind}')



# %% MTS.2 Separate pipeline just for looking at pre and post conditions for individual flies
fly_id = astar1_fly2_pre_prox
layer_name = 'astar1_fly2_pre_prox'
display_fix = False
save_fig = True
darkmode = True

spatial_periods = [10, 20, 40, 80]
temporal_frequencies = [0.5, 1, 2, 4]

file_path = os.path.join(fly_id[0][0], fly_id[0][1] + ".hdf5")
if display_fix == True:
  cfg_dict = {'timing_channel_ind': 1}
  ID = imaging_data.ImagingDataObject(file_path, fly_id[0][2], quiet=True, cfg_dict = cfg_dict)
else:  
  ID = imaging_data.ImagingDataObject(file_path, fly_id[0][2], quiet=True)
roi_data = ID.getRoiResponses(fly_id[0][3])
time_vector, epoch_response = ID.getEpochResponseMatrix(np.vstack(roi_data['roi_response']))
parameter_keys = ('current_spatial_period', 'current_temporal_frequency')
unique_parameter_values, mean_response, sem_response, _ = ID.getTrialAverages(epoch_response, parameter_key=parameter_keys)
unique_parameter_values = np.array(unique_parameter_values)
# instantiate empty arrays
reordered_fly_mean = np.zeros((mean_response.shape[0], len(spatial_periods), len(temporal_frequencies), mean_response.shape[-1]))
reordered_fly_mean[:] = np.nan
reordered_fly_sem = np.zeros((sem_response.shape[0], len(spatial_periods), len(temporal_frequencies), sem_response.shape[-1]))
reordered_fly_sem[:] = np.nan

# Put mean responses into reordered_mean based on spatial_periods and temporal_frequencies
for spatial_ind, spatial in enumerate(spatial_periods):
  for temporal_ind, temporal in enumerate(temporal_frequencies):
    tmp_ind = np.intersect1d(np.where(spatial == unique_parameter_values[:, 0]),
                              np.where(temporal == unique_parameter_values[:, 1]))
    if len(tmp_ind) == 0:
      pass # skip
    elif len(tmp_ind) == 1:
      reordered_fly_mean[:, spatial_ind, temporal_ind, :] = mean_response[:, tmp_ind[0], :]
      reordered_fly_sem[:, spatial_ind, temporal_ind, :] = sem_response[:, tmp_ind[0], :]
    else:
      print('This should never happen')
if np.any(np.isnan(reordered_fly_mean)):
  print('Nans found - missing param combo')

# calculate the number of ROIs
num_rois = len(reordered_fly_mean)
# calculate the mean and sem across ROIs
fly_mean = np.nanmean(reordered_fly_mean, axis=0)
# calculate the standard error of the mean across ROIs
fly_sem = np.nanstd(reordered_fly_mean, axis=0) / np.sqrt(np.sum(~np.isnan(reordered_fly_mean), axis=0))
# calculate the sem_plus and sem_minus
fly_sem_plus = fly_mean + fly_sem
fly_sem_minus = fly_mean - fly_sem

# Plot the average trace across ROIs for each spatial_period and temporal_frequency
# define the plot color
c = [193/255, 70/255, 255/255]

if darkmode == True:
  # Set the plots to a dark grid background
  with plt.style.context('dark_background'):
    fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(20,20))
    for sp_ind, spatial in enumerate(spatial_periods):
      for tf_ind, temporal in enumerate(temporal_frequencies):

        ax[sp_ind, tf_ind].plot(time_vector, fly_mean[sp_ind, tf_ind, :], color='w')
        ax[sp_ind, tf_ind].fill_between(time_vector, fly_sem_plus[sp_ind, tf_ind, :], fly_sem_minus[sp_ind, tf_ind, :], color='w', alpha=0.3)
        
        ax[sp_ind, tf_ind].set_title('Spatial Period = ' + str(spatial) + ' |  Temporal Frequency = ' + str(temporal))
        ax[sp_ind, tf_ind].set_xlabel('time (s)')
        #ax[sp_ind, tf_ind].set_ylabel('dF/F')

        ax[sp_ind, tf_ind].grid(False)
    fh.suptitle(f'{layer_name} Mean Response Across {str(num_rois)} ROIs', y=1.01, fontsize=16)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fh.set_tight_layout(True)
    fh.text(0.0, 0.5, 'dF/F', ha='center', va='center', rotation='vertical', fontsize=14)
    #fh.tight_layout(rect=[0, 0.03, 1, 0.8])
    #plt.subplots_adjust(top=0.85)


else: # not darkmode
  fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(8*len(spatial_periods), 8*len(temporal_frequencies)))
  for sp_ind, spatial in enumerate(spatial_periods):
    for tf_ind, temporal in enumerate(temporal_frequencies):

      ax[sp_ind, tf_ind].plot(time_vector, fly_mean[sp_ind, tf_ind, :], color='black')
      ax[sp_ind, tf_ind].fill_between(time_vector, fly_sem_plus[sp_ind, tf_ind, :], fly_sem_minus[sp_ind, tf_ind, :], color=c, alpha=0.5)
      
      ax[sp_ind, tf_ind].set_title('Spatial Period = ' + str(spatial) + ' Temporal Frequency = ' + str(temporal))
      ax[sp_ind, tf_ind].set_xlabel('time (s)')
      ax[sp_ind, tf_ind].set_ylabel('dF/F')
  fh.suptitle(f'{layer_name} Mean Response Across {str(num_rois)} ROIs')

# save figure
if save_fig == True:
  fh.savefig(os.path.join(save_directory, f'{layer_name}_mean_response_across_{str(num_rois)}_ROIs.pdf'), dpi=300, bbox_inches='tight', transparent=True)

# plt.close('all')
 







# %% Functions for extracting metrics from a window across flies and plotting those metrics

# New function to extract metrics from individual ROIs
# takes in flies_yopto_mean_response and flies nopto_mean_response, computes metrics for them, and returns the differences between them
def getMetricsFromROIs(flies_nopto_mean_response, flies_yopto_mean_response):
# optoless_unique_parameter_values | flies_nopto_mean_response | flies_yopto_mean_response
# mean_response_matrix is ROIs x spatial_periods x temporal_frequencies x time

  # Define stimulus window to collect responses
  vis_start = ID.getRunParameters('pre_time')
  vis_length = ID.getRunParameters('stim_time')
  getting_going_time = 0.3 # time to wait to start the window responding to vis stimuluxs
  early_cutoff_time = 0.0 # time to cut off the window before the vis stimulus ends
  buffer_from_interp = 1.0 # time to buffer the window from the interpolation 'adding time' to the trial length
  window_length = vis_length - getting_going_time - early_cutoff_time + buffer_from_interp # length of the window
  window_time = vis_start + getting_going_time # time to start the window
  window_start_frame = int(np.ceil(window_time / ID.getResponseTiming().get('sample_period'))) # time to start the window in frames
  window_frames = int(np.ceil(window_length / ID.getResponseTiming().get('sample_period'))) # number of frames in the window
  
  nopto_windows = np.zeros((flies_nopto_mean_response.shape[0], flies_nopto_mean_response.shape[1], flies_nopto_mean_response.shape[2], window_frames))
  yopto_windows = np.zeros((flies_yopto_mean_response.shape[0], flies_yopto_mean_response.shape[1], flies_yopto_mean_response.shape[2], window_frames))

  # Loop through spatial_periods and temporal_frequences, and for each parameter, get the window of responses and calculate the min, max, and mean of that window
  # Loop through each ROI
  for roi_ind in range(flies_nopto_mean_response.shape[0]):
    for sp_ind in range(flies_nopto_mean_response.shape[1]):
      for tf_ind in range(flies_nopto_mean_response.shape[2]):
        # pull the window frames out of the mean response matrix
        nopto_windows[roi_ind, sp_ind, tf_ind, :] = flies_nopto_mean_response[roi_ind, sp_ind, tf_ind, window_start_frame:window_start_frame+window_frames]
        yopto_windows[roi_ind, sp_ind, tf_ind, :] = flies_yopto_mean_response[roi_ind, sp_ind, tf_ind, window_start_frame:window_start_frame+window_frames]
  
  # calculate the mean and max of the window for each ROI
  nopto_windows_mean = np.mean(nopto_windows, axis=3)
  yopto_windows_mean = np.mean(yopto_windows, axis=3)
  nopto_windows_max = np.max(nopto_windows, axis=3)
  yopto_windows_max = np.max(yopto_windows, axis=3)
  
  # return the windows and the metrics
  return nopto_windows, yopto_windows, nopto_windows_mean, yopto_windows_mean, nopto_windows_max, yopto_windows_max


# Function to compare and summarize the metrics from getMetricsFromROIs
def compareMetrics(nopto_windows_mean, yopto_windows_mean, nopto_windows_max, yopto_windows_max):
  # Takes in metrics that getMetricsFromROIs returns and compares them
  # returns the differences between the metrics, their mean, and their SEM across ROIs
  # The order of operations is such that each ROI is processed individually. 
  # For each ROI, the opto value and the no opto value are compared, and the difference is normalized by the no opto value.
  # Then, the mean and SEM of the normalized differences are calculated across ROIs.


  # calculate the differences between the metrics and normalize the difference
  mean_diff = yopto_windows_mean - nopto_windows_mean
  mean_diff_norm = mean_diff / nopto_windows_mean
  max_diff = yopto_windows_max - nopto_windows_max
  max_diff_norm = max_diff / nopto_windows_max

  # find the mean and sem of the normalized differences
  mean_diff_norm_ROI_avg = np.nanmean(mean_diff_norm, axis=0)
  mean_diff_norm_ROI_sem = np.nanstd(mean_diff_norm, axis=0) / np.sqrt(mean_diff_norm.shape[0])
  max_diff_norm_ROI_avg = np.nanmean(max_diff_norm, axis=0)
  max_diff_norm_ROI_sem = np.nanstd(max_diff_norm, axis=0) / np.sqrt(max_diff_norm.shape[0])

  # return the differences and their mean and sem
  return mean_diff_norm, mean_diff_norm_ROI_avg, mean_diff_norm_ROI_sem, max_diff_norm, max_diff_norm_ROI_avg, max_diff_norm_ROI_sem

# %% testing getMetricsFromROIs and compareMetrics
# get the windows and metrics
peek_at_progress = False

nopto_windows, yopto_windows, nopto_windows_mean, yopto_windows_mean, nopto_windows_max, yopto_windows_max = getMetricsFromROIs(flies_nopto_mean_response, flies_yopto_mean_response)

if peek_at_progress == True:
  # create a new figure
  fh, ax = plt.subplots(1, figsize=(16, 8))
  ax.plot(flies_nopto_mean_response[0, 2, 2, :])
  ax.plot(nopto_windows[0, 2, 2, :])

  # create a new figure
  fh, ax = plt.subplots(1, figsize=(16, 8))
  ax.plot(flies_nopto_mean_response[2, 1, 1, :])
  ax.plot(nopto_windows[1, 1, 1, :])

# Run compareMetrics
mean_diff_norm, mean_diff_norm_ROI_avg, mean_diff_norm_ROI_sem, max_diff_norm, max_diff_norm_ROI_avg, max_diff_norm_ROI_sem = compareMetrics(nopto_windows_mean, yopto_windows_mean, nopto_windows_max, yopto_windows_max)



# %% MTS.3 Plot the metrics  Metric Plotting - Good
# Loop through spatial_periods and temporal_frequences, and for each parameter, make a boxplot for mean_diff_norm and max_diff_norm
save_fig = True
darkmode = True

if darkmode == True:
  # Set the plots to a dark grid background
  with plt.style.context('dark_background'):

    fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(24, 16))
    for sp_ind in range(len(spatial_periods)):
      for tf_ind in range(len(temporal_frequencies)):
        # first filter out nan values from mean_diff_norm and max_diff_norm
        mean_diff_norm_filtered = mean_diff_norm[:, sp_ind, tf_ind][~np.isnan(mean_diff_norm[:, sp_ind, tf_ind])]
        max_diff_norm_filtered = max_diff_norm[:, sp_ind, tf_ind][~np.isnan(max_diff_norm[:, sp_ind, tf_ind])]
        # create a boxplot for mean_diff_norm_filtered and max_diff_norm_filtered and fill the boxplots with blue and red
        c = [167/255, 66/255, 247/255]
        ax[sp_ind, tf_ind].boxplot(mean_diff_norm_filtered, positions=[1.2], notch=True, patch_artist=True,
                                  boxprops=dict(facecolor=c, color='w'),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    flierprops=dict(markeredgecolor='w', markerfacecolor=c, marker='.', markersize=15),
                                    medianprops=dict(color='w'))
        c = [70/255, 185/255, 255/255]
        ax[sp_ind, tf_ind].boxplot(max_diff_norm_filtered, positions=[1.8], notch=True, patch_artist=True,
                                    boxprops=dict(facecolor=c, color='w'),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    flierprops=dict(markeredgecolor='w', markerfacecolor=c, marker='.', markersize=15),
                                    medianprops=dict(color='w'))
        
        # plot a dotted line at y=0
        ax[sp_ind, tf_ind].plot([0.9, 2.1], [0, 0], 'w--', alpha=0.2)
        ax[sp_ind, tf_ind].set_title('spatial_period: ' + str(spatial_periods[sp_ind]) + ', temporal_frequency: ' + str(temporal_frequencies[tf_ind]))
        #ax[sp_ind, tf_ind].set_ylabel('normalized difference')
        # create labels
        ax[sp_ind, tf_ind].set_xticks([1.2, 1.8])
        ax[sp_ind, tf_ind].set_xticklabels(['mean', 'max'])
        fh.text(0.0, 0.5, 'Normalized Difference', ha='center', va='center', rotation='vertical', fontsize=12)

    fh.suptitle('Comparison of mean and max normalized difference between opto and no opto across ROIs across spatial period and temporal frequencies', y=1.01, fontsize=16)
    fh.set_tight_layout(True)

else:
    fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(24, 16))
    for sp_ind in range(len(spatial_periods)):
      for tf_ind in range(len(temporal_frequencies)):
        # first filter out nan values from mean_diff_norm and max_diff_norm
        mean_diff_norm_filtered = mean_diff_norm[:, sp_ind, tf_ind][~np.isnan(mean_diff_norm[:, sp_ind, tf_ind])]
        max_diff_norm_filtered = max_diff_norm[:, sp_ind, tf_ind][~np.isnan(max_diff_norm[:, sp_ind, tf_ind])]
        # create a boxplot for mean_diff_norm_filtered and max_diff_norm_filtered and fill the boxplots with blue and red
        c = [193/255, 70/255, 255/255]
        ax[sp_ind, tf_ind].boxplot(mean_diff_norm_filtered, positions=[1.2], notch=True, patch_artist=True,
                                  boxprops=dict(facecolor=c, color='k'),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    flierprops=dict(color=c, markeredgecolor=c),
                                    medianprops=dict(color='k'))
        c = [70/255, 185/255, 255/255]
        ax[sp_ind, tf_ind].boxplot(max_diff_norm_filtered, positions=[1.8], notch=True, patch_artist=True,
                                    boxprops=dict(facecolor=c, color='k'),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    flierprops=dict(color=c, markeredgecolor=c),
                                    medianprops=dict(color='k'))
        
        # plot a dotted line at y=0
        ax[sp_ind, tf_ind].plot([0.9, 2.1], [0, 0], 'g--', alpha=0.2)
        ax[sp_ind, tf_ind].set_title('spatial_period: ' + str(spatial_periods[sp_ind]) + ', temporal_frequency: ' + str(temporal_frequencies[tf_ind]))
        ax[sp_ind, tf_ind].set_ylabel('normalized difference')
        # create labels
        ax[sp_ind, tf_ind].set_xticks([1.2, 1.8])
        ax[sp_ind, tf_ind].set_xticklabels(['mean', 'max'])

    fh.suptitle('Comparison of mean and max normalized difference between opto and no opto across ROIs across spatial period and temporal frequencies', y=1.01, fontsize=16)
    fh.set_tight_layout(True)

if save_fig == True:
    fh.savefig(
    save_directory
    + 'comparison_of_mean_and_max_normalized_difference_between_opto_and_no_opto.'
    + 'Darkmode='
    + str(darkmode)
    + '.pdf',
    dpi=300, bbox_inches='tight', transparent=True
    )





#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %% Metric Plotting  - meh

# Create a new figure for the mean metric, with a subplot for each spatial_period and temporal_frequency
fh, ax = plt.subplots(len(spatial_periods), len(temporal_frequencies), figsize=(24, 16))
# Loop through spatial_periods and temporal_frequences, and for each parameter, plot the mean metric as a barplot
for sp_ind in range(len(spatial_periods)):
  for tf_ind in range(len(temporal_frequencies)):
    ax[sp_ind, tf_ind].bar([0, 1], [mean_diff_norm_ROI_avg[sp_ind, tf_ind], max_diff_norm_ROI_avg[sp_ind, tf_ind]], yerr=[mean_diff_norm_ROI_sem[sp_ind, tf_ind], max_diff_norm_ROI_sem[sp_ind, tf_ind]])
    ax[sp_ind, tf_ind].set_title('spatial_period: ' + str(spatial_periods[sp_ind]) + ', temporal_frequency: ' + str(temporal_frequencies[tf_ind]))
    # create labels
    ax[sp_ind, tf_ind].set_xticks([0, 1])
    ax[sp_ind, tf_ind].set_xticklabels(['mean', 'max'])
    ax[sp_ind, tf_ind].set_ylabel('normalized difference')
fh.suptitle('Mean and max normalized difference between opto and no opto windows')

# %%
#TODO: 
# the standard error currently just scales with the normalization factor. It currently doesn't reflect the original value --> 1 scaling. 