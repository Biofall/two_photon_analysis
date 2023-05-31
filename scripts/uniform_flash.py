# uniform_flash.py
# written to analyze the UniformFlash data in which I perfused AstA

# %%
from visanalysis.analysis import imaging_data, shared_analysis

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats as st

import os
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd

# %% Outline where the data is

# Fly 1 - Proximal only
### Proximal
mi1_fly1_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "1", "mi1_proximal_multiple"]]
mi1_fly1_perf_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "2", "mi1_proximal_multiple"]]
mi1_fly1_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "3", "mi1_proximal_multiple"]]

# Fly 2 - Proximal only
## Proximal
mi1_fly2_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "4", "mi1_proximal_multiple"]]
mi1_fly2_perf_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "5", "mi1_proximal_multiple"]]
mi1_fly2_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "6", "mi1_proximal_multiple"]]

# Fly 3 - Proximal, Medial, Distal
## Proximal
mi1_fly3_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "7", "mi1_proximal_multiple"]] # wrong visual stim
mi1_fly3_perf_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "8", "mi1_proximal_multiple"]]
mi1_fly3_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "9", "mi1_proximal_multiple"]]
## Medial
mi1_fly3_pre_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "7", "mi1_medial_multiple"]] # wrong visual stim
mi1_fly3_perf_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "8", "mi1_medial_multiple"]]
mi1_fly3_post_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "9", "mi1_medial_multiple"]]
## Distal
mi1_fly3_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "7", "mi1_distal_multiple"]] # wrong visual stim
mi1_fly3_perf_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "8", "mi1_distal_multiple"]]
mi1_fly3_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "9", "mi1_distal_multiple"]]

# Fly 4 - Proximal, Medial, Distal
## Proximal
mi1_fly4_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "10", "mi1_proximal_multiple"]]
mi1_fly4_perf_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "11", "mi1_proximal_multiple"]]
mi1_fly4_post_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "12", "mi1_proximal_multiple"]]
## Medial
mi1_fly4_pre_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "10", "mi1_medial_multiple"]]
mi1_fly4_perf_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "11", "mi1_medial_multiple"]]
mi1_fly4_post_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "12", "mi1_medial_multiple"]]
## Distal
mi1_fly4_pre_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "10", "mi1_distal_multiple"]]
mi1_fly4_perf_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "11", "mi1_distal_multiple"]]
mi1_fly4_post_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "12", "mi1_distal_multiple"]]

# Fly 5 - Proximal only
## Proximal
mi1_fly5_pre_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "13", "mi1_proximal_multiple"]]
mi1_fly5_perf_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230327", "2023-03-27", "14", "mi1_proximal_multiple"]]

# Lists of fly IDs in each series
fly_list_prox = [1, 2, 3, 4, 5]
fly_list_medi = [3, 4]
fly_list_dist = [3, 4]

# Concatenate them all by block
## Proximal
mi1_prox_pre_all = np.concatenate( #removed mi1_fly3_pre_prox
                                  (mi1_fly1_pre_prox, mi1_fly2_pre_prox, mi1_fly3_pre_prox, mi1_fly4_pre_prox, mi1_fly5_pre_prox,),
                                  #mi1_fly5_pre_prox,),
                                  axis = 0,
                                 )
mi1_prox_pre_5 = np.concatenate(
                                    (mi1_fly5_pre_prox,),   
                                    axis = 0,
)
mi1_prox_pre_5_list = [5]

mi1_prox_pre_list = [1, 2, 3, 4, 5]
mi1_prox_perf_all = np.concatenate(
                                   (mi1_fly1_perf_prox, mi1_fly2_perf_prox, mi1_fly3_perf_prox, mi1_fly4_perf_prox, mi1_fly5_perf_prox,),
                                   #mi1_fly5_perf_prox,),
                                   axis = 0,
                                  )
mi1_prox_perf_list = [1, 2, 3, 4, 5]
mi1_prox_perf_5 = np.concatenate(
                                    (mi1_fly5_perf_prox,),
                                    axis = 0,
)
mi1_prox_perf_5_list = [5]

mi1_prox_post_all = np.concatenate(
                                   (mi1_fly1_post_prox, mi1_fly2_post_prox, mi1_fly3_post_prox, mi1_fly4_post_prox,),
                                   axis = 0,
                                 )
mi1_prox_post_list = [1, 2, 3, 4]
## Medial
mi1_medi_pre_all = np.concatenate(#removed mi1_fly3_pre_prox
                                    (mi1_fly3_pre_medi, mi1_fly4_pre_medi,),
                                    axis = 0,
                                    )
mi1_medi_pre_list = [3, 4]
mi1_medi_perf_all = np.concatenate(
                                    (mi1_fly3_perf_medi, mi1_fly4_perf_medi,),
                                    axis = 0,
                                    )
mi1_medi_perf_list = [3, 4]
mi1_medi_post_all = np.concatenate(
                                    (mi1_fly3_post_medi, mi1_fly4_post_medi,),
                                    axis = 0,
                                    )
mi1_medi_post_list = [3, 4]
## Distal
mi1_dist_pre_all = np.concatenate(#removed mi1_fly3_pre_prox
                                    (mi1_fly3_pre_dist, mi1_fly4_pre_dist,),
                                    axis = 0,
                                    )
mi1_dist_pre_list = [3, 4]
mi1_dist_perf_all = np.concatenate(
                                    (mi1_fly3_perf_dist, mi1_fly4_perf_dist,),
                                    axis = 0,
                                    )
mi1_dist_perf_list = [3, 4]
mi1_dist_post_all = np.concatenate(
                                    (mi1_fly3_post_dist, mi1_fly4_post_dist,),
                                    axis = 0,
                                    )
mi1_dist_post_list = [3, 4]

# concatenate all flies for proximal layer by block
mi1_prox_pre_total = np.concatenate(
                                    (mi1_fly1_pre_prox, mi1_fly2_pre_prox, mi1_fly4_pre_prox, mi1_fly5_pre_prox),
                                    axis = 0,
                                    )
mi1_prox_pre_total_list = [1, 2, 4, 5]
mi1_prox_perf_total = np.concatenate(
                                    (mi1_fly1_perf_prox, mi1_fly2_perf_prox, mi1_fly3_perf_prox, mi1_fly4_perf_prox, mi1_fly5_perf_prox),
                                    axis = 0,
                                    )
mi1_prox_perf_total_list = [1, 2, 3, 4, 5]
mi1_prox_post_total = np.concatenate(
                                    (mi1_fly1_post_prox, mi1_fly2_post_prox, mi1_fly3_post_prox, mi1_fly4_post_prox,),
                                    axis = 0,
                                    )
mi1_prox_post_total_list = [1, 2, 3, 4]



# Concatenate blocks together
mi1_pre_all = [mi1_prox_pre_all, mi1_medi_pre_all, mi1_dist_pre_all]
mi1_pre_all_list = [mi1_prox_pre_list, mi1_medi_pre_list, mi1_dist_pre_list]

mi1_perf_all = [mi1_prox_perf_all, mi1_medi_perf_all, mi1_dist_perf_all]
mi1_perf_all_list = [mi1_prox_perf_list, mi1_medi_perf_list, mi1_dist_perf_list]

mi1_post_all = [mi1_prox_post_all, mi1_medi_post_all, mi1_dist_post_all] 
mi1_post_all_list = [mi1_prox_post_list, mi1_medi_post_list, mi1_dist_post_list]

#Fly 5 only (it's missing post)
mi1_pre_5 = [mi1_prox_pre_5]
mi1_pre_5_list = [mi1_prox_pre_5_list]
mi1_perf_5 = [mi1_prox_perf_5]
mi1_perf_5_list = [mi1_prox_perf_5_list]
mi1_5_good = [mi1_pre_5, mi1_perf_5]
mi1_5_good_list = [mi1_pre_5_list, mi1_perf_5_list]

# Concatenate all together
mi1_all_good = [mi1_pre_all, mi1_perf_all, mi1_post_all]
mi1_all_good_list = [mi1_pre_all_list, mi1_perf_all_list, mi1_post_all_list]
# This list of lists contains all the data's indecies. It's shape is:
# exp_block_type x exp_layer x exp_fly

# Other lists for indexing and labeling
exp_layer = ["Proximal", "Medial", "Distal"]
exp_block_type = ["Pre", "Perfusion", "Wash"]

save_directory = "/Volumes/ABK2TBData/lab_repo/analysis/outputs/uniform_flash/"
Path(save_directory).mkdir(exist_ok=True)


# # %% Lil test space

# experiment = mi1_fly3_pre_prox
# file_path = os.path.join(experiment[0][0], experiment[0][1] + ".hdf5")
# ID = imaging_data.ImagingDataObject(file_path, experiment[0][2], quiet=True)

#  function that pulls out the mean and sem responses
def getMetrics(which_experiment):
    file_path = os.path.join(which_experiment[0], which_experiment[1] + ".hdf5")
    ID = imaging_data.ImagingDataObject(file_path, which_experiment[2], quiet=True)
    roi_data = ID.getRoiResponses(which_experiment[3])

    # epoch_response is ROI x time x trial
    time_vector, epoch_response = ID.getEpochResponseMatrix(np.vstack(roi_data['roi_response']))

    # mean_response is ROI x time (trials have been averaged)
    _, mean_response, sem_response, _ = ID.getTrialAverages(epoch_response)
    # ('current_led_intensity', 'current_led_duration')
    # calc the sem + / -
    sem_plus = mean_response + sem_response
    sem_minus = mean_response - sem_response

    # calculate the mean, max, min for each trial
    # This is ROI x trials
    mean_by_rois = ID.getResponseAmplitude(epoch_response, metric='mean')
    max_by_rois = ID.getResponseAmplitude(epoch_response, metric='max')
    min_by_rois = ID.getResponseAmplitude(epoch_response, metric='min')
    # Average across ROIs, dimensions are (trials,)
    mean_by_trial = np.mean(mean_by_rois, axis=0)
    max_by_trial = np.mean(max_by_rois, axis=0)
    min_by_trial = np.mean(min_by_rois, axis=0)
    # SEM across ROIs - dimensions are (trials,)
    sem_mean_by_trial = np.std(mean_by_rois, axis=0) / np.sqrt(len(mean_by_trial))
    sem_max_by_trial = np.std(max_by_rois, axis=0) / np.sqrt(len(max_by_trial))
    sem_min_by_trial = np.std(min_by_rois, axis=0) / np.sqrt(len(min_by_trial))

    epoch_timestamps = ID.getEpochParameters('epoch_unix_time')   # sec
    epoch_timestamps = epoch_timestamps

    return time_vector, mean_response, sem_response, sem_plus, sem_minus, mean_by_rois, max_by_rois, min_by_rois, mean_by_trial, max_by_trial, min_by_trial, sem_mean_by_trial, sem_max_by_trial, sem_min_by_trial, epoch_timestamps

# %% UNF.1 Plot individual traces by layer 
save_figures = False
darkmode = True
which_style = 'seaborn-white'

which_layer = mi1_all_good 
layer = 0

# Plotting the whole trace
exp_count = len(which_layer[0][0]) # number of experiments

if darkmode == True:
  # Set the plots to a dark grid background
  with plt.style.context(which_style):
    # set the color map
    cmap = plt.get_cmap('Set3') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.0, 1.0, 12)]

    fh, ax = plt.subplots(exp_count, 1, figsize=(16, 8*exp_count))
    for block_ind in range(len(exp_block_type)):
        for exp_ind in range(exp_count):
            time_vector, mean_response, _, _, _, _, _, _, _, _, _, _, _, _, _ = getMetrics(which_layer[block_ind][layer][exp_ind])            
            # Mean it
            mean_across_rois = np.squeeze(mean_response.mean(axis=0))
            # calculate the SEM (NOTE: Do not just avg across sem_plus and sem_minus)
            sem = np.squeeze(np.std(mean_response, axis=0) / np.sqrt(mean_response.shape[0]))
            # calc the sem + / -
            sem_plus_across_rois = mean_across_rois + sem
            sem_minus_across_rois = mean_across_rois - sem
            # Plot it
            ax[exp_ind].plot(time_vector, mean_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])
            ax[exp_ind].fill_between(time_vector, sem_plus_across_rois, sem_minus_across_rois, color=colors[block_ind], alpha=0.3)
            # axes handling
            ax[exp_ind].set_title(f'Fly {mi1_all_good_list[block_ind][layer][exp_ind]}')
            ax[exp_ind].grid(color='white', alpha=0.2)
            ax[exp_ind].legend(facecolor='black', edgecolor='black', labelcolor='white')
            ax[exp_ind].set_xlabel('Time (s)')
            ax[exp_ind].set_ylabel('Mean Response (dF/F)')

    fh.suptitle(f'All {exp_layer[layer]} Traces Averaged Across ROIs', y=1.01, fontsize=16)
    fh.set_tight_layout(True)

else: # lightmode
    cmap = plt.get_cmap('Set3') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.0, 1.0, 12)]

    fh, ax = plt.subplots(exp_count, 1, figsize=(16, 8*exp_count))
    for block_ind in range(len(exp_block_type)):
        for exp_ind in range(exp_count):
            time_vector, mean_response, sem_plus, sem_minus, mean_across_rois, max_across_rois, min_across_rois, _ = getMetrics(which_layer[block_ind][layer][exp_ind])
            # Mean it
            mean_across_rois = np.squeeze(mean_response.mean(axis=0))
            # calculate the SEM (NOTE: Do not just avg across sem_plus and sem_minus)
            sem = np.squeeze(np.std(mean_response, axis=0) / np.sqrt(mean_response.shape[0]))
            # calc the sem + / -
            sem_plus_across_rois = mean_across_rois + sem
            sem_minus_across_rois = mean_across_rois - sem
            # plot it
            ax[exp_ind].plot(time_vector, mean_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])
            ax[exp_ind].fill_between(time_vector, sem_plus_across_rois, sem_minus_across_rois, color=colors[block_ind], alpha=0.1)
            # axes handling
            ax[exp_ind].set_title(f'Fly {mi1_all_good_list[block_ind][layer][exp_ind]}')
            # set the subplot facecolor to black and add a white grid with alpha 0.2
            ax[exp_ind].set_facecolor('black')
            ax[exp_ind].grid(color='white', alpha=0.2)
            ax[exp_ind].legend(facecolor='black', edgecolor='black', labelcolor='white')
    fh.suptitle(f'All {exp_layer[layer]} Traces Averaged Across ROIs', y=1.01, fontsize=16)
    fh.set_tight_layout(True)



if save_figures:
    save_name = f'All_{exp_layer[layer]}_Traces_Averaged_Across_ROIs.Darkmode={str(darkmode)}.UNF.1.{which_style}.pdf'
    save_path = os.path.join(save_directory, save_name)
    fh.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True,)

# %% Plot max values using ID.getResponseAmplitude across all ROIs
# Proximal, Medial, or Distal (0, 1, or 2)
layer = 0
save_figures = True
which_layer = mi1_all_good 

exp_count = len(which_layer[0][layer]) # number of experiments

met_fig, met_ax = plt.subplots(exp_count, 3, figsize=(16*3, 8*exp_count))
# set the color map
cmap = plt.get_cmap('Pastel2') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, 8)]

for block_ind in range(len(exp_block_type)):
    for exp_ind in range(exp_count):
        # debug print the indecies
        print(f'block_ind: {block_ind}, exp_ind: {exp_ind}')

        time_vector, mean_response, sem_plus, sem_minus, mean_across_rois, max_across_rois, min_across_rois, _ = getMetrics(which_layer[block_ind][layer][exp_ind])

        # plot the max_across_rois
        met_ax[exp_ind, 0].plot(max_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])
        # plot the min_across_rois
        met_ax[exp_ind, 1].plot(min_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])
        # plot the mean_across_rois
        met_ax[exp_ind, 2].plot(mean_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])

        # title for subfigure
        met_ax[exp_ind, 0].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Max Response')
        met_ax[exp_ind, 1].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Min Response')
        met_ax[exp_ind, 2].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Mean Response')
        # X axis label
        met_ax[exp_ind, 0].set_xlabel('Trials')
        # Y axis label
        met_ax[exp_ind, 0].set_ylabel('Max Response')
        met_ax[exp_ind, 1].set_ylabel('Min Response')
        met_ax[exp_ind, 2].set_ylabel('Mean Response')

# show legend for all subplots and set the legend alpha to white
for i in range(3):
    # set every subplot background to have a grid with an alpha of 0.5 and white
    for j in range(exp_count):
        # set the legend to a white background
        met_ax[j, i].legend(facecolor='white', framealpha=1)
        met_ax[j, i].grid(alpha=0.5, color='white')
        # set the plots to have a black background
        met_ax[j, i].set_facecolor('black')
    
# Title for whole figure
met_fig.suptitle(f'All {exp_layer[0]} Metrics Averaged Across ROIs')

# Save the figure
if save_figures:
    save_name = f'All_{exp_layer[0]}_Metrics_Averaged_Across_ROIs.pdf'
    save_path = os.path.join(save_directory, save_name)
    met_fig.savefig(save_path, dpi=300)








# %% UNF.2 Plot max values using ID.getResponseAmplitude across all ROIs. End to end across blocks
save_fig = False
darkmode = True
which_style = 'seaborn-white'

# Proximal, Medial, or Distal (0, 1, or 2)
layer = 0
which_layer = mi1_all_good 
skip_trials = 1
incomplete_flies = [[2, 0], [4, 2]] # flies that are missing a block
incomp_index = 0

exp_count = len(which_layer[0][layer]) # number of experiments

if darkmode == True:
  # Set the plots to a dark grid background
  with plt.style.context(which_style):
        met_fig, met_ax = plt.subplots(exp_count, 2, figsize=(16*3, 8*exp_count))
        # set the color map
        cmap = plt.get_cmap('Pastel2') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
        colors = [cmap(i) for i in np.linspace(0.0, 1.0, 8)]
        t0 = 0
        for exp_ind in range(exp_count):
            for block_ind in range(len(exp_block_type)):

                if exp_ind == incomplete_flies[incomp_index][0] and block_ind == incomplete_flies[incomp_index][1]:
                    print('Skipping incomplete fly. Block: ', block_ind, ' Fly: ', exp_ind)
                    incomp_index+=1
                    pass
                else:
                    # DEBUG:
                    print(f'exp_ind: {exp_ind} | block_ind: {block_ind}')
                    print(f'which experiment: {which_layer[block_ind][layer][exp_ind]}')
                    _, _, _, _, _, _, _, _, mean_by_trial, max_by_trial, _, sem_mean_by_trial, sem_max_by_trial, _, epoch_timestamps = getMetrics(which_layer[block_ind][layer][exp_ind])
                    if block_ind == 0:
                        t0 = epoch_timestamps[0]
                        print(f't0: {t0}')
                    
                    # subtract the first timepoint from all timepoints to get a relative time to 0
                    plot_timestamps = epoch_timestamps - t0

                    # catching weird case where the first block isn't there
                    if exp_ind == incomplete_flies[0][0] and block_ind == incomplete_flies[0][1]+1:
                        _, _, _, _, _, _, _, _, _, _, _, _, _, _, epoch_timestamps = getMetrics(which_layer[block_ind+1][layer][exp_ind])
                        t0 = epoch_timestamps[0]
                        plot_timestamps = epoch_timestamps - t0
                    print(f'epoch_timestamps: {epoch_timestamps}')
                    print(f'plot_timestamps: {plot_timestamps}')

                    # plot the max_across_rois
                    met_ax[exp_ind, 0].plot(plot_timestamps[skip_trials:], max_by_trial[skip_trials:], marker='o', color=colors[block_ind], label=exp_block_type[block_ind])
                    met_ax[exp_ind, 0].fill_between(plot_timestamps[skip_trials:], max_by_trial[skip_trials:] + sem_max_by_trial[skip_trials:], max_by_trial[skip_trials:] - sem_max_by_trial[skip_trials:], color=colors[block_ind], alpha=0.4)
                    # plot linear regression of max_across_rois
                    slope, intercept, r_value, p_value, std_err = st.linregress(plot_timestamps[skip_trials:], max_by_trial[skip_trials:])
                    met_ax[exp_ind, 0].plot(plot_timestamps[skip_trials:], intercept + slope*plot_timestamps[skip_trials:], color=colors[block_ind], linestyle='dashed',)

                    # plot the mean_across_rois
                    met_ax[exp_ind, 1].plot(plot_timestamps[skip_trials:], mean_by_trial[skip_trials:], color=colors[block_ind], label=exp_block_type[block_ind])
                    met_ax[exp_ind, 1].fill_between(plot_timestamps[skip_trials:], mean_by_trial[skip_trials:] + sem_mean_by_trial[skip_trials:], mean_by_trial[skip_trials:] - sem_mean_by_trial[skip_trials:], color=colors[block_ind], alpha=0.4)
                    # plot linear regression of mean_across_rois
                    slope, intercept, r_value, p_value, std_err = st.linregress(plot_timestamps[skip_trials:], mean_by_trial[skip_trials:])
                    met_ax[exp_ind, 1].plot(plot_timestamps[skip_trials:], intercept + slope*plot_timestamps[skip_trials:], color=colors[block_ind], linestyle='dashed',)

                    # title for subfigure
                    met_ax[exp_ind, 0].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Max Response')
                    met_ax[exp_ind, 1].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Mean Response')
                    # X axis label
                    met_ax[exp_ind, 0].set_xlabel('Time (s)')
                    # Y axis label
                    met_ax[exp_ind, 0].set_ylabel('Max Response (dF/F)')
                    met_ax[exp_ind, 1].set_ylabel('Mean Response (dF/F)')

                    # Keeping track of biggest x-value

                    # set x axis to start at 0
                    met_ax[exp_ind, 0].set_xlim(-20, plot_timestamps[-1]+40)
                    met_ax[exp_ind, 1].set_xlim(-20, plot_timestamps[-1]+40)
                
        # show legend for all subplots and set the legend alpha to white
        for i in range(2):
            # set every subplot background to have a grid with an alpha of 0.5 and white
            for j in range(exp_count):
                # set the legend to a white background
                met_ax[j, i].legend()
                met_ax[j, i].grid(alpha=0.5, color='white')
                # set the plots to have a black background
                #met_ax[j, i].set_facecolor('black')
        met_fig.suptitle(f'All {exp_layer[0]} Metrics Averaged Across ROIs', y=1.01, fontsize=16)
        met_fig.set_tight_layout(True)

        # create separate figures for mean and max metrics
        met_fig_mean, met_ax_mean = plt.subplots(exp_count, 1, figsize=(8*exp_count, 8*exp_count))
        met_fig_max, met_ax_max = plt.subplots(exp_count, 1, figsize=(8*exp_count, 8*exp_count))
        for exp_ind in range(exp_count):
            for block_ind in range(len(exp_block_type)):
                # DEBUG:
                print(f'exp_ind: {exp_ind} | block_ind: {block_ind}')
                print(f'which experiment: {which_layer[block_ind][layer][exp_ind]}')

                _, _, _, _, _, _, _, _, mean_by_trial, max_by_trial, _, sem_mean_by_trial, sem_max_by_trial, _, epoch_timestamps = getMetrics(which_layer[block_ind][layer][exp_ind])
                if block_ind == 0:
                    t0 = epoch_timestamps[0]
                plot_timestamps = epoch_timestamps - t0

                if exp_ind == incomplete_flies[0] and block_ind == incomplete_flies[1]:
                    pass
                else:
                    # plot the max_across_rois
                    met_ax_max[exp_ind].plot(plot_timestamps[skip_trials:], max_by_trial[skip_trials:], color=colors[block_ind], label=exp_block_type[block_ind])
                    met_ax_max[exp_ind].fill_between(plot_timestamps[skip_trials:], max_by_trial[skip_trials:] + sem_max_by_trial[skip_trials:], max_by_trial[skip_trials:] - sem_max_by_trial[skip_trials:], color=colors[block_ind], alpha=0.4)
                    # plot linear regression of max_across_rois
                    slope, intercept, r_value, p_value, std_err = st.linregress(plot_timestamps[skip_trials:], max_by_trial[skip_trials:])
                    met_ax_max[exp_ind].plot(plot_timestamps[skip_trials:], intercept + slope*plot_timestamps[skip_trials:], color=colors[block_ind], linestyle='dashed',)

                    # plot the mean_across_rois
                    met_ax_mean[exp_ind].plot(plot_timestamps[skip_trials:], mean_by_trial[skip_trials:], color=colors[block_ind], label=exp_block_type[block_ind])
                    met_ax_mean[exp_ind].fill_between(plot_timestamps[skip_trials:], mean_by_trial[skip_trials:] + sem_mean_by_trial[skip_trials:], mean_by_trial[skip_trials:] - sem_mean_by_trial[skip_trials:], color=colors[block_ind], alpha=0.4)
                    # plot linear regression of mean_across_rois
                    slope, intercept, r_value, p_value, std_err = st.linregress(plot_timestamps[skip_trials:], mean_by_trial[skip_trials:])
                    met_ax_mean[exp_ind].plot(plot_timestamps[skip_trials:], intercept + slope*plot_timestamps[skip_trials:], color=colors[block_ind], linestyle='dashed',)

                    # title for subfigure
                    met_ax_max[exp_ind].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Max Response')
                    met_ax_mean[exp_ind].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Mean Response')
                    # X axis label
                    met_ax_max[exp_ind].set_xlabel('Time (s)')
                    # Y axis label
                    met_ax_max[exp_ind].set_ylabel('Max Response (dF/F)')
                    met_ax_mean[exp_ind].set_ylabel('Mean Response (dF/F)')
                    # set x axis to start at 0
                    met_ax_max[exp_ind].set_xlim(-20, plot_timestamps[-1]+40)
                    met_ax_mean[exp_ind].set_xlim(-20, plot_timestamps[-1]+40)

        # show legend for all subplots and set the legend alpha to white
        for i in range(2):
            # set every subplot background to have a grid with an alpha of 0.5 and white
            for j in range(exp_count):
                # set the legend to a white background
                met_ax_max[j].legend()
                met_ax_max[j].grid(alpha=0.5, color='white')
                met_ax_mean[j].legend()
                met_ax_mean[j].grid(alpha=0.5, color='white')
                # set the plots to have a black background
                #met_ax[j, i].set_facecolor('black')

        met_fig_mean.suptitle(f'All {exp_layer[0]} Means Averaged Across ROIs', y=1.01, fontsize=16)
        met_fig_max.suptitle(f'All {exp_layer[0]} Maxes Averaged Across ROIs', y=1.01, fontsize=16)
        met_fig_mean.set_tight_layout(True)
        met_fig_max.set_tight_layout(True)

        # Save the figures
        if save_fig == True:
            save_name_mean = f'All_{str(exp_layer[0])}_Just_Mean_Averaged_Across_ROIs.{str(exp_count)}Flies.UNF.2.{which_style}.pdf'
            save_path = os.path.join(save_directory, save_name_mean)
            met_fig_mean.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True,)
            save_name_max = f'All_{str(exp_layer[0])}_Just_Max_Averaged_Across_ROIs.{str(exp_count)}Flies.UNF.2.{which_style}.pdf'
            save_path = os.path.join(save_directory, save_name_max) 
            met_fig_max.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True,)

else: # light mode
    met_fig, met_ax = plt.subplots(exp_count, 3, figsize=(16, 20))
    # set the color map
    cmap = plt.get_cmap('Pastel2') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.0, 1.0, 8)]
    t0 = 0
    for exp_ind in range(exp_count):
        for block_ind in range(len(exp_block_type)):
            # debug print the indecies
            print(f'block_ind: {block_ind}, exp_ind: {exp_ind}')

            time_vector, mean_response, sem_plus, sem_minus, mean_across_rois, max_across_rois, min_across_rois, epoch_timestamps = getMetrics(which_layer[block_ind][layer][exp_ind])

            if block_ind == 0:
                t0 = epoch_timestamps[0]

            plot_timestamps = epoch_timestamps - t0

            # plot the max_across_rois
            met_ax[exp_ind, 0].plot(plot_timestamps, max_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])
            # plot the min_across_rois
            met_ax[exp_ind, 1].plot(plot_timestamps, min_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])
            # plot the mean_across_rois
            met_ax[exp_ind, 2].plot(plot_timestamps, mean_across_rois, color=colors[block_ind], label=exp_block_type[block_ind])

            # title for subfigure
            met_ax[exp_ind, 0].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Max Response')
            met_ax[exp_ind, 1].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Min Response')
            met_ax[exp_ind, 2].set_title(f'Fly {mi1_all_good_list[block_ind][0][exp_ind]} Mean Response')
            # X axis label
            met_ax[exp_ind, 0].set_xlabel('Trials')
            # Y axis label
            met_ax[exp_ind, 0].set_ylabel('Max Response')
            met_ax[exp_ind, 1].set_ylabel('Min Response')
            met_ax[exp_ind, 2].set_ylabel('Mean Response')
    # show legend for all subplots and set the legend alpha to white
    for i in range(3):
        # set every subplot background to have a grid with an alpha of 0.5 and white
        for j in range(exp_count):
            # set the legend to a white background
            met_ax[j, i].legend(facecolor='white', framealpha=1)
            met_ax[j, i].grid(alpha=0.5, color='white')
            # set the plots to have a black background
            met_ax[j, i].set_facecolor('black')
    # Title for whole figure
    met_fig.suptitle(f'All {exp_layer[0]} Metrics Averaged Across ROIs')


# Save the figure
if save_fig == True:
    save_name = f'All_{str(exp_layer[0])}_Metrics_Averaged_Across_ROIs.{str(exp_count)}Flies.UNF.2.pdf'
    save_path = os.path.join(save_directory, save_name)
    met_fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True,)




# %% Incomplete fly plots

# # same plot as above, but just for fly 5 which only has pre and perf
met_fig, met_ax = plt.subplots(1, 3, figsize=(16*3, 8*1))
# set the color map
cmap = plt.get_cmap('Pastel2') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, 8)]

# mi1_fly5_pre_prox | mi1_fly5_perf_prox
skip_trials = 1
time_vector, mean_response, sem_response, sem_plus, sem_minus, mean_by_rois, max_by_rois, min_by_rois, mean_across_rois_pre, max_across_rois_pre, min_across_rois_pre, sem_mean_by_trial, sem_max_by_trial, sem_min_by_trial, epoch_timestamps = getMetrics(mi1_fly5_pre_prox[0])
time_vector, mean_response, sem_response, sem_plus, sem_minus, mean_by_rois, max_by_rois, min_by_rois, mean_across_rois_perf, max_across_rois_perf, min_across_rois_perf, sem_mean_by_trial, sem_max_by_trial, sem_min_by_trial, epoch_timestamps = getMetrics(mi1_fly5_perf_prox[0])
# plot the max_across_rois
met_ax[0].plot(np.arange(len(max_across_rois_pre[skip_trials:])), max_across_rois_pre[skip_trials:], color='r', label='pre')
met_ax[0].plot(np.arange(len(max_across_rois_perf[skip_trials:]))+50-skip_trials, max_across_rois_perf[skip_trials:], color='b', label='perf')
# plot the min_across_rois
met_ax[1].plot(np.arange(len(min_across_rois_pre[skip_trials:])), min_across_rois_pre[skip_trials:], color='r', label='pre')
met_ax[1].plot(np.arange(len(min_across_rois_perf[skip_trials:]))+50-skip_trials, min_across_rois_perf[skip_trials:], color='b', label='perf')
# plot the mean_across_rois
met_ax[2].plot(np.arange(len(mean_across_rois_pre[skip_trials:])), mean_across_rois_pre[skip_trials:], color='r', label='pre')
met_ax[2].plot(np.arange(len(mean_across_rois_perf[skip_trials:]))+50-skip_trials, mean_across_rois_perf[skip_trials:], color='b', label='perf')
# create titles and labels
met_ax[0].set_title(f'Fly {str(mi1_all_good_list[0][0][0])} Max Response')
met_ax[1].set_title(f'Fly {str(mi1_all_good_list[0][0][0])} Min Response')
met_ax[2].set_title(f'Fly {str(mi1_all_good_list[0][0][0])} Mean Response')
# X axis label
met_ax[0].set_xlabel('Trials')
met_ax[1].set_xlabel('Trials')
met_ax[2].set_xlabel('Trials')
# Y axis label
met_ax[0].set_ylabel('Max Response')
met_ax[1].set_ylabel('Min Response')
met_ax[2].set_ylabel('Mean Response')


# # same plot as above, but just for fly 3 which only has perf and post
met_fig, met_ax = plt.subplots(1, 3, figsize=(16*3, 8*1))
# set the color map
cmap = plt.get_cmap('Pastel2') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, 8)]

# mi1_fly3_perf_prox | mi1_fly3_post_prox
skip_trials = 1
time_vector, mean_response, sem_response, sem_plus, sem_minus, mean_by_rois, max_by_rois, min_by_rois, mean_across_rois_perf, max_across_rois_perf, min_across_rois_perf, sem_mean_by_trial, sem_max_by_trial, sem_min_by_trial, epoch_timestamps = getMetrics(mi1_fly3_perf_prox[0])
time_vector, mean_response, sem_response, sem_plus, sem_minus, mean_by_rois, max_by_rois, min_by_rois, mean_across_rois_post, max_across_rois_post, min_across_rois_post, sem_mean_by_trial, sem_max_by_trial, sem_min_by_trial, epoch_timestamps = getMetrics(mi1_fly3_post_prox[0])
# plot the max_across_rois
met_ax[0].plot(np.arange(len(max_across_rois_perf[skip_trials:])), max_across_rois_perf[skip_trials:], color='r', label='perf')
met_ax[0].plot(np.arange(len(max_across_rois_post[skip_trials:]))+50-skip_trials, max_across_rois_post[skip_trials:], color='b', label='post')
# plot the min_across_rois
met_ax[1].plot(np.arange(len(min_across_rois_perf[skip_trials:])), min_across_rois_perf[skip_trials:], color='r', label='perf')
met_ax[1].plot(np.arange(len(min_across_rois_post[skip_trials:]))+50-skip_trials, min_across_rois_post[skip_trials:], color='b', label='post')
# plot the mean_across_rois
met_ax[2].plot(np.arange(len(mean_across_rois_perf[skip_trials:])), mean_across_rois_perf[skip_trials:], color='r', label='perf')
met_ax[2].plot(np.arange(len(mean_across_rois_post[skip_trials:]))+50-skip_trials, mean_across_rois_post[skip_trials:], color='b', label='post')
# create titles and labels
met_ax[0].set_title(f'Fly {str(mi1_all_good_list[0][0][0])} Max Response')
met_ax[1].set_title(f'Fly {str(mi1_all_good_list[0][0][0])} Min Response')
met_ax[2].set_title(f'Fly {str(mi1_all_good_list[0][0][0])} Mean Response')
# X axis label
met_ax[0].set_xlabel('Trials')
met_ax[1].set_xlabel('Trials')
met_ax[2].set_xlabel('Trials')
# Y axis label
met_ax[0].set_ylabel('Max Response')
met_ax[1].set_ylabel('Min Response')
met_ax[2].set_ylabel('Mean Response')

















# %% Plot max, min, mean values for a specific layer across all condtions
# concatenate all the mi1_fly4 conditions
mi1_fly4_medi = np.concatenate((mi1_fly4_pre_medi, mi1_fly4_perf_medi, mi1_fly4_post_medi,), axis=0)
mi1_fly4_dist = np.concatenate((mi1_fly4_pre_dist, mi1_fly4_perf_dist, mi1_fly4_post_dist,), axis=0)

which_layer = mi1_fly4_prox
save_figures = False

# Create figure
fig, ax = plt.subplots(1, 3, figsize=(16*3, 8))

# loop through the conditions
for block_ind in range(len(exp_block_type)):
    time_vector, mean_response, sem_plus, sem_minus, mean_across_rois, max_across_rois, min_across_rois = getMetrics(which_layer[block_ind])

    # plot the max_across_rois
    ax[0].plot(max_across_rois, color=colors[block_ind], linewidth=5, label=exp_block_type[block_ind])
    # plot the min_across_rois
    ax[1].plot(min_across_rois, color=colors[block_ind], linewidth=5, label=exp_block_type[block_ind])
    # plot the mean_across_rois
    ax[2].plot(mean_across_rois, color=colors[block_ind], linewidth=5, label=exp_block_type[block_ind])

    # For all subfigures, set axis parameters and show legend
    for i in range(3):
        # set axis labels
        ax[i].set_xlabel('Trials')
        ax[i].set_ylabel('Response Amplitude')
        # show legends
        ax[i].legend()        
    
    # title for subfigure
    ax[0].set_title(f'Fly Max response')
    ax[1].set_title(f'Fly Min response')
    ax[2].set_title(f'Fly Mean response')

# Title for whole figure, referring to which_layer
fig.suptitle(f'All Metrics Averaged Across ROIs for {which_layer[0][3]}')

# Save the figure
if save_figures:
    save_name = f'All_Metrics_Averaged_Across_ROIs_for_{which_layer[0][3]}.png'
    save_path = os.path.join(save_directory, save_name)
    fig.savefig(save_path, dpi=300)



# %% Plot every single fly's Max Value over time for a specific layer
all_prox = [mi1_prox_pre_total, mi1_prox_perf_total, mi1_prox_post_total]
all_prox_list = [mi1_prox_pre_total_list, mi1_prox_perf_total_list, mi1_prox_post_total_list]

which_layer = all_prox # mi1_prox_pre_total | mi1_prox_perf_total | mi1_prox_post_total
which_list = all_prox_list
# choose a value to start from, skipping weird initial artifacts
start_val = 1
save_figures = True

# set the color map
cmap = plt.get_cmap('Set3') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, 12)]

# Create figure
fig, ax = plt.subplots(len(all_prox), 3, figsize=(14*3, 6*len(all_prox)))

# loop through exp_block_type
for block_ind in range(len(exp_block_type)):
    # loop through the flies
    for fly_ind in range(len(which_layer[block_ind])):
        time_vector, mean_response, sem_plus, sem_minus, mean_across_rois, max_across_rois, min_across_rois = getMetrics(which_layer[block_ind][fly_ind])

        # plot the max_across_rois
        ax[block_ind, 0].plot(max_across_rois[start_val:], color=colors[which_list[block_ind][fly_ind]], linewidth=4, alpha=0.9, label=f'Fly {which_list[block_ind][fly_ind]}')
        # plot the min_across_rois
        ax[block_ind, 1].plot(min_across_rois[start_val:], color=colors[which_list[block_ind][fly_ind]], linewidth=4, alpha=0.9, label=f'Fly {which_list[block_ind][fly_ind]}')
        # plot the mean_across_rois
        ax[block_ind, 2].plot(mean_across_rois[start_val:], color=colors[which_list[block_ind][fly_ind]], linewidth=4, alpha=0.9, label=f'Fly {which_list[block_ind][fly_ind]}')

    # For all subfigures, set axis parameters and show legend
    for i in range(3):
        # set axis labels
        ax[block_ind, i].set_xlabel('Trials')
        ax[block_ind, i].set_ylabel('Response Amplitude')
        # show legends with a black background and white text
        ax[block_ind, i].legend(facecolor='black', edgecolor='black', labelcolor='white')
        # show a grid with an alpha of 0.5 and white lines
        ax[block_ind, i].grid(alpha=0.4, color='white')
        # set the plots to have a black background
        ax[block_ind, i].set_facecolor('black')
        # set subfigure titles
        ax[block_ind, 0].set_title(f'{exp_block_type[block_ind]} Block - Max response')
        ax[block_ind, 1].set_title(f'{exp_block_type[block_ind]} Block - Min response')
        ax[block_ind, 2].set_title(f'{exp_block_type[block_ind]} Block - Mean response')
                
    # Figure title
    fig.suptitle(f'Max/Min/Mean Value for {which_layer[0][0][3]} Across Blocks')

# Save the figure
if save_figures:
    save_name = f'MaxMinMean_Value_for_{which_layer[0][0][3]}_Across_Blocks.pdf'
    save_path = os.path.join(save_directory, save_name)
    fig.savefig(save_path, dpi=300)




# %% To demo for Tom

all_prox = [mi1_prox_pre_total, mi1_prox_perf_total, mi1_prox_post_total]
all_prox_list = [mi1_prox_pre_total_list, mi1_prox_perf_total_list, mi1_prox_post_total_list]
which_layer = all_prox # mi1_prox_pre_total | mi1_prox_perf_total | mi1_prox_post_total
which_list = all_prox_list

# choose a value to start from, skipping weird initial artifacts
start_val = 1
save_figures = False

# set the color map
cmap = plt.get_cmap('Pastel2') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, 8)]

# Create figure
fig, ax = plt.subplots(len(all_prox), 3, figsize=(14*3, 6*len(all_prox)))

# loop through exp_block_type
for block_ind in range(len(exp_block_type)):
    # loop through the flies
    for fly_ind in range(len(which_layer[block_ind])):
        time_vector, mean_response, sem_plus, sem_minus, mean_across_rois, max_across_rois, min_across_rois = getMetrics(which_layer[block_ind][fly_ind])

        # plot the max_across_rois
        


    # For all subfigures, set axis parameters and show legend

# Title for whole figure

# Save the figure