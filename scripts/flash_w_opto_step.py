# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import os
from pathlib import Path
import numpy as np

# %%
# Opto intensity sweep w/ flash experiments (with MoCo!) 2/8/22

# Multiple ROIs
# Fly 1
mi1_fly1_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20221122.common_moco", "2022-11-22", "3", "proximal_multiple"]]
mi1_fly1_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20221122.common_moco", "2022-11-22", "3", "medial_multiple"]]
mi1_fly1_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20221122.common_moco", "2022-11-22", "3", "distal_multiple"]]
# Fly 2
mi1_fly2_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "proximal_multiple3"]]
mi1_fly2_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "medial_multiple_sub2"]] #also 'medial_multiple_sub1", "medial_multiple_sub2"
mi1_fly2_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "distal_multiple"]]
# Fly 3
mi1_fly3_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230216", "2023-02-16", "5", "mi1_proximal_multiple"]]
mi1_fly3_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230216", "2023-02-16", "5", "mi1_medial_multiple"]] #also 'medial_multiple_sub1", "medial_multiple_sub2"
mi1_fly3_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230216", "2023-02-16", "5", "mi1_distal_multiple"]]
# Fly 4 #less good
mi1_fly4_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230223.moco", "2023-02-23", "6", "mi1_proximal_multiple"]]
mi1_fly4_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230223.moco", "2023-02-23", "6", "mi1_medial_multiple"]] 
mi1_fly4_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230223.moco", "2023-02-23", "6", "mi1_distal_multiple"]]
# Fly 5 #less good
mi1_fly5_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230223.moco", "2023-02-23", "8", "mi1_proximal_multiple"]]
mi_fly5_prox_double = [["/Volumes/ABK2TBData/data_repo/bruker/20230223.moco", "2023-02-23", "8", "mi1_proximal_multiple_double"]]
mi1_fly5_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230223.moco", "2023-02-23", "8", "mi1_medial_multiple"]] 
mi1_fly5_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230223.moco", "2023-02-23", "8", "mi1_distal_multiple"]]
# Fly 6 (moco)
mi1_fly6_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230302", "2023-03-02", "8", "mi1_proximal_multiple"]]
mi1_fly6_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230302", "2023-03-02", "8", "mi1_medial_multiple"]]
mi1_fly6_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230302", "2023-03-02", "8", "mi1_distal_multiple"]]
# Fly 7 (only prox, kind of medial)
mi1_fly7_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230302", "2023-03-02", "2", "mi1_proximal_multiple"]]
# Fly 8 (prox only) lotta motion
mi1_fly8_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230302", "2023-03-02", "6", "mi1_proximal_multiple"]]
# Fly 9
mi1_fly9_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "4", "mi1_proximal_multiple"]]
mi1_fly9_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "4", "mi_medial_multiple"]]
mi1_fly9_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "4", "mi1_distal_multiple"]]
# Fly 10
mi1_fly10_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "6", "mi1_proximal_multiple"]]
mi1_fly10_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "6", "mi_medial_multiple"]]
mi1_fly10_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "6", "mi1_distal_multiple_lowcon"]]
# Fly 11
mi1_fly11_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "7", "mi1_proximal_multiple"]]
# Fly 12
mi1_fly12_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "1", "mi1_proximal_multiple"]]
mi1_fly12_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "1", "mi_medial_multiple"]]
mi1_fly12_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "1", "mi1_distal_multiple"]]
# Fly 13 # kinda shitty b/c motion
mi1_fly13_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "2", "mi1_proximal_multiple"]]
# Fly 14 # kinda shitty b/c motion
mi1_fly14_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "3", "mi1_proximal_multiple"]]

# CONTROL FLIES
# control fly 1 - several ROI name options here: mi1_proximal_multiple_lessbi mi1_proximal_multiple_morebi
mi1_control1_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "4", "mi1_proximal_multiple_morebi"]]
mi1_control1_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "4", "mi1_medial_multiple"]]
mi1_control1_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "4", "mi1_distal_multiple"]]
# control fly 2
mi1_control2_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "5", "mi1_proximal_multiple"]]



mi1_prox_all = np.concatenate(
                             (mi1_fly1_prox, mi1_fly2_prox, mi1_fly3_prox, 
                             mi1_fly4_prox, mi1_fly5_prox, mi1_fly6_prox,
                             mi1_fly7_prox, mi1_fly8_prox, mi1_fly9_prox,
                             mi1_fly10_prox, mi1_fly11_prox, mi1_fly12_prox,
                             mi1_fly13_prox, mi1_fly14_prox,), 
                             axis = 0,
                            )

mi1_prox_max = np.concatenate(
                             (mi1_fly4_prox, mi1_fly5_prox, mi1_fly6_prox,
                             mi1_fly7_prox, mi1_fly8_prox, mi1_fly9_prox,
                             mi1_fly10_prox, mi1_fly11_prox, mi1_fly12_prox,
                             mi1_fly13_prox, mi1_fly14_prox,),
                             axis = 0,
                             )

mi1_medi_all = np.concatenate(
                       (mi1_fly1_medi, mi1_fly2_medi, mi1_fly3_medi,mi1_fly4_medi, mi1_fly5_medi, mi1_fly6_medi,), 
                        axis = 0,
                      )
mi1_dist_all = np.concatenate(
                       (mi1_fly1_dist, mi1_fly2_dist, mi1_fly3_dist, mi1_fly4_dist, mi1_fly5_dist, mi1_fly6_dist,), 
                        axis = 0,
                      )
mi1_all_multiple = np.concatenate(
                                  (mi1_fly1_prox, mi1_fly2_prox, mi1_fly3_prox, mi1_fly4_prox, mi1_fly5_prox, mi1_fly6_prox, 
                                   mi1_fly1_medi, mi1_fly2_medi, mi1_fly3_medi, mi1_fly4_medi, mi1_fly5_medi, mi1_fly6_medi,
                                   mi1_fly1_dist, mi1_fly2_dist, mi1_fly3_dist, mi1_fly4_dist, mi1_fly5_dist, mi1_fly6_dist,),
                                   axis = 0,
                                 )

mi1_control_prox = np.concatenate(
                                  (mi1_control1_prox, mi1_control2_prox,),
                                  axis = 0,
                                 )


# Single ROI
# Fly 1
mi1_fly1_prox_single = [["/Volumes/ABK2TBData/data_repo/bruker/20221122.common_moco", "2022-11-22", "3", "proximal_single"]]
mi1_fly1_medi_single = [["/Volumes/ABK2TBData/data_repo/bruker/20221122.common_moco", "2022-11-22", "3", "medial_single"]]
mi1_fly1_dist_single = [["/Volumes/ABK2TBData/data_repo/bruker/20221122.common_moco", "2022-11-22", "3", "distal_single"]]
# Fly 2
mi1_fly2_prox_single = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "proximal_single"]]
mi1_fly2_medi_single = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "medial_single"]] 
mi1_fly2_dist_single = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "distal_single"]]

mi1_prox_all_single = np.concatenate(
                       (mi1_fly1_prox_single, mi1_fly2_prox_single,), 
                        axis = 0,
                       )

mi1_medi_all_single = np.concatenate(
                       (mi1_fly1_medi_single, mi1_fly2_medi_single,), 
                        axis = 0,
                      )

mi1_dist_all_single = np.concatenate(
                       (mi1_fly1_dist_single, mi1_fly2_dist_single,), 
                        axis = 0,
                      )
mi1_all_single = np.concatenate(
                                (mi1_fly1_prox_single, mi1_fly2_prox_single, mi1_fly1_medi_single, mi1_fly2_medi_single,
                                 mi1_fly1_dist_single, mi1_fly2_dist_single,),
                                 axis = 0,
                               )


condition_name = 'current_led_intensity'

# %% Functions
# find the visual flash locations (for plotting)
def visFlash(ID):
    pre_time = ID.getRunParameters('pre_time')
    flash_times = ID.getRunParameters('flash_times')
    flash_width = ID.getRunParameters('flash_width')
    
    flash_start = flash_times + pre_time
    flash_end = flash_start + flash_width
    
    return flash_start, flash_end

save_directory = "/Volumes/ABK2TBData/lab_repo/analysis/outputs/flash_w_opto_step/" #+ experiment_file_name + "/"
Path(save_directory).mkdir(exist_ok=True)

# %%

# do outer loop here...

#all_response_max = []


# Which one to plot
# (0) mi1_fly1_prox (1) mi1_fly2_prox (2) mi1_fly1_medi (3) mi1_fly2_medi (4) mi1_fly1_dist (5) mi1_fly2_dist
#pull_ind = 14
save_fig = False
which_layer = mi1_prox_max

# Make the metric arrays - each will be experiment x unique opto params x visual flash window
n_exps = len(which_layer)
n_vis_flashes = 4 #hardcoded, sorry
n_opto_params = 3
mean_matrix = np.empty((n_exps, n_opto_params, n_vis_flashes))
mean_matrix[:] = np.nan
max_matrix = np.empty((n_exps, n_opto_params, n_vis_flashes))
max_matrix[:] = np.nan
min_matrix = np.empty((n_exps, n_opto_params, n_vis_flashes))
min_matrix[:] = np.nan
ptt_matrix = np.empty((n_exps, n_opto_params, n_vis_flashes))
ptt_matrix[:] = np.nan


for pull_ind in range(len(which_layer)):
    file_path = os.path.join(which_layer[pull_ind][0], which_layer[pull_ind][1] + ".hdf5")
    ID = imaging_data.ImagingDataObject(file_path, which_layer[pull_ind][2], quiet=True)
    roi_data = ID.getRoiResponses(which_layer[pull_ind][3], background_roi_name='bg_proximal_lessbi', background_subtraction=False)


    # Plot the average Traces of the whole trial followed by the avg traces of the windows

    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key='current_led_intensity')
    # ('current_led_intensity', 'current_led_duration')
    # calc the sem + / -
    sem_plus = mean_response + sem_response
    sem_minus = mean_response - sem_response
    trial_timepoints = range(len(roi_data['time_vector']))

    # finding vis flash locations 
    flash_start, flash_end = visFlash(ID)
    min_val = np.min(sem_minus.mean(axis=0))
    max_val = np.max(sem_plus.mean(axis=0))
    y_low = min_val-abs(0.05*min_val)
    y_high = max_val+abs(0.05*max_val)

    # Colormap setting
    cmap = plt.get_cmap('cool') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.1, 1.0, len(unique_parameter_values))]

    # Plotting the whole trace
    fh, ax = plt.subplots(1, 1, figsize=(16, 8))
    for up_ind, up in enumerate(unique_parameter_values): # up = unique parameter
        ax.plot(roi_data['time_vector'], mean_response[:, up_ind, :].mean(axis=0), color=colors[up_ind], alpha=0.9, label=up)
        ax.fill_between(roi_data['time_vector'], sem_plus[:, up_ind, :].mean(axis=0), 
                        sem_minus[:, up_ind, :].mean(axis=0),
                        color=colors[up_ind], alpha=0.1)
    # opto stim plotting
    led_start_time = ID.getRunParameters('pre_time')+ID.getRunParameters('led_time')
    led_end_time = led_start_time + ID.getRunParameters('led_duration')        
    ax.fill_between([led_start_time, led_end_time], y_low*1.01, y_high*1.01, 
                    alpha=0.5, edgecolor='r', facecolor='none', linewidth=3, label='Opto')
    # vis stim plotting
    for vis_ind in range(len(flash_start)):
        if vis_ind < 1:
            flash_label = 'Vis'
        else:
            flash_label = None
        ax.fill_between([flash_start[vis_ind], flash_end[vis_ind]], 
                        y_low, y_high,
                        alpha=0.6, edgecolor='b', facecolor='none', 
                        linewidth=1, label=flash_label)

    # Legend, Grid, Axis
    ax.legend(loc="upper right", fontsize=15)
    ax.grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
    x_locator = FixedLocator(list(range(-1, 20)))
    ax.xaxis.set_major_locator(x_locator)
    ax.tick_params(axis="x", direction="in", length=10, width=1, color="k")
    ax.grid(axis="y", color="k", alpha=.1, linewidth=.5)
    ax.set_xlabel('Time in Seconds')
    ax.set_ylabel('DF/F')
    ax.set_title(f'{which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=20)

    if save_fig == True:
        fh.savefig(
        save_directory
        + "AvgTraces."
        + str(which_layer[pull_ind][1])
        + ".Series"
        + str(which_layer[pull_ind][2])
        + ".ROI"
        + str(which_layer[pull_ind][3])
        + ".Conditions:"
        + str(condition_name)
        + ".pdf",
        dpi=300,
        )



    #  Windows to analyze for metrics
    flash_start = ID.getRunParameters('flash_times') + ID.getRunParameters('pre_time')
    flash_width = ID.getRunParameters('flash_width')
    window_lag = 0.25  # sec
    window_length = flash_width + 1.3  # sec
    window_times = flash_start - window_lag
    window_frames = int(np.ceil(window_length / ID.getResponseTiming().get('sample_period')))
    windows = np.zeros((len(unique_parameter_values), len(window_times), window_frames))
    windows_sem = np.zeros((len(unique_parameter_values), len(window_times), window_frames))
    num_windows = len(window_times)


    fh, ax = plt.subplots(1, len(unique_parameter_values), figsize=(18, 4))

    # Collect windowed responses
    cmap = plt.get_cmap('viridis') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.0, 1.0, num_windows)]
    for up_ind, up in enumerate(unique_parameter_values): # Opto intensities
        for w_ind, w in enumerate(window_times): # windows
            start_index = np.where(roi_data.get('time_vector') > window_times[w_ind])[0][0]
            windows[up_ind, w_ind, :] = mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
            windows_sem[up_ind, w_ind, :] = sem_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)

            # Plot: Each Window for a given LED Intensity
            ax[up_ind].plot(windows[up_ind, w_ind, :], color=colors[w_ind], label=w if up_ind==0 else '')
            ax[up_ind].set_title('led={}'.format(up))


    fh.legend()
    fh.suptitle(f'Windows for {which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=12)

    if save_fig == True:
        fh.savefig(
        save_directory
        + "Windows."
        + str(which_layer[pull_ind][1])
        + ".Series"
        + str(which_layer[pull_ind][2])
        + ".ROI"
        + str(which_layer[pull_ind][3])
        + ".Conditions:"
        + str(condition_name)
        + ".pdf",
        dpi=300,
        )

    #%  Plot each LED intensity for a given window
    fh, ax = plt.subplots(1, len(window_times), figsize=(20, 4))
    # Plot windowed responses
    cmap = plt.get_cmap('cool') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'#colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]
    colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]

    # Setting the values for all axes.
    custom_ylim = (y_low, y_high)
    plt.setp(ax, ylim=custom_ylim)

    for w_ind, w in enumerate(window_times):
        for up_ind, up in enumerate(unique_parameter_values):
            ax[w_ind].plot(windows[up_ind, w_ind], linewidth=2, color=colors[up_ind], alpha=0.8, label=up if w_ind==0 else '')
            ax[w_ind].set_title('Visual Flash at: {}s'.format(w))

    fh.legend()
    fh.suptitle(f'Each LED intenisty/window for {which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=12)

    if save_fig == True:
        fh.savefig(
        save_directory
        + "LED.Intensity.window."
        + str(which_layer[pull_ind][1])
        + ".Series"
        + str(which_layer[pull_ind][2])
        + ".ROI"
        + str(which_layer[pull_ind][3])
        + ".Conditions:"
        + str(condition_name)
        + ".pdf",
        dpi=300,
        )

    # % Plotting the metrics for the windows and then store those values
    response_mean = np.mean(windows, axis = -1)
    response_max = np.max(windows, axis = -1)
    response_min = np.min(windows, axis = -1)
    response_PtT = response_max - response_min
    # storage time!
    mean_matrix[pull_ind] = response_mean
    max_matrix[pull_ind] = response_max
    min_matrix[pull_ind] = response_min
    ptt_matrix[pull_ind] = response_PtT

    cmap = plt.get_cmap('cool') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]

    fh, ax = plt.subplots(3, len(window_times)-1, figsize=(16, 12))
    # Setting the values for all axes.
    #custom_ylim = (y_low, y_high)
    #plt.setp(ax, ylim=custom_ylim)

    # temp values to be used to set axes
    # temp_lower_max = 0
    # temp_upper_max = 1
    # temp_lower_min = 0
    # temp_upper_min = 1

    for w_ind in range(len(window_times)-1): # w_ind = window_indicies

        for up_ind, up in enumerate(unique_parameter_values):
            # Maximums for top row
            ax[0, w_ind].plot(response_max[up_ind, w_ind], response_max[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o', label=up if w_ind==0 else '')
            # Minimums for middle row
            ax[1, w_ind].plot(response_min[up_ind, w_ind], response_min[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o')
            # Peak to trough for bottom row
            ax[2, w_ind].plot(response_PtT[up_ind, w_ind], response_PtT[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o')
            #ax[2, w_ind].plot(response_PtT[up_ind, w_ind], response_PtT(up_ind, w_ind+1), color=colors[up_ind], markersize=10, marker='o')

            ax[0, w_ind].set_title(f'Visual Flash {w_ind+2} | Window Time: {window_times[w_ind+1]}')

        # Finding unity params - Max
        unity_lower_max = min(min(response_max[:, w_ind]), min(response_max[:, w_ind+1]))*0.8
        unity_upper_max = max(max(response_max[:, w_ind]), max(response_max[:, w_ind+1]))*1.2
        #ax[0, w_ind].plot([unity_lower_max, unity_upper_max], [unity_lower_max, unity_upper_max], 'k--', alpha=0.7)
        if w_ind == 0:
            temp_lower_max = unity_lower_max
            temp_upper_max = unity_upper_max

        if temp_lower_max > unity_lower_max:
            temp_lower_max = unity_lower_max
        if temp_upper_max < unity_upper_max:
            temp_upper_max = unity_upper_max

        # Finding unity params - Min
        unity_lower_min_raw = min(min(response_min[:, w_ind]), min(response_min[:, w_ind+1]))
        unity_lower_min = unity_lower_min_raw - abs(0.2*unity_lower_min_raw)
        unity_upper_min_raw = max(max(response_min[:, w_ind]), max(response_min[:, w_ind+1]))
        unity_upper_min = unity_upper_min_raw + abs(0.2*unity_upper_min_raw)
        #ax[1, w_ind].plot([unity_lower_min, unity_upper_min], [unity_lower_min, unity_upper_min], 'k--', alpha=0.7)
        if w_ind == 0:
            temp_lower_min = unity_lower_min
            temp_upper_min = unity_upper_min
        if temp_lower_min  > unity_lower_min:
            temp_lower_min = unity_lower_min
        if temp_upper_min < unity_upper_min:
            temp_upper_min = unity_upper_min

        # Finding unity params - PtT
        unity_lower_PtT = min(min(response_PtT[:, w_ind]), min(response_PtT[:, w_ind+1]))*0.8
        unity_upper_PtT = max(max(response_PtT[:, w_ind]), max(response_PtT[:, w_ind+1]))*1.2
        #ax[0, w_ind].plot([unity_lower_PtT, unity_upper_PtT], [unity_lower_PtT, unity_upper_PtT], 'k--', alpha=0.7)
        if w_ind == 0:
            temp_lower_PtT = unity_lower_PtT
            temp_upper_PtT = unity_upper_PtT

        if temp_lower_PtT > unity_lower_PtT:
            temp_lower_PtT = unity_lower_PtT
        if temp_upper_PtT < unity_upper_PtT:
            temp_upper_PtT = unity_upper_PtT


        ax[0, w_ind].set_xlabel('Visual Flash 1 (Pre-Opto) Peak Amplitude')
        ax[1, w_ind].set_xlabel('Visual Flash 1 (Pre-Opto) Trough Amplitude')
        ax[2, w_ind].set_xlabel('Visual Flash 1 (Pre-Opto) Peak-Trough')
    
    for w_ind in range(len(window_times)-1): # sets all the axes to be the same
        ax[0,w_ind].set_xlim(left=temp_lower_max, right=temp_upper_max)
        ax[0,w_ind].set_ylim(bottom=temp_lower_max, top=temp_upper_max)
        ax[0,w_ind].plot([temp_lower_max, temp_upper_max], [temp_lower_max, temp_upper_max], 'k--', alpha=0.7)

        ax[1,w_ind].set_xlim(left=temp_lower_min, right=temp_upper_min)
        ax[1,w_ind].set_ylim(bottom=temp_lower_min, top=temp_upper_min)
        ax[1,w_ind].plot([temp_lower_min, temp_upper_min], [temp_lower_min, temp_upper_min], 'k--', alpha=0.7)

        ax[2,w_ind].set_xlim(left=temp_lower_PtT, right=temp_upper_PtT)
        ax[2,w_ind].set_ylim(bottom=temp_lower_PtT, top=temp_upper_PtT)
        ax[2,w_ind].plot([temp_lower_PtT, temp_upper_PtT], [temp_lower_PtT, temp_upper_PtT], 'k--', alpha=0.7)

    #ax[0,0].setp(xlim=temp_upper_max, ylim=temp_upper_max)
    #ax[0,1].setp(xlim=temp_upper_max, ylim=temp_upper_max)

    ax[0, 0].set_ylabel('Comparison Visual Flash Peak Amplitude')
    ax[1, 0].set_ylabel('Comparison Visual Flash Trough Amplitude')
    ax[2, 0].set_ylabel('Comparison Visual Flash Peak-Trough')

    #ax.set_ylabel(row, rotation=0, size='large
    #fh.tight_layout()
    fh.legend(title = 'Opto Intensities')
    fh.suptitle(f'Windows Metrics for {which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=15)

    if save_fig == True:
        fh.savefig(
        save_directory
        + "WindowsMetrics."
        + str(which_layer[pull_ind][1])
        + ".Series"
        + str(which_layer[pull_ind][2])
        + ".ROI"
        + str(which_layer[pull_ind][3])
        + ".Conditions:"
        + str(condition_name)
        + ".pdf",
        dpi=300,
        )

#plt.close('all')

# %% Plotting the cross-animal data

# first make a function that gives normalized differences for each metric
def metricDifNormalizer(metric_in):
    # initialize nan-filled output matrix
    metric_out = np.empty((metric_in.shape[0], metric_in.shape[1], metric_in.shape[2]-1))
    metric_out[:] = np.nan

    for win_ind in range(mean_matrix.shape[-1] - 1):

        metric_out[:, :, win_ind] = (metric_in[:, :, win_ind + 1] - metric_in[:, :, 0]) / (metric_in[:, :, win_ind + 1] + metric_in[:, :, 0])

    return metric_out

mean_diff_matrix = metricDifNormalizer(mean_matrix)
max_diff_matrix = metricDifNormalizer(max_matrix)
min_diff_matrix = metricDifNormalizer(min_matrix)
ptt_diff_matrix = metricDifNormalizer(ptt_matrix)

# Histograms of these values
# reminder: each matrix be experiment x unique opto params x visual flash window
# NOTE: this needs unique_parameter_values
fh_mean, ax_mean = plt.subplots(3, len(window_times)-1, figsize=(16, 12))
fh_max, ax_max = plt.subplots(3, len(window_times)-1, figsize=(16, 12))
fh_min, ax_min = plt.subplots(3, len(window_times)-1, figsize=(16, 12))
fh_ptt, ax_ptt = plt.subplots(3, len(window_times)-1, figsize=(16, 12))

for win_ind in range(mean_dif_matrix.shape[2]):
    for opto_ind in range(mean_dif_matrix.shape[1]):
        ax_mean[opto_ind, win_ind].hist(mean_diff_matrix[:, opto_ind, win_ind])
        ax_max[opto_ind, win_ind].hist(max_diff_matrix[:, opto_ind, win_ind])
        ax_min[opto_ind, win_ind].hist(min_diff_matrix[:, opto_ind, win_ind])
        ax_ptt[opto_ind, win_ind].hist(ptt_diff_matrix[:, opto_ind, win_ind])

        ax_mean[opto_ind, win_ind].set_title(f'Opto intensity = {unique_parameter_values[opto_ind]} | Vis Flash {win_ind+2} - Vis Flash 1')
        ax_max[opto_ind, win_ind].set_title(f'Opto intensity = {unique_parameter_values[opto_ind]} | Vis Flash {win_ind+2} - Vis Flash 1')
        ax_min[opto_ind, win_ind].set_title(f'Opto intensity = {unique_parameter_values[opto_ind]} | Vis Flash {win_ind+2} - Vis Flash 1')
        ax_ptt[opto_ind, win_ind].set_title(f'Opto intensity = {unique_parameter_values[opto_ind]} | Vis Flash {win_ind+2} - Vis Flash 1')

fh_mean.suptitle(f'Mean Value of X Window - Mean Value of First Window')
fh_max.suptitle(f'Max Value of X Window - Max Value of First Window')
fh_min.suptitle(f'Min Value of X Window - Min Value of First Window')
fh_ptt.suptitle(f'Peak-to-Trough Value of X Window - Peak-to-Trough Value of First Window')

# %%
