# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats

import os
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd

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
mi1_fly9_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "4", "mi1_medial_multiple"]]
mi1_fly9_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "4", "mi1_distal_multiple"]]
# Fly 10
mi1_fly10_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "6", "mi1_proximal_multiple"]]
mi1_fly10_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "6", "mi1_medial_multiple"]]
mi1_fly10_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "6", "mi1_distal_multiple_lowcon"]]
# Fly 11
mi1_fly11_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230316", "2023-03-16", "7", "mi1_proximal_multiple"]]
# Fly 12
mi1_fly12_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "1", "mi1_proximal_multiple"]]
mi1_fly12_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "1", "mi1_medial_multiple"]]
mi1_fly12_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "1", "mi1_distal_multiple"]]
# Fly 13 # kinda shitty b/c motion
mi1_fly13_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "2", "mi1_proximal_multiple"]]
# Fly 14 # kinda shitty b/c motion
mi1_fly14_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "3", "mi1_proximal_multiple"]]
# Fly 15
mi1_fly15_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230403.moco", "2023-04-03", "1", "mi1_proximal_multiple"]]
# Fly 16
mi1_fly16_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230403.moco", "2023-04-03", "3", "mi1_proximal_multiple"]]
# Fly 17
mi1_fly17_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230403.moco", "2023-04-03", "4", "mi1_proximal_multiple"]]


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
                             mi1_fly13_prox, mi1_fly14_prox, mi1_fly15_prox,), 
                             axis = 0,
                            )

mi1_prox_good = np.concatenate(
                             (mi1_fly4_prox, mi1_fly5_prox, mi1_fly6_prox,
                             mi1_fly7_prox, mi1_fly8_prox, mi1_fly9_prox,
                             mi1_fly10_prox, mi1_fly11_prox, mi1_fly12_prox,
                             mi1_fly13_prox, mi1_fly14_prox, mi1_fly15_prox, 
                             mi1_fly16_prox, mi1_fly17_prox,),
                             axis = 0,
                             )
fly_list_prox = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

mi1_dist_good = np.concatenate(
                             (mi1_fly4_dist, mi1_fly5_dist, mi1_fly6_dist,
                             mi1_fly9_dist, mi1_fly10_dist, mi1_fly12_dist,),
                             axis = 0,
                             )
fly_list_dist = [4, 5, 6, 9, 10, 12]

mi1_medi_good = np.concatenate(
                             (mi1_fly4_medi, mi1_fly5_medi, mi1_fly6_medi,
                             mi1_fly9_medi, mi1_fly10_medi, mi1_fly12_medi,),
                             axis = 0,
                             )
fly_list_medi = [4, 5, 6, 9, 10, 12]
                             
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

# control flies
mi1_control_prox = np.concatenate(
                                  (mi1_control1_prox, mi1_control2_prox,),
                                  axis = 0,
                                 )
fly_list_control_prox = [1, 2]
fly_list_control_medi = [1]
fly_list_control_dist = [1]

# all good flies
mi1_all_good = [mi1_prox_good, mi1_medi_good, mi1_dist_good]

#all flies for mi1_control
mi1_control_all = [mi1_control_prox, mi1_control1_medi, mi1_control1_dist]

# Hardcoded fly indecies. Must be updated above when fly identies added/changed:
fly_list_exp = [fly_list_prox, fly_list_medi, fly_list_dist]
fly_list_control = [fly_list_control_prox, fly_list_control_medi, fly_list_control_dist]
layer_list = ('Proximal', 'Medial', 'Distal')

# Housekeeping:
condition_name = 'current_led_intensity'
save_directory = "/Volumes/ABK2TBData/lab_repo/analysis/outputs/flash_w_opto_step/" #+ experiment_file_name + "/"
Path(save_directory).mkdir(exist_ok=True)


#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
#-------------------------------------Function Definitions-----------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#

# find the visual flash locations (for plotting)
def visFlash(ID):
    pre_time = ID.getRunParameters('pre_time')
    flash_times = ID.getRunParameters('flash_times')
    flash_width = ID.getRunParameters('flash_width')
    
    flash_start = flash_times + pre_time
    flash_end = flash_start + flash_width
    
    return flash_start, flash_end

# first make a function that gives normalized differences for each metric
def metricDifNormalizer(metric_in, normalize_to = "sum"):
    # initialize nan-filled output matrix
    metric_out = np.empty((metric_in.shape[0], metric_in.shape[1], metric_in.shape[2]-1))
    metric_out[:] = np.nan

    #take the absolute of the metric. This helps catch the min case. Everything else should be pos all the time anyway
    #metric_in = np.absolute(metric_in)

    for win_ind in range(metric_in.shape[-1] - 1):
        if normalize_to == "sum":
            metric_out[:, :, win_ind] = (metric_in[:, :, win_ind + 1] - metric_in[:, :, 0]) / (np.absolute(metric_in[:, :, win_ind + 1]) +np.absolute(metric_in[:, :, 0]))
        elif normalize_to == "first":
            metric_out[:, :, win_ind] = (metric_in[:, :, win_ind + 1] - metric_in[:, :, 0]) / (metric_in[:, :, 0])
    return metric_out

# Giant function that takes in layer and outputs metric matrices
def getWindowMetricsFromLayer(layer, condition_name, normalize_to=False, plot_trial_figs=False, save_fig=False):
    which_layer = layer
    # Make the metric arrays - each will be experiment x unique opto params x visual flash window
    n_exps = len(which_layer)
    n_vis_flashes = 4 #hardcoded, sorry
    n_opto_params = 3
    mean_matrix = np.empty((n_exps, n_opto_params, n_vis_flashes))
    mean_matrix[:] = np.nan
    sem_mean_matrix = np.empty((n_exps, n_opto_params, n_vis_flashes))
    sem_mean_matrix[:] = np.nan
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

        if plot_trial_figs == True:
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

        if plot_trial_figs == True:
            fh, ax = plt.subplots(1, len(unique_parameter_values), figsize=(18, 4))

        # Collect windowed responses
        cmap = plt.get_cmap('viridis') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
        colors = [cmap(i) for i in np.linspace(0.0, 1.0, num_windows)]
        for up_ind, up in enumerate(unique_parameter_values): # Opto intensities
            for w_ind, w in enumerate(window_times): # windows
                start_index = np.where(roi_data.get('time_vector') > window_times[w_ind])[0][0]
                windows[up_ind, w_ind, :] = mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
                windows_sem[up_ind, w_ind, :] = sem_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)

                if plot_trial_figs == True:
                    # Plot: Each Window for a given LED Intensity
                    ax[up_ind].plot(windows[up_ind, w_ind, :], color=colors[w_ind], label=w if up_ind==0 else '')
                    ax[up_ind].set_title('led={}'.format(up))

        if plot_trial_figs == True:
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
        
        if plot_trial_figs == True:
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
        # This step takes windows (averaged across ROIs already) = UniqueOptoParams X Window X Time) and averages across time, so:
        # response_mean = Unique Opto Params x Windows
        response_mean = np.mean(windows, axis = -1)
        # logic here for sem is that it's already a series of sem's across time. Take the avg to find 1 value per  UOPxWindow
        response_sem = np.mean(windows_sem, axis = -1)
        response_max = np.max(windows, axis = -1)
        response_min = np.min(windows, axis = -1)
        response_PtT = response_max - response_min

        # storage time!
        mean_matrix[pull_ind] = response_mean
        sem_mean_matrix[pull_ind] = response_sem
        max_matrix[pull_ind] = response_max
        min_matrix[pull_ind] = response_min
        ptt_matrix[pull_ind] = response_PtT

        if plot_trial_figs == True:
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
                    ax[0, w_ind].plot(response_max[up_ind, 0], response_max[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o', label=up if w_ind==0 else '')
                    # Minimums for middle row
                    ax[1, w_ind].plot(response_min[up_ind, 0], response_min[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o')
                    # Peak to trough for bottom row
                    ax[2, w_ind].plot(response_PtT[up_ind, 0], response_PtT[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o')
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
    
    if normalize_to == 'sum':
        mean_matrix = metricDifNormalizer(mean_matrix, normalize_to)
        sem_mean_matrix = metricDifNormalizer(sem_mean_matrix, normalize_to)
        max_matrix = metricDifNormalizer(max_matrix, normalize_to)
        min_matrix = metricDifNormalizer(min_matrix, normalize_to)
        ptt_matrix = metricDifNormalizer(ptt_matrix, normalize_to)
    elif normalize_to == 'first':
        mean_matrix = metricDifNormalizer(mean_matrix, normalize_to)
        sem_mean_matrix = metricDifNormalizer(sem_mean_matrix, normalize_to)
        max_matrix = metricDifNormalizer(max_matrix, normalize_to)
        min_matrix = metricDifNormalizer(min_matrix, normalize_to)
        ptt_matrix = metricDifNormalizer(ptt_matrix, normalize_to)
    else:
        print("NO NORMALIZING HAPPENED BTW")

    return mean_matrix, sem_mean_matrix, max_matrix, min_matrix, ptt_matrix

# %% RUN ALL THE SHIT
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
#----------------------------------------RUN All That Shit-----------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Set meeeeeeeeeeee
data_list = mi1_control_all # mi1_all_good | mi1_control_all
list_to_use = fly_list_control # fly_list_exp | fly_list_control

# Making a data frame the way it's supposed to be....
# Currently have:
# metric_matrix = Fly x unique_opto_param x window which is 11 x 3 x 3
# going for: 
# Index | Fly | Mean | Max | Etc Metric | Unique_opto_param | Window

# Loop through everything!

# Defines dataframe with desired columns
metric_df = pd.DataFrame(columns=['Fly', 'Layer', 'Mean', 'SEM_Mean', 'Min', 'Max', 'PtT', 'Opto', 'Window'])

# Adds all metrics to dataframe one row at a time
row_idx = 0
# Print statement that explains loop is starting
print(f"\nBeginning Loops for extracting data.")
for layer_ind in range(len(layer_list)):
    mean_diff_matrix, sem_mean_diff_matrix, max_diff_matrix, min_diff_matrix, ptt_diff_matrix = getWindowMetricsFromLayer(data_list[layer_ind], layer_list[layer_ind], normalize_to='sum', plot_trial_figs=True, save_fig=False)
    fly_indicies = list_to_use[layer_ind]
    # Gets dimensions of metric arrays 
    num_flies, num_opto, num_windows = mean_diff_matrix.shape   
    for fly in range(num_flies):
        for opto in range(num_opto):
            for window in range(num_windows):
                # print progress
                print(f'Fly: {fly} of {num_flies} | Opto: {opto} | Window: {window}')
                metric_df.loc[row_idx] = [
                    fly_indicies[fly], layer_list[layer_ind], mean_diff_matrix[fly, opto, window], sem_mean_diff_matrix[fly, opto, window],
                    min_diff_matrix[fly, opto, window], max_diff_matrix[fly, opto, window], 
                    ptt_diff_matrix[fly, opto, window], opto, window,
                    ]
                row_idx += 1

# Convert the floats which were indicies to ints           
metric_df['Fly'] = metric_df['Fly'].astype(int)
metric_df['Opto'] = metric_df['Opto'].astype(int)
metric_df['Window'] = metric_df['Window'].astype(int)

print('\n\n------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------')
print('---------------------------------FUCKING DONE---------------------------------------------')
print('------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------')



# %% Seaborn plots for values over windows (Fig 4)
#save_fig=True
# Subplot grid where each row is a metric and each column is a layer
bigfig, bigaxes = plt.subplots(4,3, figsize=(36,27))
bigfig.suptitle(f'PLACEHOLDER SUPERTITLE')

# Proximal = Col 1
# Mean
prox_df = metric_df[metric_df.Layer=='Proximal']
sns.lineplot(
    ax=bigaxes[0,0], x="Window", y="Mean", hue="Opto", data=prox_df, palette="pastel", 
    )
bigaxes[0,0].set(title='Proximal - Mean')
# Max
sns.lineplot(
    ax=bigaxes[1,0], x="Window", y="Max", hue="Opto", data=prox_df, palette="pastel"
    )
bigaxes[1,0].set(title='Proximal - Max')
# Min
sns.lineplot(
    ax=bigaxes[2,0], x="Window", y="Min", hue="Opto", data=prox_df, palette="pastel"
    )
bigaxes[2,0].set(title='Proximal - Min')
# PtT
sns.lineplot(
    ax=bigaxes[3,0], x="Window", y="PtT", hue="Opto", data=prox_df, palette="pastel"
    )
bigaxes[3,0].set(title='Proximal - Peak-to-Trough')

# Medial = Col 2
# Mean
med_df = metric_df[metric_df.Layer=='Medial']
sns.lineplot(
    ax=bigaxes[0,1], x="Window", y="Mean", hue="Opto", data=med_df, palette="pastel"
    )
bigaxes[0,1].set(title='Medial - Mean')
# Max
sns.lineplot(
    ax=bigaxes[1,1], x="Window", y="Max", hue="Opto", data=med_df, palette="pastel"
    )
bigaxes[1,1].set(title='Medial - Max')
# Min
sns.lineplot(
    ax=bigaxes[2,1], x="Window", y="Min", hue="Opto", data=med_df, palette="pastel"
    )
bigaxes[2,1].set(title='Medial - Min')
# PtT
sns.lineplot(
    ax=bigaxes[3,1], x="Window", y="PtT", hue="Opto", data=med_df, palette="pastel"
    )
bigaxes[3,1].set(title='Medial - Peak-to-Trough')

# Distal = Col 3
# Mean
dist_df = metric_df[metric_df.Layer=='Medial']
sns.lineplot(
    ax=bigaxes[0,2], x="Window", y="Mean", hue="Opto", data=dist_df, palette="pastel"
    )
bigaxes[0,2].set(title='Distal - Mean')
# Max
sns.lineplot(
    ax=bigaxes[1,2], x="Window", y="Max", hue="Opto", data=dist_df, palette="pastel"
    )
bigaxes[1,2].set(title='Distal - Max')
# Min
sns.lineplot(
    ax=bigaxes[2,2], x="Window", y="Min", hue="Opto", data=dist_df, palette="pastel"
    )
bigaxes[2,2].set(title='Distal - Min')
# PtT
sns.lineplot(
    ax=bigaxes[3,2], x="Window", y="PtT", hue="Opto", data=dist_df, palette="pastel"
    )
bigaxes[3,2].set(title='Distal - Peak-to-Trough')

# Set for whole plot
plt.setp(bigaxes, xticks=[0, 1, 2], xticklabels=['Second-First', 'Third-First', 'Fourth-First'])

if save_fig == True:
    bigfig.savefig(
    save_directory
    + "Cross-Fly.Cross-Layer.control.LinePlot."
    + ".pdf",
    dpi=300,
    )



# %% Plotting the cross-animal data
#savefig = True
# normalize to value:
normalize_to = "sum" # to divide the diff of windows by the sum of the windows

mean_diff_matrix = metricDifNormalizer(mean_matrix, normalize_to)
sem_mean_diff_matrix = metricDifNormalizer(sem_mean_matrix, normalize_to)
max_diff_matrix = metricDifNormalizer(max_matrix, normalize_to)
min_diff_matrix = metricDifNormalizer(min_matrix, normalize_to)
ptt_diff_matrix = metricDifNormalizer(ptt_matrix, normalize_to)


# flatten everything, have list of names, add it all as row with columns in pd
m_shape = max_diff_matrix.shape
flattened_means = np.reshape(mean_diff_matrix, (m_shape[0], m_shape[1]*m_shape[2]))
mean_labels = ['Mean_opto0_win01', 'Mean_opto0_win02', 'Mean_opto0_win03', 
               'Mean_opto1_win01', 'Mean_opto1_win02', 'Mean_opto1_win03',
               'Mean_opto2_win01', 'Mean_opto2_win02', 'Mean_opto2_win03']
flattened_sems = np.reshape(sem_mean_diff_matrix, (m_shape[0], m_shape[1]*m_shape[2]))
sem_labels = ['SEM_opto0_win01', 'SEM_opto0_win02', 'SEM_opto0_win03', 
               'SEM_opto1_win01', 'SEM_opto1_win02', 'SEM_opto1_win03',
               'SEM_opto2_win01', 'SEM_opto2_win02', 'SEM_opto2_win03']

flattened_maxes = np.reshape(max_diff_matrix, (m_shape[0], m_shape[1]*m_shape[2]))
max_labels = ['Max_opto0_win01', 'Max_opto0_win02', 'Max_opto0_win03', 
               'Max_opto1_win01', 'Max_opto1_win02', 'Max_opto1_win03',
               'Max_opto2_win01', 'Max_opto2_win02', 'Max_opto2_win03']
flattened_mins = np.reshape(min_diff_matrix, (m_shape[0], m_shape[1]*m_shape[2]))
min_labels = ['Min_opto0_win01', 'Min_opto0_win02', 'Min_opto0_win03', 
               'Min_opto1_win01', 'Min_opto1_win02', 'Min_opto1_win03',
               'Min_opto2_win01', 'Min_opto2_win02', 'Min_opto2_win03']
flattened_ptts = np.reshape(ptt_diff_matrix, (m_shape[0], m_shape[1]*m_shape[2]))
PtT_labels = ['PtT_opto0_win01', 'PtT_opto0_win02', 'PtT_opto0_win03', 
               'PtT_opto1_win01', 'PtT_opto1_win02', 'PtT_opto1_win03',
               'PtT_opto2_win01', 'PtT_opto2_win02', 'PtT_opto2_win03']
# Set it all up to be made into DF
concat_met = np.concatenate((flattened_means, flattened_sems, flattened_maxes, flattened_mins, flattened_ptts), axis=1)
concat_labels = mean_labels + sem_labels + max_labels + min_labels + PtT_labels
# Make that DataFrame
big_daddy_df = pd.DataFrame(data=concat_met, columns=concat_labels)


############################################################################################################
#               Seaborn plots of all metrics in histogram form for:                                        #
############################################################################################################

# Initialize some shit:
kde = True
element = 'poly' # bars | poly | step
multiple = 'stack' # dodge | stack | layer | fill
bins = 10
binwidth = 0.05 #0.025 # supercedes bins |  0.025
savefig = False

# Second window - First Window
fig2, axes2 = plt.subplots(2, 2, figsize = (12,9))
fig2.suptitle(f'{layer_name} Metrics for Comparing The First and Second Visual Rsponses Across Flies')
# Maxes
df_maxes = big_daddy_df[['Max_opto0_win01', 'Max_opto1_win01', 'Max_opto2_win01']]
sns.histplot(ax=axes2[0,0], data=df_maxes, element=element, kde=kde, multiple=multiple, binwidth=binwidth, bins=bins)
axes2[0,0].set_title(f'Max Value of Second Window - Max Value of First Window')
# Mins
df_mins = big_daddy_df[['Min_opto0_win01', 'Min_opto1_win01', 'Min_opto2_win01']]
sns.histplot(ax=axes2[0,1], data=df_mins, element=element, kde=kde, multiple=multiple, bins=bins)
axes2[0,1].set_title(f'Min Value of Second Window - Min Value of First Window')
# Peak to Troughs
df_ptt = big_daddy_df[['PtT_opto0_win01', 'PtT_opto1_win01', 'PtT_opto2_win01']]
sns.histplot(ax=axes2[1,0], data=df_ptt, element=element, kde=kde, multiple=multiple, binwidth=binwidth, bins=bins)
axes2[1,0].set_title(f'PtT Value of Second Window - PtT Value of First Window')
# Means
df_means = big_daddy_df[['Mean_opto0_win01', 'Mean_opto1_win01', 'Mean_opto2_win01']]
sns.histplot(ax=axes2[1,1], data=df_means, element=element, kde=kde, multiple=multiple, bins=bins)
axes2[1,1].set_title(f'Mean Value of Second Window - Mean Value of First Window')

# Third window - First Window
fig3, axes3 = plt.subplots(2, 2, figsize = (12,9))
fig3.suptitle(f'{layer_name} Metrics for Comparing The First and Third Visual Rsponses Across Flies')
# Maxes
df_maxes = big_daddy_df[['Max_opto0_win02', 'Max_opto1_win02', 'Max_opto2_win02']]
sns.histplot(ax=axes3[0,0], data=df_maxes, element=element, kde=kde, multiple=multiple, binwidth=binwidth, bins=bins)
axes3[0,0].set_title(f'Max Value of Third Window - Max Value of First Window')
# Mins
df_mins = big_daddy_df[['Min_opto0_win02', 'Min_opto1_win02', 'Min_opto2_win02']]
sns.histplot(ax=axes3[0,1], data=df_mins, element=element, kde=kde, multiple=multiple, bins=bins)
axes3[0,1].set_title(f'Min Value of Third Window - Min Value of First Window')
# Peak to Troughs
df_ptt = big_daddy_df[['PtT_opto0_win02', 'PtT_opto1_win02', 'PtT_opto2_win02']]
sns.histplot(ax=axes3[1,0], data=df_ptt, element=element, kde=kde, multiple=multiple, binwidth=binwidth, bins=bins)
axes3[1,0].set_title(f'PtT Value of Third Window - PtT Value of First Window')
# Means
df_means = big_daddy_df[['Mean_opto0_win02', 'Mean_opto1_win02', 'Mean_opto2_win02']]
sns.histplot(ax=axes3[1,1], data=df_means, element=element, kde=kde, multiple=multiple, bins=bins)
axes3[1,1].set_title(f'Mean Value of Third Window - Mean Value of First Window')

# Fourth window - First Window
fig4, axes4 = plt.subplots(2, 2, figsize = (12,9))
fig4.suptitle(f'{layer_name} Metrics for Comparing The First and Fourth Visual Rsponses Across Flies')
# Maxes
df_maxes = big_daddy_df[['Max_opto0_win03', 'Max_opto1_win03', 'Max_opto2_win03']]
sns.histplot(ax=axes4[0,0], data=df_maxes, element=element, kde=kde, multiple=multiple, binwidth=binwidth, bins=bins)
axes4[0,0].set_title(f'Max Value of Fourth Window - Max Value of First Window')
# Mins
df_mins = big_daddy_df[['Min_opto0_win03', 'Min_opto1_win03', 'Min_opto2_win03']]
sns.histplot(ax=axes4[0,1], data=df_mins, element=element, kde=kde, multiple=multiple, bins=bins)
axes4[0,1].set_title(f'Min Value of Fourth Window - Min Value of First Window')
# Peak to Troughs
df_ptt = big_daddy_df[['PtT_opto0_win03', 'PtT_opto1_win03', 'PtT_opto2_win03']]
sns.histplot(ax=axes4[1,0], data=df_ptt, element=element, kde=kde, multiple=multiple, binwidth=binwidth, bins=bins)
axes4[1,0].set_title(f'PtT Value of Fourth Window - PtT Value of First Window')
# Means
df_means = big_daddy_df[['Mean_opto0_win03', 'Mean_opto1_win03', 'Mean_opto2_win03']]
sns.histplot(ax=axes4[1,1], data=df_means, element=element, kde=kde, multiple=multiple, bins=bins)
axes4[1,1].set_title(f'Mean Value of Fourth Window - Mean Value of First Window')

if save_fig == True:
    fig2.savefig(
    save_directory
    + "Cross-FlyMetricHistograms."
    + layer_name
    + ".2Minus1"
    + ".pdf",
    dpi=300,
    )
    fig3.savefig(
    save_directory
    + "Cross-FlyMetricHistograms."
    + layer_name
    + ".3Minus1"
    + ".pdf",
    dpi=300,
    )
    fig4.savefig(
    save_directory
    + "Cross-FlyMetricHistograms."
    + layer_name
    + ".4Minus1"
    + ".pdf",
    dpi=300,
    )


# %% Close the figs after

plt.close('all')


# %% Deprecated: Histograms of these values
# reminder: each matrix be experiment x unique opto params x visual flash window
# NOTE: this needs unique_parameter_values
fh_mean, ax_mean = plt.subplots(3, len(window_times)-1, figsize=(16, 12))
fh_max, ax_max = plt.subplots(3, len(window_times)-1, figsize=(16, 12))
fh_min, ax_min = plt.subplots(3, len(window_times)-1, figsize=(16, 12))
fh_ptt, ax_ptt = plt.subplots(3, len(window_times)-1, figsize=(16, 12))

for win_ind in range(mean_dif_matrix.shape[2]):
    for opto_ind in range(mean_dif_matrix.shape[1]):
        ax_mean[opto_ind, win_ind].hist(mean_diff_matrix[:, opto_ind, win_ind])
        ax_mean
        #ax_mean.plot(kind = "hist", density = True)
        #ax_mean.plot(kind = "kde")
        ax_max[opto_ind, win_ind].hist(max_diff_matrix[:, opto_ind, win_ind], density = True)
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
