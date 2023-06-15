# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats
from scipy.stats import wilcoxon
import pingouin as pg


import os
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd

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
# Fly 18
mi1_fly18_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230427.moco", "2023-04-27", "1", "mi1_proximal_multiple"]]
# Fly 19
mi1_fly19_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230427.moco", "2023-04-27", "2", "mi1_proximal_multiple"]]
# Fly 20
mi1_fly20_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230427.moco", "2023-04-27", "3", "mi1_proximal_multiple"]]
# Fly 21
mi1_fly21_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230427.moco", "2023-04-27", "5", "mi1_proximal_multiple"]]
# Fly 22 #5/09/23
mi1_fly22_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "3", "mi1_proximal_multiple"]]
mi1_fly22_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "3", "mi1_distal_multiple"]]
# Fly 23 #5/09/23
mi1_fly23_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "4", "mi1_proximal_multiple"]]
mi1_fly23_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "4", "mi1_distal_multiple"]]
# Fly 24 #5/09/23
mi1_fly24_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "5", "mi1_proximal_multiple"]]
mi1_fly24_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "5", "mi1_distal_multiple"]]


# CONTROL FLIES
# control fly 1 - several ROI name options here: mi1_proximal_multiple_lessbi mi1_proximal_multiple_morebi
mi1_control1_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "4", "mi1_proximal_multiple_morebi"]]
mi1_control1_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "4", "mi1_medial_multiple"]]
mi1_control1_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "4", "mi1_distal_multiple"]]
# control fly 2
mi1_control2_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230317", "2023-03-17", "5", "mi1_proximal_multiple"]]
# control fly 3
mi1_control3_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "1", "mi1_proximal_multiple"]]
mi1_control3_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "1", "mi1_distal_multiple"]]
# control fly 4
mi1_control4_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "2", "mi1_proximal_multiple"]]
# control fly 5
mi1_control5_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "6", "mi1_proximal_multiple"]]
mi1_control5_dist = [["/Volumes/ABK2TBData/data_repo/bruker/20230509.selected", "2023-05-09", "6", "mi1_distal_multiple"]]

# RNAI FLIES
# RNAi fly 1
mi1_rnai_fly1_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230531.moco", "2023-05-31", "4", "proximal_multiple"]]
mi1_rnai_fly2_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230531.moco", "2023-05-31", "6", "proximal_multiple"]]
mi1_rnai_fly3_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230531.moco", "2023-05-31", "13", "proximal_multiple"]]
mi1_rnai_fly4_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230531.moco", "2023-05-31", "8", "proximal_multiple"]]
mi1_rnai_fly5_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20230531.moco", "2023-05-31", "1", "proximal_multiple"]]

mi1_prox_all = np.concatenate(
                             (mi1_fly1_prox, mi1_fly2_prox, mi1_fly3_prox, 
                             mi1_fly4_prox, mi1_fly5_prox, mi1_fly6_prox,
                             mi1_fly7_prox, mi1_fly8_prox, mi1_fly9_prox,
                             mi1_fly10_prox, mi1_fly11_prox, mi1_fly12_prox,
                             mi1_fly13_prox, mi1_fly14_prox, mi1_fly15_prox,
                             ), 
                             axis = 0,
                            )

mi1_prox_goodish = np.concatenate(
                             (mi1_fly4_prox, mi1_fly5_prox, mi1_fly6_prox,
                             mi1_fly7_prox, mi1_fly8_prox, mi1_fly9_prox,
                             mi1_fly10_prox, mi1_fly11_prox, mi1_fly12_prox,
                             mi1_fly13_prox, mi1_fly14_prox, mi1_fly15_prox, 
                             mi1_fly16_prox, mi1_fly17_prox, mi1_fly18_prox,
                             mi1_fly19_prox, mi1_fly20_prox, mi1_fly21_prox,),
                             axis = 0,
                             )

mi1_prox_good = np.concatenate(
                             (mi1_fly6_prox, mi1_fly8_prox, mi1_fly16_prox, mi1_fly17_prox,
                              mi1_fly18_prox, mi1_fly19_prox, mi1_fly20_prox, mi1_fly21_prox,
                              mi1_fly22_prox, mi1_fly23_prox, mi1_fly24_prox,),
                             axis = 0,
                             )

fly_list_prox = [6, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24]

mi1_dist_good = np.concatenate(
                             (mi1_fly6_dist, mi1_fly22_dist, mi1_fly23_dist, mi1_fly24_dist,
                             ),
                             axis = 0,
                             )
fly_list_dist = [6, 22, 23, 24]

mi1_medi_good = np.concatenate(
                             (mi1_fly6_medi,
                             ),
                             axis = 0,
                             )
fly_list_medi = [6]
                             
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
                                  (mi1_control1_prox, mi1_control2_prox, mi1_control3_prox, mi1_control4_prox, mi1_control5_prox,),
                                  axis = 0,
                                 )
fly_list_control_prox = [1, 2, 3, 4, 5]

mi1_control_dist = np.concatenate(
                                  (mi1_control1_prox, mi1_control3_prox, mi1_control5_prox,),
                                  axis = 0,
                                 )
fly_list_control_dist = [1, 3, 5]

fly_list_control_medi = [1]

# RNAi Flies
mi1_rnai_prox = np.concatenate(
                                (mi1_rnai_fly1_prox, mi1_rnai_fly2_prox, mi1_rnai_fly3_prox, mi1_rnai_fly4_prox),
                                axis = 0,
                                )
mi1_rnai_prox_list = [1, 2, 3, 4]

mi1_rnai_test = np.concatenate(
                                (mi1_rnai_fly4_prox, mi1_rnai_fly5_prox,),
                                axis = 0,
                                )
mi1_rnai_test_list = [4, 5]

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
def getWindowMetricsFromLayer(layer, condition_name, per_ROI=False, normalize_to=False, plot_trial_figs=False, save_fig=False):
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

    if per_ROI == True: # initializing. Going to have to append b/c we don't know how many ROIs there are
        mean_matrix_per_ROI = np.empty((1, n_exps, n_opto_params, n_vis_flashes))
        mean_matrix_per_ROI[:] = np.nan
        sem_mean_matrix_per_ROI = np.empty((1, n_exps, n_opto_params, n_vis_flashes))
        sem_mean_matrix_per_ROI[:] = np.nan
        max_matrix_per_ROI = np.empty((1, n_exps, n_opto_params, n_vis_flashes))
        max_matrix_per_ROI[:] = np.nan
        min_matrix_per_ROI = np.empty((1, n_exps, n_opto_params, n_vis_flashes))
        min_matrix_per_ROI[:] = np.nan
        ptt_matrix_per_ROI = np.empty((1, n_exps, n_opto_params, n_vis_flashes))
        ptt_matrix_per_ROI[:] = np.nan


    for pull_ind in range(len(which_layer)):
        file_path = os.path.join(which_layer[pull_ind][0], which_layer[pull_ind][1] + ".hdf5")
        #ID = imaging_data.ImagingDataObject(file_path, which_layer[pull_ind][2], quiet=True)
        ## DEBUG
        cfg_dict = {'timing_channel_ind': 1}
        ID = imaging_data.ImagingDataObject(file_path,
                                        which_layer[pull_ind][2],
                                        quiet=True,
                                        cfg_dict=cfg_dict)

        roi_data = ID.getRoiResponses(which_layer[pull_ind][3], background_roi_name='bg_proximal_lessbi', background_subtraction=False)
        ID.getStimulusTiming(plot_trace_flag=True)
        

        # Plot the average Traces of the whole trial followed by the avg traces of the windows

        unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key='current_led_intensity')
        # ('current_led_intensity', 'current_led_duration')
        # calc the sem + / - NOTE This is a sem across trials. Probably not what you're looking for unless plotting individual ROIs
        # instead calculate the SEM when averaging across ROIs
        # sem_plus = mean_response + sem_response
        # sem_minus = mean_response - sem_response
        print(unique_parameter_values)
        # Calculate the mean and SEM of mean_response across ROIs
        cross_roi_mean_response = np.mean(mean_response, axis=0)
        cross_roi_sem_response = np.std(mean_response, axis=0) / np.sqrt(mean_response.shape[0])
        cross_roi_sem_plus = cross_roi_mean_response + cross_roi_sem_response
        cross_roi_sem_minus = cross_roi_mean_response - cross_roi_sem_response

        trial_timepoints = range(len(roi_data['time_vector']))

        # finding vis flash locations 
        flash_start, flash_end = visFlash(ID)
        min_val = np.min(cross_roi_sem_minus.mean(axis=0))
        max_val = np.max(cross_roi_sem_plus.mean(axis=0))
        y_low = min_val-abs(0.05*min_val)
        y_high = max_val+abs(0.05*max_val)

        # Set the plots to a dark grid background
        which_style = "seaborn-white" # | 'seaborn-white' ' darkgrid'
        plt.style.use(which_style)

        if plot_trial_figs == True:
            # Colormap setting
            cmap = plt.get_cmap('cool') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
            colors = [cmap(i) for i in np.linspace(0.1, 1.0, len(unique_parameter_values))]
            # Plotting the whole trace
            fh, ax = plt.subplots(1, 1, figsize=(16, 8))
            for up_ind, up in enumerate(unique_parameter_values): # up = unique parameter
                ax.plot(roi_data['time_vector'], cross_roi_mean_response[up_ind, :], color=colors[up_ind], alpha=0.9, label=up)
                ax.fill_between(roi_data['time_vector'], cross_roi_sem_plus[up_ind, :], 
                                cross_roi_sem_minus[up_ind, :],
                                color=colors[up_ind], alpha=0.2)
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
                                alpha=0.9, edgecolor='b', facecolor='none', 
                                linewidth=1, label=flash_label)

            # Legend, Grid, Axis
            ax.legend(loc="upper right", fontsize=15)
            ax.grid(axis="x", color="k", alpha=.2, linewidth=1, linestyle=":")
            x_locator = FixedLocator(list(range(-1, 20)))
            ax.xaxis.set_major_locator(x_locator)
            ax.tick_params(axis="x", direction="in", length=10, width=1, color="w")
            ax.grid(axis="y", color="w", alpha=.1, linewidth=.5)
            ax.set_xlabel('Time in Seconds')
            ax.set_ylabel('DF/F')
            ax.set_title(f'{which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=20)

            if save_fig == True:
                fh.savefig(
                save_directory
                + "AvgTraces.Dark"
                + str(which_layer[pull_ind][1])
                + ".Series"
                + str(which_layer[pull_ind][2])
                + ".ROI"
                + str(which_layer[pull_ind][3])
                + ".Conditions:"
                + str(condition_name)
                + "."
                + which_style
                + ".pdf",
                dpi=300, bbox_inches='tight', transparent=True,
                )

            # make a subplot fig for each ROI
            fig, ax = plt.subplots(len(mean_response), 1, figsize=(16, 4*len(mean_response)))
            for roi_ind in range(len(mean_response)):
                for up_ind, up in enumerate(unique_parameter_values):
                    ax[roi_ind].plot(roi_data['time_vector'], mean_response[roi_ind, up_ind, :], color=colors[up_ind], alpha=0.9, label=up)
            # title
            fig.suptitle(f'{which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=20)
                    


        #  Windows to analyze for metrics
        flash_start = ID.getRunParameters('flash_times') + ID.getRunParameters('pre_time')
        flash_width = ID.getRunParameters('flash_width')
        window_lag = 0.25  # sec
        window_length = flash_width + 1.3  # sec
        window_times = flash_start - window_lag
        window_frames = int(np.ceil(window_length / ID.getResponseTiming().get('sample_period')))
        windows = np.zeros((len(unique_parameter_values), len(window_times), window_frames))
        windows_sem = np.zeros((len(unique_parameter_values), len(window_times), window_frames))
        if per_ROI == True:
            windows_by_ROI = np.zeros((len(mean_response), len(unique_parameter_values), len(window_times), window_frames))
            windows_by_ROI[:] = np.nan
            windows_sem_by_ROI = np.zeros((len(mean_response), len(unique_parameter_values), len(window_times), window_frames))
            windows_sem_by_ROI[:] = np.nan

        num_windows = len(window_times)
            
        if plot_trial_figs == True:
            fh, ax = plt.subplots(1, len(unique_parameter_values), figsize=(18, 4))

        # Collect windowed responses - calculate the mean and sem the correct way
        cmap = plt.get_cmap('viridis') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
        colors = [cmap(i) for i in np.linspace(0.0, 1.0, num_windows)]
        for up_ind, up in enumerate(unique_parameter_values): # Opto intensities
            for w_ind, w in enumerate(window_times): # windows
                start_index = np.where(roi_data.get('time_vector') > window_times[w_ind])[0][0]
                # Collect windows responses for the cross-ROI
                windows[up_ind, w_ind, :] = cross_roi_mean_response[up_ind, start_index:(start_index+window_frames)]
                windows_sem[up_ind, w_ind, :] = cross_roi_sem_response[up_ind, start_index:(start_index+window_frames)] # This is using the correct cross-ROI SEM calculation

                if per_ROI == True:
                    # Collect windows responses for each individual ROI
                    windows_by_ROI[:, up_ind, w_ind, :] = mean_response[:, up_ind, start_index:(start_index+window_frames)]
                    windows_sem_by_ROI[:, up_ind, w_ind, :] = sem_response[:, up_ind, start_index:(start_index+window_frames)] # This is using the correct cross-ROI SEM calculation

                if plot_trial_figs == True:
                    # Plot: Each Window for a given LED Intensity
                    ax[up_ind].plot(windows[up_ind, w_ind, :], color=colors[w_ind], label=w if up_ind==0 else '')
                    ax[up_ind].set_title('led={}'.format(up))
                    ax[up_ind].grid(False)

        if plot_trial_figs == True:
            fh.legend()
            fh.suptitle(f'Windows for {which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=12)

            if save_fig == True:
                fh.savefig(
                save_directory
                + "Windows.Dark"
                + str(which_layer[pull_ind][1])
                + ".Series"
                + str(which_layer[pull_ind][2])
                + ".ROI"
                + str(which_layer[pull_ind][3])
                + ".Conditions:"
                + str(condition_name)
                + "."
                + which_style
                + ".pdf",
                dpi=300, bbox_inches='tight', transparent=True,
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
                    ax[w_ind].grid(False)

            fh.legend()
            fh.suptitle(f'Each LED intenisty/window for {which_layer[pull_ind][1]} Series: {which_layer[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={which_layer[pull_ind][3]}', fontsize=12)

            if save_fig == True:
                fh.savefig(
                save_directory
                + "LED.Intensity.window.Dark"
                + str(which_layer[pull_ind][1])
                + ".Series"
                + str(which_layer[pull_ind][2])
                + ".ROI"
                + str(which_layer[pull_ind][3])
                + ".Conditions:"
                + str(condition_name)
                + "."
                + which_style
                + ".pdf",
                dpi=300, bbox_inches='tight', transparent=True,
                )

        # % Plotting the metrics for the windows and then store those values
        # This step takes windows (averaged across ROIs already) = UniqueOptoParams X Window X Time) and averages across time, so:
        # response_mean = Unique Opto Params x Windows
        response_mean = np.mean(windows, axis = -1)
        # NOTE: easy to use the wrong SEM calculation again. If mean(windows_sem) is used, then it's the wrong SEM calculation. Recalculate. 
        response_sem = np.std(windows, axis = -1) / np.sqrt(window_frames) #NOTE - this is the correct SEM calculation, avg SEM across time windows
        response_max = np.max(windows, axis = -1)
        response_min = np.min(windows, axis = -1)
        response_PtT = response_max - response_min

        # storage time!
        mean_matrix[pull_ind] = response_mean
        sem_mean_matrix[pull_ind] = response_sem
        max_matrix[pull_ind] = response_max
        min_matrix[pull_ind] = response_min
        ptt_matrix[pull_ind] = response_PtT

        if per_ROI == True:
            # This step takes windows = ROIs X UniqueOptoParams X Window X Time) and averages across time, so:
            # response_mean = ROIs X Unique Opto Params x Windows
            response_mean_by_ROI = np.mean(windows_by_ROI, axis = -1)
            # NOTE: easy to use the wrong SEM calculation again. If mean(windows_sem) is used, then it's the wrong SEM calculation. Recalculate. 
            response_sem_by_ROI = np.std(windows_by_ROI, axis = -1) / np.sqrt(window_frames)
            response_max_by_ROI = np.max(windows_by_ROI, axis = -1)
            response_min_by_ROI = np.min(windows_by_ROI, axis = -1)
            response_PtT_by_ROI = response_max_by_ROI - response_min_by_ROI

            # storage time!
            if pull_ind == 0:
                mean_matrix_by_ROI = response_mean_by_ROI
                sem_mean_matrix_by_ROI = response_sem_by_ROI
                max_matrix_by_ROI = response_max_by_ROI
                min_matrix_by_ROI = response_min_by_ROI
                ptt_matrix_by_ROI = response_PtT_by_ROI
            else:
                mean_matrix_by_ROI = np.append(mean_matrix_by_ROI, response_mean_by_ROI, axis=0)
                sem_mean_matrix_by_ROI = np.append(sem_mean_matrix_by_ROI, response_sem_by_ROI, axis=0)
                max_matrix_by_ROI = np.append(max_matrix_by_ROI, response_max_by_ROI, axis=0)
                min_matrix_by_ROI = np.append(min_matrix_by_ROI, response_min_by_ROI, axis=0)
                ptt_matrix_by_ROI = np.append(ptt_matrix_by_ROI, response_PtT_by_ROI, axis=0)


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
                    ax[up_ind, w_ind].grid(False)


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
                + "WindowsMetrics.Dark"
                + str(which_layer[pull_ind][1])
                + ".Series"
                + str(which_layer[pull_ind][2])
                + ".ROI"
                + str(which_layer[pull_ind][3])
                + ".Conditions:"
                + str(condition_name)
                + "."
                + which_style
                + ".pdf",
                dpi=300, bbox_inches='tight', transparent=True,
                )
    
    if normalize_to == 'sum' or normalize_to == 'first':
        mean_matrix = metricDifNormalizer(mean_matrix, normalize_to)
        sem_mean_matrix = metricDifNormalizer(sem_mean_matrix, normalize_to)
        max_matrix = metricDifNormalizer(max_matrix, normalize_to)
        min_matrix = metricDifNormalizer(min_matrix, normalize_to)
        ptt_matrix = metricDifNormalizer(ptt_matrix, normalize_to)
    else:
        print("NO NORMALIZING HAPPENED BTW")

    if per_ROI == True:
        # if normalize_to exists, then normalize the mean_matrix and sem_mean_matrix
        if normalize_to == 'sum' or normalize_to == 'first':
            mean_matrix_by_ROI = metricDifNormalizer(mean_matrix_by_ROI, normalize_to)
            sem_mean_matrix_by_ROI = metricDifNormalizer(sem_mean_matrix_by_ROI, normalize_to)
            max_matrix_by_ROI = metricDifNormalizer(max_matrix_by_ROI, normalize_to)
            min_matrix_by_ROI = metricDifNormalizer(min_matrix_by_ROI, normalize_to)
            ptt_matrix_by_ROI = metricDifNormalizer(ptt_matrix_by_ROI, normalize_to)
        else:
            print("NO NORMALIZING HAPPENED BTW")
    
        return mean_matrix_by_ROI, sem_mean_matrix_by_ROI, max_matrix_by_ROI, min_matrix_by_ROI, ptt_matrix_by_ROI
    else:
        return mean_matrix, sem_mean_matrix, max_matrix, min_matrix, ptt_matrix

# %% RUN ALL THE SHIT
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
#----------------------------------------RUN All That Shit-----------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Set meeeeeeeeeeee
data_list = mi1_all_good # mi1_all_good | mi1_control_all | [mi1_rnai_prox] | [mi1_rnai_test]
which_str = 'mi1_all_good'
list_to_use = fly_list_exp # fly_list_exp | fly_list_control | fly_list_prox | mi1_rnai_prox_list | mi1_rnai_test_list
per_ROI = False
#layer_list = ['Proximal'] # only for RNAi
# Making a data frame the way it's supposed to be....
# Currently have:
# metric_matrix = Fly x unique_opto_param x window which is 11 x 3 x 3
# going for: 
# Index | Fly | Mean | Max | Etc Metric | Unique_opto_param | Window

# Loop through everything!

# Defines dataframe with desired columns
if per_ROI == True:
    metric_df = pd.DataFrame(columns=['ROI', 'Layer', 'Mean', 'SEM_Mean', 'Min', 'Max', 'PtT', 'Opto', 'Window'])
else:
    metric_df = pd.DataFrame(columns=['Fly', 'Layer', 'Mean', 'SEM_Mean', 'Min', 'Max', 'PtT', 'Opto', 'Window'])

# Adds all metrics to dataframe one row at a time
row_idx = 0
# Print statement that explains loop is starting
print(f"\nBeginning Loops for extracting data.")
for layer_ind in range(len(layer_list)):
#for layer_ind in [0]:
    mean_diff_matrix, sem_mean_diff_matrix, max_diff_matrix, min_diff_matrix, ptt_diff_matrix = getWindowMetricsFromLayer(data_list[layer_ind], layer_list[layer_ind], per_ROI = per_ROI, normalize_to='sum', plot_trial_figs=True, save_fig=False)
    fly_indicies = list_to_use[layer_ind]
    if data_list == [mi1_rnai_prox]:
        fly_indicies = list_to_use
    # Gets dimensions of metric arrays 
    num_flies, num_opto, num_windows = mean_diff_matrix.shape   
    print(f'mean_diff_matrix.shape = {mean_diff_matrix.shape}')
    for fly in range(num_flies):
    #for fly in range(len(fly_indicies)):
        for opto in range(num_opto):
            for window in range(num_windows):
                if per_ROI == True:
                    metric_df.loc[row_idx] = [
                        fly, layer_list[layer_ind], mean_diff_matrix[fly, opto, window], sem_mean_diff_matrix[fly, opto, window],
                        min_diff_matrix[fly, opto, window], max_diff_matrix[fly, opto, window],
                        ptt_diff_matrix[fly, opto, window], opto, window,
                        ]
                else:
                    print(f'layer_list[layer_ind] = {layer_list[layer_ind]}')
                    print(f'fly_indicies[fly] = {fly_indicies[fly]}')
                    print(f'opto = {opto}')
                    print(f'window = {window}')

                    metric_df.loc[row_idx] = [
                        fly_indicies[fly], layer_list[layer_ind], mean_diff_matrix[fly, opto, window], sem_mean_diff_matrix[fly, opto, window],
                        min_diff_matrix[fly, opto, window], max_diff_matrix[fly, opto, window], 
                        ptt_diff_matrix[fly, opto, window], opto, window,
                        ]
                row_idx += 1

# Convert the floats which were indicies to ints   
if per_ROI == True:
    metric_df['ROI'] = metric_df['ROI'].astype(int)   
else:
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



# %%
plt.close('all')
#control_metric_df_by_fly = metric_df.copy()
#exp_metric_df_by_fly = metric_df.copy()
rnai_metric_df_by_fly = metric_df.copy()

# %% Combine experimental and control data, then export as pickle. also lets you unpickle

# 1) run the above loop for control data

# 2) save control data
# metric_df_control_byroi = metric_df # metric_df_control_byfly | metric_df_control_byoi
#rnai_metric_df_by_roi = metric_df.copy()

# 3) run the above loop for experimental data

# 4) save the experimental data
# metric_df_exp_byroi = metric_df # metric_df_exp_byfly | metric_df_exp_byroi

# 5) append a column label for experimental / control designations
# exp_metric_df_by_roi['Type'] = 'Experimental' # metric_df_exp_byfly | metric_df_exp_byroi
# control_metric_df_by_roi['Type'] = 'Control' # metric_df_control_byfly | metric_df_control_byoi
rnai_metric_df_by_fly['Type'] = 'RNAi2'
#exp_metric_df_by_fly['Type'] = 'Experimental' # metric_df_exp_byfly | metric_df_exp_byroi
#control_metric_df_by_fly['Type'] = 'Control' # metric_df_control_byfly | metric_df_control_byoi
#rnai_metric_df_by_fly['Type'] = 'RNAi'

# 6) concat them
# both_metric_df_byroi = pd.concat([metric_df_exp_byroi, metric_df_control_byroi]) # metric_df_exp_byfly | metric_df_exp_byroi
# exp_control_rnai_by_roi = pd.concat([both_metric_df_byroi, rnai_df])
#exp_control_rnai_by_roi = pd.concat([exp_metric_df_by_roi,control_metric_df_by_roi, rnai_metric_df_by_roi])
#exp_control_rnai_by_roi3 = pd.concat([exp_control_rnai_by_roi2, rnai_metric_df_by_roi])
çexp_control_rnai_by_fly3 = pd.concat([exp_control_rnai_by_fly2, rnai_metric_df_by_fly])


# 7) To save/read in summarized data as .pkl file
# both_metric_df_byroi.to_pickle(save_directory + 'both_metric_df_byroi.pkl')  # metric_df_exp_byfly | metric_df_exp_byroi
#rnai_by_fly_df.to_pickle(save_directory + 'rnai_metric_df_byfly.pkl')  # metric_df_exp_byfly | metric_df_exp_byroi
#exp_control_rnai_by_roi3.to_pickle(save_directory + 'exp_control_rnai_by_roi3.pkl') 
#exp_control_rnai_by_fly.to_pickle(save_directory + 'exp_control_rnai_by_fly2.pkl') 
exp_control_rnai_by_fly3.to_pickle(save_directory + 'exp_control_rnai_by_fly3.pkl') 
# To unpickle: 
#both_metric_df_byfly = pd.read_pickle(save_directory + 'both_metric_df_byfly.pkl')  # metric_df_exp_byfly | metric_df_exp_byroi
#exp_control_rnai_by_roi = pd.read_pickle(save_directory + 'exp_control_rnai_by_roi.pkl')  # metric_df_exp_byfly | metric_df_exp_byroi

# exp_control_rnai_by_roi.to_pickle(save_directory + 'exp_control_rnai_by_roi.pkl')

# %% Seaborn plots for values over windows (Fig 4)

# FIGURE: Subplot grid where each row is a metric and each column is a layer
save_fig=True
bigfig, bigaxes = plt.subplots(4,3, figsize=(36,27))
bigfig.suptitle(f'Metric x Layer for {which_str}')

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
    dpi=300, bbox_inches='tight', transparent=True,
    )


# %% FIGURE: Create a histogram of the difference between the first and third window for each fly (that's window 2)
# seaborn histogram guide: https://seaborn.pydata.org/generated/seaborn.histplot.html 
# Initialize some shit:
# subset df for layer and window
prox_win02_df = rnai_df.loc[(metric_df['Window'] == 1) & (metric_df['Layer'] == 'Proximal')]
df_to_plot = prox_win02_df
which_str2 = 'RNAi'
plot_style = 'seaborn-white' # 'dark_background' or 'seaborn-white'


# Parameters for histplots
kde = True
element = 'step' # bars | poly | step
multiple = 'stack' # dodge | stack | layer | fill
#bins = 10
#binwidth = 0.05 #0.025 # supercedes bins |  0.025
pally = 'Set3' # Set3 pastel

save_fig = False
# Histogram of Mean values for third - first windows
sns.set_theme()
with plt.style.context(plot_style):
    sns.set_context("talk")

    fig_hist_mean, ax_hist_mean = plt.subplots(1, figsize = (12,9))
    sns.histplot(
        ax=ax_hist_mean, x="Mean", hue="Opto", data=df_to_plot, multiple=multiple, element=element, kde=kde, palette=pally, 
        )
    ax_hist_mean.set(xlabel='Mean Value of Third Window - Mean Value of First Window')
    ax_hist_mean.set_title(f'Mean Response for {which_str2} condition - Mi1 Proximal Layer - Opto Effect')
    ax_hist_mean.grid(False)
    sns.despine(fig=fig_hist_mean)

if save_fig == True:
    fig_hist_mean.savefig(
    save_directory
    + "SummaryMeanHistogramThirdWindow.Proximal."
    + plot_style
    + "."
    + which_str2
    + ".pdf",
    dpi=300, bbox_inches='tight', transparent=True,
    )

# Histogram of Max values for third - first windows
# save_fig = True
with plt.style.context(plot_style):
    sns.set_context("talk")

    fig_hist_max, ax_hist_max = plt.subplots(1, figsize = (12,9))
    sns.histplot(
        ax=ax_hist_max, x="Max", hue="Opto", data=df_to_plot, multiple=multiple, element=element, kde=kde, palette=pally, 
        )
    ax_hist_max.set(xlabel='Max Value of Third Window - Max Value of First Window')
    ax_hist_max.set_title(f'Max Response for {which_str2} condition - Mi1 Proximal Layer - Opto Effect')
    ax_hist_max.grid(False)
    sns.despine(fig=fig_hist_max)
if save_fig == True:
    fig_hist_max.savefig(
    save_directory
    + "SummaryMaxHistogramThirdWindow.Proximal."
    + plot_style
    + "."
    + which_str2
    + ".pdf",
    dpi=300, bbox_inches='tight', transparent=True,
    )

# Histogram of Peak-to-Trough values for third - first windows
# save_fig = True
with plt.style.context(plot_style):
    sns.set_context("talk")

    fig_hist_ptt, ax_hist_ptt = plt.subplots(1, figsize = (12,9))
    sns.histplot(
        ax=ax_hist_ptt, x="PtT", hue="Opto", data=df_to_plot, multiple=multiple, element=element, kde=kde, palette=pally, 
        )
    ax_hist_ptt.set(xlabel='Peak-to-Trough Value of Third Window - Peak-to-Trough Value of First Window')
    ax_hist_ptt.set_title(f'Peak-to-Trough Response for {which_str2} condition - Mi1 Proximal Layer - Opto Effect')
    ax_hist_ptt.grid(False)
    sns.despine(fig=fig_hist_ptt)
if save_fig == True:
    fig_hist_ptt.savefig(
    save_directory
    + "SummaryPtTHistogramThirdWindow.Proximal."
    + plot_style
    + "."
    + which_str2
    + ".pdf",
    dpi=300, bbox_inches='tight', transparent=True,
    )

#%% FIGURE: Create a series of violin plots with the differences between the first and third window for each fly
which_df = exp_control_rnai_by_roi
which_str2 = 'RNAi'
save_fig = False

prox_win02_df = which_df.loc[(which_df['Window'] == 1) & (which_df['Layer'] == 'Proximal') & (which_df['Type'] == which_str2)]
df_to_plot = prox_win02_df

pally = 'Set3' # Set3 pastel
with plt.style.context(plot_style):
    sns.set_context("talk")
    sns.axes_style("whitegrid", {'grid.linestyle': '--'})
    contrast_color = "coral"

    fig_box, ax_box_mean = plt.subplots(1, figsize = (10, 10))
    sns.violinplot(
        ax=ax_box_mean, data=df_to_plot, x="Mean", y="Opto", palette=pally, orient="h", order=[2,1,0],
        medianprops={"color": contrast_color}, whiskerprops={"color":contrast_color}, capprops={"color":contrast_color}, 
        flierprops={"markerfacecolor":contrast_color, "markeredgecolor":contrast_color, 'markersize': 10}, boxprops={"edgecolor":contrast_color},        
        )
    ax_box_mean.set(xlabel='Mean Value of Third Window - Mean Value of First Window')
    ax_box_mean.set(ylabel='Opto Intensity')
    ax_box_mean.set_title(f'Mean Response for {which_str2} condition - Mi1 Proximal Layer - Opto Effect')
    sns.despine(fig=fig_box, left=True)
    sns.set_context("talk")
    ax_box_mean.grid(alpha=0.3)

    # calculate the number of obs per group & median to position labels
    medians = df_to_plot.groupby(['Opto'])['Mean'].median().values
    nobs = df_to_plot['Opto'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    pos = range(len(nobs))
    for tick, label in zip(pos, ax_box_mean.get_yticklabels()):
        ax_box_mean.text(
            0.1, pos[tick], nobs[tick], horizontalalignment='center', size='small', color='w', weight='semibold'
        )  
if save_fig == True:
    fig_box.savefig(
    save_directory
    + "SummaryMeanBoxPlotThirdWindow.Proximal."
    + plot_style
    + "."
    + which_str2
    + ".pdf",
    dpi=300, bbox_inches='tight', transparent=True,
    )

# same figure as above, but max
# save_fig = True
with plt.style.context(plot_style):
    sns.set_context("talk")
    sns.axes_style("whitegrid", {'grid.linestyle': '--'})
    contrast_color = "coral"

    fig_box_max, ax_box_max = plt.subplots(1, figsize = (10, 10))
    sns.violinplot(
        ax=ax_box_max, data=df_to_plot, x="Max", y="Opto", palette=pally, orient="h", order=[2,1,0],
        medianprops={"color": contrast_color}, whiskerprops={"color":contrast_color}, capprops={"color":contrast_color}, 
        flierprops={"markerfacecolor":contrast_color, "markeredgecolor":contrast_color, 'markersize': 10}, boxprops={"edgecolor":contrast_color},
        )
    ax_box_max.set(xlabel='Max Value of Third Window - Max Value of First Window')
    ax_box_max.set(ylabel='Opto Intensity')
    ax_box_max.set_title(f'Max Response for {which_str2} condition - Mi1 Proximal Layer - Opto Effect')
    sns.despine(fig=fig_box_max, left=True)
    sns.set_context("talk")
    ax_box_max.grid(alpha=0.3)
    # calculate the number of obs per group & median to position labels
    medians = df_to_plot.groupby(['Opto'])['Max'].median().values
    nobs = df_to_plot['Opto'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    pos = range(len(nobs))
    for tick, label in zip(pos, ax_box_max.get_yticklabels()):
        ax_box_max.text(
            0.1, pos[tick], nobs[tick], horizontalalignment='center', size='small', color='w', weight='semibold'
        )
if save_fig == True:
    fig_box_max.savefig(
    save_directory
    + "SummaryMaxBoxPlotThirdWindow.Proximal."
    + plot_style
    + "."
    + which_str2
    + ".pdf",
    dpi=300, bbox_inches='tight', transparent=True,
    )

# same figure as above, but PtT
# save_fig = True
with plt.style.context(plot_style):
    sns.set_context("talk")
    sns.axes_style("whitegrid", {'grid.linestyle': '--'})
    contrast_color = "coral"

    fig_box_ptt, ax_box_ptt = plt.subplots(1, figsize = (10, 10))
    sns.violinplot(
        ax=ax_box_ptt, data=df_to_plot, x="PtT", y="Opto", palette=pally, orient="h", order=[2,1,0],
        medianprops={"color": contrast_color}, whiskerprops={"color":contrast_color}, capprops={"color":contrast_color}, 
        flierprops={"markerfacecolor":contrast_color, "markeredgecolor":contrast_color, 'markersize': 10}, boxprops={"edgecolor":contrast_color},
        )
    ax_box_ptt.set(xlabel='Peak to Trough Value of Third Window - Peak to Trough Value of First Window')
    ax_box_ptt.set(ylabel='Opto Intensity')
    ax_box_ptt.set_title(f'Peak to Trough Response for {which_str2} condition - Mi1 Proximal Layer - Opto Effect')
    sns.despine(fig=fig_box_ptt, left=True)
    sns.set_context("talk")
    ax_box_ptt.grid(alpha=0.3)

    # calculate the number of obs per group & median to position labels
    medians = df_to_plot.groupby(['Opto'])['PtT'].median().values
    nobs = df_to_plot['Opto'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    pos = range(len(nobs))
    for tick, label in zip(pos, ax_box_ptt.get_yticklabels()):
        ax_box_ptt.text(
            0.1, pos[tick], nobs[tick], horizontalalignment='center', size='small', color='w', weight='semibold'
        )
if save_fig == True:
    fig_box_ptt.savefig(
    save_directory
    + "SummaryPtTBoxPlotThirdWindow.Proximal."
    + plot_style
    + "."
    + which_str2
    + ".pdf",
    dpi=300, bbox_inches='tight', transparent=True,
    )


# %% Plot the scatter of the control vs experimental data for window 2
which_df_placeholder = exp_control_rnai_by_roi
save_fig = False
plot_style = 'seaborn-whitegrid' # 'dark_background' or 'seaborn-white'

which_opto = 2
prox_win02_opto2_df = which_df_placeholder.loc[(which_df_placeholder['Window'] == 1) & (which_df_placeholder['Layer'] == 'Proximal') & (which_df_placeholder['Opto'] != 1)]
#prox_win02_opto2_df = which_df_placeholder.loc[(which_df_placeholder['Window'] == 1) & (which_df_placeholder['Layer'] == 'Proximal')]

df_to_plot = prox_win02_opto2_df

#  Violin plots of exp vs control data vs rnai
# create a figure
with plt.style.context(plot_style):
    fig_box_means, ax_box_means = plt.subplots(1, figsize = (14, 8))
    sns.violinplot(ax=ax_box_means, data = df_to_plot, x = 'Mean', y = 'Type', hue='Opto', split=True, medianprops={"color": contrast_color}, whiskerprops={"color":contrast_color}, capprops={"color":contrast_color}, 
        flierprops={"markerfacecolor":contrast_color, "markeredgecolor":contrast_color, 'markersize': 10}, boxprops={"edgecolor":contrast_color},)
    # calculate the number of obs per group & median to position labels
    medians = df_to_plot.groupby(['Type'])['Mean'].median().values
    nobs = df_to_plot['Type'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    # add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, ax_box_means.get_yticklabels()):
        ax_box_means.text(
            0.3, pos[tick], nobs[tick], horizontalalignment='center', size='large', color=contrast_color, weight='semibold'
        )

    fig_box_maxes, ax_box_maxes = plt.subplots(1, figsize = (14, 8))
    sns.violinplot(ax=ax_box_maxes, data = df_to_plot, x = 'Max', y = 'Type', hue='Opto', split=False, bw=.3, medianprops={"color": contrast_color}, whiskerprops={"color":contrast_color}, capprops={"color":contrast_color}, 
        flierprops={"markerfacecolor":contrast_color, "markeredgecolor":contrast_color, 'markersize': 10}, boxprops={"edgecolor":contrast_color},)
    # add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, ax_box_maxes.get_yticklabels()):
        ax_box_maxes.text(
            0.3, pos[tick], nobs[tick], horizontalalignment='center', size='large', color=contrast_color, weight='semibold'
        )

    fig_box_ptt, ax_box_ptt = plt.subplots(1, figsize = (14, 8))
    sns.violinplot(ax=ax_box_ptt, data = df_to_plot, x = 'PtT', y = 'Type', hue='Opto', split=True, medianprops={"color": contrast_color}, whiskerprops={"color":contrast_color}, capprops={"color":contrast_color}, 
        flierprops={"markerfacecolor":contrast_color, "markeredgecolor":contrast_color, 'markersize': 10}, boxprops={"edgecolor":contrast_color},)
    # add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, ax_box_ptt.get_yticklabels()):
        ax_box_ptt.text(
            0.3, pos[tick], nobs[tick], horizontalalignment='center', size='large', color=contrast_color, weight='semibold'
        )

    # titles for the figures
    ax_box_means.set_title(f'Mean Response - Mi1 Proximal Layer - Opto Effect for Window 3 - Window 1')
    ax_box_maxes.set_title(f'Max Response - Mi1 Proximal Layer - Opto Effect for Window 3 - Window 1')
    ax_box_ptt.set_title(f'Peak to Trough Response - Mi1 Proximal Layer - Opto Effect for Window 3 - Window 1')

    # save the figures
    if save_fig == True:
        fig_box_means.savefig(
            save_directory
            + "ConVExpVRNAi.SummaryMeanBoxPlotThirdWindow.Proximal."
            + plot_style
            + ".pdf",
            dpi=300, bbox_inches='tight', transparent=True,
        )
        fig_box_maxes.savefig(
            save_directory
            + "ConVExpVRNAi.SummaryMaxBoxPlotThirdWindow.Proximal."
            + plot_style
            + ".pdf",
            dpi=300, bbox_inches='tight', transparent=True,
        )
        fig_box_ptt.savefig(
            save_directory
            + "ConVExpVRNAi.SummaryPtTBoxPlotThirdWindow.Proximal."
            + plot_style
            + ".pdf",
            dpi=300, bbox_inches='tight', transparent=True,
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
savefig = False

kde = True
element = 'poly' # bars | poly | step
multiple = 'stack' # dodge | stack | layer | fill
bins = 10
binwidth = 0.05 #0.025 # supercedes bins |  0.025

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
# %% Stats time baby

# Assumption testing
which_df = exp_control_rnai_by_roi
win = 1
opto = 2
dv = 'PtT' # Max, Min, PtT, Mean
exp_sub_df = which_df.loc[(which_df['Window'] == win) & (which_df['Layer'] == 'Proximal') & (which_df['Opto'] == opto) & (which_df['Type'] == 'Experimental')][dv]
con_sub_df = which_df.loc[(which_df['Window'] == win) & (which_df['Layer'] == 'Proximal') & (which_df['Opto'] == opto) & (which_df['Type'] == 'Control')][dv]
rnai_sub_df = which_df.loc[(which_df['Window'] == win) & (which_df['Layer'] == 'Proximal') & (which_df['Opto'] == opto) & (which_df['Type'] == 'RNAi')][dv]

# Testing for homogeneity. P-value > 0.05 means we can assume homogeneity of variance
print('-------------------------------------------')
print('HOMOGENEITY TESTING')
expXcon_stat, expXcon_pval = stats.levene(exp_sub_df, con_sub_df)
print(f'Experimental and Control conditions:')
print(f'    P-value = {expXcon_pval}')
print(f'    Test-stat = {expXcon_stat}\n')

expXrnai_stat, expXrnai_pval = stats.levene(exp_sub_df, rnai_sub_df)
print(f'Experimental and RNAi conditions:')
print(f'    P-value = {expXrnai_pval}')
print(f'    Test-stat = {expXrnai_stat}\n')

conXrnai_stat, conXrnai_pval = stats.levene(con_sub_df, rnai_sub_df)
print(f'Control and RNAi conditions:')
print(f'    P-value = {conXrnai_pval}')
print(f'    Test-stat = {conXrnai_stat}\n')

# - Exp v Control not homogeneous, exp v rnai not homogeneous, control v rnai YES homogeneous

print('-------------------------------------------')
# Testing for normality - Shapiro-Wilk test
print('NORMALITY TESTING')
exp_norm_stat, exp_norm_pval = stats.shapiro(exp_sub_df)
print('Experimental condition:')
print(f'    P-value = {exp_norm_pval}')
print(f'    Test-stat = {exp_norm_stat}\n')
con_norm_stat, con_norm_pval = stats.shapiro(con_sub_df)
print('Control condition:')
print(f'    P-value = {con_norm_pval}')
print(f'    Test-stat = {con_norm_stat}\n')
rnai_norm_stat, rnai_norm_pval = stats.shapiro(rnai_sub_df)
print('RNAi condition:')
print(f'    P-value = {rnai_norm_pval}')
print(f'    Test-stat = {rnai_norm_stat}\n')

# - All conditions are normal

# NOTE: Do this again but combine opto intensity 1 and 2
# When combining opto intensity 1 and 2, we need to make sure that the data is still normal and homogeneous
# I did it. It's not. Already wasn't homogenous, but when combined, also not normal 


# %% from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html 

which_df = exp_control_rnai_by_roi
# All experimental types, one window and opto
win_set = [0, 1, 2]
opto_set = [0, 1, 2]
type_set = ['Experimental', 'Control', 'RNAi']
dv = 'Max' # 'Max', 'Mean', 'PtT'
for win_ind, win in enumerate(win_set):
    print(f'Stats for {dv} metric, single conditions, window {win_set[win_ind]+2} - window 1')
    for opto_ind, opto in enumerate(opto_set):
        print(f'            - Opto intensity value = {opto} -')
        for type_ind, type_i in enumerate(type_set):
            prox_df = which_df.loc[(which_df['Window'] == win) & (which_df['Layer'] == 'Proximal') & (which_df['Opto'] == opto)]
            # Test of differences for one specific condition and opto, for a single metric
            res_exp = wilcoxon(prox_df[prox_df['Type'] == type_set[type_ind]][[dv]], alternative='greater', correction=True)
            print(f'{type_set[type_ind]} Condition:         P-value = {res_exp.pvalue}  |   statistic = {res_exp.statistic}')
        #print('\n')
    print('\n')
# %%
print('--------------------------------------------------------------------------------------')
print('When straight combining opto intensity 1 and 2...')
# Again but combining opto intensity 1 and 2
comb_df = exp_control_rnai_by_roi.copy()
comb_df.loc[comb_df['Opto'] == 2, 'Opto'] = 1
win_set = [0, 1, 2]
opto_set = [0, 1]
type_set = ['Experimental', 'Control', 'RNAi']
dv = 'Max' # 'Max', 'Mean', 'PtT'
for win_ind, win in enumerate(win_set):
    print(f'Stats for {dv} metric, single conditions, window {win_set[win_ind]+2} - window 1')
    for opto_ind, opto in enumerate(opto_set):
        print(f'            - Opto intensity value = {opto} -')
        for type_ind, type_i in enumerate(type_set):
            prox_df = comb_df.loc[(comb_df['Window'] == win) & (comb_df['Layer'] == 'Proximal') & (comb_df['Opto'] == opto)]
            # Test of differences for one specific condition and opto, for a single metric
            res_exp = wilcoxon(prox_df[prox_df['Type'] == type_set[type_ind]][[dv]], alternative='greater', correction=True)
            print(f'{type_set[type_ind]} Condition:         P-value = {res_exp.pvalue}  |   statistic = {res_exp.statistic}')
        #print('\n')
    print('\n')

# %%
# Test of differences across two conditions, single opto, for a single metric

# the Exp v control and exp vs RNAi violate homogeneity assumption, so we should use a Kruskal-Wallis test

# But I'll do an ANOVA anyway, just to see what happens
opto = 2
dv = 'PtT' # 'Max'
only_types = which_df.loc[(which_df['Window'] == win) & (which_df['Layer'] == 'Proximal') & (which_df['Opto'] == opto)][["Type", dv]]
aov = pg.anova(data=only_types, dv=dv, between='Type', detailed=True)
print('--------------------------------------------------------------------------------------')
print(f'ANOVA Test for differences across conditions, window 3-1, opto {opto}')
print(aov)

# Kruskal-Wallis test
kruskal = pg.kruskal(data=only_types, dv=dv, between='Type', detailed=True) # Neat
print('--------------------------------------------------------------------------------------')
print(f'KRUSKAL-WALLIS Test for differences across conditions, window 3-1, opto {opto}')
print(kruskal)

# Paired T-tests and Wilcoxon signed-rank tests
is_par = True
adj_type = 'none' # 'bonf'
alt_hypo = 'less' # 'two-sided' or 'greater'
paired_t_results = pg.pairwise_tests(data=only_types, dv=dv, between='Type', parametric=is_par, padjust=adj_type, effsize='cohen', alternative=alt_hypo)

print('--------------------------------------------------------------------------------------')
print('PAIRED T-TESTS for differences across conditions, window 3-1, opto {opto}')
print('Wilcoxon signed-rank test results b/c non-parametric:')
print(f'Parameters: parametric = {is_par}, padjust = {adj_type}, alternative = {alt_hypo}')
print(f'Experimental v Control: {paired_t_results.loc[0, "p-unc"]}')
print(f'Control v RNAi: {paired_t_results.loc[1, "p-unc"]}')
print(f'Experimental v RNAi: {1-paired_t_results.loc[2, "p-unc"]}')



# %% Paired t-tests like above, but combining opto intensities 1 and 2
dv = 'Max' # 'Max' 'Mean' 'PtT'
opto_1 = which_df.loc[(which_df['Window'] == win) & (which_df['Layer'] == 'Proximal') & (which_df['Opto'] == 1)][["Type", dv]]
opto_2 = which_df.loc[(which_df['Window'] == win) & (which_df['Layer'] == 'Proximal') & (which_df['Opto'] == 2)][["Type", dv]]
# concatenate the two dataframes
opto_1_2 = pd.concat([opto_1, opto_2])

aov = pg.anova(data=opto_1_2, dv=dv, between='Type', detailed=True)
print('--------------------------------------------------------------------------------------')
print(f'ANOVA Test for {dv} differences across conditions, window 3-1, opto 1 and 2 combined')
print(aov)

# Kruskal-Wallis test
kruskal = pg.kruskal(data=opto_1_2, dv=dv, between='Type', detailed=True) # Neat
print('--------------------------------------------------------------------------------------')
print(f'KRUSKAL-WALLIS Test for {dv} differences across conditions, window 3-1, opto 1 and 2 combined')
print(kruskal)

# Paired T-tests and Wilcoxon signed-rank tests
is_par = False
adj_type = 'bonf' # 'bonf'
alt_hypo = 'less' # 'two-sided' or 'greater'
paired_t_results = pg.pairwise_tests(data=opto_1_2, dv=dv, between='Type', parametric=is_par, padjust=adj_type, effsize='cohen', alternative=alt_hypo)

print('--------------------------------------------------------------------------------------')
print(f'PAIRED T-TESTS for {dv} differences across conditions, window 3-1, opto 1 and 2 combined')
print('This is Wilcoxon signed-rank test results b/c non-parametric:')
print(f'Parameters - parametric = {is_par}, padjust = {adj_type}, alternative = {alt_hypo}')
print(f'Exp v Con:      P-Value: {paired_t_results.loc[0, "p-unc"]}, effect size = {paired_t_results.loc[0, "cohen"]}')
print(f'Con v RNAi:     P-Value: {paired_t_results.loc[1, "p-unc"]}, effect size = {paired_t_results.loc[1, "cohen"]}')
print(f'Exp v RNAi:     P-Value: {1-paired_t_results.loc[2, "p-unc"]}, effect size = {paired_t_results.loc[2, "cohen"]}')



# %%
