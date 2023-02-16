# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import os
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
mi1_prox_all = np.concatenate(
                       (mi1_fly1_prox, mi1_fly2_prox,), 
                        axis = 0,
                      )
mi1_medi_all = np.concatenate(
                       (mi1_fly1_medi, mi1_fly2_medi,), 
                        axis = 0,
                      )
mi1_dist_all = np.concatenate(
                       (mi1_fly1_dist, mi1_fly2_dist,), 
                        axis = 0,
                      )
mi1_all_multiple = np.concatenate(
                                  (mi1_fly1_prox, mi1_fly2_prox, mi1_fly1_medi, mi1_fly2_medi, mi1_fly1_dist, mi1_fly2_dist,),
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

# %%

# do outer loop here...

all_response_max = []


# Which one to plot
# (0) mi1_fly1_prox (1) mi1_fly2_prox (2) mi1_fly1_medi (3) mi1_fly2_medi (4) mi1_fly1_dist (5) mi1_fly2_dist
pull_ind = 0
file_path = os.path.join(mi1_all_multiple[pull_ind][0], mi1_all_multiple[pull_ind][1] + ".hdf5")
ID = imaging_data.ImagingDataObject(file_path, mi1_all_multiple[pull_ind][2], quiet=True)
roi_data = ID.getRoiResponses(mi1_all_multiple[pull_ind][3], background_roi_name='bg_proximal', background_subtraction=False)


# %% Plot the average Traces of the whole trial followed by the avg traces of the windows

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
cmap = plt.get_cmap('PRGn') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
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
ax.set_title(f'{mi1_all_multiple[pull_ind][1]} Series: {mi1_all_multiple[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={mi1_all_multiple[pull_ind][3]}', fontsize=20)


# %% Windows to analyze for metrics
flash_start = ID.getRunParameters('flash_times') + ID.getRunParameters('pre_time')
flash_width = ID.getRunParameters('flash_width')
window_lag = 0.25  # sec
window_length = flash_width + 1.3  # sec
window_times = flash_start - window_lag
window_frames = int(np.ceil(window_length / ID.getResponseTiming().get('sample_period')))
windows = np.zeros((len(unique_parameter_values), len(window_times), window_frames))
windows_sem = np.zeros((len(unique_parameter_values), len(window_times), window_frames))

fh, ax = plt.subplots(1, 5, figsize=(18, 4))

# Collect windowed responses
cmap = plt.get_cmap('cool') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
#colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]
for up_ind, up in enumerate(unique_parameter_values): # Opto intensities
    for w_ind, w in enumerate(window_times): # windows
        start_index = np.where(roi_data.get('time_vector') > window_times[w_ind])[0][0]
        windows[up_ind, w_ind, :] = mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)
        windows_sem[up_ind, w_ind, :] = sem_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)

        # Plot: Each Window for a given LED Intensity
        ax[up_ind].plot(windows[up_ind, w_ind, :], color=colors[w_ind], label=w if up_ind==0 else '')
        ax[up_ind].set_title('led={}'.format(up))


fh.legend()
fh.suptitle(f'Windows for {mi1_all_multiple[pull_ind][1]} Series: {mi1_all_multiple[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={mi1_all_multiple[pull_ind][3]}', fontsize=12)

# %% Plot each LED intensity for a given window

fh, ax = plt.subplots(1, 5, figsize=(20, 4))
# Plot windowed responses
cmap = plt.get_cmap('viridis') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'#colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]
colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]

for w_ind, w in enumerate(window_times):
    for up_ind, up in enumerate(unique_parameter_values):
        ax[w_ind].plot(windows[up_ind, w_ind], linewidth=2, color=colors[up_ind], alpha=0.8, label=up if w_ind==0 else '')
        ax[w_ind].set_title('Visual Flash at: {}s'.format(w))

fh.legend()
fh.suptitle(f'Each LED intenisty/window for {mi1_all_multiple[pull_ind][1]} Series: {mi1_all_multiple[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={mi1_all_multiple[pull_ind][3]}', fontsize=12)

# %% Plotting the metrics for the windows

response_max = np.max(windows, axis=-1)
response_min = np.min(windows, axis=-1)

cmap = plt.get_cmap('viridis') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]

fh, ax = plt.subplots(2, len(window_times)-1, figsize=(16, 8))

for w_ind in range(len(window_times)-1): # w_ind = window_indicies
    for up_ind, up in enumerate(unique_parameter_values):
        # Maximums for top row
        ax[0, w_ind].plot(response_max[up_ind, w_ind], response_max[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o', label=up if w_ind==0 else '')
        # Minimums for bottom row
        ax[1, w_ind].plot(response_min[up_ind, w_ind], response_min[up_ind, w_ind+1], color=colors[up_ind], markersize=10, marker='o')

        ax[0, w_ind].set_title(f'Visual Flash {w_ind+2} | Window Time: {window_times[w_ind+1]}')

    # Finding unity params - Max
    unity_lower_max = min(min(response_max[:, w_ind]), min(response_max[:, w_ind+1]))*0.9
    unity_upper_max = max(max(response_max[:, w_ind]), max(response_max[:, w_ind+1]))*1.1
    ax[0, w_ind].plot([unity_lower_max, unity_upper_max], [unity_lower_max, unity_upper_max], 'k--', alpha=0.7)
    # Finding unity params - Min
    unity_lower_min = min(min(response_min[:, w_ind]), min(response_min[:, w_ind+1]))*1.1
    unity_upper_min = max(max(response_min[:, w_ind]), max(response_min[:, w_ind+1]))*0.9
    ax[1, w_ind].plot([unity_lower_min, unity_upper_min], [unity_lower_min, unity_upper_min], 'k--', alpha=0.7)

    ax[0, w_ind].set_xlabel('Visual Flash 1 (Pre-Opto) Peak Amplitude')
    ax[1, w_ind].set_xlabel('Visual Flash 1 (Pre-Opto) Trough Amplitude')
        
ax[0, 0].set_ylabel('Comparison Visual Flash Peak Amplitude')
ax[1, 0].set_ylabel('Comparison Visual Flash Trough Amplitude')

#ax.set_ylabel(row, rotation=0, size='large
#fh.tight_layout()
fh.legend(title = 'Opto Intensities')
fh.suptitle(f'Windows Metrics for {mi1_all_multiple[pull_ind][1]} Series: {mi1_all_multiple[pull_ind][2]} | DFF=True | Conditions: {condition_name} | ROI={mi1_all_multiple[pull_ind][3]}', fontsize=15)


# %%
all_response_max.append(response_max)
all_response_max = np.stack(response_max, axis=-1)  # shape = param values, window times, flies




# %%
