# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt

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
mi1_fly2_prox = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "proximal_multiple"]]
mi1_fly2_medi = [["/Volumes/ABK2TBData/data_repo/bruker/20221129.common_moco", "2022-11-29", "4", "medial_multiple_all"]] #also 'medial_multiple_sub1", "medial_multiple_sub2"
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
                                  (mi1_fly1_prox, mi1_fly2_prox,mi1_fly1_medi, mi1_fly2_medi,mi1_fly1_dist, mi1_fly2_dist,),
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

alt_pre_time = 2
single_roi = True

condition_name = 'current_led_intensity'


# %%

# do outer loop here...

all_response_max = []

pull_ind = 0
file_path = os.path.join(mi1_all_multiple[pull_ind][0], mi1_all_multiple[pull_ind][1] + ".hdf5")
ID = imaging_data.ImagingDataObject(file_path, mi1_all_multiple[pull_ind][2], quiet=True)
roi_data = ID.getRoiResponses(mi1_all_multiple[pull_ind][3], background_subtraction=True)


# %%

unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key='current_led_intensity')



fh, ax = plt.subplots(1, 1, figsize=(8, 4))



for up_ind, up in enumerate(unique_parameter_values):
    ax.plot(mean_response[:, up_ind, :].mean(axis=0))


flash_start = ID.getRunParameters('flash_times') + ID.getRunParameters('pre_time')

window_lag = 0.25  # sec
window_length = 1.8  # sec
window_times = flash_start - window_lag
window_frames = np.int(np.ceil(window_length / ID.getResponseTiming().get('sample_period')))

windows = np.zeros((len(unique_parameter_values), len(window_times), window_frames))

fh, ax = plt.subplots(1, 5, figsize=(18, 4))

# Collect windowed responses

cmap = plt.get_cmap('cool') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]
for up_ind, up in enumerate(unique_parameter_values):
    for w_ind, w in enumerate(window_times):
        start_index = np.where(roi_data.get('time_vector') > window_times[w_ind])[0][0]
        windows[up_ind, w_ind, :] = mean_response[:, up_ind, start_index:(start_index+window_frames)].mean(axis=0)


        # Plot
        ax[up_ind].plot(windows[up_ind, w_ind, :], color=colors[w_ind], label=w if up_ind==0 else '')
        ax[up_ind].set_title('led={}'.format(up))


fh.legend()

# %%
response_max = np.max(windows, axis=-1)
response_min = np.min(windows, axis=-1)



fh, ax = plt.subplots(2, len(window_times)-1, figsize=(16, 8))

for w_ind in range(len(window_times)-1):
    for up_ind, up in enumerate(unique_parameter_values):
        ax[0, w_ind].plot(response_max[up_ind, 0], response_max[up_ind, w_ind+1], color=colors[up_ind], marker='o', label=up if w_ind==0 else '')
        ax[0, w_ind].plot([0, 0.5], [0, 0.5], 'k--')
        ax[0, w_ind].set_title(window_times[w_ind+1])

        ax[1, w_ind].plot(response_min[up_ind, 0], response_min[up_ind, w_ind+1], color=colors[up_ind], marker='o')
        ax[1, w_ind].plot([-0.50, 0.1], [-0.5, 0.1], 'k--')

        
ax[1, 0].set_xlabel('baseline')
ax[1, 0].set_ylabel('comparison')

fh.legend()


all_response_max.append(response_max)


all_response_max = np.stack(response_max, axis=-1)  # shape = param values, window times, flies




# %%
