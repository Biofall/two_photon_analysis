# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import os
import numpy as np

# Single ROI
mi1_prox_single = ["/Volumes/ABK2TBData/data_repo/bruker/20221129", "2022-11-29", "1", "Mi1_proximal"]
mi1_medi_single = ["/Volumes/ABK2TBData/data_repo/bruker/20221129", "2022-11-29", "1", "Mi1_medial"]
mi1_dist_single = ["/Volumes/ABK2TBData/data_repo/bruker/20221129", "2022-11-29", "1", "Mi1_distal"]

# also Mi1_cell_bodies, Mi1_cell_bodies1, distal, medial, proximal, proximal_2



# %%
# Which one to plot
layer = mi1_dist_single

file_path = os.path.join(layer[0], layer[1] + ".hdf5")
ID = imaging_data.ImagingDataObject(file_path, layer[2], quiet=True)
roi_data = ID.getRoiResponses(layer[3])



# %% Pull the trial's data based on the parameter key
start_trial = 0

# For both
parameter_keys = ('current_led_intensity', 'current_led_duration')
if start_trial < 1:
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key=parameter_keys)
else: # for un motion corrected ONLY
    #start_trial = 40
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(np.expand_dims(roi_data.get('epoch_response')[0][start_trial::][:], axis=0), parameter_key=parameter_keys)
# calc the sem + / -
sem_plus = mean_response + sem_response
sem_minus = mean_response - sem_response

# Just current_led_intensity
parameter_key_intensity = 'current_led_intensity'
if start_trial < 1:
    unique_parameter_values_intensity, mean_response_intensity, sem_response_intensity, trial_response_by_stimulus_intensity = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key=parameter_key_intensity)
else: # for un motion corrected ONLY
    #start_trial = 40
    unique_parameter_values_intensity, mean_response_intensity, sem_response_intensity, trial_response_by_stimulus_intensity = ID.getTrialAverages(np.expand_dims(roi_data.get('epoch_response')[0][start_trial::][:], axis=0), parameter_key=parameter_key_intensity)
# calc the sem + / -
sem_plus_intensity = mean_response_intensity + sem_response_intensity
sem_minus_intensity = mean_response_intensity - sem_response_intensity

# Just current_led_duration
parameter_key_duration = 'current_led_duration'
if start_trial < 1:
    unique_parameter_values_duration, mean_response_duration, sem_response_duration, trial_response_by_stimulus_duration = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key=parameter_key_duration)
else: # for un motion corrected ONLY
    #start_trial = 40
    unique_parameter_value_duration, mean_response_duration, sem_response_duration, trial_response_by_stimulus_duration = ID.getTrialAverages(np.expand_dims(roi_data.get('epoch_response')[0][start_trial::][:], axis=0), parameter_key=parameter_key_duration)
# calc the sem + / -
sem_plus_duration = mean_response_duration + sem_response_duration
sem_minus_duration = mean_response_duration - sem_response_duration


# %% Plotting functions
def plotOptoStepSeries(parameter_keys, unique_parameter_values, mean_response, sem_plus, sem_minus):
    opto_on_time = 2
    cmap = plt.get_cmap('winter') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.0, 1.0, len(unique_parameter_values))]
    x_locator = FixedLocator(list(range(-1, int(np.ceil(roi_data['time_vector'][-1]))+1)))
    n_params = len(unique_parameter_values)

    # Plotting
    fh, ax = plt.subplots(1, 1, figsize=(16, 8))
    figg, axiss = plt.subplots(n_params, 1, figsize=(16, 8*n_params))
    for up_ind, up in enumerate(unique_parameter_values): # up = unique parameter
        # Plot each individual unique parameter value on the same plot
        ax.fill_between(roi_data['time_vector'], sem_plus[:, up_ind, :].mean(axis=0), 
                        sem_minus[:, up_ind, :].mean(axis=0),
                        color=colors[up_ind], alpha=0.1)
        ax.plot(roi_data['time_vector'], mean_response[:, up_ind, :].mean(axis=0), color=colors[up_ind], alpha=1.0, label=up)

        # Plot each individual unique parameter value on its own sub plot
        #figg, axiss = plt.subplots(1, 1, figsize=(16, 8))
        axiss[up_ind].plot(roi_data['time_vector'], mean_response[:, up_ind, :].mean(axis=0), color='white', alpha=1)
        axiss[up_ind].fill_between(roi_data['time_vector'], sem_plus[:, up_ind, :].mean(axis=0), 
                        sem_minus[:, up_ind, :].mean(axis=0),
                        color=colors[up_ind], alpha=0.9)

        
        # Opto Stim Plotting
        response_ind = 11 #the first index after the response
        min_val = np.min(mean_response[:, up_ind, response_ind:].mean(axis=0))
        min_x = np.where(mean_response[:, up_ind, :].mean(axis=0)==min_val)[0][0]
        max_val = np.max(sem_plus[:, up_ind, response_ind:].mean(axis=0))
        max_x = np.where(sem_plus[:, up_ind, :].mean(axis=0)==max_val)[0][0]
        y_low = min_val-abs(0.01*min_val)
        y_high = max_val+abs(0.01*max_val)
        
        # Max and min annotation
        axiss[up_ind].annotate('Max', xy =(roi_data['time_vector'][max_x], max_val), 
                               xytext =(roi_data['time_vector'][max_x]+1.5, max_val), 
                               arrowprops = dict(facecolor ='green', alpha=0.5, shrink = 0.1))
        axiss[up_ind].annotate('Min', xy =(roi_data['time_vector'][min_x], min_val), 
                        xytext =(roi_data['time_vector'][min_x]+1.5, min_val), 
                        arrowprops = dict(facecolor ='black', alpha=0.5, shrink = 0.1))
        
        led_start_time = ID.getRunParameters('pre_time')
        if type(parameter_keys) == tuple:
            led_end_time = led_start_time + up[1]
        else: 
            led_end_time = led_start_time + up[0]
        axiss[up_ind].fill_between([led_start_time, led_end_time], y_low, y_high, 
                        alpha=0.1, edgecolor='r', facecolor='r', linewidth=2)
        
        axiss[up_ind].grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
        axiss[up_ind].xaxis.set_major_locator(x_locator)
        axiss[up_ind].tick_params(axis="x", direction="in", length=10, width=1, color="k")
        axiss[up_ind].grid(axis="y", color="k", alpha=.1, linewidth=.5)
        
        if type(parameter_keys) == tuple:
            axiss[up_ind].set_title(f'LED Intensity: {up[0]}  |  LED Duration: {up[1]}')
            ax.set_title('LED Intensity and Duration Sweep')
        else: 
            axiss[up_ind].set_title(f'{parameter_keys} = {up}')
            ax.set_title(f'{parameter_keys} Sweep')
        figg.legend()
        fh.legend()



# %% Call the plot functions
plotOptoStepSeries(parameter_keys=parameter_keys, unique_parameter_values=unique_parameter_values, mean_response= mean_response, sem_plus= sem_plus, sem_minus=sem_minus)
# %%
# Plots for duration sweep
plotOptoStepSeries(parameter_keys=parameter_key_duration, unique_parameter_values=unique_parameter_values_duration, mean_response= mean_response_duration, sem_plus= sem_plus_duration, sem_minus=sem_minus_duration)

# %%
# Plots for intensity sweep
plotOptoStepSeries(parameter_keys=parameter_key_intensity, unique_parameter_values=unique_parameter_values_intensity, mean_response= mean_response_intensity, sem_plus= sem_plus_intensity, sem_minus=sem_minus_intensity)

# %%
