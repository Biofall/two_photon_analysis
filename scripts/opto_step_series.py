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
layer = mi1_prox_single

file_path = os.path.join(layer[0], layer[1] + ".hdf5")
ID = imaging_data.ImagingDataObject(file_path, layer[2], quiet=True)
roi_data = ID.getRoiResponses(layer[3])



# %% Pull the trial's data based on the parameter key
start_trial = 40

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
    start_trial = 40
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
    cmap = plt.get_cmap('Pastel1') # also 'cool' 'winter' 'PRGn' 'Pastel1' 'YlGnBu' 'twilight'
    colors = [cmap(i) for i in np.linspace(0.1, 1.0, len(unique_parameter_values))]
    x_locator = FixedLocator(list(range(-1, int(np.ceil(roi_data['time_vector'][-1]))+1)))


    # Plotting
    fh, ax = plt.subplots(1, 1, figsize=(16, 8))
    for up_ind, up in enumerate(unique_parameter_values): # up = unique parameter
        # Plot each individual unique parameter value on the same plot
        ax.plot(roi_data['time_vector'], mean_response[:, up_ind, :].mean(axis=0), color=colors[up_ind], alpha=0.9, label=up)
        ax.fill_between(roi_data['time_vector'], sem_plus[:, up_ind, :].mean(axis=0), 
                        sem_minus[:, up_ind, :].mean(axis=0),
                        color=colors[up_ind], alpha=0.1)


        # Plot each individual unique parameter value on its own plot
        figg, axiss = plt.subplots(1, 1, figsize=(16, 8))
        axiss.plot(roi_data['time_vector'], mean_response[:, up_ind, :].mean(axis=0), color='k', alpha=1)
        axiss.fill_between(roi_data['time_vector'], sem_plus[:, up_ind, :].mean(axis=0), 
                        sem_minus[:, up_ind, :].mean(axis=0),
                        color=colors[up_ind], alpha=0.8)
        # Opto Stim Plotting
        min_val = np.min(sem_minus[:, up_ind, :].mean(axis=0))
        max_val = np.max(sem_plus[:, up_ind, :].mean(axis=0))
        y_low = min_val-abs(0.01*min_val)
        y_high = max_val+abs(0.01*max_val)
        led_start_time = ID.getRunParameters('pre_time')
        if type(parameter_keys) == tuple:
            led_end_time = led_start_time + up[1]
        else: 
            led_end_time = led_start_time + up[0]
        axiss.fill_between([led_start_time, led_end_time], y_low, y_high, 
                        alpha=0.5, edgecolor='r', facecolor='none', linewidth=3, label='Opto')
        axiss.grid(axis="x", color="k", alpha=.1, linewidth=1, linestyle=":")
        axiss.xaxis.set_major_locator(x_locator)
        axiss.tick_params(axis="x", direction="in", length=10, width=1, color="k")
        axiss.grid(axis="y", color="k", alpha=.1, linewidth=.5)
        
        if type(parameter_keys) == tuple:
            axiss.set_title(f'LED Intensity: {up[0]}  |  LED Duration: {up[1]}')
            ax.set_title('LED Intensity and Duration Sweep')
        else: 
            axiss.set_title(f'{parameter_keys} = {up}')
            ax.set_title(f'{parameter_keys} Sweep')
        figg.legend()
        fh.legend()



# %% Call the plot functions
plotOptoStepSeries(parameter_keys=parameter_keys, unique_parameter_values=unique_parameter_values, mean_response= mean_response, sem_plus= sem_plus, sem_minus=sem_minus)
# %%
# Plots for duration sweep
plotOptoStepSeries(parameter_keys=parameter_key_duration, unique_parameter_values=unique_parameter_values_duration, mean_response= mean_response_duration, sem_plus= sem_plus_duration, sem_minus=sem_minus_duration)

# %%
# Plots for duration sweep
plotOptoStepSeries(parameter_keys=parameter_key_intensity, unique_parameter_values=unique_parameter_values_intensity, mean_response= mean_response_intensity, sem_plus= sem_plus_intensity, sem_minus=sem_minus_intensity)

# %%
