#!/usr/bin/env python
"""
Script to interact with ImagingDataObject, and extract data.

using MHT's visanalysis: https://github.com/ClandininLab/visanalysis
stripped down code to analyzie medulla responses from bruker experiments
Avery Krieger 6/6/22
"""

# %% Imports
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem as sem
import os

# %% LOADING UP THE DATA

experiment_file_directory = '/Users/averykrieger/Documents/local_data_repo/20220527'
experiment_file_name = '2022-05-27'
series_number = 2
opto_condition = True
roi_name = 'distal medulla-2'

displayFix = False

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
save_path='/Users/averykrieger/Documents/local_data_repo/figs/'
metric_save_path = save_path='/Users/averykrieger/Documents/local_data_repo/figs/metrics/'
saveFig = True

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
# displayFix is for the few trials in which something is blocking the display,
# specify for the ID to only use one display photodiode for timing
if displayFix == False:
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=False)
else:
    cfg_dict = {'timing_channel_ind': 1}
    ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True,
                                    cfg_dict=cfg_dict)

# getRoiResponses() wants a ROI set name, returns roi_data (dict)
roi_data = ID.getRoiResponses(roi_name)

roi_number = len(roi_data['roi_response'])

# %% PLOT functions

# For plotting the various conditions in the experiment
def plotConditionedROIResponses(ID, roi_data, opto_condition, saveFig):
    """
    -opto_condition: True or False
            True used when there is an optogenetic stimulus applied
            False used when there is no opto, only visual getStimulusTiming
    -saveFig: True or False
            True used when a 300dpi pdf of the plot is saved
            False when no plot saved
    """

    # Find the unique parameter space for this experiment
    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))
    tf_number = len(target_tf)
    roi_number = len(roi_data['roi_response'])
    color = plt.cm.tab20(np.linspace(0, 1, roi_number))

    # Make the figure axes
    fh, ax = plt.subplots(len(target_sp), len(target_tf) * roi_number, figsize=(25*roi_number, 25))
    [x.set_ylim([-0.55, +1.6]) for x in ax.ravel()] # list comprehension makes each axis have the same limit for comparisons
    for roi_ind in range(roi_number):
        for sp_ind, sp in enumerate(target_sp):
            for tf_ind, tf in enumerate(target_tf):
                # For No Opto Condition (light off)
                query = {'current_spatial_period': sp,
                         'current_temporal_frequency': tf,
                         'opto_stim': False}
                no_opto_trials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=query)
                noOptoMean = np.mean(no_opto_trials[roi_ind, :, :], axis=0)
                nopt_sem_plus = noOptoMean + sem(no_opto_trials[roi_ind, :, :], axis=0)
                nopt_sem_minus = noOptoMean - sem(no_opto_trials[roi_ind, :, :], axis=0)
                ax[sp_ind, tf_ind + (tf_number*roi_ind)].plot(roi_data['time_vector'],
                    noOptoMean, color='m', alpha=1.0)
                ax[sp_ind, tf_ind + (tf_number*roi_ind)].fill_between(roi_data['time_vector'],
                    nopt_sem_plus, nopt_sem_minus, color=color[roi_ind], alpha=0.8)

                if opto_condition == True:
                    # For Opto Condition (light on)
                    query = {'current_spatial_period': sp,
                             'current_temporal_frequency': tf,
                             'opto_stim': True}
                    opto_trials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=query)
                    optoMean = np.mean(opto_trials[roi_ind, :, :], axis=0)
                    opto_sem_plus = optoMean + sem(opto_trials[roi_ind, :, :], axis=0)
                    opto_sem_minus = optoMean - sem(opto_trials[roi_ind, :, :], axis=0)
                    ax[sp_ind, tf_ind + (tf_number*roi_ind)].plot(roi_data['time_vector'], optoMean, color='k', alpha=0.9)
                    ax[sp_ind, tf_ind + (tf_number*roi_ind)].fill_between(roi_data['time_vector'],
                        opto_sem_plus, opto_sem_minus, color='g', alpha=0.6)

                ax[sp_ind, tf_ind + (tf_number*roi_ind)].set_title(f'ROI:{roi_ind}|SpatPer:{sp}|TempFreq:{tf}', fontsize=20)
                #struther and riser paper has temporal frequency tuning in on-off layers of medulla

    if saveFig == True:
        fh.savefig(save_path+str(experiment_file_name)+' | SeriesNumber'+str(series_number)+' | '+str(roi_name) + '.pdf', dpi=300)

# %% CALLING THAT PLOT FUNCTION
plotConditionedROIResponses(ID, roi_data, opto_condition, saveFig)

# %% Find Mean and Maxing by condition
def getResponseMetrics(ID, roi_name, opto_condition, silent = True):

    roi_data = ID.getRoiResponses(roi_name)
    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))
    tf_number = len(target_tf)
    sp_number = len(target_sp)
    roi_number = len(roi_data['roi_response'])
    allMaxes = np.empty((roi_number, sp_number, tf_number), float)
    allMeans = np.empty((roi_number, sp_number, tf_number), float)

    for roi_ind in range(roi_number):
        for sp_ind, sp in enumerate(target_sp):
            for tf_ind, tf in enumerate(target_tf):
                query = {'current_spatial_period': sp,
                         'current_temporal_frequency': tf,
                         'opto_stim': opto_condition}
                filtered_trials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=query)
                #print(f'ROI: {roi_ind} | filtered_trials shape = {filtered_trials[roi_ind,:,:].shape}')
                mean_filtered_trials = np.mean(filtered_trials[roi_ind, :, :], axis=0)

                trials_amp = ID.getResponseAmplitude(mean_filtered_trials, metric= 'max')
                trials_mean = ID.getResponseAmplitude(mean_filtered_trials, metric= 'mean')

                allMaxes[roi_ind, sp_ind, tf_ind] = trials_amp
                allMeans[roi_ind, sp_ind, tf_ind] = trials_mean

    if silent == False:
        print('======================Response Metrics======================')
        print(f'Number of ROIs = {roi_number}')
        print(f'Spatial Periods = {target_sp}')
        print(f'Temporal Frequencies = {target_tf}')
        print(f'Size should thus be: ({roi_number}, {tf_number}, {sp_number})')
        print(f'Size of allMaxes:    {allMaxes.shape}')
        print('============================================================')

    return allMaxes, allMeans

# %%
# now, import to pandas DF. in this df, the  maxes and means would be appended to a df
# the fileName, series Number, opto condition, and roi_name should be added  as columns

# %% Plotting the max and mean metrics

def plotROIResponsesMetrics(noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number, saveFig=True):

    saveFig = True
    # Collapse across stimulus space.
    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))

    # First, plot max and avgs of spatial period, collapsed across temp freq:
    fh, ax = plt.subplots(2, roi_number, figsize=(20*roi_number, 20))
    for roi_ind in range(roi_number):
        # Max Values
        nopto_spatial_per_max_max = np.max(noptoMaxes[roi_ind,:,:], axis = 1)
        ax[0, roi_ind].plot(target_sp, noptoMaxes[roi_ind,:,:], 'b^', alpha=0.4, markersize=20)
        ax[0, roi_ind].plot(target_sp, nopto_spatial_per_max_max, '-bo', linewidth=6, alpha=0.8)

        yopto_spatial_per_max_max = np.max(yoptoMaxes[roi_ind,:,:], axis = 1)
        ax[0, roi_ind].plot(target_sp, yoptoMaxes[roi_ind,:,:], 'r^', alpha=0.4, markersize=20)
        ax[0, roi_ind].plot(target_sp, yopto_spatial_per_max_max, '-ro', linewidth=6, alpha=0.8)

        ax[0, roi_ind].set_title(f'ROI:{roi_ind}| Max Respone by SpatPer', fontsize=20)

        # Mean Values
        nopto_spatial_per_mean = np.mean(noptoMeans[roi_ind,:,:], axis = 1)
        ax[1, roi_ind].plot(target_sp, noptoMeans[roi_ind,:,:], 'bP', alpha=0.4, markersize=20)
        ax[1, roi_ind].plot(target_sp, nopto_spatial_per_mean, '-bo', linewidth=6, alpha=0.8)

        yopto_spatial_per_mean = np.mean(yoptoMeans[roi_ind,:,:], axis = 1)
        ax[1, roi_ind].plot(target_sp, yoptoMeans[roi_ind,:,:], 'rP', alpha=0.4, markersize=20)
        ax[1, roi_ind].plot(target_sp, yopto_spatial_per_mean, '-ro', linewidth=6, alpha=0.8)

        ax[1, roi_ind].set_title(f'ROI:{roi_ind}| Mean Respone by SpatPer', fontsize=20)

    fh.suptitle(f'{experiment_file_name} | {series_number} | {roi_name} | SpatPer', fontsize=20)
    if saveFig == True:
        fh.savefig(metric_save_path+str(experiment_file_name)+' | SeriesNumber'+str(series_number)+' | '+str(roi_name) +' | ' + 'SpatPer' '.pdf', dpi=300)

    # Second, plot max and avgs of temporal frequencies, collapsed across spatial_periods:
    fh, ax = plt.subplots(2, roi_number, figsize=(20*roi_number, 20))
    # Note the transpose in the scatterplot because we're looking at the flipped matrix for temporal frequency
    for roi_ind in range(roi_number):
        # Max Values
        nopto_temp_freq_max_max = np.max(noptoMaxes[roi_ind,:,:], axis = 0)
        ax[0, roi_ind].plot(target_tf, noptoMaxes[roi_ind,:,:].T, 'b^', alpha=0.4, markersize=20)
        ax[0, roi_ind].plot(target_tf, nopto_temp_freq_max_max, '-bo', linewidth=6, alpha=0.8)

        yopto_temp_freq_max_max = np.max(yoptoMaxes[roi_ind,:,:], axis = 0)
        ax[0, roi_ind].plot(target_tf, yoptoMaxes[roi_ind,:,:].T, 'r^', alpha=0.4, markersize=20)
        ax[0, roi_ind].plot(target_tf, yopto_temp_freq_max_max, '-ro', linewidth=6, alpha=0.8)

        ax[0, roi_ind].set_title(f'ROI:{roi_ind}| Max Respone by TempFreq', fontsize=20)

        # Mean Values
        nopto_temp_freq_mean = np.mean(noptoMeans[roi_ind,:,:], axis = 0)
        ax[1, roi_ind].plot(target_tf, noptoMeans[roi_ind,:,:].T, 'bP', alpha=0.4, markersize=20)
        ax[1, roi_ind].plot(target_tf, nopto_temp_freq_mean, '-bo', linewidth=6, alpha=0.8)

        yopto_temp_freq_mean = np.mean(yoptoMeans[roi_ind,:,:], axis = 0)
        ax[1, roi_ind].plot(target_tf, yoptoMeans[roi_ind,:,:].T, 'rP', alpha=0.4, markersize=20)
        ax[1, roi_ind].plot(target_tf, yopto_temp_freq_mean, '-ro', linewidth=6, alpha=0.8)

        ax[1, roi_ind].set_title(f'ROI:{roi_ind}| Mean Respone by TempFreq', fontsize=20)

    fh.suptitle(f'{experiment_file_name} | {series_number} | {roi_name} | TempFreq', fontsize=20)
    if saveFig == True:
        fh.savefig(metric_save_path+str(experiment_file_name)+' | SeriesNumber'+str(series_number)+' | '+str(roi_name) + ' | ' + 'TempFreq' '.pdf', dpi=300)

# %% Call those functions from above
noptoMaxes, noptoMeans = getResponseMetrics(ID, roi_name, opto_condition = False, silent = True)
yoptoMaxes, yoptoMeans = getResponseMetrics(ID, roi_name, opto_condition = True, silent = True)

# call the below function for convenience
plotROIResponsesMetrics(noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number)
