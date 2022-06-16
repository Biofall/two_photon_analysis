#!/usr/bin/env python
"""
Script to interact with ImagingDataObject, and extract data.

using MHT's visanalysis: https://github.com/ClandininLab/visanalysis
stripped down code to analyzie medulla responses from bruker experiments
Avery Krieger 6/6/22
"""

from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem as sem
import os

# %% LOADING UP THE DATA

experiment_file_directory = '/Users/averykrieger/Documents/local_data_repo/20220527'
experiment_file_name = '2022-05-27'
series_number = 3
opto_condition = False
roi_name = 'distal_medulla-2'

displayFix = False

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
save_path='/Users/averykrieger/Documents/local_data_repo/figs/'
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

                ax[sp_ind, tf_ind + (tf_number*roi_ind)].set_title(f'ROI:{roi_ind}|SpatPer:{sp}|TempFreq:{tf}', fontsize=14)
                #struther and riser paper has temporal frequency tuning in on-off layers of medulla

    if saveFig == True:
        fh.savefig(save_path+str(experiment_file_name)+' | SeriesNumber'+str(series_number)+' | '+str(roi_name) + '.pdf', dpi=300)

# %% CALLING THAT PLOT FUNCTION
plotConditionedROIResponses(ID, roi_data, opto_condition, saveFig)


# %% Visualing the entire ROI response
# fh, ax = plt.subplots(1, 1, figsize=(20, 4))
# ax.plot(roi_data.get('roi_response')[0].T)

# %% Plot trial-average responses by specified parameter name

plotRoiResponsesByCondition(opto_unique_parameter_values, opto_plot=True)
# %% Deprecated plotting function
def plotRoiResponsesByCondition(unique_parameter_values, opto_plot=False):

    # Find Number of ROIs
    roi_number = mean_response.shape[0]
    color = plt.cm.rainbow(np.linspace(0, 1, roi_number))

    # plot that thing
    fh, ax = plt.subplots(roi_number, len(unique_parameter_values), figsize=(40, 10))
    [plot_tools.cleanAxes(x) for x in ax.ravel()]
    for u_ind, up in enumerate(unique_parameter_values):
        for r in range(roi_number):
            ax[r, u_ind].plot(roi_data['time_vector'], mean_response[r, u_ind, :].T, color=color[r])
            ax[r, u_ind].set_title('rv = {}'.format(up))
            if opto_plot==False:
                fh.suptitle(f'Experiment Name: {experiment_file_name} | Series Number: {series_number} | Opto Off | Mean [Spatial Period, Temporal Frequency] combinations for each ROI')
            else:
                fh.suptitle(f'Experiment Name: {experiment_file_name} | Series Number: {series_number} | Opto On | Mean [Spatial Period, Temporal Frequency] combinations for each ROI')

    plt.savefig(save_path+str(experiment_file_name)+' | SeriesNumber'+str(series_number)+'.png', dpi=300)

# %% Some other convenience methods...

# Quickly look at roi responses (averaged across all trials)
shared_analysis.plotRoiResponses(ID, roi_name='medulla_rois')

shared_analysis.plotResponseByCondition(ID, roi_name='medulla_rois', condition='current_temporal_frequency')
