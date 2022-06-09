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
import os

# %% LOADING UP THE DATA

experiment_file_directory = '/Users/averykrieger/Documents/local_data_repo/20220527'
#experiment_file_directory = '/Users/averykrieger/Documents/local_data_repo/example_data/responses/Bruker'

experiment_file_name = '2022-05-27'
series_number = 16
opto_condition = True
file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
save_path='/Users/averykrieger/Documents/local_data_repo/figs/'

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=False)

# %% ROIS AND RESPONSES

# getRoiResponses() wants a ROI set name, returns roi_data (dict)
roi_data = ID.getRoiResponses('medulla_rois')
roi_data.keys()

np.unique(ID.getEpochParameters('current_spatial_period'))

# %%
target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
target_tf = [0.5, 1, 2, 4]
fh, ax = plt.subplots((len(target_sp)), len(target_tf), figsize=(12, 12))
[x.set_ylim([-0.1, +0.9]) for x in ax.ravel()] # list comprehension makes each axis have the same limit for comparisons
for sp_ind, sp in enumerate(target_sp):
    for tf_ind, tf in enumerate(target_tf):
        query = {'current_spatial_period': sp,
                 'current_temporal_frequency': tf,
                 'opto_stim': False}
        no_opto_trials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=query)

        query = {'current_spatial_period': sp,
                 'current_temporal_frequency': tf,
                 'opto_stim': True}
        opto_trials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=query)

        ax[sp_ind, tf_ind].plot(roi_data['time_vector'], np.mean(no_opto_trials[0, :, :], axis=0), color='k', alpha=0.8)
        ax[sp_ind, tf_ind].plot(roi_data['time_vector'], np.mean(opto_trials[0, :, :], axis=0), color='r', alpha=0.8)
        #pyplot fill_between
        #struther and riser paper has temporal frequency tuning in on-off layers of medulla

ID.getRunParameters('pre_time')
roi_data.keys()

roi_data.get('roi_response')[0].shape
# %%
fh, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.plot(roi_data.get('roi_response')[0].T)

# %% Plot trial-average responses by specified parameter name
ID.getEpochParameters()
if opto_condition == True:
    optoQuery = {'opto_stim': True}
    optoTrials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=optoQuery)
    optoTrials.shape

    noOptoQuery = {'opto_stim': False}
    noOptoTrials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=noOptoQuery)
    noOptoTrials.shape

    opto_unique_parameter_values, opto_mean_response, opto_sem_response, opto_trial_response_by_stimulus = ID.getTrialAverages(optoTrials, parameter_key=('current_spatial_period', 'current_temporal_frequency', 'opto_stim'))
    print(len(opto_unique_parameter_values))
    opto_uni
    print(len(optoless_sem_response))


    optoless_unique_parameter_values, optoless_mean_response, optoless_sem_response, optoless_trial_response_by_stimulus = imaging_data.ImagingDataObject.getTrialAverages(epoch_response_matrix=noOptoTrials, parameter_key=('current_spatial_period', 'current_temporal_frequency', 'opto_stim'))
    print(len(opto_unique_parameter_values))
    opto_unique_parameter_values
    print(len(optoless_sem_response))
    optoless_unique_parameter_values


else:
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key=('current_spatial_period', 'current_temporal_frequency', 'opto_stim'))
    print(len(trial_response_by_stimulus))
    unique_parameter_values

ID.getTrialAverages
print(type(unique_parameter_values[1][2]))
len(unique_parameter_values)
mean_response.shape
# roi_data.get('epoch_response').shape
# roi_data.keys()


plotRoiResponsesByCondition(optoless_unique_parameter_values, opto_plot=False)
plotRoiResponsesByCondition(opto_unique_parameter_values, opto_plot=True)

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

# %% FILTERING FOR OPTO

query = {'opto_stim': True}
optoOff = shared_analysis.filterTrials(roi_data.get('epoch_resopnse'), ID, query=query)
unique_parameter_values2, mean_response2, sem_response2, trial_response_by_stimulus2 = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key=('current_spatial_period', 'current_temporal_frequency', 'opto_stim'))




# %% Some other convenience methods...

# Quickly look at roi responses (averaged across all trials)
shared_analysis.plotRoiResponses(ID, roi_name='medulla_rois')

# %%
shared_analysis.plotResponseByCondition(ID, roi_name='medulla_rois', condition='current_temporal_frequency')
