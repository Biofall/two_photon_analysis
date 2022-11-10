#!/usr/bin/env python
"""
Functions for loading data, manipulating it, and plotting it -
 specifically the visanalysis ImagingData object

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
import warnings
from pathlib import Path
from scipy import interpolate
from two_photon_analysis.utils import getRandVal

# %% Data Loader
# takes in all the file and exp deets and returns an ID and roi_data
def dataLoader(file_directory, save_path, file_name, series_number, roi_name, opto_condition, displayFix = False, saveFig = True, dff = False):

    displayFix = displayFix
    file_path = os.path.join(file_directory, file_name + '.hdf5')
    metric_save_path = os.path.join(save_path, 'metrics/')

    # ImagingDataObject wants a path to an hdf5 file and a series number from that file
    # displayFix is for the few trials in which something is blocking the display,
    # specify for the ID to only use one display photodiode for timing
    if displayFix == False:
        ID = imaging_data.ImagingDataObject(file_path,
                                            series_number,
                                            quiet=True)
    else:
        cfg_dict = {'timing_channel_ind': 1}
        ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True,
                                        cfg_dict=cfg_dict)

    # getRoiResponses() wants a ROI set name, returns roi_data (dict)
    roi_data = ID.getRoiResponses(roi_name)

    return ID, roi_data

# %% Stimulus Combo Checker

def stimComboChecker(ID):
    from collections import Counter

    # First find the numbers of unique stimuli:
    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))
    opto_cons = np.unique(ID.getEpochParameters('opto_stim'))
    sp_number = len(target_sp)
    tf_number = len(target_tf)
    opto_number = len(opto_cons)
    total_supposed_combos = sp_number*tf_number*opto_number

    # Now see what was actually shown
    current_opto_stim = ID.getEpochParameters('opto_stim')
    spatial_period_list = ID.getEpochParameters('current_spatial_period')
    temporal_frequency_list = ID.getEpochParameters('current_temporal_frequency')

    tab_dict = Counter(zip(spatial_period_list,temporal_frequency_list, current_opto_stim))
    uniqueComboNumber = len(tab_dict)

    matches = True
    if total_supposed_combos != uniqueComboNumber:
        matches = False

    print('\n\n')
    print('================================================================')
    print('````````````````````Stim Combo Checker Start````````````````````')
    print('................................................................')
    print('The possible number of stimuli are:')
    print(f'Spatial Period = {sp_number}')
    print(f'Temporal Frequency = {tf_number}')
    print(f'Opto Conditions = {opto_number}')
    print(f'Total Possible Conditions = {total_supposed_combos}')
    print(f'----- The ACTUAL shown stimuli total = {uniqueComboNumber}-----')
    if matches == True:
        print('These numbers MATCH! Yayyyy')
    elif matches == False:
        print('These numbers DO NOT Match! This is bad and/or sad')
    print('``````````````````````````````````````````` `````````````````````')
    print(f'````````````````````Stim Combo Checker End``````````````````````')
    print('================================================================')
    print('\n\n')

# %% Plotting things

# Plot conditioned ROI Responses
# For plotting the various conditions in the experiment
def plotConditionedROIResponses(ID, roi_data, opto_condition, vis_stim_type, saveFig, save_path, saveName, alt_pre_time = 0, dff=True, df = False):
    """
    -opto_condition: True or False
            True used when there is an optogenetic stimulus applied
            False used when there is no opto, only visual getStimulusTiming
    -saveFig: True or False
            True used when a 300dpi pdf of the plot is saved
            False when no plot saved
    """
    time_vector, epoch_response = getAltEpochResponseMatrix(ID, np.vstack(roi_data['roi_response']), alt_pre_time=alt_pre_time, dff=dff, df=df)
    roi_number = len(roi_data['roi_response'])
    # Find the unique parameter space for this experiment
    if vis_stim_type == 'spatiotemporal':
        target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
        target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))
        tf_number = len(target_tf)

        # Make the figure axes
        fh, ax = plt.subplots(len(target_sp), len(target_tf) * roi_number, figsize=(25*roi_number, 25))
    elif vis_stim_type == 'single':
        opto_start_times = np.unique(ID.getEpochParameters('current_opto_start_time'))

        # Make the figure axes
        fh, ax = plt.subplots(len(opto_start_times), roi_number, figsize=(25*roi_number, 25))
    else:
        raise Exception('vis_stim_type should be spatiotemporal or single. It was: {}'.format(vis_stim_type))

    color = plt.cm.tab20(np.linspace(0, 1, roi_number))


    #[x.set_ylim([-0.55, +1.6]) for x in ax.ravel()] # list comprehension makes each axis have the same limit for comparisons
    for roi_ind in range(roi_number):
        # in the spatiotemporal condition:
        if vis_stim_type == 'spatiotemporal':
            for sp_ind, sp in enumerate(target_sp):
                for tf_ind, tf in enumerate(target_tf):
                    # For No Opto Condition (light off)
                    query = {'current_spatial_period': sp,
                             'current_temporal_frequency': tf,
                             'opto_stim': False}
                    no_opto_trials = shared_analysis.filterTrials(epoch_response, ID, query=query)
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
                        opto_trials = shared_analysis.filterTrials(epoch_response, ID, query=query)
                        optoMean = np.mean(opto_trials[roi_ind, :, :], axis=0)
                        opto_sem_plus = optoMean + sem(opto_trials[roi_ind, :, :], axis=0)
                        opto_sem_minus = optoMean - sem(opto_trials[roi_ind, :, :], axis=0)
                        ax[sp_ind, tf_ind + (tf_number*roi_ind)].plot(roi_data['time_vector'], optoMean, color='k', alpha=0.9)
                        ax[sp_ind, tf_ind + (tf_number*roi_ind)].fill_between(roi_data['time_vector'],
                            opto_sem_plus, opto_sem_minus, color='g', alpha=0.6)

                    ax[sp_ind, tf_ind + (tf_number*roi_ind)].set_title(f'ROI:{roi_ind}|SpatPer:{sp}|TempFreq:{tf}', fontsize=20)
                    #struther and riser paper has temporal frequency tuning in on-off layers of medulla
        # in the single spatiotemporal visual stimuli condition
        elif vis_stim_type == 'single':
            for opto_time_ind, oi in enumerate(opto_start_times):
                query = {'current_opto_start_time': oi, 'opto_stim': False}
                # For no opto condition
                no_opto_trials = shared_analysis.filterTrials(epoch_response, ID, query=query)
                noOptoMean = np.mean(no_opto_trials[roi_ind, :, :], axis=0)
                nopt_sem_plus = noOptoMean + sem(no_opto_trials[roi_ind, :, :], axis=0)
                nopt_sem_minus = noOptoMean - sem(no_opto_trials[roi_ind, :, :], axis=0)
                ax[opto_time_ind, roi_ind].plot(roi_data['time_vector'],
                    noOptoMean, color='m', alpha=1.0)
                ax[opto_time_ind, roi_ind].fill_between(roi_data['time_vector'],
                    nopt_sem_plus, nopt_sem_minus, color=color[roi_ind], alpha=0.8)

                if opto_condition == True:
                    query = {'current_opto_start_time': oi, 'opto_stim': True}
                    # For no opto condition
                    opto_trials = shared_analysis.filterTrials(epoch_response, ID, query=query)
                    optoMean = np.mean(opto_trials[roi_ind, :, :], axis=0)
                    opto_sem_plus = optoMean + sem(opto_trials[roi_ind, :, :], axis=0)
                    opto_sem_minus = optoMean - sem(opto_trials[roi_ind, :, :], axis=0)
                    ax[opto_time_ind, roi_ind].plot(roi_data['time_vector'],
                        optoMean, color='k', alpha=0.9)
                    ax[opto_time_ind, roi_ind].fill_between(roi_data['time_vector'],
                        opto_sem_plus, opto_sem_minus, color='g', alpha=0.6)

                ax[opto_time_ind, roi_ind].set_title(f'ROI:{roi_ind} | Opto Start:{oi}', fontsize=20)

    if saveFig == True:
        #save_path = '/Users/averykrieger/Documents/local_data_repo/figs/'

        if df == True:
            fh.savefig(saveName, dpi=300)
        else:
            fh.savefig(saveName, dpi=300)

# Plotting heatmaps for opto and no opto comparisons
def plotOptoHeatmaps(ID, roi_data, noptoMaxes, yoptoMaxes, noptoMeans, yoptoMeans, saveName, dff = True, df = False):

    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))

    # The abs catches when a neg and pos value are close, inflating the denominator
    norm_max = (yoptoMaxes-noptoMaxes) / np.nanmax(np.abs(yoptoMaxes)+np.abs(noptoMaxes))
    norm_mean = (yoptoMeans-noptoMeans) / np.nanmax(np.abs(yoptoMeans)+np.abs(noptoMeans))

    norm = MidPointNorm(midpoint=0, vmin=-.45, vmax=.45, clip=False)

    saveFig = True

    roi_number = len(roi_data['roi_response'])

    fh, ax = plt.subplots(2, roi_number, figsize=(10*roi_number, 20))
    for roi_ind in range(roi_number):
        # the maxes
        maxImage = ax[0, roi_ind].imshow(norm_max[roi_ind,:,:], cmap = 'coolwarm', norm = norm)
        ax[0, roi_ind].set_xticks(np.arange(len(target_sp)), labels=target_sp)
        ax[0, roi_ind].set_yticks(np.arange(len(target_tf)), labels=target_tf)
        ax[0, roi_ind].set_title(f'ROI:{roi_ind} | Max', fontsize=40)
        # Loop over data dimensions and create text annotations.
        for i in range(len(target_sp)):
            for j in range(len(target_tf)):
                text = ax[0, roi_ind].text(j, i, round(norm_max[roi_ind, i, j], 3),
                               ha="center", va="center", color="w")

        # The means
        meanImage = ax[1, roi_ind].imshow(norm_mean[roi_ind,:,:], cmap = 'coolwarm', norm = norm)
        ax[1, roi_ind].set_xticks(np.arange(len(target_sp)), labels=target_sp)
        ax[1, roi_ind].set_yticks(np.arange(len(target_tf)), labels=target_tf)
        ax[1, roi_ind].set_title(f'ROI:{roi_ind} | Mean', fontsize=40)
        # Loop over data dimensions and create text annotations.
        for i in range(len(target_sp)):
            for j in range(len(target_tf)):
                text = ax[1, roi_ind].text(j, i, round(norm_mean[roi_ind, i, j], 3),
                               ha="center", va="center", color="w")
    fh.suptitle(f'{saveName} | Heatmap of No Opto / Opto', fontsize=40)

    if saveFig == True:
        fh.savefig(saveName+'heatmap.pdf', dpi=300)



#  Cross-roi heatmap summaries
def plotOptoHeatmapsAcrossROIs(ID, roi_data, noptoMaxes, yoptoMaxes, noptoMeans, yoptoMeans, saveName, dff = True, df = False, saveFig = True):
    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))

    # The abs catches when a neg and pos value are close, inflating the denominator
    norm_max = (yoptoMaxes-noptoMaxes) / np.nanmax(np.abs(yoptoMaxes)+np.abs(noptoMaxes))
    norm_mean = (yoptoMeans-noptoMeans) / np.nanmax(np.abs(yoptoMeans)+np.abs(noptoMeans))

    norm_max_mean = np.mean(norm_max, axis = 0)
    norm_mean_mean = np.mean(norm_mean, axis = 0)

    # This centers the colormap at midpoint
    #   The most intense color is set at vmin/vmax
    #
    norm = MidPointNorm(midpoint=0, vmin=-.45, vmax=.45, clip=False)

    fh, ax = plt.subplots(2,1, figsize=(10,20))
    plt.subplots_adjust(bottom=0.1, right=0.9, wspace=0.2, hspace=0.2)
    maxImage = ax[0].imshow(norm_max_mean, cmap = 'coolwarm', norm = norm)
    ax[0].set_xticks(np.arange(len(target_sp)), labels=target_sp)
    ax[0].set_yticks(np.arange(len(target_tf)), labels=target_tf)
    ax[0].set_title(f'Max Normalized Diff Across ROIs')
    ax[0].set_xlabel('Spatial Period')
    ax[0].set_ylabel('Temporal Frequency')
    for i in range(len(target_sp)):
        for j in range(len(target_tf)):
            text = ax[0].text(j, i, round(norm_max_mean[i, j], 3),
                           ha="center", va="center", color="w")

    # Means
    meanImage = ax[1].imshow(norm_mean_mean, cmap = 'coolwarm', norm = norm)
    ax[1].set_xticks(np.arange(len(target_sp)), labels=target_sp)
    ax[1].set_yticks(np.arange(len(target_tf)), labels=target_tf)
    ax[1].set_title(f'Mean Normalized Diff Across ROIs')
    ax[1].set_xlabel('Spatial Period')
    ax[1].set_ylabel('Temporal Frequency')
    for i in range(len(target_sp)):
        for j in range(len(target_tf)):
            text = ax[1].text(j, i, round(norm_mean_mean[i, j], 3),
                           ha="center", va="center", color="w")

    fh.suptitle(f'{saveName} | Heatmap of (No Opto-Yes Opto)/(No Opto+Yes Opto)')

    if saveFig == True:
        fh.savefig(saveName+'heatmapAcrossRois.pdf', dpi=300)

# Plotting the max and mean metrics
def plotROIResponsesMetrics(ID, plotTitle, figTitle, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number, vis_stim_type='spatiotemporal', saveFig=True):

    # First, plot max and avgs of spatial period, collapsed across temp freq:
    if vis_stim_type == 'spatiotemporal':
        fh, ax = plt.subplots(2, roi_number, figsize=(20*roi_number, 20))
        # Collapse across stimulus space.
        target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
        target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))

        for roi_ind in range(roi_number):
            # Max Values
            nopto_spatial_per_max_max = np.nanmean(noptoMaxes[roi_ind,:,:], axis = 1)
            ax[0, roi_ind].plot(target_sp, noptoMaxes[roi_ind,:,:], 'k^', alpha=0.4, markersize=20)
            ax[0, roi_ind].plot(target_sp, nopto_spatial_per_max_max, '-ko', linewidth=6, alpha=0.8)

            yopto_spatial_per_max_max = np.nanmean(yoptoMaxes[roi_ind,:,:], axis = 1)
            ax[0, roi_ind].plot(target_sp, yoptoMaxes[roi_ind,:,:], 'g^', alpha=0.4, markersize=20)
            ax[0, roi_ind].plot(target_sp, yopto_spatial_per_max_max, '-go', linewidth=6, alpha=0.8)

            ax[0, roi_ind].set_title(f'ROI:{roi_ind}| Max Respone by SpatPer', fontsize=20)

            # Mean Values
            nopto_spatial_per_mean = np.nanmean(noptoMeans[roi_ind,:,:], axis = 1)
            ax[1, roi_ind].plot(target_sp, noptoMeans[roi_ind,:,:], 'kP', alpha=0.4, markersize=20)
            ax[1, roi_ind].plot(target_sp, nopto_spatial_per_mean, '-ko', linewidth=6, alpha=0.8)

            yopto_spatial_per_mean = np.nanmean(yoptoMeans[roi_ind,:,:], axis = 1)
            ax[1, roi_ind].plot(target_sp, yoptoMeans[roi_ind,:,:], 'gP', alpha=0.4, markersize=20)
            ax[1, roi_ind].plot(target_sp, yopto_spatial_per_mean, '-go', linewidth=6, alpha=0.8)

            ax[1, roi_ind].set_title(f'ROI:{roi_ind}| Mean Respone by SpatPer', fontsize=20)

        fh.suptitle(plotTitle + f' | SpatPer', fontsize=20)
        if saveFig == True:
            fh.savefig(figTitle + 'EachRoiSpatPer.pdf', dpi=300)

        # Second, plot max and avgs of temporal frequencies, collapsed across spatial_periods:
        fh, ax = plt.subplots(2, roi_number, figsize=(20*roi_number, 20))
        # Note the transpose in the scatterplot because we're looking at the flipped matrix for temporal frequency
        for roi_ind in range(roi_number):
            # Max Values
            nopto_temp_freq_max_max = np.nanmean(noptoMaxes[roi_ind,:,:], axis = 0)
            ax[0, roi_ind].plot(target_tf, noptoMaxes[roi_ind,:,:].T, 'k^', alpha=0.4, markersize=20)
            ax[0, roi_ind].plot(target_tf, nopto_temp_freq_max_max, '-ko', linewidth=6, alpha=0.8)

            yopto_temp_freq_max_max = np.nanmean(yoptoMaxes[roi_ind,:,:], axis = 0)
            ax[0, roi_ind].plot(target_tf, yoptoMaxes[roi_ind,:,:].T, 'g^', alpha=0.4, markersize=20)
            ax[0, roi_ind].plot(target_tf, yopto_temp_freq_max_max, '-go', linewidth=6, alpha=0.8)

            ax[0, roi_ind].set_title(f'ROI:{roi_ind}| Max Respone by TempFreq', fontsize=20)

            # Mean Values
            nopto_temp_freq_mean = np.nanmean(noptoMeans[roi_ind,:,:], axis = 0)
            ax[1, roi_ind].plot(target_tf, noptoMeans[roi_ind,:,:].T, 'kP', alpha=0.4, markersize=20)
            ax[1, roi_ind].plot(target_tf, nopto_temp_freq_mean, '-ko', linewidth=6, alpha=0.8)

            yopto_temp_freq_mean = np.nanmean(yoptoMeans[roi_ind,:,:], axis = 0)
            ax[1, roi_ind].plot(target_tf, yoptoMeans[roi_ind,:,:].T, 'gP', alpha=0.4, markersize=20)
            ax[1, roi_ind].plot(target_tf, yopto_temp_freq_mean, '-go', linewidth=6, alpha=0.8)

            ax[1, roi_ind].set_title(f'ROI:{roi_ind}| Mean Respone by TempFreq', fontsize=20)

        if saveFig == True:
            fh.savefig(figTitle + 'EachRoiTempFreq.pdf', dpi=300)

    elif vis_stim_type == 'single':
        opto_start_times = np.unique(ID.getEpochParameters('current_opto_start_time'))

        # Plot max and avgs of for each ROI and each opto start time:
        fh, ax = plt.subplots(2, roi_number, figsize=(20*roi_number, 20))
        for roi_ind in range(roi_number):
            # Max Values
            ax[0, roi_ind].plot(opto_start_times, noptoMaxes[roi_ind,:], '-k^', alpha=0.8, markersize=20)
            ax[0, roi_ind].plot(opto_start_times, yoptoMaxes[roi_ind,:], '-g^', alpha=0.8, markersize=20)
            ax[0, roi_ind].set_title(f'ROI:{roi_ind}| Max Respone by Opto Start Time', fontsize=20)

            # Mean Values
            ax[1, roi_ind].plot(opto_start_times, noptoMeans[roi_ind,:], '-kP', alpha=0.8, markersize=20)
            ax[1, roi_ind].plot(opto_start_times, yoptoMeans[roi_ind,:], '-gP', alpha=0.8, markersize=20)
            ax[1, roi_ind].set_title(f'ROI:{roi_ind}| Mean Respone by Opto Start Time', fontsize=20)


        fh.suptitle(plotTitle + f' | Each ROI | opto start time', fontsize=20)
        if saveFig == True:
            figTitle = figTitle + ' | EachRoi.pdf'
            fh.savefig(figTitle, dpi=300)


# %%
def plotReponseMetricsAcrossROIs(ID, plotTitle, figTitle, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number, vis_stim_type, saveFig=True):
    import matplotlib.patches as mpatches

    # Collapse across stimulus space.

    fh, ax = plt.subplots(2, 1) #, figsize=(40, 40))
    plt.subplots_adjust(bottom=0.1, right=0.9, wspace=0.4, hspace=0.6)

    if vis_stim_type == 'spatiotemporal':
        target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
        target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))
        #calculate mean across rois. optoMaxes = ROI x SpatPer x TempFreq
        nopto_spatial_mean_max_across_rois= np.nanmean(np.nanmean(noptoMaxes, axis = 0), axis = 1)
        yopto_spatial_mean_max_across_rois= np.nanmean(np.nanmean(yoptoMaxes, axis = 0), axis = 1)
        nopto_spatial_mean_mean_across_rois = np.nanmean(np.nanmean(noptoMeans, axis = 0), axis = 1)
        yopto_spatial_mean_mean_across_rois = np.nanmean(np.nanmean(yoptoMeans, axis = 0), axis = 1)
        for roi_ind in range(roi_number):
            # Max Values for each ROI get plotted, avg across TF
            nopto_spatial_per_max_max = np.nanmean(noptoMaxes[roi_ind,:,:], axis = 1)
            ax[0].plot(target_sp, nopto_spatial_per_max_max, 'k^', alpha=0.4)

            yopto_spatial_per_max_max = np.nanmean(yoptoMaxes[roi_ind,:,:], axis = 1)
            ax[0].plot(target_sp, yopto_spatial_per_max_max, 'g^', alpha=0.4)

            ax[0].set_title('Max Respone by SpatPer')

            # Mean Values for each ROI get plotted, avg across TF
            nopto_spatial_per_mean = np.nanmean(noptoMeans[roi_ind,:,:], axis = 1)
            ax[1].plot(target_sp, nopto_spatial_per_mean, 'k^', alpha=0.4)

            yopto_spatial_per_mean = np.nanmean(yoptoMeans[roi_ind,:,:], axis = 1)
            ax[1].plot(target_sp, yopto_spatial_per_mean, 'g^', alpha=0.4)

            ax[1].set_title('Mean Respone by SpatPer')

        ax[0].plot(target_sp, nopto_spatial_mean_max_across_rois, '-ko', alpha=0.8)
        ax[0].plot(target_sp, yopto_spatial_mean_max_across_rois, '-go', alpha=0.8)

        ax[1].plot(target_sp, nopto_spatial_mean_mean_across_rois, '-ko', alpha=0.8)
        ax[1].plot(target_sp, yopto_spatial_mean_mean_across_rois, '-go', alpha=0.8)

        green_patch = mpatches.Patch(color='green', label='With Opto')
        black_patch = mpatches.Patch(color='black', label='No Opto')
        ax[1].legend(handles=[green_patch, black_patch])

        fh.suptitle(plotTitle + ' | SpatPer')
        if saveFig == True:
            fh.savefig(figTitle + 'AcrossROISpatPer.pdf', dpi=300)

        # Second, plot max and avgs of temporal frequencies, collapsed across spatial_periods:
        fh, ax = plt.subplots(2, 1)
        plt.subplots_adjust(bottom=0.1, right=0.9, wspace=0.4, hspace=0.6)
        nopto_temporal_mean_max_across_rois= np.nanmean(np.nanmean(noptoMaxes[:,:,:], axis = 0), axis = 0)
        yopto_temporal_mean_max_across_rois= np.nanmean(np.nanmean(yoptoMaxes[:,:,:], axis = 0), axis = 0)
        nopto_temporal_mean_mean_across_rois = np.nanmean(np.nanmean(noptoMeans[:,:,:], axis = 0), axis = 0)
        yopto_temporal_mean_mean_across_rois = np.nanmean(np.nanmean(yoptoMeans[:,:,:], axis = 0), axis = 0)

        for roi_ind in range(roi_number):
            # Max Values
            nopto_temporal_per_max_max = np.nanmean(noptoMaxes[roi_ind,:,:], axis = 0)
            ax[0].plot(target_tf, nopto_temporal_per_max_max, 'k^', alpha=0.4)

            yopto_temporal_per_max_max = np.nanmean(yoptoMaxes[roi_ind,:,:], axis = 0)
            ax[0].plot(target_tf, yopto_temporal_per_max_max, 'g^', alpha=0.4)

            ax[0].set_title('Max Respone by TempFreq')

            # Mean Values
            nopto_temporal_per_mean = np.nanmean(noptoMeans[roi_ind,:,:], axis = 0)
            ax[1].plot(target_tf, nopto_temporal_per_mean, 'k^', alpha=0.4)

            yopto_temporal_per_mean = np.nanmean(yoptoMeans[roi_ind,:,:], axis = 0)
            ax[1].plot(target_tf, yopto_temporal_per_mean, 'g^', alpha=0.4)

            ax[1].set_title('Mean Respone by Temp Freq')

        ax[0].plot(target_tf, nopto_temporal_mean_max_across_rois, '-ko', alpha=0.8)
        ax[0].plot(target_tf, yopto_temporal_mean_max_across_rois, '-g', alpha=0.8)

        ax[1].plot(target_tf, nopto_temporal_mean_mean_across_rois, '-ko', alpha=0.8)
        ax[1].plot(target_tf, yopto_temporal_mean_mean_across_rois, '-go', alpha=0.8)

        green_patch = mpatches.Patch(color='green', label='With Opto')
        black_patch = mpatches.Patch(color='black', label='No Opto')
        ax[1].legend(handles=[green_patch, black_patch])

        fh.suptitle(plotTitle + ' | TempFreq')
        if saveFig == True:
            fh.savefig(figTitle + 'AcrossROITempFreq.pdf', dpi=300)

    elif vis_stim_type == 'single':
        opto_start_times = np.unique(ID.getEpochParameters('current_opto_start_time'))
        # max_norm_val = np.max(np.vstack([np.max(noptoMaxes, axis=1), np.max(yoptoMaxes, axis=1)]), axis=0)
        # mean_norm_val = np.max(np.vstack([np.max(noptoMeans, axis=1), np.max(yoptoMeans, axis=1)]), axis=0)
        # normalize_curves = True
        # if normalize_curves:
        #     noptoMaxes = noptoMaxes / max_norm_val[:, np.newaxis]
        #     yoptoMaxes = yoptoMaxes / max_norm_val[:, np.newaxis]
        #
        #     noptoMeans = noptoMeans / mean_norm_val[:, np.newaxis]
        #     yoptoMeans = yoptoMeans / mean_norm_val[:, np.newaxis]

        # calculate mean across ROIs. optoMaxes = ROI x opto_start_times
        nopto_max_across_rois= np.mean(noptoMaxes, axis = 0)
        yopto_max_across_rois= np.mean(yoptoMaxes, axis = 0)
        nopto_mean_across_rois = np.mean(noptoMeans, axis = 0)
        yopto_mean_across_rois = np.mean(yoptoMeans, axis = 0)

        for roi_ind in range(roi_number):
            # Max Values for each ROI get plotted
            ax[0].plot(opto_start_times, noptoMaxes[roi_ind, :], 'k^', alpha=0.4)
            ax[0].plot(opto_start_times, yoptoMaxes[roi_ind, :], 'g^', alpha=0.4)
            ax[0].set_title('Max Respone by Opto Start Time')

            # Mean Values for each ROI get plotted
            ax[1].plot(opto_start_times, noptoMeans[roi_ind, :], 'k^', alpha=0.4)
            ax[1].plot(opto_start_times, yoptoMeans[roi_ind, :], 'g^', alpha=0.4)
            ax[1].set_title('Mean Respone by Opto Start Time')

        ax[0].plot(opto_start_times, nopto_max_across_rois, '-ko', alpha=0.8)
        ax[0].plot(opto_start_times, yopto_max_across_rois, '-go', alpha=0.8)

        ax[1].plot(opto_start_times, nopto_mean_across_rois, '-ko', alpha=0.8)
        ax[1].plot(opto_start_times, yopto_mean_across_rois, '-go', alpha=0.8)

        green_patch = mpatches.Patch(color='green', label='With Opto')
        black_patch = mpatches.Patch(color='black', label='No Opto')
        ax[1].legend(handles=[green_patch, black_patch])

        fh.suptitle(plotTitle + ' | Opto Start Time')
        if saveFig == True:
            print('figTitle is:')
            print(figTitle)
            fh.savefig(figTitle, dpi=300)

            #fh.savefig(figTitle + 'Opto Start Time.pdf', dpi=300)

    else:
        raise Exception('vis_stim_type should be spatiotemporal or single. It was: {}'.format(vis_stim_type))


# %% Find Mean and Maxing by condition
def getResponseMetrics(ID, roi_data, dff, df, vis_stim_type, alt_pre_time = 0, opto_condition = True, silent = True):
    time_vector, epoch_response = getAltEpochResponseMatrix(ID, np.vstack(roi_data['roi_response']), alt_pre_time=alt_pre_time, dff=dff, df=df)

    roi_number = len(roi_data['roi_response'])

    if vis_stim_type == 'spatiotemporal':
        target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
        target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))
        tf_number = len(target_tf)
        sp_number = len(target_sp)
        allMaxes = np.empty((roi_number, sp_number, tf_number), float)
        allMaxes[:] = np.NaN
        allMeans = np.empty((roi_number, sp_number, tf_number), float)
        allMeans[:] = np.NaN

        for roi_ind in range(roi_number):
            for sp_ind, sp in enumerate(target_sp):
                for tf_ind, tf in enumerate(target_tf):
                    query = {'current_spatial_period': sp,
                             'current_temporal_frequency': tf,
                             'opto_stim': opto_condition}
                    filtered_trials = shared_analysis.filterTrials(epoch_response, ID, query=query)
                    #print(f'ROI: {roi_ind} | filtered_trials shape = {filtered_trials[roi_ind,:,:].shape}')
                    # Old Way:
                    # mean_filtered_trials = np.mean(filtered_trials[roi_ind, :, :], axis=0)
                    #
                    # trials_amp = ID.getResponseAmplitude(mean_filtered_trials, metric= 'max')
                    # trials_mean = ID.getResponseAmplitude(mean_filtered_trials, metric= 'mean')

                    # NEW WAY:
                    trials_max = ID.getResponseAmplitude(filtered_trials[roi_ind,:,:], metric = 'max')
                    trials_max_mean = np.nanmean(trials_max)
                    trials_mean = ID.getResponseAmplitude(filtered_trials[roi_ind,:,:], metric = 'mean')
                    trials_mean_mean = np.nanmean(trials_mean)

                    # # # DEBUGGING:
                    # print('\n\n')
                    # print('================== getRresponseMetrics Debugging! ============================')
                    # print('trials_max:')
                    # print(trials_max)
                    # print('\n\n')
                    # print('trials_max_mean:')
                    # print(trials_max_mean)
                    # print('\n\n')

                    # print(f'ROI: {roi_ind} | OT: {ot}')
                    # print(f'filtered trials size = {filtered_trials.shape}')
                    # print(f'trials_max shape: {trials_max.shape}')
                    # print(f'trials_max_mean: {trials_max_mean}')
                    # print(f'trials_mean shape: {trials_mean.shape}')
                    # print(f'trials_mean_mean: {trials_mean_mean}')

                    allMaxes[roi_ind, sp_ind, tf_ind] = trials_max_mean
                    allMeans[roi_ind, sp_ind, tf_ind] = trials_mean_mean

        if silent == False:
            print('\n\n')
            print('======================Response Metrics======================')
            print(f'Number of ROIs = {roi_number}')
            print(f'Spatial Periods = {target_sp}')
            print(f'Temporal Frequencies = {target_tf}')
            print(f'Size should thus be: ({roi_number}, {tf_number}, {sp_number})')
            print(f'Size of allMaxes:    {allMaxes.shape}')
            print('============================================================')
            print('\n\n')

    elif vis_stim_type == 'single':
        opto_start_times = np.unique(ID.getEpochParameters('current_opto_start_time'))
        opto_time_number = len(opto_start_times)
        allMaxes = np.empty((roi_number, opto_time_number), float)
        allMeans = np.empty((roi_number, opto_time_number), float)

        for roi_ind in range(roi_number):
            for opto_time_ind, ot in enumerate(opto_start_times):
                    query = {'current_opto_start_time': ot, 'opto_stim': opto_condition}
                    filtered_trials = shared_analysis.filterTrials(epoch_response, ID, query=query)
                    #OLD WAY
                    # mean_filtered_trials = np.mean(filtered_trials[roi_ind, :], axis=0)
                    #
                    # trials_amp = ID.getResponseAmplitude(mean_filtered_trials, metric= 'max')
                    # trials_mean = ID.getResponseAmplitude(mean_filtered_trials, metric= 'mean')
                    #
                    # allMaxes[roi_ind, opto_time_ind] = trials_amp
                    # allMeans[roi_ind, opto_time_ind] = trials_mean

                    #NEW WAY:
                    trials_max = ID.getResponseAmplitude(epoch_response_matrix=filtered_trials[roi_ind,:], metric= 'max')
                    trials_max_mean = np.mean(trials_max)

                    trials_mean = ID.getResponseAmplitude(epoch_response_matrix=filtered_trials[roi_ind,:], metric= 'mean')
                    trials_mean_mean = np.mean(trials_mean)

                    # # DEBUGGING:
                    # print(f'OPTO CONDITION: {opto_condition}')
                    # print(f'ROI: {roi_ind} | OT: {ot}')
                    # print(f'filtered trials size = {filtered_trials.shape}')
                    # print(f'trials_max shape: {trials_max.shape}')
                    # print(f'trials_max_mean: {trials_max_mean}')
                    # print(f'trials_mean shape: {trials_mean.shape}')
                    # print(f'trials_mean_mean: {trials_mean_mean}')

                    allMaxes[roi_ind, opto_time_ind] = trials_max_mean
                    allMeans[roi_ind, opto_time_ind] = trials_mean_mean

        if silent == False:
            print('\n\n')
            print('======================Response Metrics======================')
            print(f'Number of ROIs = {roi_number}')
            print(f'Opto Start Times = {opto_start_times}')
            print(f'Size should thus be: ({roi_number}, {opto_time_number})')
            print(f'Size of allMaxes:    {allMaxes.shape}')
            print('============================================================')
            print('\n\n')

    else:
        raise Exception('vis_stim_type should be spatiotemporal or single. It was: {}'.format(vis_stim_type))

    return allMaxes, allMeans

# %% Normalize Means and Maxes for each ROI
# Only designed to work in the alternating case
# Finds the Max across opto conditions for each ROI

def normalizeMetrics(noptoMaxes, yoptoMaxes, noptoMeans, yoptoMeans, vis_stim_type):

    if vis_stim_type == 'spatiotemporal':
        max_norm_val = np.nanmax(np.hstack([np.max(noptoMaxes, axis=1), np.max(yoptoMaxes, axis=1)]), axis=1)
        mean_norm_val = np.nanmax(np.hstack([np.max(noptoMeans, axis=1), np.max(yoptoMeans, axis=1)]), axis=1)

        noptoMaxes = noptoMaxes / max_norm_val[:, np.newaxis, np.newaxis]
        yoptoMaxes = yoptoMaxes / max_norm_val[:, np.newaxis, np.newaxis]

        noptoMeans = noptoMeans / mean_norm_val[:, np.newaxis, np.newaxis]
        yoptoMeans = yoptoMeans / mean_norm_val[:, np.newaxis, np.newaxis]

    elif vis_stim_type == 'single':
        max_norm_val = np.nanmax(np.vstack([np.max(noptoMaxes, axis=1), np.max(yoptoMaxes, axis=1)]), axis = 0)
        mean_norm_val = np.nanmax(np.vstack([np.max(noptoMeans, axis=1), np.max(yoptoMeans, axis=1)]), axis = 0)

        noptoMaxes = noptoMaxes / max_norm_val[:, np.newaxis]
        yoptoMaxes = yoptoMaxes / max_norm_val[:, np.newaxis]

        noptoMeans = noptoMeans / mean_norm_val[:, np.newaxis]
        yoptoMeans = yoptoMeans / mean_norm_val[:, np.newaxis]

    # Norm Checker prints
    noptoMaxSum = np.sum(noptoMaxes==1)
    yoptoMaxSum = np.sum(yoptoMaxes==1)
    maxSum = noptoMaxSum + yoptoMaxSum
    noptoMeanSum = np.sum(noptoMeans==1)
    yoptoMeanSum = np.sum(yoptoMeans==1)
    meanSum = noptoMeanSum + yoptoMeanSum
    totalLength = len(mean_norm_val)

    print('\n\n')
    print('================================================================')
    print('=====================Normalization Checker======================')
    print(f'No Opto maxes==1: {noptoMaxSum}')
    print(f'Yes Opto maxes==1: {yoptoMaxSum}')
    print(f'Maxes add up to: {maxSum}')
    print(f'No Opto means==1: {noptoMeanSum}')
    print(f'Yes Opto means==1: {yoptoMeanSum}')
    print(f'Means add up to: {meanSum}')
    print(f'Both of the sums should be: {totalLength}')
    print('================================================================')
    print('\n\n')

    return noptoMaxes, yoptoMaxes, noptoMeans, yoptoMeans


# %% Midpoint Normalize function - from https://stackoverflow.com/a/7746125
# Used to center heatmap colors, key on zero
from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize

class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (value - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

# %% My own version of getEpochResponseMatrix that allows for different
#    flourescence calculations than df/f

def getAltEpochResponseMatrix(ID, region_response, alt_pre_time=0, dff=True, df=False):
        """
        getEpochReponseMatrix(self, region_response, dff=True)
            Takes in long stack response traces and splits them up into each stimulus epoch
            Params:
                region_response: Matrix of region/voxel responses. Shape = (n regions, time)
                dff: (Bool) convert from raw intensity value to dF/F based on mean of pre_time

            Returns:
                time_vector (1d array): 1d array, time values for epoch_response traces (sec)
                response_matrix (ndarray): response for each roi in each epoch.
                    shape = (rois, epochs, frames per epoch)
        """
        no_regions, t_dim = region_response.shape

        run_parameters = ID.getRunParameters()
        response_timing = ID.getResponseTiming()
        stimulus_timing = ID.getStimulusTiming()

        epoch_start_times = stimulus_timing['stimulus_start_times'] - run_parameters['pre_time']
        epoch_end_times = stimulus_timing['stimulus_end_times'] + run_parameters['tail_time']
        epoch_time = (run_parameters['pre_time']
                      + run_parameters['stim_time']
                      + run_parameters['tail_time'])  # sec

        # find how many acquisition frames correspond to pre, stim, tail time
        epoch_frames = int(epoch_time / response_timing['sample_period'])  # in acquisition frames
        pre_frames = int(run_parameters['pre_time'] / response_timing['sample_period'])  # in acquisition frames
        time_vector = np.arange(0, epoch_frames) * response_timing['sample_period']  # sec

        # #DEBUGGGING
        # print('\n\n======================================')
        # print(f'The size of epoch_frames is: {epoch_frames}\n')
        # print(f'The shape of pre_frames is: {pre_frames}\n')
        # samplePeriod = response_timing['sample_period']
        # print(f'The response_timing[sample_period] is: {samplePeriod}\n')
        # print(f'The shape of time_vector is {time_vector}\n\n\n')

        no_trials = len(epoch_start_times)
        response_matrix = np.empty(shape=(no_regions, no_trials, epoch_frames), dtype=float)
        response_matrix[:] = np.nan
        cut_inds = np.empty(0, dtype=int)  # trial/epoch indices to cut from response_matrix
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(response_timing['time_vector'] < epoch_end_times[idx],
                                                 response_timing['time_vector'] >= epoch_start_times[idx]))[0]
            if len(stack_inds) == 0:  # no imaging acquisitions happened during this epoch presentation
                cut_inds = np.append(cut_inds, idx)
                continue
            if np.any(stack_inds > region_response.shape[1]):
                cut_inds = np.append(cut_inds, idx)
                continue
            if idx == no_trials:
                if len(stack_inds) < epoch_frames:  # missed images for the end of the stimulus
                    cut_inds = np.append(cut_inds, idx)
                    print('Missed acquisition frames at the end of the stimulus!')
                    continue
            # pull out Roi values for these scans. shape of newRespChunk is (nROIs,nScans)
            new_resp_chunk = region_response[:, stack_inds]

            if dff:
                # calculate baseline using pre frames
                if alt_pre_time == 0: # standard, use the pre_frames
                    baseline = np.mean(new_resp_chunk[:, 0:pre_frames], axis=1, keepdims=True)
                # Allows for the specification of an alternative amount of
                # seconds to use for the pre-time or baseline in df/f
                elif alt_pre_time > 0:
                    samplePeriod = response_timing['sample_period']
                    alt_pre_frames = int(alt_pre_time/samplePeriod)
                    baseline = np.mean(new_resp_chunk[:, 0:alt_pre_frames], axis=1, keepdims=True)

                # to dF/F
                with warnings.catch_warnings():  # Warning to catch divide by zero or nan. Will return nan or inf
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    new_resp_chunk = (new_resp_chunk - baseline) / baseline

            elif df:
                # calculate baseline using pre frames, don't divide by f

                baseline = np.mean(new_resp_chunk[:, 0:pre_frames], axis=1, keepdims=True)
                # to dF/F
                with warnings.catch_warnings():  # Warning to catch divide by zero or nan. Will return nan or inf
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    new_resp_chunk = (new_resp_chunk - baseline)

            if epoch_frames > new_resp_chunk.shape[-1]:
                print('Warnging: Size mismatch idx = {}'.format(idx))  # the end of a response clipped off
                response_matrix[:, idx, :new_resp_chunk.shape[-1]] = new_resp_chunk[:, 0:]

            else:
                response_matrix[:, idx, :] = new_resp_chunk[:, 0:epoch_frames]
            # except:
            #     print('Size mismatch idx = {}'.format(idx))  # the end of a response clipped off
            #     print(response_matrix.shape)
            #     print(new_resp_chunk.shape)
            #     print(epoch_frames)
                # cut_inds = np.append(cut_inds, idx)


        if len(cut_inds) > 0:
            print('Warning: cut {} epochs from epoch response matrix'.format(len(cut_inds)))
        response_matrix = np.delete(response_matrix, cut_inds, axis=1)  # shape = (region, trial, time)

        return time_vector, response_matrix

# TRF Utils Below:

def trfMaker(
             experiment_file_directory, experiment_file_name,
             series_number, roi_set_name, filter_length, dff, savefig = True, silent = False
            ):
    # join path to proper format for ImagingDataObject()
    file_path = os.path.join(experiment_file_directory, experiment_file_name + ".hdf5")
    print(file_path)
    # create save directory
    save_directory = "/Volumes/ROG2TBAK/data/bruker/trfs/" + experiment_file_name + "/"
    print(save_directory)
    Path(save_directory).mkdir(exist_ok=True)
    # create ImagingDataObject (wants a path to an hdf5 file and a series number from that file)
    ID = imaging_data.ImagingDataObject(file_path, series_number, quiet=True)
    # get ROI timecourses and stimulus parameters
    roi_data = ID.getRoiResponses(roi_set_name)
    ID.getRoiSetNames()

    # Interpolation of the ROI response trace into sample period time
    roi_ind = 0
    response_trace = roi_data.get('roi_response')[roi_ind][0, :]
    response_time = np.arange(1, len(response_trace)+1) * ID.getAcquisitionMetadata('sample_period')
    # Interpolation function to interpolate response. Fxn built using sample period
    f_interp_response = interpolate.interp1d(response_time, response_trace)

    # Establish relevant variables
    epoch_parameters = ID.getEpochParameters()
    ideal_frame_rate = 120  # Hz
    sample_period = ID.getAcquisitionMetadata('sample_period') # (sec), bruker imaging acquisition period
    filter_len = filter_length * ideal_frame_rate
    stimulus_timing = ID.getStimulusTiming(plot_trace_flag=False)
    stimulus_start_times = stimulus_timing['stimulus_start_times']
    run_parameters = ID.getRunParameters
    stim_frames = run_parameters('stim_time') * ideal_frame_rate

    # frame flip times in a stimulus presentation:
    stim_times = np.arange(1, stim_frames+1) * 1/ideal_frame_rate

    # Initialize roi_trfs - ROI x Filter x Trials
    roi_trfs = np.zeros(
        (
            roi_data["epoch_response"].shape[0],
            int(filter_len),
            int(run_parameters("num_epochs")),
        )
    )
    all_stims = []
    all_responses = []

    if silent == False:
        print('\n----------------------------------------------------------------------------------')
        print('Initializing: (1) visual stim recreation (2) fft responses (3) generate filters...\n')

    for epoch_ind in range(int(ID.getRunParameters('num_epochs'))):
        if (epoch_ind%10 == 0) and silent == False:
            print(f'...Starting Trial {epoch_ind} of {int(ID.getRunParameters("num_epochs"))}...')
        for roi_ind in range(0, roi_data["epoch_response"].shape[0]):
            # initalize trf by trial array (T, trial)
            roi_trf = np.zeros((int(filter_len), int(run_parameters("num_epochs"))))

            response_trace = roi_data.get('roi_response')[roi_ind][0, :]
            response_time = np.arange(1, len(response_trace)+1) * ID.getAcquisitionMetadata('sample_period')
            # Interpolation function to interpolate response. Fxn built using sample period
            f_interp_response = interpolate.interp1d(response_time, response_trace)

            # Regenerate the stimulus
            start_seed = epoch_parameters[epoch_ind]['start_seed']
            rand_min = eval(epoch_parameters[epoch_ind]['distribution_data'])['kwargs']['rand_min']
            rand_max = eval(epoch_parameters[epoch_ind]['distribution_data'])['kwargs']['rand_max']
            update_rate = epoch_parameters[epoch_ind]['update_rate']

            new_stim = np.array([getRandVal(rand_min, rand_max, start_seed, update_rate, t) for t in stim_times])
            current_frame_times = stimulus_start_times[epoch_ind] + stim_times  # In Prairie View time (sec)

            baseline_time = 1 # (sec) generally could be pre_time, but for opto only take previous 1 sec
            baseline_times = np.linspace(current_frame_times[0]-baseline_time, current_frame_times[0], int(1/sample_period))
            baseline = np.mean(f_interp_response(baseline_times))

            if dff == True:
                # Convert to dF/F
                current_interp_response = (f_interp_response(current_frame_times) - baseline) / baseline
            else: # don't df/f
                current_interp_response = f_interp_response(current_frame_times)

            filter_fft = np.fft.fft(current_interp_response - np.mean(current_interp_response)) * np.conj(
              np.fft.fft(new_stim - np.mean(new_stim)))

            filt = np.real(np.fft.ifft(filter_fft))[0 : int(filter_len)]

            trf = np.flip(filt)

            #roi_trf[:, epoch_ind] = trf
            roi_trfs[roi_ind, :, epoch_ind] = trf
    if silent == False:
    #all_trfs = np.stack(all_trfs, axis=-1)
        print('\n-------')
        print('DONE!')
        print('-------\n')

    # Run optoSplit
    (roi_mean_trf, nopto_mean_trf, nopto_sem_plus, nopto_sem_minus,
     yopto_mean_trf, yopto_sem_plus, yopto_sem_minus) = optoSplitAndMean(roi_trfs)

    return (roi_mean_trf, nopto_mean_trf, nopto_sem_plus, nopto_sem_minus,
            yopto_mean_trf, yopto_sem_plus, yopto_sem_minus)

def optoSplitAndMean(roi_trfs, silent = True):
    # SPLIT into NO Opto and YES Opto trials (alternating)
    no_slice = roi_trfs[:, :, 0::2]  # Every 2 trials, starting at 0
    yes_slice = roi_trfs[:, :, 1::2]  # Every 2 trials, starting at 1

    # compute mean TRF across trials
    # We go from ROI x TRF x Trial --> ROI x TRF
    roi_mean_trf = np.mean(roi_trfs, 2)
    nopto_mean_trf = np.mean(no_slice, 2)
    yopto_mean_trf = np.mean(yes_slice, 2)

    # Standard Error of the Mean calculations
    nopto_sem = sem(no_slice, axis=2)  # calculate the no opto sem
    nopto_sem_plus = np.squeeze(nopto_mean_trf + nopto_sem)
    nopto_sem_minus = np.squeeze(nopto_mean_trf - nopto_sem)
    yopto_sem = sem(yes_slice, axis=2)  # calculate the yes opto sem
    yopto_sem_plus = np.squeeze(yopto_mean_trf + yopto_sem)
    yopto_sem_minus = np.squeeze(yopto_mean_trf - yopto_sem)


    if silent == False:
        # Checking ouputs for separating no opto from opto trials
        print("\n----------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------")
        print("||    Checking the shape of the various trfs to ensure opto/no opto split:!")
        print("||    Shape of roi_trfs is: " + str(roi_trfs.shape))
        print("||    Shape of no_slice is: " + str(no_slice.shape))
        print("||    Shape of yes_slice is: " + str(yes_slice.shape))
        print("||")
        print("||    Shape of roi_mean_trf is: " + str(roi_mean_trf.shape))
        print("||    Shape of nopto_mean_trf is: " + str(nopto_mean_trf.shape))
        print("||    Shape of yopto_mean_trf is: " + str(yopto_mean_trf.shape))
        print("----------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------\n")

        print(f'shape of std error is {nopto_sem_plus.shape}')

    return (roi_mean_trf, nopto_mean_trf, nopto_sem_plus, nopto_sem_minus,
            yopto_mean_trf, yopto_sem_plus, yopto_sem_minus
           )

def avgAcrossROIs(nopto_mean_trf, nopto_sem_plus, nopto_sem_minus, yopto_mean_trf, yopto_sem_plus, yopto_sem_minus):
    # We go from ROI x TRF --> TRF
    across_roi_nopto_trf = np.mean(nopto_mean_trf, axis = 0)
    across_roi_nopto_sem_plus = np.mean(nopto_sem_plus, axis = 0)
    across_roi_nopto_sem_minus = np.mean(nopto_sem_minus, axis = 0)
    across_roi_yopto_trf = np.mean(yopto_mean_trf, axis = 0)
    across_roi_yopto_sem_plus = np.mean(yopto_sem_plus, axis = 0)
    across_roi_yopto_sem_minus = np.mean(yopto_sem_minus, axis = 0)

    return across_roi_nopto_trf, across_roi_nopto_sem_plus, across_roi_nopto_sem_minus, across_roi_yopto_trf, across_roi_yopto_sem_plus, across_roi_yopto_sem_minus

# SEM Plot for single ROIs!
def plotSingleTRFComparison(
                  nopto_trf, nopto_sem_plus, nopto_sem_minus,
                  yopto_trf, yopto_sem_plus, yopto_sem_minus,
                  filter_len, ideal_frame_rate = 120, savefig = False
                 ):
    fh, ax = plt.subplots(1, 1, figsize=(20, 10))
    #time = np.arange(0, roi_trfs.shape[1])
    filter_time = np.flip(np.arange(0, filter_len) * 1/ideal_frame_rate)

    ax.plot(filter_time, yopto_trf, color="r")
    ax.fill_between(filter_time, yopto_sem_plus, yopto_sem_minus, color="r", alpha=0.4)
    ax.plot(filter_time, nopto_trf, color="g")
    ax.fill_between(filter_time, nopto_sem_plus, nopto_sem_minus, color="g", alpha=0.4)
    ax.axhline(y=0, color="k", alpha=0.5)
    red_patch = mpatches.Patch(color="red", label="Opto Condition")
    green_patch = mpatches.Patch(color="green", label="No Opto Condition")

    ax.legend(handles=[green_patch, red_patch], fontsize=20)
    # ax.set_title(
    #     f"Temporal Receptive Field for {experiment_file_name} | Series {series_number} | {roi_set_name} | Filter Length: {filter_length}",
    #     fontsize=25,
    # )
    ax.set_xlabel("Time (s)", fontweight="bold", fontsize=13)
    ax.grid(True)

    if savefig == True:
        fh.savefig(
            save_directory
            + "Cross-ROI.TemporalReceptiveField."
            + str(experiment_file_name)
            + "."
            + str(series_number)
            + "."
            + str(roi_set_name)
            + ".FilterLength"
            + str(filter_length)
            + ".pdf",
            dpi=300,
        )

# SEM Plot for multiple ROIs!
def plotMultipleTRFComparisons(
                               nopto_mean_trf, nopto_sem_plus, nopto_sem_minus,
                               yopto_mean_trf, yopto_sem_plus, yopto_sem_minus,
                               filter_len, ideal_frame_rate = 120, savefig = False
                              ):
    roi_count = nopto_mean_trf.shape[0]

    fh, ax = plt.subplots(roi_count, 1, figsize=(10,roi_count*5))
    #time = np.arange(0, roi_trfs.shape[1])
    filter_time = np.flip(np.arange(0, filter_len) * 1/ideal_frame_rate)
    for roi_ind in range(0, roi_count):
        ax[roi_ind].plot(filter_time, yopto_mean_trf[roi_ind], color="r")
        ax[roi_ind].fill_between(filter_time, yopto_sem_plus[roi_ind], yopto_sem_minus[roi_ind], color="r", alpha=0.4)
        ax[roi_ind].plot(filter_time, nopto_mean_trf[roi_ind], color="g")
        ax[roi_ind].fill_between(filter_time, nopto_sem_plus[roi_ind], nopto_sem_minus[roi_ind], color="g", alpha=0.4)
        ax[roi_ind].axhline(y=0, color="k", alpha=0.5)
        red_patch = mpatches.Patch(color="red", label="Opto Condition")
        green_patch = mpatches.Patch(color="green", label="No Opto Condition")

        ax[roi_ind].legend(handles=[green_patch, red_patch], fontsize=12)
        # ax[roi_ind].set_title(
        #     f"Temporal Receptive Field for {experiment_file_name} | Series {series_number} | {roi_set_name}: ROI_{roi_ind+1} | Filter Length: {filter_length}",
        #     fontsize=12,
        # )
        ax[roi_ind].set_xlabel("Time (s)", fontsize=11)
        ax[roi_ind].grid(True)

    if savefig == True:
        fh.savefig(
            save_directory
            + "Each-ROI.TemporalReceptiveField."
            + str(experiment_file_name)
            + "."
            + str(series_number)
            + "."
            + str(roi_set_name)
            + ".FilterLength"
            + str(filter_length)
            + ".pdf",
            dpi=300,
        )
