# %%
# MHT

import numpy as np
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
import os
from scipy import interpolate
import two_photon_analysis.medulla_analysis as ma

data_dir = '/Users/mhturner/CurrentData/avery'

do_for = 'Mi1'  # 'L2'

if do_for == 'Mi1':
    roi_name = 'Mi1_proximal'
    eg_ind = 0
    matching_series = [
                ('2022-10-25', 1),
                ('2022-10-25', 2),
                ('2022-10-25', 5),
                ('2022-10-27', 1),
                ('2022-10-27', 3),
                ('2022-10-27', 5)
                ]
elif do_for == 'L2':
    roi_name = 'L2'
    matching_series = []



# %%

for s_ind, ser in enumerate(matching_series):
    file_path = os.path.join(data_dir, ser[0] + '.hdf5')
    ID = imaging_data.ImagingDataObject(file_path, ser[1], quiet=True)

    roi_data = ID.getRoiResponses(roi_name)
    roi_ind = 0
    response_trace = roi_data.get('roi_response')[roi_ind][0, :]

    time_vec, erm = ma.getAltEpochResponseMatrix(ID, 
                                                roi_data.get('roi_response')[0], 
                                                alt_pre_time=0, 
                                                dff=True, 
                                                df=False)

    # Opto trials
    opto_trials, opto_inds = shared_analysis.filterTrials(erm, 
                                                          ID, 
                                                          query={'opto_stim': True}, 
                                                          return_inds=True)
    
    # NOpto trials
    nopto_trials, nopto_inds = shared_analysis.filterTrials(erm, 
                                                            ID, 
                                                            query={'opto_stim': False}, 
                                                            return_inds=True)
    
    diff = opto_trials - nopto_trials  # difference of subsequent trials (same seed)

    filter_duration = 1.5 # sec
    filter_results = ma.getTrialTrfs(ID, response_trace, filter_duration, dff=True, trial_subset_fraction=0.25)

    filter_time = filter_results.get('filter_time')
    trial_trfs = filter_results.get('trial_trfs')

    start_trfs = filter_results.get('start_trfs')
    end_trfs = filter_results.get('end_trfs')
    
    # if s_ind == eg_ind:
    if True:
        # plot example trial responses overlaid
        fh0, ax0 = plt.subplots(3, 1, figsize=(6, 6))
        eg_trial = 1 
        ax0[0].plot(roi_data.get('time_vector'), opto_trials[0, eg_trial, :],
                color='k')
        ax0[0].plot(roi_data.get('time_vector'), nopto_trials[0, eg_trial, :],
                color='r')
        
        # Plot across-trial average responses overlaid
        nopto_mean = np.mean(nopto_trials[0,...], axis=0)
        nopto_sem = np.std(nopto_trials[0,...], axis=0) / np.sqrt(nopto_trials.shape[1])

        opto_mean = np.mean(opto_trials[0,...], axis=0)
        opto_sem = np.std(opto_trials[0,...], axis=0) / np.sqrt(opto_trials.shape[1])
        ax0[1].plot(roi_data.get('time_vector'), nopto_mean, 'k')
        ax0[1].fill_between(roi_data.get('time_vector'), nopto_mean-nopto_sem, nopto_mean+nopto_sem, color='k')

        ax0[1].plot(roi_data.get('time_vector'), opto_mean, 'r')
        ax0[1].fill_between(roi_data.get('time_vector'), opto_mean-opto_sem, opto_mean+opto_sem, color='r')

        
        ax0[2].axhline(y=0, color='k', alpha=0.5)
        ax0[2].axvline(x=2+0.5, color='g')
        ax0[2].fill_betweenx(y=[np.nanmean(diff, axis=(0, 1)).max(), np.nanmean(diff, axis=(0, 1)).min()],
                            x1=0,
                            x2=1,
                            color='r')

        ax0[2].plot(roi_data.get('time_vector'), np.mean(diff[0, :, :], axis=0), 'k')
        ax0[2].set_xlabel('Time (s)')
        ax0[2].set_ylabel('Response difference, Opto. - No Opto.')

        # Plot filters
        fh1, ax1 = plt.subplots(1, 3, figsize=(6, 4))
        nopto_mean = np.mean(trial_trfs[:,0::2], axis=1)
        nopto_sem = np.std(trial_trfs[:,0::2], axis=1) / np.sqrt(trial_trfs.shape[1]/2)

        opto_mean = np.mean(trial_trfs[:,1::2], axis=1)
        opto_sem = np.std(trial_trfs[:,1::2], axis=1) / np.sqrt(trial_trfs.shape[1]/2)

        ax1[0].axhline(y=0, color='k', alpha=0.5)
        # ax1[0].fill_between(filter_time, nopto_mean-nopto_sem, nopto_mean+nopto_sem, 
        #                  color='k', alpha=0.5)
        ax1[0].plot(filter_time, nopto_mean, 'k')
        
        # ax1[0].fill_between(filter_time, opto_mean-opto_sem, opto_mean+opto_sem, 
        #                  color='r', alpha=0.5)
        ax1[0].plot(filter_time, opto_mean, 'r')
        

        # Plot filters for beginning and end of trials
        nopto_mean = np.mean(start_trfs[:,0::2], axis=1)
        nopto_sem = np.std(start_trfs[:,0::2], axis=1) / np.sqrt(start_trfs.shape[1]/2)

        opto_mean = np.mean(start_trfs[:,1::2], axis=1)
        opto_sem = np.std(start_trfs[:,1::2], axis=1) / np.sqrt(start_trfs.shape[1]/2)
        
        ax1[1].axhline(y=0, color='k', alpha=0.5)
        # ax1[1].fill_between(filter_time, nopto_mean-nopto_sem, nopto_mean+nopto_sem,
        #                     color='k', alpha=0.5)
        ax1[1].plot(filter_time, nopto_mean, 'k')
        
        ax1[1].plot(filter_time, opto_mean, 'r')
        # ax1[1].fill_between(filter_time, opto_mean-opto_sem, opto_mean+opto_sem, 
        #                     color='r', alpha=0.5)


        nopto_mean = np.mean(end_trfs[:,0::2], axis=1)
        nopto_sem = np.std(end_trfs[:,0::2], axis=1) / np.sqrt(end_trfs.shape[1]/2)

        opto_mean = np.mean(end_trfs[:,1::2], axis=1)
        opto_sem = np.std(end_trfs[:,1::2], axis=1) / np.sqrt(end_trfs.shape[1]/2)
        ax1[2].axhline(y=0, color='k', alpha=0.5)
        # ax1[2].fill_between(filter_time, nopto_mean-nopto_sem, nopto_mean+nopto_sem, 
        #                    color='k', alpha=0.5)
        ax1[2].plot(filter_time, nopto_mean, 'k')
        
        # ax1[2].fill_between(filter_time, opto_mean-opto_sem, opto_mean+opto_sem, 
        #                     color='r', alpha=0.5)
        ax1[2].plot(filter_time, opto_mean, 'r')
        


# %%
