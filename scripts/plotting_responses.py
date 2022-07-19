"""
Script to call medulla_analysis.py functions and plot across flies

Avery Krieger 5/20/22
"""
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import os
from two_photon_analysis import medulla_analysis as ma

# %% Analysis Parameters
file_directory = '/Users/averykrieger/Documents/local_data_repo/20220527'
save_path = '/Users/averykrieger/Documents/local_data_repo/figs/'
file_name = '2022-05-27'
series_number = 16
roi_name = 'medial_vis_responsive'
opto_condition = True
displayFix = True
saveFig = True
dff = False
# Handling the DF/F or DF or Raw options for the Ca traces
if dff == False:
    df = True
else:
    df = False

# Load up that data
ID, roi_data = ma.dataLoader(file_directory = file_directory,
                      save_path = save_path,
                      file_name = file_name,
                      series_number = series_number,
                      roi_name = roi_name,
                      opto_condition = opto_condition,
                      displayFix = displayFix,
                      saveFig = saveFig)
# saveNames for figures
if df == True:
    saveName = save_path+str(file_name)+' | SeriesNumber'+' | DF is '+str(df)+' | '+str(series_number)+' | '+str(roi_name) + '.pdf'
else:
    saveName = save_path+str(file_name)+' | SeriesNumber'+' | DFF is '+str(dff)+' | '+str(series_number)+' | '+str(roi_name) + '.pdf'

# %% PLOT traces of each ROI
ma.plotConditionedROIResponses(ID = ID, roi_data = roi_data,
                               opto_condition = opto_condition, dff=dff, df = df,
                               save_path = save_path, saveFig=saveFig,
                               saveName = saveName)

# %% Heatmaps of ROIs opto vs no opto
heatmapFigName = save_path+'metrics/heatmaps/'+'test  '+str(file_name)+' | SeriesNumber'+str(series_number)+' | '+str(roi_name) + ' | ' + 'Heatmap' '.pdf'
ma.plotOptoHeatmaps(ID, roi_data, saveName = heatmapFigName, dff = dff, df = df)


# %% Calculate Response Metrics so that they can be plotted

noptoMaxes, noptoMeans = ma.getResponseMetrics(ID = ID, roi_data = roi_data,
                                              dff = dff, df = df,
                                              opto_condition = False,
                                              silent = True)
yoptoMaxes, yoptoMeans = ma.getResponseMetrics(ID = ID, roi_data = roi_data,
                                              dff = dff, df = df,
                                              opto_condition = True,
                                              silent = True)
roi_number = len(roi_data['roi_response'])

# gotta get those title and fig names going
plotTitle=f'{file_name} | {series_number}'
if df == True:
    roiResponseFigName = save_path+'/metrics/'+str(file_name)+' | SeriesNumber '+str(series_number)+' | DF | '+str(roi_name) + ' | '
elif dff == True:
        roiResponseFigName = save_path+'/metrics/'+str(file_name)+' | SeriesNumber '+str(series_number)+' | DF/F | '+str(roi_name) + ' | '
else:
    roiResponseFigName = save_path+'/metrics/'+str(file_name)+' | SeriesNumber'+str(series_number)+'raw | '+str(roi_name) + ' | '


# %% PLOT ROI Resonse Metrics (ROIs are separate)
ma.plotROIResponsesMetrics(ID, plotTitle, roiResponseFigName, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number)

# %% PLOT ROI Resonse Metrics (ROIs are collapsed)
ma.plotReponseMetricsAcrossROIs(ID, plotTitle, roiResponseFigName, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number)

# %%
def plotReponseMetricsAcrossROIs(ID, plotTitle, figTitle, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number, saveFig=True):
    from visanalysis.analysis import imaging_data, shared_analysis
    from visanalysis.util import plot_tools
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import sem as sem
    import os
    import matplotlib.patches as mpatches

    saveFig = True
    # Collapse across stimulus space.
    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))

    # First, plot max and avgs of spatial period, collapsed across temp freq:
    fh, ax = plt.subplots(2, 1) #, figsize=(40, 40))
    plt.subplots_adjust(bottom=0.1, right=0.9, wspace=0.4, hspace=0.6)

    #calculate mean across rois. optoMaxes = ROI x SpatPer x TempFreq
    nopto_spatial_mean_max_across_rois= np.mean(np.mean(noptoMaxes[:,:,:], axis = 0), axis = 1)
    yopto_spatial_mean_max_across_rois= np.mean(np.mean(yoptoMaxes[:,:,:], axis = 0), axis = 1)
    nopto_spatial_mean_mean_across_rois = np.mean(np.mean(noptoMeans[:,:,:], axis = 0), axis = 1)
    yopto_spatial_mean_mean_across_rois = np.mean(np.mean(yoptoMeans[:,:,:], axis = 0), axis = 1)

    for roi_ind in range(roi_number):
        # Max Values for each ROI get plotted, avg across TF
        nopto_spatial_per_max_max = np.mean(noptoMaxes[roi_ind,:,:], axis = 1)
        ax[0].plot(target_sp, nopto_spatial_per_max_max, 'k^', alpha=0.4)

        yopto_spatial_per_max_max = np.mean(yoptoMaxes[roi_ind,:,:], axis = 1)
        ax[0].plot(target_sp, yopto_spatial_per_max_max, 'r^', alpha=0.4)

        ax[0].set_title('Max Respone by SpatPer')

        # Mean Values for each ROI get plotted, avg across TF
        nopto_spatial_per_mean = np.mean(noptoMeans[roi_ind,:,:], axis = 1)
        ax[1].plot(target_sp, nopto_spatial_per_mean, 'k^', alpha=0.4)

        yopto_spatial_per_mean = np.mean(yoptoMeans[roi_ind,:,:], axis = 1)
        ax[1].plot(target_sp, yopto_spatial_per_mean, 'r^', alpha=0.4)

        ax[1].set_title('Mean Respone by SpatPer')

    ax[0].plot(target_sp, nopto_spatial_mean_max_across_rois, '-ko', alpha=0.8)
    ax[0].plot(target_sp, yopto_spatial_mean_max_across_rois, '-ro', alpha=0.8)

    ax[1].plot(target_sp, nopto_spatial_mean_mean_across_rois, '-ko', alpha=0.8)
    ax[1].plot(target_sp, yopto_spatial_mean_mean_across_rois, '-ro', alpha=0.8)

    red_patch = mpatches.Patch(color='red', label='With Opto')
    black_patch = mpatches.Patch(color='black', label='No Opto')
    ax[1].legend(handles=[red_patch, black_patch])

    fh.suptitle(plotTitle + ' | SpatPer')
    if saveFig == True:
        fh.savefig(figTitle + 'SpatPer.pdf', dpi=300)

    # Second, plot max and avgs of temporal frequencies, collapsed across spatial_periods:
    fh, ax = plt.subplots(2, 1)
    plt.subplots_adjust(bottom=0.1, right=0.9, wspace=0.4, hspace=0.6)
    nopto_temporal_mean_max_across_rois= np.mean(np.mean(noptoMaxes[:,:,:], axis = 0), axis = 0)
    yopto_temporal_mean_max_across_rois= np.mean(np.mean(yoptoMaxes[:,:,:], axis = 0), axis = 0)
    nopto_temporal_mean_mean_across_rois = np.mean(np.mean(noptoMeans[:,:,:], axis = 0), axis = 0)
    yopto_temporal_mean_mean_across_rois = np.mean(np.mean(yoptoMeans[:,:,:], axis = 0), axis = 0)

    for roi_ind in range(roi_number):
        # Max Values
        nopto_temporal_per_max_max = np.mean(noptoMaxes[roi_ind,:,:], axis = 0)
        ax[0].plot(target_tf, nopto_temporal_per_max_max, 'k^', alpha=0.4)

        yopto_temporal_per_max_max = np.mean(yoptoMaxes[roi_ind,:,:], axis = 0)
        ax[0].plot(target_tf, yopto_temporal_per_max_max, 'r^', alpha=0.4)

        ax[0].set_title('Max Respone by TempFreq')

        # Mean Values
        nopto_temporal_per_mean = np.mean(noptoMeans[roi_ind,:,:], axis = 0)
        ax[1].plot(target_tf, nopto_temporal_per_mean, 'k^', alpha=0.4)

        yopto_temporal_per_mean = np.mean(yoptoMeans[roi_ind,:,:], axis = 0)
        ax[1].plot(target_tf, yopto_temporal_per_mean, 'r^', alpha=0.4)

        ax[1].set_title('Mean Respone by Temp Freq')

    ax[0].plot(target_tf, nopto_temporal_mean_max_across_rois, '-ko', alpha=0.8)
    ax[0].plot(target_tf, yopto_temporal_mean_max_across_rois, '-ro', alpha=0.8)

    ax[1].plot(target_tf, nopto_temporal_mean_mean_across_rois, '-ko', alpha=0.8)
    ax[1].plot(target_tf, yopto_temporal_mean_mean_across_rois, '-ro', alpha=0.8)

    red_patch = mpatches.Patch(color='red', label='With Opto')
    black_patch = mpatches.Patch(color='black', label='No Opto')
    ax[1].legend(handles=[red_patch, black_patch])

    fh.suptitle(plotTitle + ' | TempFreq')
    if saveFig == True:
        fh.savefig(figTitle + 'TempFreq.pdf', dpi=300)


# %% Collapse across ROIs

for i in roi_set_list:
    roi_name = roi_set_list[i]
    ID, roi_data = ma.dataLoader(file_directory = file_directory,
                          save_path = save_path,
                          file_name = file_name,
                          series_number = series_number,
                          roi_name = roi_name,
                          opto_condition = opto_condition,
                          displayFix = displayFix,
                          saveFig = saveFig)


    noptoMaxes, noptoMeans = ma.getResponseMetrics(ID = ID, roi_data = roi_data,
                                                  dff = dff, df = df,
                                                  opto_condition = False,
                                                  silent = True)
    yoptoMaxes, yoptoMeans = ma.getResponseMetrics(ID = ID, roi_data = roi_data,
                                                  dff = dff, df = df,
                                                  opto_condition = True,
                                                  silent = True)




# %% PARAMETERS & METADATA

# all run_parameters as a dict
run_parameters = ID.getRunParameters()

# specified run parameter
protocol_ID = ID.getRunParameters('protocol_ID')
print(protocol_ID)

# epoch_parameters: list of dicts of all epoch parameters, one for each epoch (trial)
epoch_parameters = ID.getEpochParameters()
print(epoch_parameters)
# Pass a param key to return a list of specified param values, one for each trial
current_opto_stim = ID.getEpochParameters('opto_stim')

# fly_metadata: dict
fly_metadata = ID.getFlyMetadata()
prep = ID.getFlyMetadata('prep')
print(prep)

# acquisition_metadata: dict
acquisition_metadata = ID.getAcquisitionMetadata()
print(acquisition_metadata)
#sample_period = ID.getAcquisitionMetadata('sample_period')
#print(sample_period)
# %% ROIS AND RESPONSES

# Get list of rois present in the hdf5 file for this series
roi_set_names = ID.getRoiSetNames()
# getRoiResponses() wants a ROI set name, returns roi_data (dict)
roi_data = ID.getRoiResponses('medulla_rois')
roi_data.keys()

# See the ROI overlaid on top of the image
#ID.generateRoiMap(roi_name='medulla_rois', z=1)

# Plot whole ROI response across entire series
fh0, ax0 = plt.subplots(1, 1, figsize=(12, 4))
ax0.plot(roi_data.get('roi_response')[0].T)
ax0.set_xlabel('Frame')
ax0.set_ylabel('Avg ROI intensity')

# Plot ROI response for trials 10 thru 15
# 'epoch_response' is shape (rois, trials, time)
fh1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
ax1.plot(roi_data.get('time_vector'), roi_data.get('epoch_response')[0, 10:15, :].T)
ax1.set_ylabel('Response (dF/F)')
ax1.set_xlabel('Time (s)')


ID.getStimulusTiming(plot_trace_flag=True)

v = ID.getVoltageData()
v[0].shape
v[1]

plt.plot(v[0][2, :])

# %% Plot trial-average responses by specified parameter name
ID.getEpochParameters()
query = {'current_spatial_period': 20, 'current_temporal_frequency': 1.0}
filtered_trials = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query=query)
filtered_trials.shape
unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key=('current_spatial_period', 'current_temporal_frequency'))
ID.getTrialAverages
print(unique_parameter_values)
mean_response.shape
roi_data.get('epoch_response').shape

roi_data.keys()

fh, ax = plt.subplots(1, len(unique_parameter_values), figsize=(10, 2))
#[x.set_ylim([-0.15, 0.25]) for x in ax.ravel()]
[plot_tools.cleanAxes(x) for x in ax.ravel()]
for u_ind, up in enumerate(unique_parameter_values):
    ax[u_ind].plot(roi_data['time_vector'], mean_response[:, u_ind, :].T, color='k')
    ax[u_ind].set_title('rv = {}'.format(up))
plot_tools.addScaleBars(ax[0], dT=2.0, dF=0.10, T_value=-0.5, F_value=-0.14)

# %% Some other convenience methods...



# Quickly look at roi responses (averaged across all trials)
shared_analysis.plotRoiResponses(ID, roi_name='glom')

# %%
shared_analysis.plotResponseByCondition(ID, roi_name='set_2', condition='current_intensity', eg_ind=0)
