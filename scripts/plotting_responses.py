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
fileTwo = '/Users/averykrieger/Documents/local_data_repo/20220527'
fileTwoName = '2022-05-27'
fileThree = '/Volumes/ROG2TBAK/data/bruker/20220718'
fileThreeName = '2022-07-18' #Alt = 5, 9, 14, 18

file_directory = fileThree
file_name = fileThreeName
alt_pre_time = 1
save_path = '/Users/averykrieger/Documents/local_data_repo/figs/'
series_number = 18
roi_name = 'proximal_visual_response'
opto_condition = True
vis_stim_type = 'single' # 'spatiotemporal' or 'single'
displayFix = True
saveFig = True
dff = True
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
    saveName = save_path+str(file_name)+' | SeriesNumber'+' | DFF is '+str(dff)+' | '+str(series_number)+' | '+str(roi_name) + ' | alt pre time: ' + str(alt_pre_time)+ '.pdf'

#ID.getStimulusTiming(plot_trace_flag=True)

# PLOT traces of each ROI
ma.plotConditionedROIResponses(ID = ID, roi_data = roi_data,
                               opto_condition = opto_condition,
                               vis_stim_type = vis_stim_type,
                               alt_pre_time = alt_pre_time,
                               dff=dff, df = df, save_path = save_path,
                               saveFig=saveFig, saveName = saveName)



# Calculate Response Metrics so that they can be plotted
noptoMaxes, noptoMeans = ma.getResponseMetrics(ID = ID, roi_data = roi_data,
                                              dff = dff, df = df,
                                              vis_stim_type = vis_stim_type,
                                              alt_pre_time = alt_pre_time,
                                              opto_condition = False,
                                              silent = False)
yoptoMaxes, yoptoMeans = ma.getResponseMetrics(ID = ID, roi_data = roi_data,
                                              dff = dff, df = df,
                                              vis_stim_type = vis_stim_type,
                                              alt_pre_time = alt_pre_time,
                                              opto_condition = True,
                                              silent = False)

#Normalize Max and Mean values for each ROI
noptoMaxes, yoptoMaxes, noptoMeans, yoptoMeans = ma.normalizeMetrics(noptoMaxes, yoptoMaxes, noptoMeans, yoptoMeans)


# gotta get those title and fig names going
plotTitle=f'{file_name} | {series_number}'
roi_number = len(roi_data['roi_response'])
if df == True:
    roiResponseFigName = save_path+'metrics/'+str(file_name)+' | SeriesNumber '+str(series_number)+' | DF | '+str(roi_name) + ' | '
elif dff == True:
    roiResponseFigName = save_path+'metrics/'+str(file_name)+' | SeriesNumber'+' | DFF is '+str(dff)+' | '+str(series_number)+' | '+str(roi_name) + ' | alt pre time: ' + str(alt_pre_time)+ '.pdf'
else:
    roiResponseFigName = save_path+'metrics/'+str(file_name)+' | SeriesNumber'+str(series_number)+'raw | '+str(roi_name) + ' | '


# # PLOT ROI Resonse Metrics (ROIs are separate)
ma.plotROIResponsesMetrics(ID, plotTitle, roiResponseFigName, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number, vis_stim_type)

#% # PLOT ROI Resonse Metrics (ROIs are collapsed)
ma.plotReponseMetricsAcrossROIs(ID, plotTitle, roiResponseFigName, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans, roi_number, vis_stim_type)


# %% Compute the means across ROIs for temporal and spatial frequencies
def getResponseMeansAcrossROIs(ID, noptoMaxes, noptoMeans, yoptoMaxes, yoptoMeans):
    import numpy as np
    from scipy.stats import sem as sem

    roi_number = len(roi_data['roi_response'])
    target_sp = np.unique(ID.getEpochParameters('current_spatial_period'))
    target_tf = np.unique(ID.getEpochParameters('current_temporal_frequency'))

    emptySizeSpatial = [roi_number, len(target_sp)]
    emptySizeTemporal = [roi_number, len(target_tf)]

    nopto_spatial_per_max_max = np.empty(emptySizeSpatial,dtype=object)
    opto_spatial_per_max_max = np.empty(emptySizeSpatial,dtype=object)
    nopto_temporal_per_max_max = np.empty(emptySizeTemporal,dtype=object)
    opto_temporal_per_max_max  = np.empty(emptySizeTemporal,dtype=object)
    nopto_spatial_per_mean_mean  = np.empty(emptySizeSpatial,dtype=object)
    opto_spatial_per_mean_mean  = np.empty(emptySizeSpatial,dtype=object)
    nopto_temporal_per_mean_mean  = np.empty(emptySizeTemporal,dtype=object)
    opto_temporal_per_mean_mean = np.empty(emptySizeTemporal,dtype=object)

    for roi_ind in range(roi_number):
        # Max Values
        nopto_spatial_per_max_max[roi_ind, :] = np.mean(noptoMaxes[roi_ind,:], axis = 1)
        opto_spatial_per_max_max[roi_ind, :] = np.mean(yoptoMaxes[roi_ind,:], axis = 1)

        nopto_temporal_per_max_max[roi_ind, :] = np.mean(noptoMaxes[roi_ind,:], axis = 0)
        opto_temporal_per_max_max[roi_ind, :] = np.mean(yoptoMaxes[roi_ind,:], axis = 0)

        # Mean Values
        nopto_spatial_per_mean_mean[roi_ind, :] = np.mean(noptoMeans[roi_ind,:], axis = 1)
        opto_spatial_per_mean_mean[roi_ind, :] = np.mean(yoptoMeans[roi_ind,:], axis = 1)

        nopto_temporal_per_mean_mean[roi_ind, :] = np.mean(noptoMeans[roi_ind,:], axis = 0)
        opto_temporal_per_mean_mean[roi_ind, :] = np.mean(yoptoMeans[roi_ind,:], axis = 0)

# Stats on the above
from scipy import stats as st

test_stat_temporal_max = np.empty(len(target_tf),dtype=object)
test_stat_temporal_mean = np.empty(len(target_tf),dtype=object)
test_stat_spatial_max = np.empty(len(target_sp),dtype=object)
test_stat_spatial_mean = np.empty(len(target_sp),dtype=object)

test_pvalue_temporal_max = np.empty(len(target_tf),dtype=object)
test_pvalue_temporal_mean = np.empty(len(target_tf),dtype=object)
test_pvalue_spatial_max = np.empty(len(target_sp),dtype=object)
test_pvalue_spatial_mean = np.empty(len(target_sp),dtype=object)

for sp_ind in range(len(target_sp)):
    print('sp_ind = '+ str(sp_ind))
    test_stat_spatial_max[sp_ind], test_pvalue_spatial_max[sp_ind] = \
    st.ttest_ind(a = nopto_spatial_per_max_max[:,sp_ind],
                 b = opto_spatial_per_max_max[:,sp_ind])
    test_stat_spatial_mean[sp_ind], test_pvalue_spatial_mean[sp_ind] = \
    st.ttest_ind(a = nopto_spatial_per_mean_mean[:,sp_ind],
                 b = opto_spatial_per_mean_mean[:,sp_ind])

for tf_ind in range(len(target_tf)):
    test_stat_temporal_max[tf_ind], test_pvalue_temporal_max[tf_ind] = \
    st.ttest_ind(a = nopto_temporal_per_max_max[:,tf_ind],
                 b = opto_temporal_per_max_max[:,tf_ind])
    test_stat_temporal_mean[tf_ind], test_pvalue_temporal_mean[tf_ind] = \
    st.ttest_ind(a = nopto_temporal_per_mean_mean[:,tf_ind],
                 b = opto_temporal_per_mean_mean[:,tf_ind])

# test_stat_temporal_max
# test_pvalue_temporal_max
#
# test_stat_temporal_mean
# test_pvalue_temporal_mean
#
# test_stat_spatial_max
# test_pvalue_spatial_max
#
# test_stat_spatial_mean
# test_pvalue_spatial_mean


# %% ToDo:

# Do paired t tests somehow. Across ROIs, one combo of TempFreq and SpatPer, opto or not differences2154


#%% Heatmaps of ROIs opto vs no opto
# heatmapFigName = save_path+'metrics/heatmaps/'+'test  '+str(file_name)+' | SeriesNumber'+str(series_number)+' | '+str(roi_name) + ' | ' + 'Heatmap' '.pdf'
# ma.plotOptoHeatmaps(ID, roi_data, saveName = heatmapFigName, dff = dff, df = df)
import numpy as np
roi_data['epoch_response'].shape
roi_data['epoch_response'][0, 0, :]
ID.getEpochParameters('current_opto_start_time')[0]
np.sum(np.array(ID.getEpochParameters('current_opto_start_time')) == 9.0)

epoch_response
ot = 2.
query = {'current_opto_start_time': ot, 'opto_stim': True}
filtered_trials = shared_analysis.filterTrials(roi_data['epoch_response'], ID, query=query)
filtered_trials.shape

voltage_trace, voltage_time_vector, voltage_sampling_rate = ID.getVoltageData()
voltage_trace.shape
plt.plot(voltage_trace[1, :])

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
print(run_parameters)
# specified run parameter
protocol_ID = ID.getRunParameters('protocol_ID')
print(protocol_ID)

# epoch_parameters: list of dicts of all epoch parameters, one for each epoch (trial)
epoch_parameters = ID.getEpochParameters()
print(epoch_parameters[0])
# Pass a param key to return a list of specified param values, one for each trial
current_opto_stim = ID.getEpochParameters('opto_stim')
current_opto_stim
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
