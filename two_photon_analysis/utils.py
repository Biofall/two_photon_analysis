import numpy as np

# Function to randomly draw a value
def getRandVal(rand_min, rand_max, start_seed, update_rate, t):
  seed = int(round(start_seed + t*update_rate))
  np.random.seed(seed)
  return np.random.choice([rand_min, (rand_min + rand_max)/2, rand_max], size=1)[0]

# FFT frequency sample points
def plotFFTFilter(filter_fft, ideal_frame_rate):
    freq = np.fft.fftfreq(n=len(filter_fft), d=1/ideal_frame_rate)

    # Filter power spectrum
    fh, ax = plt.subplots(1, 1, figsize=(6, 3))
    # FFT shift before visualizing
    ax.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(filter_fft))**2)
    # ax.plot(np.fft.fftshift(freq), np.fft.fftshift(filter_fft))

    ax.set_xlim([0, 60])
    # ax.set_yscale('log')
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('FFT Filter baby')

def dropTrials(roi_trfs, n_trials_to_drop):
    # multiply by 2 for no opto+opto pairing
    double_drop = n_trials_to_drop*2
    dropped_roi_trfs = roi_trfs[:, :, double_drop:]

    return dropped_roi_trfs
    print(f'{double_drop} trials dropped!')


# % Lil functin to test the voltage sampling
# fh, ax = plt.subplots(1, 1, figsize=(12, 4))
# ax.plot(voltage_trace[0, :])
# voltage_sampling_rate
# ax.set_xlim([0,200000])
#
#
# ID.getStimulusTiming().keys()
# stimulus_start_times = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times']
#
# opto_on = []
# opto_off = []
# voltage_time_vector[-1]
# # epoch_time in seconds
# epoch_time = ID.getRunParameters('pre_time') + ID.getRunParameters('stim_time') + ID.getRunParameters('tail_time')
# epoch_len = epoch_time * voltage_sampling_rate  # sec -> data points of voltage trace
#
# opto_traces = []
# no_opto_traces = []
# for ss_ind, ss in enumerate(stimulus_start_times):
#     start_index = np.where(voltage_time_vector > (ss - ID.getRunParameters('pre_time')))[0][0]
#     trial_voltage = voltage_trace[0, start_index:np.int32(start_index+epoch_len)]
#     if ID.getEpochParameters('opto_stim')[ss_ind]:
#         opto_traces.append(trial_voltage)
#     else:
#         no_opto_traces.append(trial_voltage)
#
# opto_traces = np.vstack(opto_traces)
# no_opto_traces = np.vstack(no_opto_traces)
#
# np.max(opto_traces, axis=1)
#
# fh, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].plot(opto_traces.T, 'k')
# ax[1].plot(no_opto_traces.T, 'r')

# Plotting functions

def config_matplotlib():
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Helvetica'})
    
def clean_axes(ax):
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
