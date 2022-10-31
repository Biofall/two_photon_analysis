# save_strfs.py
# author: Tim Currier; last updated: 2022-05-25
# saves numpy arrays and 2-color mp4 movies of the STRFs for all ROIs in a set
# array filter is 4 sec; movie of the first 2 sec at 1/2 real-time speed

# argumenmts: [1] date (yyyy-mm-dd); [2] series_number; [3] roi_set_name
# implementation: save_strfs.py 2022-03-17 1 roi_set_post

# import relevant packages
from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import sys
import cv2
import warnings
from tqdm import tqdm
from tifffile import imsave
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter

# disable runtime and deprecation warnings - dangerous! turn this off while working on the function
warnings.filterwarnings("ignore")

# define recording series to analyze
experiment_file_directory = '/Volumes/TimBigData/Bruker/StimData'
experiment_file_name = sys.argv[1]
series_number = int(sys.argv[2])
roi_set_name = sys.argv[3]

# join path to proper format for ImagingDataObject()
file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# create save directory
save_directory = '/Volumes/TimBigData/Bruker/STRFs/' + experiment_file_name + '/'
Path(save_directory).mkdir(exist_ok=True)

# create ImagingDataObject (wants a path to an hdf5 file and a series number from that file)
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

# get ROI timecourses and stimulus parameters
roi_data = ID.getRoiResponses(roi_set_name);
epoch_parameters = ID.getEpochParameters();
run_parameters = ID.getRunParameters();

# pull run parameters (same across all trials)
update_rate = run_parameters['update_rate']
rand_min = run_parameters['rand_min']
rand_max = run_parameters['rand_max']

# calculate size of noise grid, in units of patches (H,W)
output_shape = (int(np.floor(run_parameters['grid_height']/run_parameters['patch_size'])), int(np.floor(run_parameters['grid_width']/run_parameters['patch_size'])));

# calculate number of time points needed
n_frames = update_rate*(run_parameters['stim_time']);

# initialize array that will contain stimuli for all trials
all_stims = np.zeros(output_shape+(int(n_frames),int(run_parameters['num_epochs'])))

# populate all-trial stimulus array
for trial_num in range(1, int(run_parameters['num_epochs']+1)):
    # pull start_seed for trial
    start_seed = epoch_parameters[(trial_num-1)]['start_seed']
    # initialize stimulus frames variable with full idle color
    stim = np.full(output_shape+(int(n_frames),),run_parameters['idle_color'])
    # populate stim array (H,W,T) specifically during "stim time"
    for stim_ind in range(0, stim.shape[2]):
        # find time in sec at stim_ind
        t = stim_ind/update_rate;
        # define seed at each timepoint
        seed = int(round(start_seed + t*update_rate))
        np.random.seed(seed)
        # find random values for the current seed and write to pre-initialized stim array
        if run_parameters['rgb_texture']: # this variable tracks if a UV stim is being played
            # if this is a UV series, need to populate full [uv,g,b] stim data and subsample only the UV portion
            rand_values = np.random.choice([rand_min, (rand_min + rand_max)/2, rand_max], (output_shape+(3,)));
            stim[:,:,stim_ind] = rand_values[:,:,0];
        else:
            rand_values = np.random.choice([rand_min, (rand_min + rand_max)/2, rand_max], output_shape);
            stim[:,:,stim_ind] = rand_values;
    # save trial stimulus to all_stims(Height, Width, Time, Trial)
    all_stims[:,:,:,(trial_num-1)] = stim;

# define filter length in seconds, convert to samples
filter_length = 4;
filter_len = filter_length*run_parameters['update_rate'];

# iterate over ROIs to save STRFs and movies
print('Calculating STRFs...')
for roi_id in range(0, roi_data['epoch_response'].shape[0]):
    # initialize strf by trial array (H,W,T,Tr)
    roi_strf = np.zeros(output_shape+(int(filter_len),int(run_parameters['num_epochs'])));
    for trial_num in range(0, int(run_parameters['num_epochs'])):
        current_resp = roi_data['epoch_response'][roi_id,trial_num]
        # initialize strf and full time series for update rate of stimulus
        full_t = np.arange(run_parameters['pre_time'],run_parameters['stim_time'] + run_parameters['pre_time'],1 / run_parameters['update_rate'])
        strf = np.zeros(output_shape+(int(filter_len),))
        # linearly interpolate response to match stimulus timing
        full_resp = np.interp(full_t,roi_data['time_vector'],current_resp)
        resp_mean = np.mean(full_resp)
        resp_var = np.var(full_resp)
        # compute TRF for mean-subtracted stimulus and response; then compile STRF across patches
        n=all_stims.shape[2];
        ext_size=2*n-1
        fsize=2**np.ceil(np.log2(ext_size)).astype('int')
        for phi in range(0,output_shape[0]):
            for theta in range(0,output_shape[1]):
                patch_stim = all_stims[phi,theta,:,trial_num];
                patch_mean = np.mean(patch_stim)
                filter_fft = np.fft.fft(full_resp-resp_mean,fsize) * np.conj(np.fft.fft(patch_stim-patch_mean,fsize));
                filt = np.real(np.fft.ifft(filter_fft))[0:int(filter_len)];
                trf = np.flip(filt);
                strf[phi,theta,:] = trf;
        # add trial strf to roi_strf array
        roi_strf[:,:,:,trial_num] = strf;
    # compute mean STRF
    roi_mean_strf = np.mean(roi_strf,3);
    # flip again to have t=0 at beginning
    roi_mean_strf = np.flip(roi_mean_strf,2)
    # calculate std for each patch
    STRF_std = np.std(roi_strf,(2,3))
    # divide by std to generate z-scored STRF
    roi_mean_strf_z = np.zeros(roi_mean_strf.shape)
    for frame in range(0,roi_mean_strf.shape[2]):
        roi_mean_strf_z[:,:,frame] = roi_mean_strf[:,:,frame]/STRF_std
    # print(roi_mean_strf_z[:,:,1])
    # save mean, dc-subtracted (AND a z-scored) STRF
    np.save('/Volumes/TimBigData/Bruker/STRFs/' + experiment_file_name + '/' + experiment_file_name + '-' + roi_set_name + '_' + str(roi_id) + '_STRF.npy', roi_mean_strf)
    np.save('/Volumes/TimBigData/Bruker/STRFs/' + experiment_file_name + '/' + experiment_file_name + '-' + roi_set_name + '_' + str(roi_id) + '_STRFz.npy', roi_mean_strf_z)
    # oversample z-scored STRF in x and y so video looks better
    big_strf=roi_mean_strf_z.repeat(20,axis=0)
    bigger_strf=big_strf.repeat(20,axis=1)
    # oversample z-scored STRF in t so framerate can be reduced
    biggest_strf=bigger_strf.repeat(2,axis=2)
    # convert z-scored STRF to -1 to 1 scale (units are standard deviations)
    low_lim = -3
    high_lim = 3
    new_strf = ((biggest_strf - low_lim) * (2/(high_lim - low_lim))) - 1
    new_strf=np.where(new_strf>1,1,new_strf)
    new_strf=np.where(new_strf<-1,-1,new_strf)
    # make empty rgb array and populate with positive or negative values
    rgb_strf=np.zeros((new_strf.shape+(3,)))
    pos_strf=np.where(new_strf>0,new_strf,0)
    neg_strf=np.where(new_strf<0,new_strf*-1,0)
    rgb_strf[:,:,:,0]=1-(pos_strf*1)-(neg_strf*.3)
    rgb_strf[:,:,:,2]=1-(pos_strf*1)-(neg_strf*0)
    rgb_strf[:,:,:,1]=1-(pos_strf*.3)-(neg_strf*1)
    rgb_strf=np.where(rgb_strf>1,1,rgb_strf)
    # scale rgb_strf to 0-255
    rgb_strf = (rgb_strf*255).astype('uint8')
    # save multicolor video
    fps = 10
    video = cv2.VideoWriter('/Volumes/TimBigData/Bruker/STRFs/' + experiment_file_name + '/' + experiment_file_name + '-' + roi_set_name + '_' + str(roi_id) + '_movie.mp4', cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (new_strf.shape[1],new_strf.shape[0]))
    for frame_count in range(int(new_strf.shape[2]/2)):
        img = rgb_strf[:,:,frame_count,:]
        video.write(img)
    video.release()
print('Done.')
