# collapse_nii_across_time 
# takes in a X, Y, Z, T nifty file and outputs collapsed across n_frames
# Handy for anatomical images

import nibabel as nib
import numpy as np

def collapse_nii(filepath, savepath, n_frames, ouptut_name):

    brain = np.asanyarray(nib.load(filepath).dataobj).astype('uint16')

    if n_frames == 'all':
        n_frames = brain.shape[-1]

    nib.save(nib.Nifti1Image(np.mean(brain[:, :, :, :n_frames], axis=-1), np.eye(4)), savepath+output_name)

# Set yo variables
filepath = '/Volumes/ABK2TBData/data_repo/bruker/20230223/TSeries-20230223-anat-004/TSeries-20230223-anat-004_channel_2.nii'
savepath = '/Volumes/ABK2TBData/data_repo/bruker/20230223/TSeries-20230223-anat-004/'
n_frames = 'all'
output_name = 'channel2'

# run the function
collapse_nii(filepath, savepath, n_frames, output_name)
