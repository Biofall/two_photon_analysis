# %%
import numpy as np
import os
import nibabel as nib



filename = 'TSeries-20221122-003_channel_1_aff.nii.gz'
data_dir = '/Users/mhturner/CurrentData/krieger/20221122-manual'


filepath = os.path.join(data_dir, filename)

# %%




brain = np.asanyarray(nib.load(filepath).dataobj).astype('uint16')

# %%

save_path = os.path.join(data_dir, 'TSeries-20221122-003_channel_1_aff_16.nii')

nib.save(nib.Nifti1Image(brain, np.eye(4)), save_path)

# %%
